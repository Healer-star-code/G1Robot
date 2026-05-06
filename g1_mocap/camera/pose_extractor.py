"""
RealSense + YOLO Pose 人体 3D 关键点提取
========================================
替代 MediaPipe，解决 Python 3.12 兼容性问题
适配: RealSense D435i/D405 深度相机 + YOLOv8 Pose
"""

import time
import numpy as np
import cv2
from ultralytics import YOLO
import pyrealsense2 as rs


class RealSensePoseExtractor:
    """
    使用 Intel RealSense 深度相机 + YOLOv8 Pose
    提取 17 个人体关键点的 3D 坐标（单位: 米）

    YOLO Pose 关键点索引 (与 MediaPipe 兼容映射):
         0: 鼻尖
         5: 左肩    6: 右肩
         7: 左肘    8: 右肘
         9: 左腕    10: 右腕
        11: 左髋    12: 右髋
        13: 左膝    14: 右膝
        15: 左踝    16: 右踝

    注意: YOLO 只有 17 个关键点，少于 MediaPipe 的 33 个
    关键点编号直接映射到 MediaPipe 的: 0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
    """

    # YOLO 索引 -> MediaPipe 索引 映射
    YOLO_TO_MP = {
        0: 0,    # 鼻尖
        5: 11,   # 左肩
        6: 12,   # 右肩
        7: 13,   # 左肘
        8: 14,   # 右肘
        9: 15,   # 左腕
        10: 16,  # 右腕
        11: 23,  # 左髋
        12: 24,  # 右髋
        13: 25,  # 左膝
        14: 26,  # 右膝
        15: 27,  # 左踝
        16: 28,  # 右踝
    }

    def __init__(self,
                 width=640,
                 height=480,
                 fps=30,
                 serial=None,
                 complexity=2,
                 detection_conf=0.5,
                 tracking_conf=0.5):
        """
        参数:
            width, height: 彩色图像分辨率
            fps: 目标帧率
            serial: RealSense 序列号，None 表示自动选择
            complexity: 兼容参数（YOLO 自动忽略）
            detection_conf: 检测置信度阈值
            tracking_conf: 兼容参数（YOLO 自动忽略）
        """
        # ---- RealSense 初始化 ----
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height,
                             rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height,
                             rs.format.z16, fps)
        profile = self.pipeline.start(config)

        # 获取深度内参
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # m/unit
        depth_intrin = profile.get_stream(
            rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.fx = depth_intrin.fx
        self.fy = depth_intrin.fy
        self.cx = depth_intrin.ppx
        self.cy = depth_intrin.ppy
        self.img_w = width
        self.img_h = height

        # 深度与彩色对齐
        self.align = rs.align(rs.stream.color)

        # 关闭激光发射器（纯被动深度，适用于近距离人体）
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)

        # ---- YOLO Pose 初始化 ----
        print("[pose] Loading YOLOv8 pose model...")
        self.model = YOLO("yolov8n-pose.pt")
        self.detection_conf = detection_conf
        print("[pose] YOLO model ready.")

        # 跳过几帧让相机自动曝光稳定
        for _ in range(30):
            self.pipeline.wait_for_frames()

        self._frame_count = 0
        self._last_time = time.time()
        self._fps_display = 0.0
        self._annotated_frame = None  # 缓存最近一帧的标注图像
        self._connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 12), (5, 11), (6, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]

    def _pixel_to_3d(self, px, py, depth_m):
        """像素坐标 → 相机坐标系 3D 坐标 (单位: 米)

        RealSense 坐标系:
            X: 右, Y: 下, Z: 前
        此处转换为更直观的:
            X: 右, Y: 上, Z: 前
        """
        z = depth_m
        x = (px - self.cx) * z / self.fx
        y = -(py - self.cy) * z / self.fy  # 翻转 Y 向上
        return np.array([x, y, z], dtype=np.float32)

    def get_landmarks_3d(self):
        """
        获取 33 个人体关键点的 3D 坐标，同时缓存标注帧供 get_annotated_frame() 使用
        （返回数组大小为 33 以保持与 MediaPipe 版本的接口兼容，未使用的关键点设为 0）

        返回:
            success:  是否成功检测到人体
            landmarks_3d:  (33, 3) numpy float32 数组，单位米
            timestamp:  时间戳
        """
        success = False
        landmarks_3d = np.zeros((33, 3), dtype=np.float32)
        timestamp = time.time()

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                self._annotated_frame = None
                return success, landmarks_3d, timestamp

            # YOLO 推理
            image = np.asanyarray(color_frame.get_data())
            results = self.model(image, verbose=False, conf=self.detection_conf)

            # ---- 构建标注帧（缓存） ----
            annotated = image.copy()

            if len(results) > 0 and results[0].keypoints is not None:
                kps = results[0].keypoints.data
                if len(kps) > 0:
                    kp = kps[0].cpu().numpy()  # shape: (17, 3) - x, y, confidence

                    # 绘制骨架连接线
                    for i, j in self._connections:
                        conf_i, conf_j = kp[i, 2], kp[j, 2]
                        if conf_i > self.detection_conf and conf_j > self.detection_conf:
                            pt1 = (int(kp[i, 0]), int(kp[i, 1]))
                            pt2 = (int(kp[j, 0]), int(kp[j, 1]))
                            cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)

                    # 绘制关键点
                    for i in range(17):
                        conf = kp[i, 2]
                        if conf > self.detection_conf:
                            pt = (int(kp[i, 0]), int(kp[i, 1]))
                            cv2.circle(annotated, pt, 3, (0, 128, 255), -1)

                    # 获取深度图并计算 3D 坐标
                    depth_np = np.asanyarray(depth_frame.get_data())

                    for yolo_idx in range(17):
                        conf = kp[yolo_idx, 2]
                        if conf > self.detection_conf:
                            px = int(round(kp[yolo_idx, 0]))
                            py = int(round(kp[yolo_idx, 1]))

                            # 边界检查
                            px = max(0, min(px, self.img_w - 1))
                            py = max(0, min(py, self.img_h - 1))

                            # 中值滤波获取深度
                            win = 2
                            u0, v0 = max(0, px - win), max(0, py - win)
                            u1, v1 = min(self.img_w, px + win + 1), min(self.img_h, py + win + 1)
                            patch = depth_np[v0:v1, u0:u1].reshape(-1)
                            patch = patch[patch > 0]
                            if patch.size > 0:
                                depth_m = float(np.median(patch) * self.depth_scale)
                                if depth_m > 0.1 and depth_m < 5.0:
                                    mp_idx = self.YOLO_TO_MP.get(yolo_idx, yolo_idx)
                                    if mp_idx < 33:
                                        landmarks_3d[mp_idx] = self._pixel_to_3d(px, py, depth_m)

                    success = True

            self._annotated_frame = annotated
            return success, landmarks_3d, timestamp

        except Exception as e:
            print(f"[pose] Frame error: {e}")
            self._annotated_frame = None
            return success, landmarks_3d, timestamp

    def get_annotated_frame(self):
        """返回最近一帧的标注图像"""
        return self._annotated_frame
