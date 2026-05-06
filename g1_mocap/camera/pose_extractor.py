"""
RealSense + MediaPipe 人体 3D 关键点提取
========================================
参考: MonkeySee_MonkeyDo 的 camera_tracker.py（OAK-D + MediaPipe）
      Nao_Mimc 的 Pose_est.py（OpenCV + MediaPipe）
适配: RealSense D435i/D405 深度相机 + MediaPipe Pose
"""

import time
import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs


class RealSensePoseExtractor:
    """
    使用 Intel RealSense 深度相机 + MediaPipe Pose
    提取 33 个人体关键点的 3D 坐标（单位: 米）

    MediaPipe Pose 关键点索引（部分）:
         0: 鼻尖
        11: 左肩    12: 右肩
        13: 左肘    14: 右肘
        15: 左腕    16: 右腕
        23: 左髋    24: 右髋
        25: 左膝    26: 右膝
        27: 左踝    28: 右踝
        29: 左脚跟  30: 右脚跟
        31: 左脚尖  32: 右脚尖
    """

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
            complexity: MediaPipe 模型复杂度 (0/1/2)
            detection_conf: 检测置信度阈值
            tracking_conf: 跟踪置信度阈值
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
        # 如需远距离可注释掉这两行
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)

        # ---- MediaPipe Pose 初始化 ----
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=complexity,
            smooth_landmarks=True,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )

        # 跳过几帧让相机自动曝光稳定
        for _ in range(30):
            self.pipeline.wait_for_frames()

        self._frame_count = 0
        self._last_time = time.time()
        self._fps_display = 0.0
        self._annotated_frame = None  # 缓存最近一帧的标注图像

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

        返回:
            success:  是否成功检测到人体
            landmarks_3d:  (33, 3) numpy float32 数组，单位米
            timestamp:  时间戳
        """
        success = False
        landmarks_3d = None
        timestamp = time.time()

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                self._annotated_frame = None
                return success, landmarks_3d, timestamp

            # MediaPipe 推理
            image = np.asanyarray(color_frame.get_data())
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            # ---- 构建标注帧（缓存） ----
            annotated = image.copy()
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated, results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(
                        color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # 像素 → 3D 转换
                landmarks_3d = np.zeros((33, 3), dtype=np.float32)
                h, w = self.img_h, self.img_w

                for i, lm in enumerate(results.pose_landmarks.landmark):
                    px = int(np.clip(lm.x * w, 0, w - 1))
                    py = int(np.clip(lm.y * h, 0, h - 1))
                    depth_mm = depth_frame.get_distance(px, py)
                    depth_m = depth_mm  # pyrealsense2 get_distance 返回米

                    landmarks_3d[i] = self._pixel_to_3d(px, py, depth_m)

                success = True
                self._frame_count += 1

            # FPS
            now = time.time()
            dt = now - self._last_time
            if dt > 0.5:
                self._fps_display = self._frame_count / dt
                self._frame_count = 0
                self._last_time = now
            cv2.putText(annotated, f"FPS: {self._fps_display:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            self._annotated_frame = annotated

        except Exception as e:
            print(f"[WARN] 帧处理异常: {e}")
            self._annotated_frame = None

        return success, landmarks_3d, timestamp

    def get_annotated_frame(self):
        """获取最近一帧带骨骼标注的彩色图像（与 get_landmarks_3d() 共用同一帧，无额外推理）"""
        return self._annotated_frame

    def stop(self):
        """释放资源"""
        self.pose.close()
        self.pipeline.stop()


# ============================================================
# 测试独立运行
# ============================================================
if __name__ == "__main__":
    print("[TEST] 测试 RealSense + MediaPipe 姿态提取...")
    extractor = RealSensePoseExtractor()

    try:
        while True:
            success, landmarks, ts = extractor.get_landmarks_3d()

            # 显示标注帧
            frame = extractor.get_annotated_frame()
            if frame is not None:
                status = "DETECTED" if success else "NO PERSON"
                color = (0, 255, 0) if success else (0, 0, 255)
                cv2.putText(frame, status, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, color, 2)
                cv2.imshow("G1 Motion Capture", frame)

            # 打印关键点（调试）
            if success:
                print(f"\r 左腕: ({landmarks[15][0]:.2f}, {landmarks[15][1]:.2f}, {landmarks[15][2]:.2f})"
                      f"  右腕: ({landmarks[16][0]:.2f}, {landmarks[16][1]:.2f}, {landmarks[16][2]:.2f})",
                      end="")

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        extractor.stop()
        cv2.destroyAllWindows()
        print("\n[TEST] 测试结束")
