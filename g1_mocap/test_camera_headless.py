#!/usr/bin/env python3
"""
RealSense 摄像头 + YOLO Pose 无 GUI 测试脚本
运行 5 秒后自动退出，输出统计结果
"""

import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

def main():
    print("=" * 60)
    print("  📷 RealSense + YOLO Pose 无 GUI 测试")
    print("=" * 60)

    # 检查设备
    ctx = rs.context()
    devices = ctx.devices
    if len(devices) == 0:
        print("❌ 未找到 RealSense 设备")
        return False
    else:
        for dev in devices:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            print(f"✅ 找到设备: {name}")
            print(f"   序列号: {serial}")

    print("\n[1/4] 加载 YOLO 模型...")
    model = YOLO("yolov8n-pose.pt")
    print("✅ 模型加载完成")

    print("\n[2/4] 启动 RealSense 摄像头...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(config)
        print("✅ 摄像头启动成功")
    except Exception as e:
        print(f"❌ 摄像头启动失败: {e}")
        return False

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"   深度缩放系数: {depth_scale} m/unit")

    print("\n[3/4] 开始采集 5 秒钟...")

    frame_count = 0
    pose_detected_count = 0
    t0 = time.time()
    test_duration = 5.0

    try:
        while time.time() - t0 < test_duration:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())

            # YOLO 姿态检测
            results = model(color, verbose=False)

            frame_count += 1
            has_pose = False

            if len(results) > 0 and results[0].keypoints is not None:
                kps = results[0].keypoints.data
                if len(kps) > 0:
                    kp = kps[0].cpu().numpy()
                    visible = np.sum(kp[:, 2] > 0.5)
                    if visible > 5:
                        pose_detected_count += 1
                        has_pose = True

            if frame_count % 10 == 0:
                status = "✅" if has_pose else "⚪"
                elapsed = time.time() - t0
                print(f"   帧 {frame_count:3d} | 姿态: {status} | 已运行: {elapsed:.1f}s")

    except Exception as e:
        print(f"❌ 采集出错: {e}")
        return False
    finally:
        pipeline.stop()

    print("\n[4/4] 测试结果统计")
    print("-" * 40)
    elapsed = time.time() - t0
    fps = frame_count / elapsed

    print(f"   总运行时间:  {elapsed:.2f} 秒")
    print(f"   处理帧数:    {frame_count} 帧")
    print(f"   平均 FPS:    {fps:.1f}")
    print(f"   检测到姿态:  {pose_detected_count} 帧")
    print(f"   姿态检出率:  {pose_detected_count/max(1,frame_count)*100:.1f}%")

    print("\n" + "=" * 60)
    if pose_detected_count > 0 and fps > 10:
        print("  ✅✅✅ 摄像头 + 姿态检测全部正常！")
    else:
        print("  ⚠️  基本正常，可调整检测距离和角度")
    print("=" * 60)

    return True

if __name__ == "__main__":
    main()
