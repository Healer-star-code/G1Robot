#!/usr/bin/env python3
"""
RealSense 摄像头 + YOLO Pose 测试脚本
自动检测 GUI 环境，SSH 下自动切换到无头模式
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['XDG_SESSION_TYPE'] = 'x11'

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

def check_gui_available():
    """检查是否可以显示 GUI 窗口"""
    # 检查是否有 DISPLAY 环境变量
    if 'DISPLAY' not in os.environ:
        return False
    
    # 尝试创建一个测试窗口
    try:
        cv2.namedWindow("test", cv2.WINDOW_HIDDEN)
        cv2.destroyWindow("test")
        return True
    except Exception:
        return False

def main():
    print("=" * 60)
    print("  📷 RealSense + YOLO Pose 测试")
    print("=" * 60)

    # 检测运行环境
    FORCE_GUI = True  # 设置为 True 强制显示窗口
    if FORCE_GUI:
        HAS_GUI = True
        print("✅ 强制 GUI 模式，启用窗口显示")
    else:
        HAS_GUI = check_gui_available()
        if HAS_GUI:
            print("✅ 检测到 GUI 环境，启用窗口显示")
        else:
            print("ℹ️  无头模式运行（SSH 或无显示环境）")
            print("   每 2 秒输出一次检测状态")
    print()

    # 检查设备
    ctx = rs.context()
    devices = ctx.devices
    if len(devices) == 0:
        print("❌ 未找到 RealSense 设备")
        print("请检查:")
        print("  1. USB 3.0 接口连接")
        print("  2. lsusb | grep -i camera")
        return
    else:
        for dev in devices:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            print(f"✅ 找到设备: {name} (SN: {serial})")

    print("\n[1/3] 加载 YOLO 模型...")
    model = YOLO("yolov8n-pose.pt")
    print("✅ 模型加载完成")

    print("\n[2/3] 启动 RealSense 摄像头...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(config)
        print("✅ 摄像头启动成功")
    except Exception as e:
        print(f"❌ 摄像头启动失败: {e}")
        print("\n可能原因:")
        print("  1. 摄像头被其他程序占用")
        print("  2. 尝试重新插拔 USB")
        return

    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

    print("\n[3/3] 开始采集...")
    if HAS_GUI:
        print("按 ESC 或 q 退出")
    else:
        print("按 Ctrl+C 退出")
    print("-" * 50)

    frame_count = 0
    pose_detected_count = 0
    fps = 0
    t0 = cv2.getTickCount()
    last_print = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_color = np.asanyarray(colorizer.colorize(depth_frame).get_data())

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

            # 计算 FPS
            if frame_count % 30 == 0:
                t = cv2.getTickCount()
                fps = 30 * cv2.getTickFrequency() / (t - t0)
                t0 = t

            if HAS_GUI:
                annotated = results[0].plot()
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                status = "POSE OK" if has_pose else "searching..."
                cv2.putText(annotated, status, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 0) if has_pose else (0, 128, 255), 1)
                combo = np.hstack([annotated, depth_color])
                cv2.imshow("Camera Test - RealSense + YOLO Pose", combo)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
            else:
                # 无头模式：每2秒打印一次状态
                if time.time() - last_print > 2.0:
                    rate = pose_detected_count / max(1, frame_count) * 100
                    status = "✅ 检测到人体" if has_pose else "⚪ 搜寻中..."
                    print(f"  帧 {frame_count:4d} | FPS: {fps:4.1f} | {status} | 检出率: {rate:3.0f}%")
                    last_print = time.time()
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        pipeline.stop()
        if HAS_GUI:
            cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("  📊 测试统计")
        print("=" * 60)
        print(f"  总帧数:    {frame_count}")
        print(f"  姿态检出:  {pose_detected_count} 帧 ({pose_detected_count/max(1,frame_count)*100:.1f}%)")
        print(f"  平均 FPS:  {fps:.1f}")
        print()
        print("  ✅ 摄像头 + YOLO 姿态检测工作正常！")
        print()

if __name__ == "__main__":
    main()
