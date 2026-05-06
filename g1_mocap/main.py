"""
G1 实时动作映射主程序
=====================
Intel RealSense + MediaPipe Pose → G1 全身关节映射

管线:
  相机采集(3D关键点) → 关节映射(下肢+腰部) + IK求解(上肢) → 平滑滤波 → G1 DDS指令

键盘控制:
  ESC   - 安全退出 (发送阻尼指令)
  SPACE - 暂停/恢复 映射
  L     - 切换下肢映射 开/关
  W     - 切换腰部映射 开/关
  U     - 切换上肢映射 开/关
  S     - 切换 Kp 配置 (safe/default)
"""

import sys
import os
import time
import math
import argparse
import numpy as np
import cv2

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    NETWORK_INTERFACE, G1_IP,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_SERIAL,
    MEDIAPIPE_COMPLEXITY, MEDIAPIPE_DETECTION_CONFIDENCE, MEDIAPIPE_TRACKING_CONFIDENCE,
    CONTROL_FREQUENCY, SMOOTHING_ALPHA, DEAD_ZONE,
    ENABLE_LOWER_BODY, ENABLE_WAIST, ENABLE_UPPER_BODY,
    KP_SAFE, KD_SAFE, KP_DEFAULT, KD_DEFAULT,
    JOINT_NAMES,
)

from camera.pose_extractor import RealSensePoseExtractor
from mapping.joint_mapper import JointMapper
from mapping.arm_ik import ArmIK
from robot.g1_controller import G1Controller


class MotionCaptureApp:
    """G1 动作映射主应用"""

    def __init__(self, args):
        self.args = args

        # ---- 运行状态 ----
        self.running = True
        self.paused = False
        self.loss_count = 0       # 连续丢失人体帧数
        self.max_loss_frames = 30  # 超过此帧数执行保护策略
        self.use_safe_kp = True   # 当前使用 safe 还是 default Kp

        # ---- 映射开关 (运行时可变) ----
        self.enable_lower = args.enable_lower_body if args.enable_lower_body is not None else ENABLE_LOWER_BODY
        self.enable_waist = ENABLE_WAIST
        self.enable_upper = ENABLE_UPPER_BODY

        # ---- 统计 ----
        self.frame_count = 0
        self.start_time = time.time()
        self.latency_sum = 0.0
        self.loop_period = 1.0 / CONTROL_FREQUENCY

        print("=" * 60)
        print("  G1 Motion Capture — RealSense + MediaPipe → G1")
        print("=" * 60)
        print(f"  控制频率:     {CONTROL_FREQUENCY} Hz")
        print(f"  平滑系数:     {SMOOTHING_ALPHA}")
        print(f"  死区阈值:     {DEAD_ZONE} rad")
        print(f"  下肢映射:     {'开' if self.enable_lower else '关'}")
        print(f"  腰部映射:     {'开' if self.enable_waist else '关'}")
        print(f"  上肢映射:     {'开' if self.enable_upper else '关'}")
        print(f"  IK 模式:      {args.ik_mode}")
        print(f"  URDF 路径:    {args.urdf or '(未指定)'}")
        print(f"  模拟模式:     {'开 (不连机器人)' if args.sim else '关'}")
        print()

        # ---- 初始化模块 ----
        self._init_camera()
        self._init_mapping(args)
        self._init_controller()

        # ---- 状态变量 ----
        self.smoothed_q = np.zeros(29, dtype=np.float32)  # 平滑后的目标
        self.last_sent_q = np.zeros(29, dtype=np.float32)  # 上次发送的指令

    def _init_camera(self):
        """初始化 RealSense + MediaPipe"""
        print("[INIT] 初始化相机...")
        try:
            self.extractor = RealSensePoseExtractor(
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=CAMERA_FPS,
                serial=CAMERA_SERIAL,
                complexity=MEDIAPIPE_COMPLEXITY,
                detection_conf=MEDIAPIPE_DETECTION_CONFIDENCE,
                tracking_conf=MEDIAPIPE_TRACKING_CONFIDENCE,
            )
            print("[INIT] 相机就绪")
        except Exception as e:
            print(f"[ERROR] 相机初始化失败: {e}")
            sys.exit(1)

    def _init_mapping(self, args):
        """初始化关节映射 + 手臂 IK"""
        self.joint_mapper = JointMapper()
        self.arm_ik = ArmIK(urdf_path=args.urdf, mode=args.ik_mode)

    def _init_controller(self):
        """初始化 G1 DDS 控制器"""
        if self.args.sim:
            print("[INIT] 模拟模式，跳过控制器连接")
            self.controller = None
            return

        kp = KP_SAFE if self.use_safe_kp else KP_DEFAULT
        kd = KD_SAFE if self.use_safe_kp else KD_DEFAULT

        print("[INIT] 连接 G1 机器人...")
        try:
            self.controller = G1Controller(
                network_interface=NETWORK_INTERFACE,
                kp=kp,
                kd=kd,
            )
            self.controller.start()
            print("[INIT] G1 控制器就绪")
        except Exception as e:
            print(f"[ERROR] 控制器连接失败: {e}")
            resp = input("是否以模拟模式继续? (y/n): ").strip().lower()
            if resp == 'y':
                self.controller = None
                print("[INFO] 进入模拟模式")
            else:
                sys.exit(1)

    # ================================================================
    #  平滑 & 滤波
    # ================================================================
    def apply_smoothing(self, q_raw: np.ndarray) -> np.ndarray:
        """指数滑动平均 (EMA) 滤波"""
        if self.frame_count == 0:
            self.smoothed_q = q_raw.copy()
        else:
            self.smoothed_q = (SMOOTHING_ALPHA * q_raw +
                               (1.0 - SMOOTHING_ALPHA) * self.smoothed_q)
        return self.smoothed_q

    def apply_deadzone(self, q_target: np.ndarray) -> np.ndarray:
        """死区过滤：微小变化不发送"""
        q_filtered = q_target.copy()
        for i in range(29):
            diff = abs(q_target[i] - self.last_sent_q[i])
            if diff < DEAD_ZONE:
                q_filtered[i] = self.last_sent_q[i]
        return q_filtered

    # ================================================================
    #  主循环单步
    # ================================================================
    def step(self, landmarks_3d):
        """处理一帧人体关键点 → 输出 29 关节目标"""
        t0 = time.time()

        # 1. 手臂 IK
        left_arm = np.zeros(7, dtype=np.float32)
        right_arm = np.zeros(7, dtype=np.float32)
        if self.enable_upper:
            left_arm, right_arm = self.arm_ik.solve_both_arms(landmarks_3d)

        # 2. 组装全部关节
        q_raw = self.joint_mapper.get_all_joint_targets(
            landmarks_3d,
            left_arm, right_arm,
            enable_lower_body=self.enable_lower,
            enable_waist=self.enable_waist,
            enable_upper_body=self.enable_upper,
        )

        # 3. 平滑
        q_smooth = self.apply_smoothing(q_raw)

        # 4. 死区
        q_out = self.apply_deadzone(q_smooth)

        self.last_sent_q = q_out

        t1 = time.time()
        self.latency_sum += (t1 - t0) * 1000  # ms

        return q_out

    # ================================================================
    #  主循环
    # ================================================================
    def run(self):
        """主控制循环"""
        print("[INFO] 开始运行...")
        print("       按 ESC 退出  |  空格 暂停/恢复  |  L/W/U 切换映射  |  S 切换刚度")
        print()

        cv2.namedWindow("G1 Motion Capture", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("G1 Motion Capture", 960, 720)

        try:
            while self.running:
                loop_start = time.time()

                # ---- 键盘处理 ----
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key)

                # ---- 获取人体关键点 ----
                success, landmarks_3d, ts = self.extractor.get_landmarks_3d()

                # ---- 获取预览画面 ----
                frame = self.extractor.get_annotated_frame()
                if frame is not None:
                    self._draw_hud(frame, success)
                    cv2.imshow("G1 Motion Capture", frame)

                # ---- 暂停时保持画面 ----
                if self.paused:
                    self._sleep_remainder(loop_start)
                    continue

                # ---- 人体丢失处理 ----
                if not success:
                    self.loss_count += 1
                    if self.loss_count >= self.max_loss_frames:
                        print(f"\r[WARN] 人体丢失 {self.loss_count} 帧，保持最后位姿", end="")
                        # 不重置关节，保持最后位置
                        if self.controller:
                            self.controller.send_command(self.last_sent_q)
                    self._sleep_remainder(loop_start)
                    continue

                self.loss_count = 0
                self.frame_count += 1

                # ---- 解算 → 发送 ----
                q_target = self.step(landmarks_3d)

                if self.controller and self.controller._running:
                    self.controller.send_command(q_target)

                # ---- 延迟日志（每100帧） ----
                if self.frame_count % 100 == 0:
                    avg_latency = self.latency_sum / max(self.frame_count, 1)
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / max(elapsed, 0.001)
                    print(f"\r[STAT] FPS={fps:.1f} | "
                          f"延迟={avg_latency:.1f}ms | "
                          f"下肢={'ON' if self.enable_lower else 'OFF'} | "
                          f"腰部={'ON' if self.enable_waist else 'OFF'} | "
                          f"上肢={'ON' if self.enable_upper else 'OFF'} | "
                          f"Kp={'SAFE' if self.use_safe_kp else 'DEFAULT'}",
                          end="")

                # ---- 帧率控制 ----
                self._sleep_remainder(loop_start)

        except KeyboardInterrupt:
            print("\n[INFO] 收到中断信号")
        except Exception as e:
            print(f"\n[ERROR] 运行异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def _handle_key(self, key):
        """键盘事件处理"""
        if key == 27:  # ESC
            self.running = False
        elif key == 32:  # SPACE
            self.paused = not self.paused
            status = "暂停" if self.paused else "恢复"
            print(f"\r[INFO] {status}                                 ", end="")
        elif key == ord('l') or key == ord('L'):
            self.enable_lower = not self.enable_lower
            print(f"\r[INFO] 下肢映射: {'ON' if self.enable_lower else 'OFF'}        ", end="")
        elif key == ord('w') or key == ord('W'):
            self.enable_waist = not self.enable_waist
            print(f"\r[INFO] 腰部映射: {'ON' if self.enable_waist else 'OFF'}        ", end="")
        elif key == ord('u') or key == ord('U'):
            self.enable_upper = not self.enable_upper
            print(f"\r[INFO] 上肢映射: {'ON' if self.enable_upper else 'OFF'}        ", end="")
        elif key == ord('s') or key == ord('S'):
            self.use_safe_kp = not self.use_safe_kp
            if self.controller:
                self.controller.Kp = np.array(KP_DEFAULT if not self.use_safe_kp else KP_SAFE, dtype=np.float32)
                self.controller.Kd = np.array(KD_DEFAULT if not self.use_safe_kp else KD_SAFE, dtype=np.float32)
            print(f"\r[INFO] Kp: {'SAFE' if self.use_safe_kp else 'DEFAULT'}        ", end="")

    def _draw_hud(self, frame, person_detected):
        """在预览画面上绘制状态信息"""
        # 状态文字
        if self.paused:
            cv2.putText(frame, "PAUSED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        status = "DETECTED" if person_detected else "NO PERSON"
        color = (0, 255, 0) if person_detected else (0, 0, 255)
        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 映射状态
        y = 90
        for label, enabled in [
            ("Lower Body", self.enable_lower),
            ("Waist", self.enable_waist),
            ("Upper Body", self.enable_upper),
        ]:
            c = (0, 255, 0) if enabled else (100, 100, 100)
            cv2.putText(frame, f"{label}: {'ON' if enabled else 'OFF'}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1)
            y += 22

        # Kp 模式
        kp_label = f"Kp: {'SAFE' if self.use_safe_kp else 'DEFAULT'}"
        cv2.putText(frame, kp_label, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20

        if self.controller is None:
            cv2.putText(frame, "SIM MODE", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # 实时关节角度（右侧栏）
        if person_detected and hasattr(self, 'last_sent_q'):
            x0 = frame.shape[1] - 240
            y0 = 10
            cv2.putText(frame, "Joint Angles (deg):", (x0, y0 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            # 仅显示上肢 14 个关节
            for i in range(15, 29):
                deg = math.degrees(self.last_sent_q[i])
                name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"J{i}"
                row = i - 15
                cv2.putText(frame, f"{name}: {deg:6.1f}",
                            (x0, y0 + 30 + row * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (200, 200, 200), 1)

    def _sleep_remainder(self, loop_start):
        """帧率控制：补足剩余时间"""
        elapsed = time.time() - loop_start
        remaining = self.loop_period - elapsed
        if remaining > 0.001:
            time.sleep(remaining)

    def cleanup(self):
        """安全清理"""
        print("\n[INFO] 正在安全关闭...")

        if self.controller:
            self.controller.stop()

        if hasattr(self, 'extractor'):
            self.extractor.stop()

        cv2.destroyAllWindows()

        # 统计
        elapsed = time.time() - self.start_time
        if self.frame_count > 0:
            avg_latency = self.latency_sum / self.frame_count
            fps = self.frame_count / max(elapsed, 0.001)
            print(f"[STAT] 运行时间:  {elapsed:.1f}s")
            print(f"[STAT] 总帧数:    {self.frame_count}")
            print(f"[STAT] 平均 FPS:  {fps:.1f}")
            print(f"[STAT] 平均延迟:  {avg_latency:.1f}ms")

        print("[INFO] 程序已退出")


def parse_args():
    parser = argparse.ArgumentParser(
        description="G1 Motion Capture — RealSense + MediaPipe → G1 Robot"
    )
    parser.add_argument("--sim", action="store_true",
                        help="模拟模式 (不连接机器人)")
    parser.add_argument("--ik-mode", type=str, default="analytical",
                        choices=["analytical", "pinocchio"],
                        help="IK 求解模式 (默认: analytical)")
    parser.add_argument("--urdf", type=str, default=None,
                        help="G1 URDF 模型路径 (pinocchio 模式需要)")
    parser.add_argument("--enable-lower-body", type=lambda x: x.lower() == 'true',
                        default=None,
                        help="是否启用下肢映射 (true/false)")
    parser.add_argument("--interface", type=str, default=None,
                        help="网络接口名称 (覆盖 config.py 设置)")
    return parser.parse_args()


# ================================================================
#  入口
# ================================================================
if __name__ == "__main__":
    args = parse_args()

    if args.interface:
        import config
        config.NETWORK_INTERFACE = args.interface

    app = MotionCaptureApp(args)
    app.run()
