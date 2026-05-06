"""
手臂逆运动学（Pinocchio）
=========================
参考: MonkeySee_MonkeyDo 的 OCRA 优化算法（rx200_kinematics.py）
      使用 URDF 模型 + Pinocchio 进行 IK 求解

方法: 阻尼最小二乘法 (Damped Least Squares)
      以腕部 3D 位置为目标，从肩关节坐标系出发求解 7-DOF 手臂

简化版实现（不依赖 Pinocchio 时使用解析法）
"""

import numpy as np


class ArmIK:
    """
    G1 手臂逆运动学求解器

    提供两种模式:
      1. pinocchio 模式: 精确 IK（需要安装 pinocchio）
      2. analytical 模式: 解析近似（不需要 pinocchio，作为 fallback）
    """

    def __init__(self, urdf_path=None, mode="analytical"):
        """
        参数:
            urdf_path: G1 URDF 模型路径（pinocchio 模式需要）
            mode: "pinocchio" 或 "analytical"
        """
        self.mode = mode
        self._pin_model = None
        self._pin_data = None
        self._left_frame_id = None
        self._right_frame_id = None

        if mode == "pinocchio":
            self._init_pinocchio(urdf_path)

        # 参考零位（T-Pose 时的关节角度）
        self.q_ref = np.array([
            0.0, 0.0, 0.0,     # shoulder pitch/roll/yaw
            0.0,                # elbow
            0.0, 0.0, 0.0,     # wrist roll/pitch/yaw
        ])

    def _init_pinocchio(self, urdf_path):
        """初始化 Pinocchio 模型"""
        try:
            import pinocchio as pin
            if urdf_path is None:
                print("[WARN] 未提供 URDF 路径，回退到解析模式")
                self.mode = "analytical"
                return

            self._pin_model = pin.buildModelFromUrdf(urdf_path)
            self._pin_data = self._pin_model.createData()

            # 查找手腕 frame
            try:
                self._left_frame_id = self._pin_model.getFrameId("left_wrist")
                self._right_frame_id = self._pin_model.getFrameId("right_wrist")
            except Exception:
                # 尝试其他命名
                for name in self._pin_model.frames:
                    print(f"  Frame: {name.name}")

            print(f"[INFO] Pinocchio 模型加载成功: {urdf_path}")
            self.mode = "pinocchio"

        except ImportError:
            print("[WARN] Pinocchio 未安装，使用解析模式")
            self.mode = "analytical"
        except Exception as e:
            print(f"[WARN] Pinocchio 初始化失败: {e}，使用解析模式")
            self.mode = "analytical"

    def solve_arm(self, shoulder_pos, wrist_pos, elbow_pos=None,
                  is_left=True, max_iter=100, eps=1e-4):
        """
        求解单臂 IK

        参数:
            shoulder_pos: (3,) 肩关节在相机坐标系下的位置
            wrist_pos:    (3,) 腕部在相机坐标系下的位置
            elbow_pos:    (3,) 肘部在相机坐标系下的位置（可选的肘部约束）
            is_left:      左臂=True, 右臂=False
            max_iter:     最大迭代次数（pinocchio 模式）
            eps:          收敛阈值（pinocchio 模式）

        返回:
            q: (7,) 手臂关节角度 [shoulder_pitch, shoulder_roll, shoulder_yaw,
                                  elbow, wrist_roll, wrist_pitch, wrist_yaw]
        """
        if self.mode == "pinocchio" and self._pin_model is not None:
            return self._solve_pinocchio(shoulder_pos, wrist_pos,
                                         is_left, max_iter, eps)
        else:
            return self._solve_analytical(shoulder_pos, wrist_pos,
                                          elbow_pos, is_left)

    def _solve_pinocchio(self, shoulder_pos, wrist_pos,
                         is_left, max_iter, eps):
        """Pinocchio IK 求解（阻尼最小二乘法）

        参考 MonkeySee_MonkeyDo 的 JAX 梯度法，但用 Pinocchio 代替 PyRoki
        """
        import pinocchio as pin

        # 腕部在肩坐标系下的目标位置
        target_pos = wrist_pos - shoulder_pos
        target_pose = pin.SE3(np.eye(3), target_pos)

        frame_id = self._left_frame_id if is_left else self._right_frame_id
        if frame_id is None:
            print("[WARN] 未找到手腕 frame，回退解析模式")
            return self._solve_analytical(shoulder_pos, wrist_pos, None, is_left)

        start_idx = 15 if is_left else 22  # G1 关节向量中的起始索引

        q = self.q_ref.copy()
        full_q = np.zeros(self._pin_model.nq)
        full_q[start_idx:start_idx + 7] = q

        damping = 0.1
        dt = 0.5

        for _ in range(max_iter):
            pin.forwardKinematics(self._pin_model, self._pin_data, full_q)
            pin.updateFramePlacements(self._pin_model, self._pin_data)

            current_pose = self._pin_data.oMf[frame_id]
            error = target_pose.translation - current_pose.translation

            if np.linalg.norm(error) < eps:
                break

            # 位置雅可比
            J = pin.computeFrameJacobian(
                self._pin_model, self._pin_data, full_q, frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )[:3, start_idx:start_idx + 7]

            # 阻尼最小二乘: dq = J^T (J J^T + λI)^(-1) error
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(
                JJt + damping * np.eye(3), error
            )
            q += dq * dt
            full_q[start_idx:start_idx + 7] = q

        return q

    def _solve_analytical(self, shoulder_pos, wrist_pos,
                          elbow_pos, is_left):
        """
        解析法 IK（不需要 Pinocchio）

        基于 G1 手臂的简化运动学模型:
          肩 3DOF(Pitch/Roll/Yaw) → 肘 1DOF → 腕 3DOF(Roll/Pitch/Yaw)

        目前实现: 计算肩和肘的近似角度，腕部留零。
        完整 IK 建议安装 Pinocchio 后切换到 pinocchio 模式。
        """
        # 腕部在肩坐标系下的位置
        rel_wrist = wrist_pos - shoulder_pos

        # 上臂长度估算（G1 上臂约 0.25m）
        upper_arm_len = 0.25
        lower_arm_len = 0.25

        # ---- 肩关节 ----
        # Shoulder Pitch: 上臂相对竖直方向的角度
        shoulder_pitch = np.arctan2(rel_wrist[2], -rel_wrist[1])
        if not is_left:
            shoulder_pitch = shoulder_pitch

        # Shoulder Roll
        shoulder_roll = np.arctan2(rel_wrist[0], -rel_wrist[1])
        if not is_left:
            shoulder_roll = -shoulder_roll

        # Shoulder Yaw
        shoulder_yaw = np.arctan2(rel_wrist[0], rel_wrist[2]) * 0.5

        # ---- 肘关节 ----
        total_len = np.linalg.norm(rel_wrist)
        # 余弦定理求肘角
        cos_elbow = (upper_arm_len**2 + lower_arm_len**2 - total_len**2) / \
                    (2 * upper_arm_len * lower_arm_len + 1e-8)
        cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
        elbow_angle = np.pi - np.arccos(cos_elbow)

        return np.array([
            shoulder_pitch,
            shoulder_roll,
            shoulder_yaw,
            elbow_angle,
            0.0,  # wrist_roll  (解析法暂不计算)
            0.0,  # wrist_pitch
            0.0,  # wrist_yaw
        ])

    def solve_both_arms(self, lm: np.ndarray) -> tuple:
        """
        同时求解双臂

        参数:
            lm: (33, 3) MediaPipe 3D 关键点

        返回:
            left:  (7,) 左臂关节角度
            right: (7,) 右臂关节角度
        """
        # 左臂
        l_shoulder = lm[11]
        l_elbow = lm[13]
        l_wrist = lm[15]
        left_q = self.solve_arm(l_shoulder, l_wrist, l_elbow, is_left=True)

        # 右臂
        r_shoulder = lm[12]
        r_elbow = lm[14]
        r_wrist = lm[16]
        right_q = self.solve_arm(r_shoulder, r_wrist, r_elbow, is_left=False)

        return left_q, right_q


# ============================================================
# 测试独立运行
# ============================================================
if __name__ == "__main__":
    print("[TEST] 测试手臂 IK 求解器...")

    ik = ArmIK(mode="analytical")

    # 模拟 T-Pose 腕部位置
    l_sh = np.array([-0.2, 0.0, 0.3])
    l_wr = np.array([-0.7, 0.1, 0.3])
    r_sh = np.array([0.2, 0.0, 0.3])
    r_wr = np.array([0.7, 0.1, 0.3])

    left = ik.solve_arm(l_sh, l_wr, is_left=True)
    right = ik.solve_arm(r_sh, r_wr, is_left=False)

    print(f"左臂: {np.round(left, 3)}")
    print(f"右臂: {np.round(right, 3)}")

    # 测试双臂同时求解
    lm_fake = np.zeros((33, 3))
    lm_fake[11] = l_sh
    lm_fake[13] = (l_sh + l_wr) / 2 + np.array([0, 0, 0.05])
    lm_fake[15] = l_wr
    lm_fake[12] = r_sh
    lm_fake[14] = (r_sh + r_wr) / 2 + np.array([0, 0, 0.05])
    lm_fake[16] = r_wr

    l, r = ik.solve_both_arms(lm_fake)
    print(f"\n双臂求解:")
    print(f"左: {np.round(l, 3)}")
    print(f"右: {np.round(r, 3)}")
