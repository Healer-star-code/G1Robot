"""
人 → G1 关节映射（下肢 + 腰部）
================================
参考: Nao_Mimc 的 Pose_est.py（MediaPipe 关键点直接提取）
      OmniH2O 的 keypoint mapping 策略
      MonkeySee_MonkeyDo 的 arm length normalization

方法: 利用 MediaPipe 人体 3D 关键点的几何关系直接计算关节角度
      不需要 RL 或复杂优化，适合实时映射

映射策略:
  下肢: 髋(23/24) → 膝(25/26) → 踝(27/28) 的 3D 向量计算角度
  腰部: 肩中点(11/12中点) → 髋中点(23/24中点) 的躯干向量
"""

import numpy as np
from config import JOINT_LIMITS


class JointMapper:
    """
    将 MediaPipe 33 个人体 3D 关键点映射为 G1 29 个关节角度
    """

    def __init__(self):
        self.num_joints = 29

        # 人体关键点索引
        self.L_SHOULDER = 11
        self.R_SHOULDER = 12
        self.L_ELBOW = 13
        self.R_ELBOW = 14
        self.L_WRIST = 15
        self.R_WRIST = 16
        self.L_HIP = 23
        self.R_HIP = 24
        self.L_KNEE = 25
        self.R_KNEE = 26
        self.L_ANKLE = 27
        self.R_ANKLE = 28
        self.L_HEEL = 29
        self.R_HEEL = 30
        self.L_TOE = 31
        self.R_TOE = 32

    def clamp(self, value, joint_id):
        """限幅到关节安全范围"""
        lo, hi = JOINT_LIMITS.get(joint_id, (-np.inf, np.inf))
        return np.clip(value, lo, hi)

    # ------------------------------------------------------------------
    #  下肢映射
    # ------------------------------------------------------------------
    def map_lower_body(self, lm: np.ndarray) -> tuple:
        """
        下肢映射
        lm: (33, 3) 关键点数组

        返回:
            left : (6,)  左腿 [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
            right: (6,)  右腿
        """
        left = self._compute_leg(lm, self.L_HIP, self.L_KNEE,
                                 self.L_ANKLE, self.L_HEEL, self.L_TOE,
                                 is_left=True, base_idx=0)
        right = self._compute_leg(lm, self.R_HIP, self.R_KNEE,
                                  self.R_ANKLE, self.R_HEEL, self.R_TOE,
                                  is_left=False, base_idx=6)
        return left, right

    def _compute_leg(self, lm, hip_id, knee_id, ankle_id,
                     heel_id, toe_id, is_left, base_idx):
        """
        单腿关节角度计算

        计算方法（基于 MonkeySee_MonkeyDo 和 Nao_Mimc 的思路）:
          从 3D 关键点构建骨骼向量，用 atan2 和 arccos 计算关节角度

        大腿向量 = knee - hip    (髋→膝)
        小腿向量 = ankle - knee  (膝→踝)
        脚掌向量 = toe - heel   (跟→尖)
        """
        hip = lm[hip_id]
        knee = lm[knee_id]
        ankle = lm[ankle_id]
        heel = lm[heel_id]
        toe = lm[toe_id]

        # 安全检查：深度为 0 时跳过
        if np.any(np.isnan(hip)) or np.any(np.isnan(knee)):
            return np.zeros(6)

        thigh = knee - hip       # 大腿向量
        shank = ankle - knee     # 小腿向量
        foot = toe - heel        # 脚掌向量

        thigh_len = np.linalg.norm(thigh) + 1e-8
        shank_len = np.linalg.norm(shank) + 1e-8
        foot_len = np.linalg.norm(foot) + 1e-8

        thigh_n = thigh / thigh_len
        shank_n = shank / shank_len
        foot_n = foot / foot_len if foot_len > 0.001 else np.array([1, 0, 0])

        # ---- 髋关节 Pitch（大腿前后摆动）----
        # atan2(dz, -dy): 大腿在竖直面(YZ平面)的投影角度
        hip_pitch = np.arctan2(thigh_n[2], -thigh_n[1])

        # ---- 髋关节 Roll（大腿左右摆动）----
        hip_roll = np.arctan2(thigh_n[0], -thigh_n[1])

        # ---- 髋关节 Yaw（大腿内外旋转）----
        hip_yaw = np.arctan2(np.dot(thigh_n, np.array([1, 0, 0])),
                             np.dot(thigh_n, np.array([0, 0, 1])))

        # 左右对称修正
        if not is_left:
            hip_roll = -hip_roll
            hip_yaw = -hip_yaw

        # ---- 膝关节（大腿与小腿夹角）----
        dot_val = np.clip(np.dot(thigh_n, shank_n), -1.0, 1.0)
        knee_angle = np.pi - np.arccos(dot_val)
        # G1 膝关节: 0=伸直, 正值=弯曲
        knee_angle = np.clip(knee_angle, JOINT_LIMITS[base_idx + 3][0],
                             JOINT_LIMITS[base_idx + 3][1])

        # ---- 踝关节 Pitch（脚掌上下）----
        ankle_pitch = np.arcsin(np.clip(-shank_n[2], -1.0, 1.0))

        # ---- 踝关节 Roll（脚掌左右倾）----
        ankle_roll = np.arctan2(foot_n[0], -foot_n[1]) * 0.3

        angles = np.array([
            self.clamp(hip_pitch, base_idx + 0),
            self.clamp(hip_roll,  base_idx + 1),
            self.clamp(hip_yaw,   base_idx + 2),
            self.clamp(knee_angle, base_idx + 3),
            self.clamp(ankle_pitch, base_idx + 4),
            self.clamp(ankle_roll,  base_idx + 5),
        ])
        return angles

    # ------------------------------------------------------------------
    #  腰部映射
    # ------------------------------------------------------------------
    def map_waist(self, lm: np.ndarray) -> np.ndarray:
        """
        腰部映射 → G1 WAIST_YAW(12), WAIST_ROLL(13), WAIST_PITCH(14)

        肩中点 = (L_SHOULDER + R_SHOULDER) / 2
        髋中点 = (L_HIP + R_HIP) / 2
        躯干向量 = 髋中点 → 肩中点
        """
        l_sh = lm[self.L_SHOULDER]
        r_sh = lm[self.R_SHOULDER]
        l_hi = lm[self.L_HIP]
        r_hi = lm[self.R_HIP]

        shoulder_mid = (l_sh + r_sh) / 2
        hip_mid = (l_hi + r_hi) / 2

        torso = shoulder_mid - hip_mid  # 躯干向量
        torso_len = np.linalg.norm(torso) + 1e-8
        torso_n = torso / torso_len

        # Yaw: 躯干在 XZ 平面的旋转
        waist_yaw = np.arctan2(torso_n[0], torso_n[2])

        # Roll: 肩线偏离水平的角度
        shoulder_line = r_sh - l_sh
        waist_roll = np.arctan2(shoulder_line[1], abs(shoulder_line[0]) + 1e-8)

        # Pitch: 躯干前后倾
        waist_pitch = np.arcsin(np.clip(-torso_n[2], -1.0, 1.0))

        return np.array([
            self.clamp(waist_yaw, 12),
            self.clamp(waist_roll, 13),
            self.clamp(waist_pitch, 14),
        ])

    # ------------------------------------------------------------------
    #  组装
    # ------------------------------------------------------------------
    def get_all_joint_targets(self, lm: np.ndarray,
                              left_arm_angles: np.ndarray,
                              right_arm_angles: np.ndarray,
                              enable_lower_body: bool = True,
                              enable_waist: bool = True,
                              enable_upper_body: bool = True) -> np.ndarray:
        """
        组装全部 29 个关节目标

        参数:
            lm:             (33, 3) 人体关键点
            left_arm_angles:  (7,) 左臂 IK 结果
            right_arm_angles: (7,) 右臂 IK 结果
            enable_lower_body: 是否启用下肢
            enable_waist:      是否启用腰部
            enable_upper_body: 是否启用上肢

        返回:
            targets: (29,) float32 目标关节角度 (rad)
        """
        targets = np.zeros(29, dtype=np.float32)

        if enable_lower_body:
            left_leg, right_leg = self.map_lower_body(lm)
            targets[0:6] = left_leg
            targets[6:12] = right_leg

        if enable_waist:
            targets[12:15] = self.map_waist(lm)

        if enable_upper_body:
            targets[15:22] = left_arm_angles
            targets[22:29] = right_arm_angles

        return targets
