"""
G1 机器人 DDS 控制模块
======================
通过 unitree_sdk2_python 的 DDS LowCmd 通道控制 G1 全身关节

DDS Topic:
  发送: rt/lowcmd  (LowCmd_)
  接收: rt/lowstate (LowState_)

参考: unitree_sdk2 C++ 示例 g1_ankle_swing_example.cpp
      unitree_sdk2_python 示例 lowlevel_control.py
"""

import time
import numpy as np


class G1Controller:
    """
    G1 底层关节控制器

    用法:
        ctrl = G1Controller("enp2s0")
        ctrl.start()

        while True:
            q_target = ...  # (29,) 目标角度
            ctrl.send_command(q_target)

        ctrl.stop()
    """

    def __init__(self, network_interface="enp2s0", kp=None, kd=None):
        """
        参数:
            network_interface: 连接 G1 的网卡名
            kp: (29,) 位置刚度数组
            kd: (29,) 速度阻尼数组
        """
        self.network_interface = network_interface
        self._running = False

        # 默认刚度阻尼
        if kp is None:
            kp = np.array([
                30, 30, 30, 50, 20, 20,
                30, 30, 30, 50, 20, 20,
                30, 20, 20,
                20, 20, 20, 20, 20, 20, 20,
                20, 20, 20, 20, 20, 20, 20
            ], dtype=np.float32)
        if kd is None:
            kd = np.array([
                0.5, 0.5, 0.5, 1, 0.5, 0.5,
                0.5, 0.5, 0.5, 1, 0.5, 0.5,
                0.5, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3
            ], dtype=np.float32)

        self.Kp = kp
        self.Kd = kd
        self.mode_machine = 0
        self.mode_pr = 0  # PR 模式

    def start(self):
        """初始化 DDS 连接"""
        try:
            from unitree_sdk2py.core.channel import (
                ChannelPublisher, ChannelSubscriber, ChannelFactory
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

            self._LowCmd = LowCmd_
            self._LowState = LowState_

            # 初始化 DDS
            ChannelFactory.Instance().Init(0, self.network_interface)

            # 发布 LowCmd
            self._cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
            self._cmd_pub.Init()

            # 订阅 LowState（获取 mode_machine）
            self._state_sub = ChannelSubscriber("rt/lowstate", LowState_)
            self._state_sub.Init()

            # 读取一次状态，获取 mode_machine
            time.sleep(0.5)
            self._read_state()

            self._running = True
            print(f"[INFO] G1 控制器已连接 (网卡: {self.network_interface})")
            print(f"       mode_machine = {self.mode_machine}")

        except ImportError as e:
            print(f"[ERROR] unitree_sdk2_python 未安装: {e}")
            print(f"        请执行: pip install -e /path/to/unitree_sdk2_python")
            raise
        except Exception as e:
            print(f"[ERROR] DDS 初始化失败: {e}")
            print(f"        请检查: 网线是否连接？IP 是否正确？G1 是否开机？")
            raise

    def _read_state(self):
        """读取 LowState（非阻塞）"""
        try:
            state = self._LowState()
            if self._state_sub.Read(state, timeout_ms=10):
                self.mode_machine = state.mode_machine
        except Exception:
            pass

    def send_command(self, q_target: np.ndarray, mode_pr=0):
        """
        发送关节目标位置

        参数:
            q_target: (29,) float32 目标关节角度 (rad)
            mode_pr:   0=PR, 1=AB（踝关节模式）
        """
        if not self._running:
            return

        cmd = self._LowCmd()
        cmd.mode_pr = mode_pr
        cmd.mode_machine = self.mode_machine

        for i in range(29):
            cmd.motor_cmd[i].mode = 1                    # 使能电机
            cmd.motor_cmd[i].q = float(q_target[i])
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp = float(self.Kp[i])
            cmd.motor_cmd[i].kd = float(self.Kd[i])

        self._cmd_pub.Write(cmd)

    def damp_all(self):
        """紧急停止: 所有关节 Kp=Kd=0，进入阻尼"""
        if not self._running:
            return

        cmd = self._LowCmd()
        cmd.mode_pr = 0
        cmd.mode_machine = self.mode_machine
        for i in range(29):
            cmd.motor_cmd[i].mode = 0    # 禁用
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0

        self._cmd_pub.Write(cmd)
        print("[INFO] 紧急停止: 所有关节已进入阻尼模式")

    def stop(self):
        """安全关闭"""
        if self._running:
            self.damp_all()
            self._running = False
            print("[INFO] G1 控制器已断开")
