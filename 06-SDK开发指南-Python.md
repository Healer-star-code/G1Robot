# 宇树 G1 SDK 开发指南 (Python)

> 资料来源：unitree_sdk2_python GitHub、LeRobot 文档

---

## 一、环境搭建

### 1.1 系统要求

- Python >= 3.8
- cyclonedds == 0.10.2
- numpy
- opencv-python（摄像头功能需要）

### 1.2 安装

```bash
# 从 GitHub 源码安装
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .

# 或使用 conda
conda create -y -n unitree python=3.12
conda activate unitree
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

### 1.3 常见问题：CycloneDDS 找不到

```
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

解决方法：
```bash
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
cd ~/unitree_sdk2_python
export CYCLONEDDS_HOME=~/cyclonedds/install
pip3 install -e .
```

---

## 二、示例程序列表

Python SDK 的所有示例位于 `example/` 目录下。

| 示例 | 文件路径 | 功能 |
|------|----------|------|
| Hello World | `example/helloworld/` | DDS 发布/订阅基础 |
| 高层状态 | `example/high_level/read_highstate.py` | 读取 SportModeState |
| 高层控制 | `example/high_level/sportmode_test.py` | 站立、速度、姿态、轨迹控制 |
| 底层控制 | `example/low_level/lowlevel_control.py` | 直接电机位置/力矩控制 |
| 遥控器 | `example/wireless_controller/wireless_controller.py` | 读取遥控器状态 |
| 前部摄像头 | `example/front_camera/camera_opencv.py` | OpenCV 获取图像 |
| 避障开关 | `example/obstacles_avoid_switch/obstacles_avoid_switch.py` | 开关避障服务 |
| 灯光音量 | `example/vui_client/vui_client_example.py` | 控制灯光和音量 |

---

## 三、运行示例

所有示例运行时需指定连接 G1 的网卡名（如 `enp2s0`）。

### 3.1 DDS 通信测试

```bash
# 终端 1：发布者
python3 ./example/helloworld/publisher.py

# 终端 2：订阅者
python3 ./example/helloworld/subscriber.py
```

### 3.2 高层状态读取

```bash
python3 ./example/high_level/read_highstate.py enp2s0
```

### 3.3 高层控制

```bash
python3 ./example/high_level/sportmode_test.py enp2s0

# 在代码中选择测试项：
# test.StandUpDown()      # 站立和趴下
# test.VelocityMove()     # 速度控制
# test.BalanceAttitude()  # 姿态控制
# test.TrajectoryFollow() # 轨迹追踪
# test.SpecialMotions()   # 特殊动作
```

### 3.4 底层状态读取

```bash
python3 ./example/low_level/lowlevel_control.py enp2s0
# 输出右前腿髋关节状态、IMU、电池电压
```

### 3.5 底层电机控制

> 先用 App 关闭高层运动服务（sport_mode），避免指令冲突

```bash
python3 ./example/low_level/lowlevel_control.py enp2s0
```

### 3.6 遥控器状态

```bash
python3 ./example/wireless_controller/wireless_controller.py enp2s0
# 终端输出各按键状态
```

### 3.7 前部摄像头

```bash
# 需要图形界面环境
python3 ./example/front_camera/camera_opencv.py enp2s0
# 按 ESC 退出
```

---

## 四、数据模型

Python SDK 的数据结构定义在 `unitree_sdk2py/idl/` 目录下。

### 4.1 导入

```python
from unitree_sdk2py.idl.default import *

# 创建默认消息
low_cmd = unitree_hg_msg_dds__LowCmd_()
low_state = unitree_hg_msg_dds__LowState_()
motor_cmd = unitree_hg_msg_dds__MotorCmd_()
joy = unitree_go_msg_dds__WirelessController_()
```

### 4.2 模块层次

```
unitree_sdk2py/idl/
├── default.py                 # 统一导入入口 + 默认值工厂函数
├── unitree_go/msg/           # Go2/B2/H1 等机器人通用消息
│   └── dds_/
│       ├── WirelessController_.py
│       ├── SportModeState_.py
│       ├── IMUState_.py     (unitree_go 版本)
│       └── ...
├── unitree_hg/msg/           # G1/H2 人形机器人专用消息
│   └── dds_/
│       ├── LowCmd_.py
│       ├── LowState_.py
│       ├── MotorCmd_.py
│       ├── MotorState_.py
│       ├── IMUState_.py     (unitree_hg 版本)
│       ├── HandCmd_.py
│       └── HandState_.py
└── unitree_api/msg/          # API 请求/响应
    └── dds_/
        ├── Request_.py
        └── Response_.py
```

### 4.3 创建和填充消息

```python
# 方式一：使用默认值工厂
motor_cmd = unitree_hg_msg_dds__MotorCmd_()
motor_cmd.mode = 1
motor_cmd.q = 0.5
motor_cmd.dq = 0.0
motor_cmd.tau = 0.0
motor_cmd.kp = 60.0
motor_cmd.kd = 1.0

# 方式二：直接实例化
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MotorCmd_
cmd = MotorCmd_(
    mode=1,
    q=0.5,
    dq=0.0,
    tau=0.0,
    kp=60.0,
    kd=1.0,
    reserve=0
)
```

---

## 五、DDS 通信编程

### 5.1 订阅者

```python
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactory
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

# 初始化
ChannelFactory.Instance().Init(0, "enp2s0")

# 创建订阅
sub = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
sub.Init()

# 读取消息
msg = WirelessController_()
while True:
    if sub.Read(msg):
        print(f"lx={msg.lx}, ly={msg.ly}, rx={msg.rx}, ry={msg.ry}")
        print(f"keys={msg.keys:016b}")
    time.sleep(0.02)
```

### 5.2 发布者

```python
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactory
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

pub = ChannelPublisher("rt/lowcmd", LowCmd_)
pub.Init()

cmd = LowCmd_()
# ... 填充指令 ...
pub.Write(cmd)
```

---

## 六、完整数据采集示例

```python
import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactory
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

def main():
    # 初始化
    ChannelFactory.Instance().Init(0, "enp2s0")

    # 订阅 LowState
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()

    state = LowState_()
    while True:
        if sub.Read(state):
            print(f"Tick: {state.tick}")
            print(f"IMU RPY: {list(state.imu_state.rpy)}")

            # 读取各关节位置
            for i in range(29):
                q = state.motor_state[i].q
                dq = state.motor_state[i].dq
                tau = state.motor_state[i].tau_est
                print(f"  Joint {i}: q={q:.3f}, dq={dq:.3f}, tau={tau:.3f}")

        time.sleep(0.01)

if __name__ == "__main__":
    main()
```
