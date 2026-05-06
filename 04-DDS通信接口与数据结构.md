# 宇树 G1 DDS 通信接口与数据结构

> 资料来源：unitree_sdk2 C++ SDK、unitree_sdk2_python IDL 定义

---

## 一、通信架构

G1 采用 **CycloneDDS** 作为消息中间件，支持两种通信模式：

| 模式 | 适用场景 | 示例 |
|------|----------|------|
| **订阅/发布 (Pub/Sub)** | 实时状态流、持续控制指令 | LowState, LowCmd, WirelessController |
| **请求/响应 (Req/Rep)** | 服务调用 | LocoClient, VuiClient, ObstaclesAvoid |

---

## 二、DDS Topic 列表

### 2.1 底层控制相关 (unitree_hg)

| Topic | 方向 | 消息类型 | 说明 |
|-------|------|----------|------|
| `rt/lowcmd` | 用户 → 机器人 | `LowCmd_` | 底层电机控制指令 |
| `rt/lowstate` | 机器人 → 用户 | `LowState_` | 底层电机状态 + IMU + 遥控器 |
| `rt/secondary_imu` | 机器人 → 用户 | `IMUState_` | 躯干 IMU 数据 |

### 2.2 通用接口 (unitree_go)

| Topic | 方向 | 消息类型 | 说明 |
|-------|------|----------|------|
| `rt/wirelesscontroller` | 机器人 → 用户 | `WirelessController_` | 遥控器摇杆+按键 |

### 2.3 高层服务 (unitree_api)

高层控制通过 `unitree_api::msg::Request_` / `Response_` 进行 RPC 调用，不直接使用 Topic。

---

## 三、核心数据结构：unitree_hg 模块

### 3.1 LowCmd_ — 底层控制指令

**Topic**: `rt/lowcmd`

```python
class LowCmd_:
    mode_pr: uint8           # 0=PR模式, 1=AB模式（踝关节并联机构）
    mode_machine: uint8       # 机器人类型（自动识别，只读）
    motor_cmd: MotorCmd_[35]  # 35 个电机控制指令数组
    reserve: uint32[4]        # 保留
    crc: uint32               # CRC32 校验码（必填！）
```

**字段说明**：
- `mode_pr = 0`：PR 模式，直接控制踝关节 Pitch/Roll
- `mode_pr = 1`：AB 模式，直接控制并联机构 A/B 电机
- `mode_machine`：由机器人自动设置，用户应读取后原样回传
- **CRC 必须正确**，否则指令被丢弃

### 3.2 LowState_ — 底层状态

**Topic**: `rt/lowstate`

```python
class LowState_:
    version: uint32[2]           # 版本号
    mode_pr: uint8               # 当前踝关节模式
    mode_machine: uint8          # 机器人类型
    tick: uint32                  # 时钟滴答
    imu_state: IMUState_         # 骨盆 IMU 数据
    motor_state: MotorState_[35]  # 35 个电机状态
    wireless_remote: uint8[40]    # 遥控器原始数据 (40字节)
    reserve: uint32[4]
    crc: uint32                   # CRC32 校验码
```

**使用注意**：
- 读取后务必先校验 CRC
- 如果 `motor_state[i].motorstate != 0`，表示该电机故障
- `wireless_remote` 的解析参见《03-遥控器数据结构》

### 3.3 MotorCmd_ — 单电机指令

```python
class MotorCmd_:
    mode: uint8     # 1=使能, 0=禁用
    q: float32      # 目标关节位置 (rad)
    dq: float32     # 目标关节速度 (rad/s)
    tau: float32    # 前馈力矩 (Nm)
    kp: float32     # 位置刚度增益
    kd: float32     # 速度阻尼增益
    reserve: uint32 # 保留
```

**控制律**：
```
tau_out = kp * (q_des - q_cur) + kd * (dq_des - dq_cur) + tau_ff
```

### 3.4 MotorState_ — 单电机状态

```python
class MotorState_:
    mode: uint8           # 当前模式
    q: float32            # 当前位置 (rad)
    dq: float32           # 当前速度 (rad/s)
    ddq: float32          # 当前加速度 (rad/s²)
    tau_est: float32      # 估计力矩 (Nm)
    temperature: int16[2] # 温度 [电机温度, 驱动器温度]
    vol: float32          # 电压
    sensor: uint32[2]     # 传感器数据
    motorstate: uint32    # 0=正常, 非0=故障码
    reserve: uint32[4]
```

### 3.5 IMUState_ — IMU 状态

```python
class IMUState_:
    quaternion: float32[4]    # 姿态四元数 [w, x, y, z]
    gyroscope: float32[3]     # 角速度 [x, y, z] (rad/s)
    accelerometer: float32[3] # 加速度 [x, y, z] (m/s²)
    rpy: float32[3]           # 欧拉角 [roll, pitch, yaw] (rad)
    temperature: int16        # 温度
```

### 3.6 BMS 电池状态

**BmsState_**:
```python
class BmsState_:
    version_high: uint8
    version_low: uint8
    status: uint8
    soc: uint8                          # 电量百分比
    current: int32                      # 电流 (mA)
    cycle: uint16
    bq_ntc: int8[2]
    mcu_ntc: int8[2]
    cell_vol: uint16[40]                # 各电芯电压
    temperature: uint32[12]
    # ... 更多字段
```

### 3.7 灵巧手

**HandCmd_**:
```python
class HandCmd_:
    motor_cmd: MotorCmd_[7]  # 手指关节电机指令（最多7个）
    reserve: uint32[4]
```

**HandState_**:
```python
class HandState_:
    motor_state: MotorState_[7]               # 手指关节状态
    press_sensor_state: PressSensorState_[7]  # 压力传感器
    imu_state: IMUState_                       # 手部 IMU
    temperature: float32[4]                    # 温度
    # ... 更多字段
```

---

## 四、CRC32 校验

低层指令必须附带正确的 CRC32 校验码。

```cpp
inline uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
    uint32_t xbit = 0;
    uint32_t data = 0;
    uint32_t CRC32 = 0xFFFFFFFF;
    const uint32_t dwPolynomial = 0x04c11db7;
    for (uint32_t i = 0; i < len; i++) {
        xbit = 1 << 31;
        data = ptr[i];
        for (uint32_t bits = 0; bits < 32; bits++) {
            if (CRC32 & 0x80000000) {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            } else
                CRC32 <<= 1;
            if (data & xbit) CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }
    return CRC32;
}
```

**使用方法**：
```cpp
LowCmd_ cmd;
// ... 填充指令 ...
// CRC 覆盖范围：整个结构体减去最后一个 crc 字段
cmd.crc() = Crc32Core((uint32_t *)&cmd, (sizeof(cmd) >> 2) - 1);
lowcmd_publisher_->Write(cmd);
```

**状态校验**：
```cpp
LowState_ state = *(const LowState_ *)message;
if (state.crc() != Crc32Core((uint32_t *)&state, (sizeof(state) >> 2) - 1)) {
    // CRC 错误，丢弃此帧
}
```

---

## 五、DDS 频道初始化

### C++ 初始化模式

```cpp
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// 1. 初始化 ChannelFactory
ChannelFactory::Instance()->Init(0, "enp2s0");  // 0 或网卡名

// 2. 创建发布器
ChannelPublisherPtr<LowCmd_> lowcmd_publisher;
lowcmd_publisher.reset(new ChannelPublisher<LowCmd_>("rt/lowcmd"));
lowcmd_publisher->InitChannel();

// 3. 创建订阅器
ChannelSubscriberPtr<LowState_> lowstate_subscriber;
lowstate_subscriber.reset(new ChannelSubscriber<LowState_>("rt/lowstate"));
lowstate_subscriber->InitChannel(MessageHandler, 1);

// 4. 发布数据
lowcmd_publisher->Write(cmd);
```
