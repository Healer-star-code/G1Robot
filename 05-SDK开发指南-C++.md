# 宇树 G1 SDK 开发指南 (C++)

> 资料来源：unitree_sdk2 C++ SDK、GitHub 示例代码

---

## 一、环境搭建

### 1.1 系统要求

- Ubuntu 20.04 LTS
- GCC 9.4.0
- CMake >= 3.10

### 1.2 安装依赖

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ build-essential libyaml-cpp-dev \
    libeigen3-dev libboost-all-dev libspdlog-dev libfmt-dev
```

### 1.3 构建 SDK

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. && make
```

### 1.4 安装到系统

```bash
cd build
sudo make install

# 或安装到指定目录
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics
sudo make install
```

---

## 二、项目结构

```
unitree_sdk2/
├── example/
│   ├── g1/
│   │   ├── low_level/
│   │   │   ├── g1_ankle_swing_example.cpp   # 踝关节摆动演示
│   │   │   ├── g1_dual_arm_example.cpp      # 双臂控制示例
│   │   │   ├── gamepad.hpp                  # 摇杆工具类
│   │   │   ├── terminations.cpp             # 安全终止
│   │   │   └── behavior_lib/                # 行为库
│   │   └── high_level/
│   │       ├── g1_loco_client_example.cpp       # 运动控制客户端
│   │       ├── g1_arm5_sdk_dds_example.cpp      # 5-DOF 臂 DDS 示例
│   │       ├── g1_arm7_sdk_dds_example.cpp      # 7-DOF 臂 DDS 示例
│   │       ├── g1_arm_action_example.cpp         # 臂预定义动作
│   │       └── g1_userctrl_dds_example.cpp       # 用户控制 DDS 示例
│   └── wireless_controller/
│       ├── main.cpp                    # 遥控器订阅示例
│       └── advanced_gamepad.hpp        # 遥控器封装类
├── include/unitree/                    # 头文件
└── lib/                                # 预编译库
```

---

## 三、低层控制示例详解

### 3.1 踝关节摆动 (g1_ankle_swing_example.cpp)

完整的 G1 低层控制示例，演示三个阶段：

#### 阶段 1：归零（前 3 秒）
将所有关节从任意姿态平滑过渡到零位：

```cpp
// 插值到零位
double ratio = clamp(time_ / duration_, 0.0, 1.0);
motor_command.q_target[i] = (1.0 - ratio) * current_position[i];
```

#### 阶段 2：PR 模式摆动
使用 PR 模式（默认），直接控制踝关节 Pitch 和 Roll：

```cpp
mode_pr_ = Mode::PR;  // PR 模式
double L_P_des = max_P * sin(2.0 * M_PI * t);   // 左踝 Pitch
double L_R_des = max_R * sin(2.0 * M_PI * t);   // 左踝 Roll
double R_P_des = max_P * sin(2.0 * M_PI * t);   // 右踝 Pitch
double R_R_des = -max_R * sin(2.0 * M_PI * t);  // 右踝 Roll（反相）

motor_command.q_target[LeftAnklePitch] = L_P_des;
motor_command.q_target[LeftAnkleRoll] = L_R_des;
motor_command.q_target[RightAnklePitch] = R_P_des;
motor_command.q_target[RightAnkleRoll] = R_R_des;
```

#### 阶段 3：AB 模式摆动
切换到 AB 模式，直接驱动并联机构 A/B 电机：

```cpp
mode_pr_ = Mode::AB;  // AB 模式
double L_A_des = +max_A * sin(M_PI * t);
double L_B_des = +max_B * sin(M_PI * t + M_PI);
double R_A_des = -max_A * sin(M_PI * t);
double R_B_des = -max_B * sin(M_PI * t + M_PI);

motor_command.q_target[LeftAnkleA] = L_A_des;
motor_command.q_target[LeftAnkleB] = L_B_des;
motor_command.q_target[RightAnkleA] = R_A_des;
motor_command.q_target[RightAnkleB] = R_B_des;
```

### 3.2 程序核心结构

```cpp
class G1Example {
public:
    G1Example(std::string networkInterface) {
        // 1. 初始化 DDS Channel
        ChannelFactory::Instance()->Init(0, networkInterface);

        // 2. 尝试释放运动控制服务（避免指令冲突）
        msc_ = std::make_shared<MotionSwitcherClient>();
        msc_->SetTimeout(5.0f);
        msc_->Init();
        std::string form, name;
        while (msc_->CheckMode(form, name), !name.empty()) {
            msc_->ReleaseMode();
            sleep(5);
        }

        // 3. 创建发布器和订阅器
        lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>("rt/lowcmd"));
        lowcmd_publisher_->InitChannel();

        lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>("rt/lowstate"));
        lowstate_subscriber_->InitChannel(
            std::bind(&G1Example::LowStateHandler, this, std::placeholders::_1), 1);

        // 4. 创建执行线程 (2ms 周期)
        command_writer_ptr_ = CreateRecurrentThreadEx(
            "command_writer", UT_CPU_ID_NONE, 2000,
            &G1Example::LowCommandWriter, this);
        control_thread_ptr_ = CreateRecurrentThreadEx(
            "control", UT_CPU_ID_NONE, 2000,
            &G1Example::Control, this);
    }

private:
    // 状态处理回调
    void LowStateHandler(const void *message);

    // 控制计算 (2ms 周期)
    void Control();

    // 指令发送 (2ms 周期)
    void LowCommandWriter();
};
```

### 3.3 指令发送

```cpp
void LowCommandWriter() {
    LowCmd_ dds_low_command;

    // 设置模式
    dds_low_command.mode_pr() = static_cast<uint8_t>(mode_pr_);
    dds_low_command.mode_machine() = mode_machine_;

    // 填充电机指令
    for (int i = 0; i < G1_NUM_MOTOR; i++) {
        dds_low_command.motor_cmd().at(i).mode() = 1;  // 使能
        dds_low_command.motor_cmd().at(i).tau() = mc->tau_ff[i];
        dds_low_command.motor_cmd().at(i).q()   = mc->q_target[i];
        dds_low_command.motor_cmd().at(i).dq()  = mc->dq_target[i];
        dds_low_command.motor_cmd().at(i).kp()  = mc->kp[i];
        dds_low_command.motor_cmd().at(i).kd()  = mc->kd[i];
    }

    // CRC 校验（必须！）
    dds_low_command.crc() = Crc32Core(
        (uint32_t *)&dds_low_command,
        (sizeof(dds_low_command) >> 2) - 1
    );

    // 发送
    lowcmd_publisher_->Write(dds_low_command);
}
```

### 3.4 运行

```bash
./build/example/g1/g1_ankle_swing_example enp2s0
# 替换 enp2s0 为连接 G1 的网卡名
```

---

## 四、高层控制示例详解

### 4.1 运动控制客户端 (g1_loco_client_example.cpp)

```cpp
#include <unitree/robot/g1/loco/g1_loco_client.hpp>

unitree::robot::g1::LocoClient client;
client.Init();
client.SetTimeout(10.f);

// 基础运动
client.Damp();                             // 阻尼模式
client.Start();                            // 启动运动
client.StandUp();                          // 站立
client.Squat();                            // 蹲下
client.Sit();                              // 坐下
client.HighStand();                        // 高位站立
client.LowStand();                         // 低位站立
client.BalanceStand();                     // 平衡站立
client.StopMove();                         // 停止移动
client.ZeroTorque();                       // 零力矩

// 速度控制
client.Move(vx, vy, omega);               // 连续速度移动
client.SetVelocity(vx, vy, omega, time);   // 定时速度移动

// 特殊动作
client.ShakeHand(0);  // 握手开始
// ... 等待 ...
client.ShakeHand(1);  // 握手结束
client.WaveHand();                         // 挥手
client.WaveHand(true);                     // 挥手+转身

// 步态控制
client.ContinuousGait(true);               // 连续步态
client.SwitchMoveMode(true);               // 切换移动模式

// 状态查询
int fsm_id;
client.GetFsmId(fsm_id);

int fsm_mode;
client.GetFsmMode(fsm_mode);

float stand_height;
client.GetStandHeight(stand_height);

float swing_height;
client.GetSwingHeight(swing_height);
```

### 4.2 命令行运行

```bash
# 站立
./g1_loco_client_example --network_interface=enp2s0 --stand_up

# 速度移动 1 秒
./g1_loco_client_example --network_interface=enp2s0 --set_velocity="0.2 0 0 1"

# 握手 10 秒
./g1_loco_client_example --network_interface=enp2s0 --shake_hand
```

### 4.3 手臂控制 (g1_arm5 / g1_arm7)

通过 DDS 直接控制手臂各关节位置，参见 `example/g1/high_level/g1_arm5_sdk_dds_example.cpp` 和 `g1_arm7_sdk_dds_example.cpp`。

---

## 五、如何在 CMake 项目中使用 SDK

```cmake
cmake_minimum_required(VERSION 3.10)
project(my_g1_app)

# 查找 unitree_sdk2
find_package(unitree_sdk2 REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app unitree_sdk2::unitree_sdk2)
```

> 如果安装到非标准路径，需将该路径加入 `CMAKE_PREFIX_PATH`。

---

## 六、安全注意事项

1. **始终先释放运动控制服务**，避免指令冲突
2. **校验 CRC**，丢弃错误状态帧
3. **监控 motorstate**，非零表示电机故障
4. **遵守关节限位**，超出可能导致硬件损坏
5. **从低 Kp/Kd 开始**，逐步调高增益
6. 测试时确保机器人周围有足够的**安全空间**
