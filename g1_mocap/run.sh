#!/bin/bash
# ================================================================
# G1 Motion Capture 启动脚本
# ================================================================
# 使用方法:
#   ./run.sh                 # 默认启动（连接机器人）
#   ./run.sh --sim           # 模拟模式（不连机器人，仅预览）
#   ./run.sh --ik-mode pinocchio --urdf /path/to/g1.urdf  # 精确IK
#   ./run.sh --enable-lower-body true  # 启用下肢映射
#   ./run.sh --interface eth0          # 指定网卡
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "  G1 Motion Capture Launcher"
echo "================================================"

# ---- 检查 Python ----
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] 未找到 python3"
    exit 1
fi

# ---- 检查 RealSense ----
echo -n "[CHECK] RealSense 相机..."
if python3 -c "import pyrealsense2" 2>/dev/null; then
    echo " OK"
else
    echo " 未安装"
    echo "  安装: pip install pyrealsense2"
    exit 1
fi

# ---- 检查 MediaPipe ----
echo -n "[CHECK] MediaPipe..."
if python3 -c "import mediapipe" 2>/dev/null; then
    echo " OK"
else
    echo " 未安装"
    echo "  安装: pip install mediapipe"
    exit 1
fi

# ---- 环境变量 ----
# 禁用 OpenCV 的 GUI 后端冲突
export OPENCV_VIDEOIO_PRIORITY_MSMF=0
export OPENCV_VIDEOIO_PRIORITY_V4L2=100

# MediaPipe 日志级别
export GLOG_minloglevel=2

# ---- 启动 ----
echo ""
echo "[INFO] 启动主程序..."
echo ""

python3 main.py "$@"
