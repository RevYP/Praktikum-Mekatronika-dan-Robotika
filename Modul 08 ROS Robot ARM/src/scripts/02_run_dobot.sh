#!/bin/bash
set -e

echo "=============================================="
echo "  MENJALANKAN DOBOT MAGICIAN (ROBOT FISIK)"
echo "=============================================="
echo ""
echo "Pastikan:"
echo "  1. Dobot terhubung via USB ke komputer"
echo "  2. Dobot sudah dinyalakan (tombol power)"
echo "  3. Sudah menjalankan 01_setup_install.sh"
echo ""

# Source ROS2 dan workspace
source /opt/ros/humble/setup.bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../install/setup.bash"

# Pilih tool yang terpasang
echo "Pilih end effector yang terpasang pada Dobot:"
echo "  1) none (tanpa tool)"
echo "  2) pen"
echo "  3) suction_cup"
echo "  4) gripper"
echo "  5) extended_gripper"
echo ""
read -p "Masukkan pilihan [1-5] (default: 1): " TOOL_CHOICE

case $TOOL_CHOICE in
    2) TOOL="pen" ;;
    3) TOOL="suction_cup" ;;
    4) TOOL="gripper" ;;
    5) TOOL="extended_gripper" ;;
    *) TOOL="none" ;;
esac

export MAGICIAN_TOOL=$TOOL
echo ""
echo "[INFO] Tool dipilih: $TOOL"
echo "[INFO] Menjalankan control system..."
echo ""
echo "=============================================="
echo "  Setelah sistem berjalan, buka terminal BARU"
echo "  lalu jalankan:"
echo "    ./scripts/04_homing.sh"
echo "=============================================="
echo ""

ros2 launch dobot_bringup dobot_magician_control_system.launch.py
