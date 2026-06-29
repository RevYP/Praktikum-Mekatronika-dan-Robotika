#!/bin/bash
set -e

echo "=============================================="
echo "  DEMO DOBOT MAGICIAN"
echo "  Pastikan 02_run_dobot.sh sudah berjalan"
echo "  dan homing sudah dilakukan (04_homing.sh)"
echo "=============================================="
echo ""

# Source ROS2 dan workspace
source /opt/ros/humble/setup.bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../install/setup.bash"

echo "Pilih demo yang ingin dijalankan:"
echo "  1) Point to Point (gerak ke beberapa titik)"
echo "  2) Pick and Place (ambil dan taruh objek)"
echo "  3) Test Gripper (buka tutup gripper)"
echo "  4) Test Suction Cup (nyala/mati suction)"
echo "  5) Test Homing"
echo ""
read -p "Masukkan pilihan [1-5]: " DEMO_CHOICE

case $DEMO_CHOICE in
    1) DEMO="test_point_to_point" ;;
    2) DEMO="test_pick_and_place" ;;
    3) DEMO="test_gripper" ;;
    4) DEMO="test_suction_cup" ;;
    5) DEMO="test_homing" ;;
    *) echo "Pilihan tidak valid"; exit 1 ;;
esac

echo ""
echo "[INFO] Menjalankan demo: $DEMO"
echo ""

ros2 run dobot_demos $DEMO
