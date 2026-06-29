#!/bin/bash
set -e

echo "=============================================="
echo "  DOBOT MAGICIAN DUMMY SIMULATOR"
echo "  Tidak perlu robot fisik!"
echo "  Semua fitur bisa dicoba secara virtual."
echo "=============================================="
echo ""

# Source ROS2 dan workspace
source /opt/ros/humble/setup.bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../install/setup.bash"

echo "Pilih konfigurasi robot:"
echo "  1) 4 DOF + gripper (default)"
echo "  2) 4 DOF + extended_gripper"
echo "  3) 4 DOF + suction_cup"
echo ""
read -p "Masukkan pilihan [1-3] (default: 1): " CONFIG_CHOICE

case $CONFIG_CHOICE in
    2) TOOL="extended_gripper" ;;
    3) TOOL="suction_cup" ;;
    *) TOOL="gripper" ;;
esac

echo ""
echo "[INFO] Menjalankan Dobot Dummy Simulator + RViz"
echo "       Tool: $TOOL"
echo ""
echo "=============================================="
echo "  Setelah RViz terbuka, buka terminal BARU"
echo "  lalu coba perintah berikut:"
echo ""
echo "  # Homing (reset posisi)"
echo "  ./scripts/04_homing.sh"
echo ""
echo "  # Gerakkan robot"
echo "  ./scripts/06_move_robot.sh"
echo ""
echo "  # Atau langsung via command:"
echo "  source install/setup.bash"
echo "  ros2 action send_goal /PTP_action dobot_msgs/action/PointToPoint \\"
echo "    \"{motion_type: 1, target_pose: [200.0, 50.0, 100.0, 0.0], velocity_ratio: 0.5, acceleration_ratio: 0.3}\" --feedback"
echo "=============================================="
echo ""

ros2 launch dobot_dummy dummy_bringup.launch.py tool:=$TOOL
