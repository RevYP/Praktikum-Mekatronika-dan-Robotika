#!/bin/bash
set -e

echo "=============================================="
echo "  HOMING DOBOT MAGICIAN"
echo "  Wajib dijalankan PERTAMA KALI setelah"
echo "  02_run_dobot.sh berjalan"
echo "=============================================="
echo ""
echo "PERINGATAN: Robot akan bergerak ke posisi home!"
echo "Pastikan area sekitar robot aman."
echo ""
read -p "Tekan ENTER untuk mulai homing..."

# Source ROS2 dan workspace
source /opt/ros/humble/setup.bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../install/setup.bash"

echo ""
echo "[INFO] Menjalankan homing procedure..."
ros2 service call /dobot_homing_service dobot_msgs/srv/ExecuteHomingProcedure

echo ""
echo "=============================================="
echo "  HOMING SELESAI!"
echo "  Robot siap digunakan."
echo "=============================================="
echo ""
echo "Selanjutnya bisa jalankan:"
echo "  ./scripts/05_demo.sh          (demo gerakan)"
echo "  ./scripts/06_move_robot.sh    (gerakkan manual)"
echo "  ./scripts/07_control_panel.sh (GUI kontrol)"
echo ""
