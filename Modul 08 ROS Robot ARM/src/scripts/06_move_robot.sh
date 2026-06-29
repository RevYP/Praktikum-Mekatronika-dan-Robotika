#!/bin/bash
set -e

echo "=============================================="
echo "  GERAKKAN DOBOT MAGICIAN MANUAL"
echo "  Pastikan 02_run_dobot.sh sudah berjalan"
echo "  dan homing sudah dilakukan (04_homing.sh)"
echo "=============================================="
echo ""

# Source ROS2 dan workspace
source /opt/ros/humble/setup.bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../install/setup.bash"

echo "Masukkan koordinat tujuan (dalam mm):"
echo "  X: jarak ke depan (150-300)"
echo "  Y: jarak ke samping (-150 sampai 150)"
echo "  Z: ketinggian (-30 sampai 150)"
echo "  R: rotasi end effector (-90 sampai 90)"
echo ""

read -p "X (default 200): " X
read -p "Y (default 0): " Y
read -p "Z (default 100): " Z
read -p "R (default 0): " R

X=${X:-200}
Y=${Y:-0}
Z=${Z:-100}
R=${R:-0}

echo ""
echo "Pilih tipe gerakan:"
echo "  1) Joint interpolated (lebih cepat)"
echo "  2) Linear (garis lurus)"
echo ""
read -p "Pilihan [1-2] (default: 1): " MOTION

case $MOTION in
    2) MOTION_TYPE=2 ;;
    *) MOTION_TYPE=1 ;;
esac

read -p "Kecepatan [0.1-1.0] (default 0.5): " VEL
read -p "Akselerasi [0.1-1.0] (default 0.3): " ACC

VEL=${VEL:-0.5}
ACC=${ACC:-0.3}

echo ""
echo "[INFO] Mengirim perintah gerak:"
echo "       Target: [$X, $Y, $Z, $R]"
echo "       Motion type: $MOTION_TYPE"
echo "       Velocity: $VEL, Acceleration: $ACC"
echo ""

ros2 action send_goal /PTP_action dobot_msgs/action/PointToPoint \
  "{motion_type: $MOTION_TYPE, target_pose: [$X, $Y, $Z, $R], velocity_ratio: $VEL, acceleration_ratio: $ACC}" --feedback
