#!/bin/bash
set -e

echo "=============================================="
echo "  DOBOT MAGICIAN CONTROL PANEL (GUI)"
echo "  Pastikan 02_run_dobot.sh sudah berjalan"
echo "=============================================="
echo ""

# Source ROS2 dan workspace
source /opt/ros/humble/setup.bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../install/setup.bash"

echo "[INFO] Membuka Control Panel..."
echo "       Gunakan GUI untuk mengontrol robot."
echo ""

rqt -s dobot_control_panel
