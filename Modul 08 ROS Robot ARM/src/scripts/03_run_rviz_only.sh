#!/bin/bash
set -e

echo "=============================================="
echo "  VISUALISASI DOBOT DI RVIZ (TANPA ROBOT)"
echo "  Bisa digunakan tanpa robot fisik terhubung"
echo "=============================================="
echo ""

# Source ROS2 dan workspace
source /opt/ros/humble/setup.bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../install/setup.bash"

echo "Pilih konfigurasi robot:"
echo "  1) 4 DOF + gripper"
echo "  2) 4 DOF + extended_gripper"
echo "  3) 4 DOF + suction_cup"
echo "  4) 4 DOF + pen"
echo "  5) 4 DOF tanpa tool"
echo ""
read -p "Masukkan pilihan [1-5] (default: 1): " CONFIG_CHOICE

case $CONFIG_CHOICE in
    2) TOOL="extended_gripper"; CAM="false" ;;
    3) TOOL="suction_cup"; CAM="false" ;;
    4) TOOL="pen"; CAM="false" ;;
    5) TOOL="none"; CAM="false" ;;
    *) TOOL="gripper"; CAM="false" ;;
esac

echo ""
read -p "Tampilkan kamera di model? [y/N]: " USE_CAM
if [[ "$USE_CAM" == "y" || "$USE_CAM" == "Y" ]]; then
    CAM="true"
fi

echo ""
echo "[INFO] Menjalankan RViz dengan:"
echo "       DOF=4, tool=$TOOL, camera=$CAM, gui=true"
echo ""
echo "Gunakan slider Joint State Publisher untuk"
echo "menggerakkan model robot di RViz."
echo ""

ros2 launch dobot_description display.launch.py DOF:=4 tool:=$TOOL use_camera:=$CAM gui:=true
