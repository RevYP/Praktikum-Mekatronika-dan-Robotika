#!/bin/bash
set -e

echo "=============================================="
echo "  SETUP DOBOT MAGICIAN ROS2 HUMBLE"
echo "  Script ini akan menginstall semua yang"
echo "  dibutuhkan dari NOL (Ubuntu 22.04)"
echo "=============================================="
echo ""

# --- 1. Update sistem ---
echo "[1/7] Update sistem..."
sudo apt update && sudo apt upgrade -y

# --- 2. Install ROS 2 Humble ---
echo "[2/7] Install ROS 2 Humble Desktop..."
sudo apt install -y software-properties-common
sudo add-apt-repository universe -y
sudo apt update
sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-desktop

# --- 3. Install colcon build tools ---
echo "[3/7] Install colcon dan tools..."
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-pip

# --- 4. Init rosdep ---
echo "[4/7] Setup rosdep..."
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update

# --- 5. Install dependencies tambahan ---
echo "[5/7] Install dependencies tambahan..."
sudo apt install -y ros-humble-diagnostic-aggregator ros-humble-rqt-robot-monitor python3-pykdl
pip3 install -r "$(dirname "$0")/../requirements.txt"

# --- 6. Tambah user ke group dialout (akses serial port) ---
echo "[6/7] Menambahkan user ke group dialout..."
sudo usermod -a -G dialout $USER

# --- 7. Build workspace ---
echo "[7/7] Build workspace dengan colcon..."
source /opt/ros/humble/setup.bash
cd "$(dirname "$0")/.."
colcon build

# --- 8. Tambah source otomatis ke bashrc ---
WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if ! grep -q "magician_ros2" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Dobot Magician ROS2" >> ~/.bashrc
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    echo "source ${WORKSPACE_DIR}/install/setup.bash" >> ~/.bashrc
    echo "[INFO] Ditambahkan source otomatis ke ~/.bashrc"
fi

echo ""
echo "=============================================="
echo "  SETUP SELESAI!"
echo "=============================================="
echo ""
echo "PENTING: Restart komputer atau logout/login"
echo "         agar akses serial port aktif."
echo ""
echo "Setelah restart, jalankan script berikutnya:"
echo "  ./scripts/02_run_dobot.sh        (kontrol robot)"
echo "  ./scripts/03_run_rviz_only.sh    (visualisasi saja)"
echo ""
