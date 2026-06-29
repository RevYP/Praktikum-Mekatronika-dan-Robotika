#!/bin/bash
# Run MediaPipe Modbus ROS2 Node
# Usage: ./run_node.sh [node_type]
# node_type: gui (default), camera, mediapipe, modbus, integrated

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if workspace is valid
if [ ! -d "$WORKSPACE_ROOT/src/mediapipe_modbus" ]; then
    print_error "Workspace tidak valid. Jalankan script dari direktori ROS_PLC_AI"
    exit 1
fi

# Change to workspace directory
cd "$WORKSPACE_ROOT"

# Check if build/install directories exist
if [ ! -d "install" ]; then
    print_warn "Directory 'install' tidak ditemukan. Melakukan build terlebih dahulu..."
    
    if [ ! -d "build" ]; then
        mkdir -p build
    fi
    
    if [ -x "$(command -v colcon)" ]; then
        print_info "Building workspace dengan colcon..."
        colcon build --symlink-install
        if [ $? -ne 0 ]; then
            print_error "Build gagal!"
            exit 1
        fi
    else
        print_error "colcon tidak ditemukan. Instalasi dengan: pip3 install colcon-common-extensions"
        exit 1
    fi
fi

# Source setup file
print_info "Sourcing ROS2 environment..."
if [ ! -f "install/setup.bash" ]; then
    print_error "setup.bash tidak ditemukan di install directory"
    exit 1
fi

source install/setup.bash

# Verify package is available
print_info "Verifying package..."
ros2 pkg list | grep -q "mediapipe_modbus"
if [ $? -ne 0 ]; then
    print_error "Package 'mediapipe_modbus' not found after build"
    exit 1
fi

print_success "Package verified"

# Get node type (default: gui)
NODE_TYPE="${1:-gui}"

case "$NODE_TYPE" in
    gui|integrated)
        print_info "Starting MediaPipe Modbus GUI Node..."
        python3 -m mediapipe_modbus.mediapipe_modbus_ros_node
        ;;
    camera)
        print_info "Starting Camera Node..."
        ros2 run mediapipe_modbus camera_node
        ;;
    mediapipe)
        print_info "Starting MediaPipe Node..."
        ros2 run mediapipe_modbus mediapipe_node
        ;;
    modbus)
        print_info "Starting Modbus Node..."
        ros2 run mediapipe_modbus modbus_node
        ;;
    all)
        print_info "Starting all nodes..."
        echo ""
        echo "Recommended: Open 4 terminals and run each:"
        echo "  Terminal 1: $0 camera"
        echo "  Terminal 2: $0 mediapipe"
        echo "  Terminal 3: $0 modbus"
        echo "  Terminal 4: $0 gui"
        exit 0
        ;;
    *)
        print_error "Unknown node type: $NODE_TYPE"
        echo ""
        echo "Usage: $0 [node_type]"
        echo ""
        echo "Available node types:"
        echo "  gui          - Run integrated GUI (default)"
        echo "  integrated   - Alias for 'gui'"
        echo "  camera       - Run only camera node"
        echo "  mediapipe    - Run only mediapipe node"
        echo "  modbus       - Run only modbus node"
        echo "  all          - Print instructions for running all nodes"
        echo ""
        echo "Example: $0 gui"
        exit 1
        ;;
esac
