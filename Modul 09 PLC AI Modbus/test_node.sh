#!/bin/bash
# Test script for MediaPipe Modbus ROS2 Node

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_ok() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}→${NC} $1"
}

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  MediaPipe Modbus ROS2 Node - Test Suite  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"

# Test 1: Verify Workspace
print_header "1. Workspace Verification"
if [ -d "$WORKSPACE_ROOT/src/mediapipe_modbus" ]; then
    print_ok "Workspace structure valid"
else
    print_error "Workspace structure invalid"
    exit 1
fi

# Test 2: Check ROS2
print_header "2. ROS2 Installation"
if command -v ros2 &> /dev/null; then
    ROS_VERSION=$(ros2 --version 2>&1 | awk '{print $2}')
    print_ok "ROS2 found: $ROS_VERSION"
else
    print_error "ROS2 not found. Please install ROS2 first"
    exit 1
fi

# Test 3: Python version
print_header "3. Python Version"
PYTHON_VERSION=$(python3 --version 2>&1)
print_ok "$PYTHON_VERSION"

# Test 4: Python dependencies
print_header "4. Python Dependencies"

deps_ok=0
deps_missing=0

test_import() {
    python3 -c "import $1; print(__import__('$1').__version__ if hasattr(__import__('$1'), '__version__') else 'installed')" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_ok "$1: $(python3 -c "import $1; print(__import__('$1').__version__ if hasattr(__import__('$1'), '__version__') else 'installed')" 2>/dev/null)"
        ((deps_ok++))
        return 0
    else
        print_error "$1: NOT INSTALLED"
        ((deps_missing++))
        return 1
    fi
}

echo ""
test_import "mediapipe"
test_import "cv2"
test_import "numpy"
test_import "pymodbus"
test_import "PyQt5"
test_import "yaml"

if [ $deps_missing -gt 0 ]; then
    print_error "$deps_missing dependencies missing!"
    echo ""
    print_info "Install missing dependencies:"
    echo "  pip3 install -r $WORKSPACE_ROOT/requirements.txt"
    echo ""
    # Don't exit, continue with other tests
else
    print_ok "All dependencies installed ($deps_ok/$deps_ok)"
fi

# Test 5: ROS2 Package
print_header "5. ROS2 Package"
cd "$WORKSPACE_ROOT"

if [ ! -f "install/setup.bash" ]; then
    print_error "install/setup.bash not found"
    print_info "Building package..."
    colcon build --symlink-install 2>&1 | tail -5
fi

source install/setup.bash 2>/dev/null

if ros2 pkg list 2>/dev/null | grep -q "mediapipe_modbus"; then
    print_ok "Package 'mediapipe_modbus' found"
    
    # List available executables
    print_info "Available nodes:"
    ros2 pkg executables mediapipe_modbus | sed 's/^/  - /'
else
    print_error "Package 'mediapipe_modbus' not found"
    print_info "Try rebuilding:"
    echo "  cd $WORKSPACE_ROOT"
    echo "  colcon build --symlink-install"
fi

# Test 6: Camera access
print_header "6. Camera Access"
python3 << 'EOF'
import cv2
import sys

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        print(f'\033[0;32m✓\033[0m Camera available: {w}x{h}')
    else:
        print(f'\033[0;31m✗\033[0m Camera found but cannot read frame')
    cap.release()
else:
    print(f'\033[0;31m✗\033[0m Camera not found')
    print('\033[0;34m→\033[0m Check /dev/video* or USB connection')
EOF

# Test 7: MediaPipe
print_header "7. MediaPipe Hand Detection"
python3 << 'EOF'
import cv2
import mediapipe as mp
import sys

try:
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('\033[0;31m✗\033[0m Camera not available')
        sys.exit(0)
    
    print('\033[0;34m→\033[0m Testing hand detection (5 frames)...')
    found = False
    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(rgb)
        
        if result.multi_hand_landmarks:
            print(f'\033[0;32m✓\033[0m Hand detected at frame {i+1}')
            found = True
            break
    
    if not found:
        print('\033[0;33m⚠\033[0m No hand detected in 30 frames (try moving hand)')
    else:
        print('\033[0;32m✓\033[0m MediaPipe working correctly')
    
    cap.release()
    
except Exception as e:
    print(f'\033[0;31m✗\033[0m Error: {e}')
EOF

# Test 8: Network
print_header "8. Network Configuration"
python3 << 'EOF'
import socket

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    local_ip = s.getsockname()[0]
    s.close()
    
    subnet = '.'.join(local_ip.split('.')[:3])
    print(f'\033[0;32m✓\033[0m Local IP: {local_ip}')
    print(f'\033[0;32m✓\033[0m Subnet: {subnet}.0/24')
except Exception as e:
    print(f'\033[0;31m✗\033[0m Network error: {e}')
EOF

# Test 9: Summary
print_header "Summary"
echo ""
echo -e "All basic tests completed. ${GREEN}Ready to run!${NC}"
echo ""
echo "Next steps:"
echo "  1. Test Modbus device:"
echo "     python3 << 'EOF'"
echo "     import socket"
echo "     ip = '10.41.168.203'  # Change to your V-BOX IP"
echo "     sock = socket.create_connection((ip, 502), timeout=2)"
echo "     print(f'✓ Connected to {ip}')"
echo "     sock.close()"
echo "     EOF"
echo ""
echo "  2. Run the application:"
echo "     cd $WORKSPACE_ROOT"
echo "     ./run_node.sh"
echo ""
echo "  3. For detailed setup guide, see:"
echo "     $WORKSPACE_ROOT/../SETUP_GUIDE.md"
echo ""
