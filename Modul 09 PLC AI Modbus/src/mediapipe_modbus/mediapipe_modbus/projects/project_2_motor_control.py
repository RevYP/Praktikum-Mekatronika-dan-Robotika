#!/usr/bin/env python3
"""
Project 2 Template: Motor Start-Stop Control with MediaPipe Gestures
Location: src/mediapipe_modbus/mediapipe_modbus/projects/project_2_motor_control.py

This is a template implementation. Students can modify this to create their own logic.
"""

import time
from enum import Enum
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List


class MotorState(Enum):
    """Possible motor states"""
    STOPPED = 0
    RUNNING = 1
    FAULT = 2
    E_STOPPED = 3  # Emergency stop


@dataclass
class GestureResult:
    """Result of gesture analysis"""
    gesture_type: str  # 'THUMB', 'FIST', 'OK', 'ALL_OPEN', 'UNKNOWN'
    confidence: float  # 0.0-1.0
    frame_count: int   # How many frames have been consistent


class MotorController:
    """
    Controls a motor via Modbus TCP based on hand gestures
    
    Gesture Mapping:
    - Jempol saja (thumb only) → START motor
    - Kepalan (fist) → STOP motor  
    - OK gesture (2 fingers) → RESET fault
    - Semua buka (all open) → EMERGENCY STOP
    """
    
    def __init__(self, modbus_client, debounce_frames: int = 3, debug: bool = False):
        """
        Initialize motor controller
        
        Args:
            modbus_client: pymodbus ModbusTcpClient instance
            debounce_frames: Frames needed to confirm gesture (default: 3)
            debug: Enable debug logging
        """
        self.client = modbus_client
        self.state = MotorState.STOPPED
        self.fault_flag = False
        self.debug = debug
        
        # Gesture debounce: track last N finger states
        self.debounce_frames = debounce_frames
        self.gesture_history = deque(maxlen=debounce_frames)
        
        # Modbus coil mapping
        self.coils = {
            'motor_run': 0,           # Output to enable motor
            'motor_direction': 1,     # Forward/reverse
            'fault_reset': 2,         # Reset fault
            'emergency_stop': 3,      # Hard stop
            'status_running': 4,      # Status feedback from PLC
            'status_fault': 5,        # Fault status from PLC
        }
        
        # Gesture names
        self.finger_names = ['Jempol', 'Telunjuk', 'Tengah', 'Manis', 'Kelingking']
    
    def analyze_gesture(self, finger_states: List[bool]) -> GestureResult:
        """
        Analyze current finger state and classify gesture
        
        finger_states: [thumb, index, middle, ring, pinky]
                       True = open, False = closed
        
        Returns: GestureResult with gesture_type and confidence
        """
        open_count = sum(finger_states)
        
        # Gesture 1: Jempol saja (thumb open, lainnya tutup)
        if finger_states[0] and open_count == 1:
            return GestureResult('THUMB', 1.0, 0)
        
        # Gesture 2: Kepalan (semua tutup)
        if open_count == 0:
            return GestureResult('FIST', 1.0, 0)
        
        # Gesture 3: OK gesture (telunjuk + tengah buka)
        if finger_states[1] and finger_states[2] and open_count == 2:
            return GestureResult('OK', 0.95, 0)
        
        # Gesture 4: Semua buka
        if open_count == 5:
            return GestureResult('ALL_OPEN', 1.0, 0)
        
        # Unknown gesture
        return GestureResult('UNKNOWN', 0.0, 0)
    
    def process_gesture(self, finger_states: List[bool]) -> Optional[GestureResult]:
        """
        Process gesture with debouncing
        
        Returns: GestureResult if gesture is confident and consistent,
                 None if insufficient frames or gesture changed
        """
        # Analyze current frame
        result = self.analyze_gesture(finger_states)
        self.gesture_history.append((finger_states, result))
        
        # Need minimum frames to make decision
        if len(self.gesture_history) < self.debounce_frames:
            return None
        
        # Check if gesture is consistent across all buffered frames
        first_gesture = self.gesture_history[0][1].gesture_type
        consistent = True
        
        for _, hist_result in self.gesture_history:
            if hist_result.gesture_type != first_gesture:
                consistent = False
                break
        
        # Return result only if consistent and confident
        if consistent and result.confidence >= 0.8:
            frame_count = len(self.gesture_history)
            return GestureResult(result.gesture_type, result.confidence, frame_count)
        
        return None
    
    def execute_action(self, gesture_result: GestureResult) -> Tuple[bool, str]:
        """
        Execute motor action based on gesture
        
        Returns: (success, message) tuple
        """
        gesture = gesture_result.gesture_type
        
        if gesture == 'THUMB':
            return self._start_motor()
        elif gesture == 'FIST':
            return self._stop_motor()
        elif gesture == 'OK':
            return self._reset_fault()
        elif gesture == 'ALL_OPEN':
            return self._emergency_stop()
        else:
            return False, "Unknown gesture"
    
    def _start_motor(self) -> Tuple[bool, str]:
        """Execute: START motor"""
        try:
            # Set motor run flag
            self.client.write_coil(self.coils['motor_run'], True, unit=1)
            self.client.write_coil(self.coils['motor_direction'], True, unit=1)
            self.state = MotorState.RUNNING
            self.fault_flag = False
            
            if self.debug:
                print("[DEBUG] Motor START command sent")
            
            return True, "Motor started"
        except Exception as e:
            return False, f"Start error: {e}"
    
    def _stop_motor(self) -> Tuple[bool, str]:
        """Execute: STOP motor normally"""
        try:
            self.client.write_coil(self.coils['motor_run'], False, unit=1)
            self.state = MotorState.STOPPED
            
            if self.debug:
                print("[DEBUG] Motor STOP command sent")
            
            return True, "Motor stopped"
        except Exception as e:
            return False, f"Stop error: {e}"
    
    def _reset_fault(self) -> Tuple[bool, str]:
        """Execute: RESET fault condition"""
        try:
            # Pulse the reset signal
            self.client.write_coil(self.coils['fault_reset'], True, unit=1)
            time.sleep(0.05)  # 50ms pulse
            self.client.write_coil(self.coils['fault_reset'], False, unit=1)
            
            self.fault_flag = False
            
            if self.debug:
                print("[DEBUG] Fault RESET command sent")
            
            return True, "Fault reset"
        except Exception as e:
            return False, f"Reset error: {e}"
    
    def _emergency_stop(self) -> Tuple[bool, str]:
        """Execute: EMERGENCY hard stop"""
        try:
            # Set all stop flags
            self.client.write_coil(self.coils['motor_run'], False, unit=1)
            self.client.write_coil(self.coils['emergency_stop'], True, unit=1)
            
            self.state = MotorState.E_STOPPED
            
            if self.debug:
                print("[DEBUG] EMERGENCY STOP!")
            
            return True, "EMERGENCY STOP activated"
        except Exception as e:
            return False, f"E-Stop error: {e}"
    
    def read_status(self) -> dict:
        """Read current motor status from PLC"""
        try:
            # Read status coils from PLC
            result = self.client.read_coils(
                self.coils['status_running'], 
                count=2,  # Read 2 coils: running, fault
                unit=1
            )
            
            if not result.isError():
                return {
                    'is_running': bool(result.bits[0]),
                    'has_fault': bool(result.bits[1]),
                }
            else:
                return {'error': 'Read failed'}
        except Exception as e:
            return {'error': str(e)}
    
    def get_state_string(self) -> str:
        """Get human-readable state"""
        states = {
            MotorState.STOPPED: "STOPPED",
            MotorState.RUNNING: "RUNNING",
            MotorState.FAULT: "FAULT",
            MotorState.E_STOPPED: "E-STOP",
        }
        return states.get(self.state, "UNKNOWN")


# ============================================================================
# USAGE EXAMPLE - How to integrate with main GUI
# ============================================================================

"""
In mediapipe_modbus_ros_node.py:

# 1. Add import at top:
from .projects.project_2_motor_control import MotorController

# 2. In VBoxController.__init__, after modbus is connected:
self._motor = None

# 3. In _on_modbus_connected callback:
if self._motor is None:
    self._motor = MotorController(
        modbus_client=modbus_client,
        debounce_frames=3,
        debug=True
    )
    self._log("Motor controller initialized", "OK")

# 4. In _on_gesture callback:
def _on_gesture(self, states: list, hand_detected: bool):
    # ... existing code ...
    
    # NEW: Motor control
    if self._motor and self._connected_ips:
        gesture_result = self._motor.process_gesture(states)
        if gesture_result:
            success, message = self._motor.execute_action(gesture_result)
            if success:
                self._log(f"Motor: {message}", "OK")
            else:
                self._log(f"Motor: {message}", "ERROR")

# 5. Optionally, periodically read motor status:
def _read_motor_status(self):
    if self._motor:
        status = self._motor.read_status()
        self._log(f"Motor status: {status}", "INFO")
"""


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Quick test of gesture recognition without real Modbus
    Usage: python3 project_2_motor_control.py
    """
    
    # Mock client for testing
    class MockModbusClient:
        def write_coil(self, addr, value, unit=1):
            print(f"  [MODBUS] Write coil {addr} = {int(value)}")
        
        def read_coils(self, addr, count=1, unit=1):
            class MockResult:
                bits = [False] * count
                def isError(self): return False
            return MockResult()
    
    print("=" * 60)
    print("PROJECT 2: Motor Control - Gesture Test")
    print("=" * 60)
    print()
    
    # Create controller with mock client
    motor = MotorController(MockModbusClient(), debounce_frames=2, debug=True)
    
    # Test cases
    test_cases = [
        (
            [[True, False, False, False, False]] * 3,
            "Jempol saja (START)"
        ),
        (
            [[False, False, False, False, False]] * 3,
            "Kepalan (STOP)"
        ),
        (
            [[False, True, True, False, False]] * 3,
            "OK gesture (RESET)"
        ),
        (
            [[True, True, True, True, True]] * 3,
            "Semua buka (E-STOP)"
        ),
    ]
    
    for frames, description in test_cases:
        print(f"\nTest: {description}")
        print("-" * 40)
        
        for i, frame_states in enumerate(frames, 1):
            result = motor.process_gesture(frame_states)
            if result:
                print(f"Frame {i}: Gesture recognized: {result.gesture_type}")
                success, msg = motor.execute_action(result)
                print(f"         Result: {msg}")
            else:
                print(f"Frame {i}: Processing... (need {motor.debounce_frames} frames)")
    
    print()
    print("=" * 60)
    print("Test completed!")
    print("=" * 60)

