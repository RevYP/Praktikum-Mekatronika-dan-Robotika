#!/usr/bin/env python3
"""
ROS Node: MediaPipe Hand Gesture + Modbus TCP Controller
Features:
1. Camera capture + MediaPipe processing
2. PyQt5 GUI with settings persistence (YAML)
3. Modbus TCP communication with custom register mapping
4. IP scanning, finger-to-register mapping, editable PLC LED display
"""

import sys
import os
import yaml
import rclpy
from rclpy.node import Node
import socket
import subprocess
import platform
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QGroupBox, QGridLayout,
    QTextEdit, QFrame, QSizePolicy, QProgressBar, QCheckBox, QDialog,
    QTabWidget, QSpinBox, QListWidget, QListWidgetItem, QDialogButtonBox,
    QFormLayout, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QImage, QPixmap

# ── MediaPipe ─────────────────────────────────────────────────────────────────
_MP_AVAILABLE = False
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    pass

# ─── pymodbus compatibility (v2.x dan v3.x) ──────────────────────────────────
_PYMODBUS_VERSION = 0   # 0 = tidak ada, 2 = v2.x, 3 = v3.x
ModbusTcpClient = None
ModbusException = Exception

try:
    from pymodbus.client import ModbusTcpClient as _C3       # pymodbus 3.x
    from pymodbus.exceptions import ModbusException as _ME
    ModbusTcpClient = _C3
    ModbusException = _ME
    _PYMODBUS_VERSION = 3
except ImportError:
    try:
        from pymodbus.client.sync import ModbusTcpClient as _C2  # pymodbus 2.x
        from pymodbus.exceptions import ModbusException as _ME
        ModbusTcpClient = _C2
        ModbusException = _ME
        _PYMODBUS_VERSION = 2
    except ImportError:
        pass

# ── Config Handling ───────────────────────────────────────────────────────────
CONFIG_PATH = os.path.expanduser('~/.ros/mediapipe_modbus_config.yaml')

DEFAULT_CONFIG = {
    'modbus': {
        'port': 502,
        'unit_id': 1,
        'known_ips': ['10.41.168.203', '10.41.168.185', '10.41.168.174'],
        'scan_subnet': None
    },
    'finger_mapping': [
        {'finger': 'Jempol', 'register_type': 'coil', 'address': 0, 'enabled': True},
        {'finger': 'Telunjuk', 'register_type': 'coil', 'address': 1, 'enabled': True},
        {'finger': 'Tengah', 'register_type': 'coil', 'address': 2, 'enabled': True},
        {'finger': 'Manis', 'register_type': 'coil', 'address': 3, 'enabled': True},
        {'finger': 'Kelingking', 'register_type': 'coil', 'address': 4, 'enabled': True}
    ],
    'plc_led_mapping': [
        {'label': 'M0', 'register_type': 'coil', 'address': 0, 'enabled': True},
        {'label': 'M1', 'register_type': 'coil', 'address': 1, 'enabled': True},
        {'label': 'M2', 'register_type': 'coil', 'address': 2, 'enabled': True},
        {'label': 'M3', 'register_type': 'coil', 'address': 3, 'enabled': True},
        {'label': 'M4', 'register_type': 'coil', 'address': 4, 'enabled': True}
    ],
    'gui': {
        'window_width': 1220,
        'window_height': 700,
        'camera_width': 640,
        'camera_height': 480
    }
}

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)
                return config if config else DEFAULT_CONFIG
        except Exception as e:
            print(f"Failed to load config: {e}")
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG

def save_config(config):
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Config saved to {CONFIG_PATH}")
    except Exception as e:
        print(f"Failed to save config: {e}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "Tidak diketahui"

def get_subnet_base() -> str:
    ip = get_local_ip()
    parts = ip.split(".")
    if len(parts) == 4:
        return ".".join(parts[:3])
    return "192.168.1"

def tcp_check(ip: str, port: int, timeout: float = 2.0) -> bool:
    try:
        conn = socket.create_connection((ip, port), timeout=timeout)
        conn.close()
        return True
    except Exception:
        return False

def _check_host(ip: str, port: int = 502, timeout: float = 2.5) -> dict:
    modbus = False
    alive  = False
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        if result == 0:
            modbus = True
            alive  = True
    except Exception:
        pass

    if not alive:
        try:
            is_win = platform.system() == "Windows"
            cmd = ["ping", "-n" if is_win else "-c", "1",
                   "-w" if is_win else "-W", "300" if is_win else "1", ip]
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=1.5)
            alive = (r.returncode == 0)
        except Exception:
            pass

    return {"ip": ip, "alive": alive, "modbus": modbus}

# ── Gesture Thread (Kamera + MediaPipe) ───────────────────────────────────────
class GestureThread(QThread):
    frame_ready     = pyqtSignal(object)
    gesture_changed = pyqtSignal(list, bool)
    cam_error       = pyqtSignal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self._running      = True
        self._last_states  = [-1] * 5
        self._hand_was_present = False

    def stop(self):
        self._running = False

    def run(self):
        if not _MP_AVAILABLE:
            self.cam_error.emit("mediapipe tidak terinstall. Jalankan: pip install mediapipe")
            return

        mp_hands = mp.solutions.hands
        mp_draw  = mp.solutions.drawing_utils
        hands    = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        cam_width = self.config['gui']['camera_width']
        cam_height = self.config['gui']['camera_height']
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.cam_error.emit("Kamera tidak ditemukan. Pastikan webcam terpasang.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            finger_states = [False] * 5
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                mp_draw.draw_landmarks(
                    frame, results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,200,255), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(255,230,0), thickness=2)
                )
                finger_states[0] = lm[4].x < lm[3].x
                finger_states[1] = lm[8].y < lm[6].y
                finger_states[2] = lm[12].y < lm[10].y
                finger_states[3] = lm[16].y < lm[14].y
                finger_states[4] = lm[20].y < lm[18].y

            overlay_labels = ["D0 Jempol", "D1 Telunjuk", "D2 Tengah", "D3 Manis", "D4 Kelingking"]
            for i, (state, lbl) in enumerate(zip(finger_states, overlay_labels)):
                color = (30,230,80) if state else (100,100,100)
                cv2.putText(frame, f"{lbl} = {'ON ' if state else 'OFF'}",
                           (8, 24 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

            hand_detected = bool(results.multi_hand_landmarks)
            int_states = [int(s) for s in finger_states]
            if hand_detected:
                if int_states != self._last_states or not self._hand_was_present:
                    self._last_states = int_states
                    self.gesture_changed.emit(finger_states, True)
            else:
                if self._hand_was_present:
                    self.gesture_changed.emit([False]*5, False)
                    self._last_states = [-1]*5
            self._hand_was_present = hand_detected
            self.frame_ready.emit(frame)

        cap.release()
        hands.close()

# ── Network Scanner Thread ────────────────────────────────────────────────────
class NetworkScannerThread(QThread):
    found    = pyqtSignal(str, bool)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(int)

    def __init__(self, subnet_base: str, port: int = 502, workers: int = 100, parent=None):
        super().__init__(parent)
        self.subnet_base = subnet_base
        self.port        = port
        self.workers     = workers
        self._stop       = False

    def stop(self):
        self._stop = True

    def run(self):
        total   = 254
        done    = 0
        found_n = 0
        ips = [f"{self.subnet_base}.{i}" for i in range(1,255)]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = {ex.submit(_check_host, ip, self.port): ip for ip in ips}
            for future in as_completed(futures):
                if self._stop:
                    break
                done += 1
                self.progress.emit(done, total)
                try:
                    res = future.result()
                    if res["alive"]:
                        self.found.emit(res["ip"], res["modbus"])
                        if res["modbus"]:
                            found_n += 1
                except Exception:
                    pass
        self.finished.emit(found_n)

# ── Modbus Worker Thread ─────────────────────────────────────────────────────
class ModbusWorkerThread(QThread):
    sig_client_connected    = pyqtSignal(str, int)
    sig_client_disconnected = pyqtSignal(str)
    sig_all_disconnected    = pyqtSignal()
    sig_coil_result         = pyqtSignal(int, bool, bool, str)
    sig_led_states          = pyqtSignal(list)
    sig_log                 = pyqtSignal(str, str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self._cmd_queue = queue.Queue()
        self._running   = True
        self.clients  = {}

    def request_connect_all(self, ips: list, port: int, unit_id: int):
        self._cmd_queue.put(('connect_all', list(ips), port, unit_id))

    def request_disconnect_all(self):
        self._cmd_queue.put(('disconnect_all',))

    def request_write(self, idx: int, state: bool, source: str = 'button'):
        if self._cmd_queue.qsize() < 30:
            self._cmd_queue.put(('write', idx, state, source))

    def stop(self):
        self._running = False
        self._cmd_queue.put(None)

    def run(self):
        clients  = {}
        unit_id  = 1
        while self._running:
            try:
                cmd = self._cmd_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if cmd is None:
                break

            action = cmd[0]
            if action == 'connect_all':
                _, ips, port, unit_id = cmd
                for ip in ips:
                    if ip in clients:
                        self.sig_log.emit(f"{ip}: sudah terkoneksi, skip.", "INFO")
                        continue
                    if not tcp_check(ip, port, timeout=3.0):
                        self.sig_log.emit(f"{ip}: port {port} tidak dapat diakses.", "WARN")
                        continue
                    try:
                        c = ModbusTcpClient(ip, port=port, timeout=3)
                        ok = c.connect()
                        if ok:
                            clients[ip] = c
                            self.sig_log.emit(f"Terhubung ke {ip}:{port} ✔", "OK")
                            self.sig_client_connected.emit(ip, port)
                            self._read_led_registers(c, unit_id)
                        else:
                            self.sig_log.emit(f"{ip}: Modbus tidak merespons.", "ERROR")
                            c.close()
                    except Exception as exc:
                        self.sig_log.emit(f"{ip}: exception: {exc}", "ERROR")

            elif action == 'disconnect_all':
                for ip, c in clients.items():
                    try:
                        c.close()
                    except Exception:
                        pass
                clients.clear()
                self.sig_all_disconnected.emit()
                self.sig_log.emit("Semua koneksi diputus.", "INFO")

            elif action == 'write':
                _, idx, state, source = cmd
                if not clients:
                    if source == 'button':
                        self.sig_log.emit("Tidak ada Modbus client terkoneksi!", "ERROR")
                    continue
                finger_map = self.config['finger_mapping'][idx]
                addr = finger_map['address']
                reg_type = finger_map['register_type']
                val = 1 if state else 0
                all_ok = True
                dead_ips = []
                for ip, c in list(clients.items()):
                    try:
                        response = None
                        for _kw in ('device_id', 'slave', 'unit'):
                            try:
                                if reg_type == 'coil':
                                    response = c.write_coil(addr, bool(state), **{_kw: int(unit_id)})
                                break
                            except TypeError:
                                continue
                        if response is None or response.isError():
                            all_ok = False
                    except Exception as exc:
                        all_ok = False
                        dead_ips.append(ip)
                for ip in dead_ips:
                    try:
                        clients[ip].close()
                    except Exception:
                        pass
                    del clients[ip]
                    self.sig_client_disconnected.emit(ip)
                if dead_ips and not clients:
                    self.sig_all_disconnected.emit()
                self.sig_coil_result.emit(idx, state, all_ok, source)
                if all_ok:
                    self.sig_log.emit(f"[{source}] {finger_map['finger']} -> {val}", "OK")

        for c in clients.values():
            try:
                c.close()
            except Exception:
                pass

    def _read_led_registers(self, client, unit_id: int):
        led_configs = [led for led in self.config['plc_led_mapping'] if led['enabled']]
        if not led_configs:
            return
        try:
            addr = led_configs[0]['address']
            count = len(led_configs)
            response = None
            for _kw in ('device_id', 'slave', 'unit'):
                try:
                    response = client.read_coils(addr, count=count, **{_kw: int(unit_id)})
                    break
                except TypeError:
                    continue
            if response is not None and not response.isError():
                bits = [bool(b) for b in response.bits[:count]]
                self.sig_led_states.emit(bits)
                self.sig_log.emit(f"LED states: {[int(b) for b in bits]}", "OK")
        except Exception as exc:
            self.sig_log.emit(f"Gagal baca LED register: {exc}", "WARN")

# ─── Settings Dialog ──────────────────────────────────────────────────────────
class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Settings")
        self.setMinimumSize(600, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        tabs = QTabWidget()

        # Modbus Settings Tab
        modbus_tab = QWidget()
        modbus_layout = QFormLayout()
        modbus_config = self.config['modbus']

        self.port_edit = QSpinBox()
        self.port_edit.setRange(1, 65535)
        self.port_edit.setValue(modbus_config['port'])
        modbus_layout.addRow("Port:", self.port_edit)

        self.unit_id_edit = QSpinBox()
        self.unit_id_edit.setRange(1, 255)
        self.unit_id_edit.setValue(modbus_config['unit_id'])
        modbus_layout.addRow("Unit ID:", self.unit_id_edit)

        self.known_ips_edit = QLineEdit(",".join(modbus_config['known_ips']))
        modbus_layout.addRow("Known IPs (comma-separated):", self.known_ips_edit)

        modbus_tab.setLayout(modbus_layout)
        tabs.addTab(modbus_tab, "Modbus")

        # Finger Mapping Tab
        finger_tab = QWidget()
        finger_layout = QVBoxLayout()
        self.finger_list = QListWidget()
        for finger in self.config['finger_mapping']:
            item = QListWidgetItem(f"{finger['finger']} -> {finger['register_type']} {finger['address']}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if finger['enabled'] else Qt.Unchecked)
            item.setData(Qt.UserRole, finger)
            self.finger_list.addItem(item)
        finger_layout.addWidget(QLabel("Finger to Register Mapping:"))
        finger_layout.addWidget(self.finger_list)
        finger_tab.setLayout(finger_layout)
        tabs.addTab(finger_tab, "Finger Mapping")

        # PLC LED Mapping Tab
        led_tab = QWidget()
        led_layout = QVBoxLayout()
        self.led_list = QListWidget()
        for led in self.config['plc_led_mapping']:
            item = QListWidgetItem(f"{led['label']} -> {led['register_type']} {led['address']}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled)
            item.setCheckState(Qt.Checked if led['enabled'] else Qt.Unchecked)
            item.setData(Qt.UserRole, led)
            self.led_list.addItem(item)
        self.led_list.setDragDropMode(QListWidget.InternalMove)
        led_layout.addWidget(QLabel("PLC LED Mapping (drag to reorder):"))
        led_layout.addWidget(self.led_list)
        led_tab.setLayout(led_layout)
        tabs.addTab(led_tab, "PLC LED Mapping")

        layout.addWidget(tabs)

        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.save_settings)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def save_settings(self):
        self.config['modbus']['port'] = self.port_edit.value()
        self.config['modbus']['unit_id'] = self.unit_id_edit.value()
        self.config['modbus']['known_ips'] = [ip.strip() for ip in self.known_ips_edit.text().split(",") if ip.strip()]

        self.config['finger_mapping'] = []
        for i in range(self.finger_list.count()):
            item = self.finger_list.item(i)
            finger = item.data(Qt.UserRole)
            finger['enabled'] = item.checkState() == Qt.Checked
            self.config['finger_mapping'].append(finger)

        self.config['plc_led_mapping'] = []
        for i in range(self.led_list.count()):
            item = self.led_list.item(i)
            led = item.data(Qt.UserRole)
            led['enabled'] = item.checkState() == Qt.Checked
            self.config['plc_led_mapping'].append(led)

        save_config(self.config)
        self.accept()

# ─── Style Helpers ────────────────────────────────────────────────────────────
_BTN_ON = """
QPushButton { background-color: #43A047; color: white; border-radius: 8px; border: 2px solid #1B5E20; font-size: 11pt; font-weight: bold; }
QPushButton:hover { background-color: #388E3C; }
QPushButton:pressed{ background-color: #2E7D32; }
"""

_BTN_OFF = """
QPushButton { background-color: #C62828; color: #FFCDD2; border-radius: 8px; border: 2px solid #7F0000; font-size: 11pt; font-weight: bold; }
QPushButton:hover { background-color: #B71C1C; }
QPushButton:pressed { background-color: #7F0000; }
QPushButton:disabled { background-color: #263238; color: #546E7A; border-color: #263238; }
"""

_BTN_CONNECT = """
QPushButton { background-color: #1E88E5; color: white; border-radius: 5px; font-weight: bold; font-size: 10pt; }
QPushButton:hover { background-color: #1565C0; }
QPushButton:pressed{ background-color: #0D47A1; }
"""

_BTN_DISCONNECT = """
QPushButton { background-color: #E53935; color: white; border-radius: 5px; font-weight: bold; font-size: 10pt; }
QPushButton:hover { background-color: #C62828; }
QPushButton:pressed{ background-color: #B71C1C; }
"""

# ─── Main Window ──────────────────────────────────────────────────────────────
class VBoxController(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.unit_id = config['modbus']['unit_id']
        self._gesture_enabled = True
        self._modbus_thread = ModbusWorkerThread(config)
        self._connected_ips = set()
        self._detected_ips = list(config['modbus']['known_ips'])
        self._scanner = None
        self._gesture_thread = None

        self.setWindowTitle("ROS MediaPipe Modbus Controller")
        self.setMinimumSize(config['gui']['window_width'], config['gui']['window_height'])
        self._build_ui()

        # Connect signals
        self._modbus_thread.sig_client_connected.connect(self._on_modbus_connected)
        self._modbus_thread.sig_client_disconnected.connect(self._on_modbus_client_disconnected)
        self._modbus_thread.sig_all_disconnected.connect(self._on_modbus_disconnected)
        self._modbus_thread.sig_coil_result.connect(self._on_coil_result)
        self._modbus_thread.sig_led_states.connect(self._on_led_states)
        self._modbus_thread.sig_log.connect(self._log)
        self._modbus_thread.start()

        self._start_scan()
        self._start_gesture_thread()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(12,10,12,10)

        # Left: Camera panel
        root.addWidget(self._make_camera_panel(), stretch=0)
        # Right: Control panel
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(8)
        rv.addWidget(self._make_connection_group())
        rv.addWidget(self._make_led_group())
        rv.addWidget(self._make_log_group())
        root.addWidget(right, stretch=1)

        self._log(f"ROS Node started. pymodbus: {_PYMODBUS_VERSION or 'TIDAK TERINSTALL'}")
        self._log(f"MediaPipe: {'tersedia' if _MP_AVAILABLE else 'TIDAK TERINSTALL'}")
        self._log(f"IP PC: {get_local_ip()}")

    def _make_camera_panel(self) -> QGroupBox:
        g = QGroupBox("Kamera & Deteksi Gestur")
        g.setFont(QFont("Arial",9,QFont.Bold))
        g.setFixedWidth(384)
        vl = QVBoxLayout()
        vl.setContentsMargins(8,12,8,10)
        vl.setSpacing(8)

        self.cam_label = QLabel()
        self.cam_label.setFixedSize(366, 274)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet("background: #0A0E14; border: 1px solid #37474F; border-radius: 4px; color: #546E7A; font-size: 9pt;")
        self.cam_label.setText("Memulai kamera..." if _MP_AVAILABLE else "mediapipe belum terinstall")
        vl.addWidget(self.cam_label)

        self.finger_leds = []
        ind_box = QGroupBox("Status Jari")
        il = QGridLayout()
        for i, finger in enumerate(self.config['finger_mapping']):
            led = QLabel("●")
            led.setFont(QFont("Arial",14))
            led.setStyleSheet("color: #37474F;")
            led.setFixedWidth(22)
            lbl = QLabel(finger['finger'])
            lbl.setStyleSheet("color: #90A4AE; font-size: 8pt;")
            state_lbl = QLabel("OFF")
            state_lbl.setFont(QFont("Consolas",8,QFont.Bold))
            state_lbl.setStyleSheet("color: #546E7A;")
            state_lbl.setFixedWidth(28)
            il.addWidget(led, i,0)
            il.addWidget(lbl, i,1)
            il.addWidget(state_lbl, i,2)
            self.finger_leds.append((led, state_lbl))
        ind_box.setLayout(il)
        vl.addWidget(ind_box)

        self.gesture_chk = QCheckBox("Aktifkan kontrol gestur")
        self.gesture_chk.setChecked(True)
        self.gesture_chk.setStyleSheet("color: #90A4AE; font-size: 8pt;")
        self.gesture_chk.stateChanged.connect(self._on_gesture_toggle)
        vl.addWidget(self.gesture_chk)

        vl.addStretch()
        g.setLayout(vl)
        return g

    def _make_connection_group(self) -> QGroupBox:
        g = QGroupBox("Koneksi Modbus TCP")
        g.setFont(QFont("Arial",10,QFont.Bold))
        vl = QVBoxLayout()
        vl.setSpacing(8)

        row0 = QHBoxLayout()
        row0.addWidget(QLabel("IP V-BOX:"))
        self.ip_combo = QComboBox()
        self.ip_combo.setEditable(True)
        self.ip_combo.addItems(self._detected_ips)
        self.ip_combo.setStyleSheet("QComboBox { background:#1C2833; color:#E0E6ED; border:1px solid #455A64; border-radius:4px; padding:3px 8px; }")
        row0.addWidget(self.ip_combo, stretch=3)

        scan_btn = QPushButton("🔍 Scan")
        scan_btn.clicked.connect(self._start_scan)
        row0.addWidget(scan_btn)

        self.port_edit = QLineEdit(str(self.config['modbus']['port']))
        self.port_edit.setFixedWidth(58)
        row0.addWidget(QLabel("Port:"))
        row0.addWidget(self.port_edit)

        self.unit_edit = QLineEdit(str(self.config['modbus']['unit_id']))
        self.unit_edit.setFixedWidth(48)
        row0.addWidget(QLabel("Unit ID:"))
        row0.addWidget(self.unit_edit)
        vl.addLayout(row0)

        self.conn_btn = QPushButton("▶ CONNECT")
        self.conn_btn.setStyleSheet(_BTN_CONNECT)
        self.conn_btn.clicked.connect(self.toggle_connection)
        vl.addWidget(self.conn_btn)

        # Settings button
        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.clicked.connect(self._open_settings)
        vl.addWidget(settings_btn)

        g.setLayout(vl)
        return g

    def _make_led_group(self) -> QGroupBox:
        g = QGroupBox("PLC LED Status")
        g.setFont(QFont("Arial",10,QFont.Bold))
        gl = QGridLayout()
        self.led_widgets = []
        for i, led_config in enumerate(self.config['plc_led_mapping']):
            led = QLabel("●")
            led.setFont(QFont("Arial",20))
            led.setStyleSheet("color: #37474F;")
            led.setAlignment(Qt.AlignCenter)
            lbl = QLabel(led_config['label'])
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #78909C; font-size: 8pt;")
            gl.addWidget(led, 0, i)
            gl.addWidget(lbl, 1, i)
            self.led_widgets.append(led)
        g.setLayout(gl)
        return g

    def _make_log_group(self) -> QGroupBox:
        g = QGroupBox("Log")
        g.setFont(QFont("Arial",9))
        vl = QVBoxLayout()
        vl.setContentsMargins(8,10,8,8)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(110)
        self.log_box.setFont(QFont("Consolas",8))
        self.log_box.setStyleSheet("background: #0D1117; color: #00E676; border: 1px solid #30363D;")
        vl.addWidget(self.log_box)
        g.setLayout(vl)
        return g

    def _open_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec_() == QDialog.Accepted:
            self._log("Settings saved. Restart to apply all changes.", "OK")

    def _start_scan(self):
        if self._scanner and self._scanner.isRunning():
            self._scanner.stop()
            self._scanner.wait(500)
        subnet = get_subnet_base()
        self.ip_combo.clear()
        self._detected_ips.clear()
        self._scanner = NetworkScannerThread(subnet, port=int(self.port_edit.text()))
        self._scanner.found.connect(self._on_host_found)
        self._scanner.start()
        self._log(f"Scanning subnet {subnet}.0/24...")

    def _on_host_found(self, ip: str, is_modbus: bool):
        if is_modbus and ip not in self._detected_ips:
            self._detected_ips.append(ip)
            self.ip_combo.addItem(ip)
            self._log(f"Ditemukan V-BOX: {ip}", "OK")

    def _start_gesture_thread(self):
        if not _MP_AVAILABLE:
            return
        self._gesture_thread = GestureThread(self.config)
        self._gesture_thread.frame_ready.connect(self._on_frame)
        self._gesture_thread.gesture_changed.connect(self._on_gesture)
        self._gesture_thread.cam_error.connect(self._on_cam_error)
        self._gesture_thread.start()

    def _on_frame(self, frame):
        h,w,ch = frame.shape
        qt_img = QImage(frame.data, w,h,ch*w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_img).scaled(366,274, Qt.KeepAspectRatio)
        self.cam_label.setPixmap(pixmap)

    def _on_gesture(self, states: list, hand_detected: bool):
        for i, (led, state_lbl) in enumerate(self.finger_leds):
            if not hand_detected:
                led.setStyleSheet("color: #37474F;")
                state_lbl.setText("---")
            elif states[i]:
                led.setStyleSheet("color: #43A047;")
                state_lbl.setText("ON")
                if self._gesture_enabled and self._connected_ips:
                    self._modbus_thread.request_write(i, True, source="gesture")
            else:
                led.setStyleSheet("color: #C62828;")
                state_lbl.setText("OFF")
                if self._gesture_enabled and self._connected_ips:
                    self._modbus_thread.request_write(i, False, source="gesture")

    def _on_cam_error(self, msg: str):
        self.cam_label.setText(f"Kamera error:\n{msg}")
        self._log(f"Kamera error: {msg}", "WARN")

    def _on_gesture_toggle(self, state: int):
        self._gesture_enabled = (state == Qt.Checked)
        self._log(f"Gesture control: {'aktif' if self._gesture_enabled else 'nonaktif'}")

    def _log(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        color_map = {"INFO":"#00E676", "OK":"#64FFDA", "WARN":"#FFAB00", "ERROR":"#FF5252"}
        color = color_map.get(level, "#00E676")
        self.log_box.append(f'<span style="color:#546E7A">[{ts}]</span> <span style="color:{color}">[{level}] {msg}</span>')
        print(f"[{level}] {msg}")

    def toggle_connection(self):
        if self._connected_ips:
            self._modbus_thread.request_disconnect_all()
        else:
            try:
                port = int(self.port_edit.text())
                unit = int(self.unit_edit.text())
                ips = [self.ip_combo.itemText(i) for i in range(self.ip_combo.count())]
                manual = self.ip_combo.currentText().strip()
                if manual and manual not in ips:
                    ips.append(manual)
                self._modbus_thread.request_connect_all(ips, port, unit)
                self.conn_btn.setText("Menghubungkan...")
                self.conn_btn.setEnabled(False)
            except ValueError:
                self._log("Port/Unit ID harus angka!", "ERROR")

    def _on_modbus_connected(self, ip: str, port: int):
        self._connected_ips.add(ip)
        self.conn_btn.setText("■ DISCONNECT")
        self.conn_btn.setStyleSheet(_BTN_DISCONNECT)
        self.conn_btn.setEnabled(True)
        self._log(f"Terhubung ke {ip}:{port}", "OK")

    def _on_modbus_client_disconnected(self, ip: str):
        self._connected_ips.discard(ip)
        self._log(f"V-BOX {ip} terputus", "WARN")

    def _on_modbus_disconnected(self):
        self._connected_ips.clear()
        self.conn_btn.setText("▶ CONNECT")
        self.conn_btn.setStyleSheet(_BTN_CONNECT)
        self.conn_btn.setEnabled(True)

    def _on_coil_result(self, idx: int, state: bool, ok: bool, source: str):
        if ok:
            self._log(f"{self.config['finger_mapping'][idx]['finger']} -> {1 if state else 0}", "OK")

    def _on_led_states(self, states: list):
        for i, state in enumerate(states):
            if i < len(self.led_widgets):
                self.led_widgets[i].setStyleSheet("color: #43A047;" if state else "color: #C62828;")

    def closeEvent(self, event):
        if self._scanner and self._scanner.isRunning():
            self._scanner.stop()
            self._scanner.wait(500)
        if self._gesture_thread and self._gesture_thread.isRunning():
            self._gesture_thread.stop()
            self._gesture_thread.wait(2000)
        self._modbus_thread.stop()
        self._modbus_thread.wait(2000)
        save_config(self.config)
        event.accept()

# ─── Dark Palette ─────────────────────────────────────────────────────────────
def apply_dark_palette(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window, QColor(30,33,40))
    p.setColor(QPalette.WindowText, QColor(210,215,225))
    p.setColor(QPalette.Base, QColor(22,25,31))
    p.setColor(QPalette.Text, QColor(210,215,225))
    p.setColor(QPalette.Button, QColor(50,54,65))
    p.setColor(QPalette.ButtonText, QColor(210,215,225))
    p.setColor(QPalette.Highlight, QColor(33,150,243))
    p.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(p)

# ─── Entry Point ──────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    config = load_config()
    app = QApplication(sys.argv)
    apply_dark_palette(app)
    win = VBoxController(config)
    win.show()
    try:
        sys.exit(app.exec_())
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
