#!/usr/bin/env python3
import json
import sys

import rclpy
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QSpinBox, QVBoxLayout, QWidget
from rclpy.node import Node
from std_msgs.msg import String

from .config import FINGER_NAMES, load_config, save_config


class GuiRosNode(Node):
    def __init__(self, window):
        super().__init__('gui_node')
        self.window = window
        self.command_pub = self.create_publisher(String, 'modbus/command', 10)
        self.create_subscription(String, 'mediapipe/fingers', self.window.on_fingers, 10)
        self.create_subscription(String, 'modbus/status', self.window.on_modbus_status, 10)
        self.create_subscription(String, 'modbus/led_states', self.window.on_led_states, 10)

    def send_command(self, data):
        msg = String()
        msg.data = json.dumps(data)
        self.command_pub.publish(msg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.node = None
        self.setWindowTitle('MediaPipe Modbus GUI')
        self.resize(1000, 650)
        self.finger_labels = []
        self.led_labels = []
        self.build_ui()

    def build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.addWidget(self.camera_panel())
        root.addWidget(self.config_panel(), 1)
        self.setCentralWidget(central)

    def camera_panel(self):
        group = QGroupBox('Status MediaPipe')
        layout = QVBoxLayout(group)
        title = QLabel('Kamera diproses oleh camera_node dan mediapipe_node')
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        grid = QGridLayout()
        for row, name in enumerate(FINGER_NAMES):
            led = QLabel('OFF')
            led.setStyleSheet('background:#444;color:white;padding:8px;border-radius:4px')
            grid.addWidget(QLabel(name), row, 0)
            grid.addWidget(led, row, 1)
            self.finger_labels.append(led)
        layout.addLayout(grid)
        led_group = QGroupBox('LED PLC')
        led_grid = QGridLayout(led_group)
        for index, mapping in enumerate(self.config['plc_led_mapping']):
            led = QLabel(mapping['label'])
            led.setAlignment(Qt.AlignCenter)
            led.setStyleSheet('background:#444;color:white;padding:14px;border-radius:6px')
            led_grid.addWidget(led, index // 5, index % 5)
            self.led_labels.append(led)
        layout.addWidget(led_group)
        return group

    def config_panel(self):
        group = QGroupBox('Setting Modbus dan Mapping')
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(int(self.config['modbus']['port']))
        self.unit = QSpinBox()
        self.unit.setRange(1, 255)
        self.unit.setValue(int(self.config['modbus']['unit_id']))
        self.ips = QLineEdit(','.join(self.config['modbus'].get('known_ips') or []))
        form.addRow('Port', self.port)
        form.addRow('Unit ID', self.unit)
        form.addRow('Known IP', self.ips)
        layout.addLayout(form)
        self.finger_address = []
        finger_group = QGroupBox('Mapping Jari ke Register Output')
        finger_grid = QGridLayout(finger_group)
        for row, mapping in enumerate(self.config['finger_mapping']):
            enabled = QCheckBox(mapping['finger'])
            enabled.setChecked(bool(mapping.get('enabled', True)))
            register_type = QComboBox()
            register_type.addItems(['coil', 'holding_register'])
            register_type.setCurrentText(mapping.get('register_type', 'coil'))
            address = QSpinBox()
            address.setRange(0, 65535)
            address.setValue(int(mapping['address']))
            finger_grid.addWidget(enabled, row, 0)
            finger_grid.addWidget(register_type, row, 1)
            finger_grid.addWidget(address, row, 2)
            self.finger_address.append((enabled, register_type, address))
        layout.addWidget(finger_group)
        self.led_address = []
        led_group = QGroupBox('Mapping PLC ke Kotak LED')
        led_grid = QGridLayout(led_group)
        for row, mapping in enumerate(self.config['plc_led_mapping']):
            label = QLineEdit(mapping['label'])
            register_type = QComboBox()
            register_type.addItems(['coil', 'holding_register'])
            register_type.setCurrentText(mapping.get('register_type', 'coil'))
            address = QSpinBox()
            address.setRange(0, 65535)
            address.setValue(int(mapping['address']))
            led_grid.addWidget(label, row, 0)
            led_grid.addWidget(register_type, row, 1)
            led_grid.addWidget(address, row, 2)
            self.led_address.append((label, register_type, address))
        layout.addWidget(led_group)
        buttons = QHBoxLayout()
        save_btn = QPushButton('Save Config')
        scan_btn = QPushButton('Scan IP')
        save_btn.clicked.connect(self.save_settings)
        scan_btn.clicked.connect(lambda: self.node.send_command({'action': 'scan'}) if self.node else None)
        buttons.addWidget(save_btn)
        buttons.addWidget(scan_btn)
        layout.addLayout(buttons)
        return group

    def save_settings(self):
        self.config['modbus']['port'] = int(self.port.value())
        self.config['modbus']['unit_id'] = int(self.unit.value())
        self.config['modbus']['known_ips'] = [ip.strip() for ip in self.ips.text().split(',') if ip.strip()]
        for index, widgets in enumerate(self.finger_address):
            enabled, register_type, address = widgets
            self.config['finger_mapping'][index]['enabled'] = enabled.isChecked()
            self.config['finger_mapping'][index]['register_type'] = register_type.currentText()
            self.config['finger_mapping'][index]['address'] = int(address.value())
        for index, widgets in enumerate(self.led_address):
            label, register_type, address = widgets
            self.config['plc_led_mapping'][index]['label'] = label.text()
            self.config['plc_led_mapping'][index]['register_type'] = register_type.currentText()
            self.config['plc_led_mapping'][index]['address'] = int(address.value())
        save_config(self.config)
        if self.node:
            self.node.send_command({'action': 'reload'})

    def on_fingers(self, msg):
        data = json.loads(msg.data)
        for index, state in enumerate(data.get('states', [])):
            label = self.finger_labels[index]
            label.setText('ON' if state else 'OFF')
            label.setStyleSheet('background:#1b8f3a;color:white;padding:8px;border-radius:4px' if state else 'background:#8f1b1b;color:white;padding:8px;border-radius:4px')

    def on_led_states(self, msg):
        data = json.loads(msg.data)
        for index, state in enumerate(data.get('states', [])):
            if index < len(self.led_labels):
                self.led_labels[index].setStyleSheet('background:#1b8f3a;color:white;padding:14px;border-radius:6px' if state else 'background:#444;color:white;padding:14px;border-radius:6px')

    def on_modbus_status(self, msg):
        data = json.loads(msg.data)
        self.statusBar().showMessage(f"Connected: {', '.join(data.get('connected', []))}")


def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    window = MainWindow()
    node = GuiRosNode(window)
    window.node = node
    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.01))
    timer.start(10)
    window.show()
    try:
        sys.exit(app.exec_())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
