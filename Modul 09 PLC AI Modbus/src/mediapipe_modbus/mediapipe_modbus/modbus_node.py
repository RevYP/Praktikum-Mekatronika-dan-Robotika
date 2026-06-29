#!/usr/bin/env python3
import json
import platform
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .config import load_config

try:
    from pymodbus.client import ModbusTcpClient
except ImportError:
    from pymodbus.client.sync import ModbusTcpClient


def local_subnet():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(('8.8.8.8', 80))
        ip = sock.getsockname()[0]
        sock.close()
        return '.'.join(ip.split('.')[:3])
    except Exception:
        return '192.168.1'


def host_open(ip, port):
    try:
        sock = socket.create_connection((ip, port), timeout=0.5)
        sock.close()
        return True
    except Exception:
        return False


def call_modbus(client, method, address, value_or_count, unit_id):
    for key in ('device_id', 'slave', 'unit'):
        try:
            if method.startswith('write'):
                return getattr(client, method)(address, value_or_count, **{key: unit_id})
            return getattr(client, method)(address, count=value_or_count, **{key: unit_id})
        except TypeError:
            continue
    return None


class ModbusNode(Node):
    def __init__(self):
        super().__init__('modbus_node')
        self.config = load_config()
        self.clients = {}
        self.status_pub = self.create_publisher(String, 'modbus/status', 10)
        self.led_pub = self.create_publisher(String, 'modbus/led_states', 10)
        self.create_subscription(String, 'mediapipe/fingers', self.on_fingers, 10)
        self.create_subscription(String, 'modbus/command', self.on_command, 10)
        self.scan_and_connect()
        self.create_timer(1.0, self.read_leds)

    def publish_status(self, data):
        msg = String()
        msg.data = json.dumps(data)
        self.status_pub.publish(msg)

    def scan_and_connect(self):
        modbus = self.config['modbus']
        port = int(modbus['port'])
        ips = list(modbus.get('known_ips') or [])
        subnet = modbus.get('scan_subnet') or local_subnet()
        targets = ips + [f'{subnet}.{i}' for i in range(1, 255)]
        found = []
        with ThreadPoolExecutor(max_workers=80) as executor:
            futures = {executor.submit(host_open, ip, port): ip for ip in dict.fromkeys(targets)}
            for future in as_completed(futures):
                if future.result():
                    found.append(futures[future])
        for ip in found:
            try:
                client = ModbusTcpClient(ip, port=port, timeout=2)
                if client.connect():
                    self.clients[ip] = client
            except Exception as exc:
                self.get_logger().warning(f'{ip}: {exc}')
        self.publish_status({'connected': list(self.clients), 'port': port})

    def on_command(self, msg):
        try:
            command = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if command.get('action') == 'scan':
            self.scan_and_connect()
        if command.get('action') == 'reload':
            self.config = load_config()

    def on_fingers(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        states = data.get('states', [])
        unit_id = int(self.config['modbus']['unit_id'])
        for index, mapping in enumerate(self.config['finger_mapping']):
            if index >= len(states) or not mapping.get('enabled', True):
                continue
            address = int(mapping['address'])
            value = bool(states[index])
            method = 'write_coil' if mapping.get('register_type') == 'coil' else 'write_register'
            payload = value if method == 'write_coil' else int(value)
            for ip, client in list(self.clients.items()):
                response = call_modbus(client, method, address, payload, unit_id)
                if response is None or response.isError():
                    self.get_logger().warning(f'Gagal tulis {ip} {address}')

    def read_leds(self):
        if not self.clients:
            return
        unit_id = int(self.config['modbus']['unit_id'])
        states = []
        first_client = next(iter(self.clients.values()))
        for mapping in self.config['plc_led_mapping']:
            if not mapping.get('enabled', True):
                states.append(False)
                continue
            method = 'read_coils' if mapping.get('register_type') == 'coil' else 'read_holding_registers'
            response = call_modbus(first_client, method, int(mapping['address']), 1, unit_id)
            if response is None or response.isError():
                states.append(False)
            elif method == 'read_coils':
                states.append(bool(response.bits[0]))
            else:
                states.append(bool(response.registers[0]))
        msg = String()
        msg.data = json.dumps({'states': states, 'mapping': self.config['plc_led_mapping']})
        self.led_pub.publish(msg)

    def destroy_node(self):
        for client in self.clients.values():
            client.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ModbusNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
