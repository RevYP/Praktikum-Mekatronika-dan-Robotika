import os
import yaml

CONFIG_PATH = os.path.expanduser('~/.ros/mediapipe_modbus_config.yaml')
FINGER_NAMES = ['Jempol', 'Telunjuk', 'Tengah', 'Manis', 'Kelingking']

DEFAULT_CONFIG = {
    'camera': {'index': 0, 'width': 640, 'height': 480, 'mirror': True},
    'mediapipe': {'min_detection_confidence': 0.7, 'min_tracking_confidence': 0.6},
    'modbus': {'port': 502, 'unit_id': 1, 'scan_subnet': None, 'known_ips': []},
    'finger_mapping': [
        {'finger': 'Jempol', 'register_type': 'coil', 'address': 0, 'enabled': True},
        {'finger': 'Telunjuk', 'register_type': 'coil', 'address': 1, 'enabled': True},
        {'finger': 'Tengah', 'register_type': 'coil', 'address': 2, 'enabled': True},
        {'finger': 'Manis', 'register_type': 'coil', 'address': 3, 'enabled': True},
        {'finger': 'Kelingking', 'register_type': 'coil', 'address': 4, 'enabled': True},
    ],
    'plc_led_mapping': [
        {'label': 'M0', 'register_type': 'coil', 'address': 0, 'enabled': True},
        {'label': 'M1', 'register_type': 'coil', 'address': 1, 'enabled': True},
        {'label': 'M2', 'register_type': 'coil', 'address': 2, 'enabled': True},
        {'label': 'M3', 'register_type': 'coil', 'address': 3, 'enabled': True},
        {'label': 'M4', 'register_type': 'coil', 'address': 4, 'enabled': True},
    ],
}


def merge_config(default, loaded):
    if not isinstance(default, dict) or not isinstance(loaded, dict):
        return loaded if loaded is not None else default
    result = dict(default)
    for key, value in loaded.items():
        result[key] = merge_config(default.get(key), value)
    return result


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_CONFIG.copy()
    with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
        return merge_config(DEFAULT_CONFIG, yaml.safe_load(file) or {})


def save_config(config):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as file:
        yaml.safe_dump(config, file, sort_keys=False)
