# MediaPipe Modbus ROS2 Node

Integrasi MediaPipe Hand Gesture Detection dengan Modbus TCP PLC menggunakan ROS2.

## Fitur

1. **Camera Capture**: Menangkap video dari webcam
2. **MediaPipe Processing**: Deteksi tangan dan status 5 jari real-time
3. **PyQt5 GUI**: Antarmuka grafis dengan dark theme
4. **Modbus TCP**: Komunikasi dengan PLC Wecon V-BOX
5. **YAML Configuration**: Settings tersimpan otomatis
6. **IP Scanning**: Scan jaringan untuk deteksi perangkat Modbus
7. **Custom Mapping**: Pemetaan jari ke register Modbus yang dapat diedit
8. **LED Display**: Tampilan status PLC dengan urutan yang dapat diubah

## Struktur Direktori

```
ROS_PLC_AI/
├── src/
│   └── mediapipe_modbus_pkg/
│       ├── package.xml
│       ├── setup.py
│       ├── resource/
│       └── mediapipe_modbus_pkg/
│           ├── __init__.py
│           └── mediapipe_modbus_ros_node.py
├── build/
├── install/
├── log/
├── test_node.sh          # Script untuk test dependencies
├── run_node.sh           # Script untuk menjalankan node
└── README.md
```

## Instalasi

### 1. Install Dependencies

```bash
# Install Python packages
pip3 install 'numpy<2' mediapipe 'opencv-python<4.10' pymodbus PyQt5 pyyaml

# Atau gunakan requirements.txt
pip3 install -r requirements.txt
```

### 2. Build ROS2 Package

```bash
cd /home/otomasi/Praktikum-Mekatronika-dan-Robotika/Modul09\ PLC\ AI\ Modbus/ROS_PLC_AI
colcon build --packages-select mediapipe_modbus_pkg
```

### 3. Test Installation

```bash
./test_node.sh
```

## Cara Menjalankan

### Metode 1: Menggunakan Script

```bash
./run_node.sh
```

### Metode 2: Manual

```bash
cd /home/otomasi/Praktikum-Mekatronika-dan-Robotika/Modul09\ PLC\ AI\ Modbus/ROS_PLC_AI
source install/setup.bash
ros2 run mediapipe_modbus_pkg mediapipe_modbus_node
```

## Penggunaan

### 1. Scan Jaringan

- Klik tombol **"🔍 Scan"** untuk memindai perangkat Modbus di subnet
- Perangkat yang ditemukan akan muncul di dropdown IP

### 2. Koneksi ke PLC

- Pilih IP V-BOX dari dropdown
- Pastikan Port = 502, Unit ID = 1
- Klik **"▶ CONNECT"**
- Status akan berubah menjadi "CONNECTED" jika berhasil

### 3. Deteksi Gestur

- Centang **"Aktifkan kontrol gestur"**
- Lakukan gestur tangan di depan kamera
- Setiap jari yang terbuka akan mengaktifkan coil PLC yang sesuai

### 4. Konfigurasi Settings

- Klik tombol **"⚙️ Settings"**
- **Tab Modbus**: Ubah Port, Unit ID, Known IPs
- **Tab Finger Mapping**: Ubah pemetaan jari ke register
- **Tab PLC LED Mapping**: Drag-drop untuk mengubah urutan LED
- Klik **"Save"** untuk menyimpan

## Konfigurasi YAML

Settings disimpan di: `~/.ros/mediapipe_modbus_config.yaml`

```yaml
modbus:
  port: 502
  unit_id: 1
  known_ips:
    - 10.41.168.203
    - 10.41.168.185

finger_mapping:
  - finger: Jempol
    register_type: coil
    address: 0
    enabled: true
  # ... dst

plc_led_mapping:
  - label: M0
    register_type: coil
    address: 0
    enabled: true
  # ... dst
```

## Troubleshooting

### MediaPipe Error

```bash
# Downgrade numpy jika ada konflik
pip3 install 'numpy<2'
```

### Kamera Tidak Terdeteksi

```bash
# Cek device kamera
ls /dev/video*

# Test kamera
ffplay /dev/video0
```

### Modbus Connection Failed

```bash
# Cek koneksi ke V-BOX
ping [IP_VBOX]

# Cek port 502 terbuka
telnet [IP_VBOX] 502
# atau
nmap -p 502 [IP_VBOX]
```

### Package Not Found

```bash
# Rebuild package
colcon build --packages-select mediapipe_modbus_pkg --symlink-install

# Source workspace
source install/setup.bash
```

## Dokumentasi Lengkap

- **Materi.md**: Teori dasar PLC, Modbus, MediaPipe, ROS
- **Project.md**: Deskripsi project dan komponen sistem
- **Jobsheet.md**: Lembar kerja praktikum step-by-step
- **Tugas Video.md**: Panduan pembuatan video demonstrasi
- **notebookllm.md**: 45 slide materi presentasi (300 kata/slide)

## Arsitektur Sistem

```
┌─────────────┐
│   Kamera    │
└──────┬──────┘
       │ Frame
       ▼
┌─────────────┐
│  MediaPipe  │ ──► Status 5 Jari
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  PyQt5 GUI  │
└──────┬──────┘
       │ Write Coil
       ▼
┌─────────────┐
│ Modbus TCP  │ ──► PLC V-BOX
└─────────────┘
```

## Lisensi

MIT License

## Kontributor

- Praktikum Mekatronika dan Robotika
- Modul 09: PLC AI Modbus Integration
