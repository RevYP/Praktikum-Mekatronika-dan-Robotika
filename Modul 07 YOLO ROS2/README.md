# ROS YOLO - YOLOv8 Integration with ROS 2

## Deskripsi
Package ROS 2 untuk integrasi YOLOv8 (Ultralytics) dengan ROS 2 Humble. Mendukung real-time object detection, training custom model, dan visualisasi.

## Struktur Package
```
yolo_ros/
├── launch/           # Launch files
├── config/           # Parameter files
├── models/           # YOLO models (yolov8n.pt)
├── yolo_ros/scripts/ # Python package nodes, aman dari overlay nama umum scripts
│   ├── yolo_node.py          # YOLO detection node
│   ├── camera_node.py        # Camera display node
│   ├── yolo_camera_node.py   # Camera + YOLO node
│   ├── training_node.py      # Training node
│   └── rviz_display.py       # RViz display node
├── training_data/    # Dataset folder (empty, untuk hasil annotation)
├── roboflow_data/    # Roboflow data (empty)
├── package.xml
└── setup.py
```

## Quick Start

### 1. Build Package
```bash
cd /home/otomasi/Praktikum-Mekatronika-dan-Robotika/Modul07\ ROS\ YOLO/ROS_YOLO
colcon build
source install/setup.bash
```

### 2. Jalankan Nodes

**YOLO Detection Node:**
```bash
ros2 run yolo_ros yolo_node
```

**Camera Display Node:**
```bash
ros2 run yolo_ros camera_node
```

**YOLO Camera Node (Kamera + Detection):**
```bash
ros2 run yolo_ros yolo_camera_node
```

**RViz Display (Raw + YOLO Side by Side):**
```bash
ros2 run yolo_ros rviz_display
```

**Launch File (Semua sekaligus):**
```bash
ros2 launch yolo_ros yolo.launch.py
```

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | sensor_msgs/Image | Input kamera |
| `/yolo/annotated` | sensor_msgs/Image | Output YOLO (gambar + bbox) |
| `/yolo/detections` | std_msgs/String | JSON detection info |
| `/yolo/training_status` | std_msgs/String | Status training |

## Training Custom Model

1. **Annotasi Data**: Gunakan Roboflow/LabelStudio/CVAT (lihat tutorial terpisah)
2. **Export ke YOLO format** (data.yaml + train/valid folders)
3. **Copy ke `training_data/`**
4. **Train di Colab** (GPU) atau **Local** (GTX/RTX)
5. **Download `best.pt`** → Copy ke `models/custom_yolov8.pt`
6. **Update kode** atau gunakan parameter `model_path`

## Tutorial Lengkap

- `../Tutorial_Roboflow.md` - Annotasi Roboflow
- `../Tutorial_LabelStudio.md` - Annotasi LabelStudio
- `../Tutorial_CVAT.md` - Annotasi CVAT
- `../Tutorial_Training_Colab_GPU.md` - Training Google Colab
- `../Tutorial_Training_VSCode_Colab_GPU.md` - Training VSCode + Colab
- `../Tutorial_Training_Local_GPU.md` - Training Local GPU
- `../Tutorial_Colcon_Build_Run.md` - Build & Run commands

## Dependencies

- ROS 2 Humble
- Python 3.10+
- ultralytics
- opencv-python
- ros-humble-cv-bridge
- ros-humble-sensor-msgs

## Install Dependencies

```bash
pip3 install ultralytics opencv-python numpy
sudo apt install ros-humble-cv-bridge ros-humble-sensor-msgs ros-humble-launch-ros
```

## Referensi

- Ultralytics Docs: https://docs.ultralytics.com
- ROS 2 Docs: https://docs.ros.org
