# Panduan Praktikum Dobot Magician ROS2

## Urutan Penggunaan Script

```
scripts/
├── 01_setup_install.sh    ← Jalankan SEKALI saja (install semua)
├── 02_run_dobot.sh        ← Jalankan untuk kontrol robot fisik
├── 03_run_rviz_only.sh    ← Visualisasi tanpa robot (latihan)
├── 04_homing.sh           ← Wajib setelah 02 (kalibrasi posisi)
├── 05_demo.sh             ← Contoh gerakan robot
├── 06_move_robot.sh       ← Gerakkan robot ke koordinat tertentu
└── 07_control_panel.sh    ← GUI kontrol lengkap
```

---

## LANGKAH PERTAMA KALI (Hanya 1x)

1. Buka Terminal (Ctrl+Alt+T)
2. Masuk ke folder project:
   ```bash
   cd "/home/otomasi/Praktikum-Mekatronika-dan-Robotika/Modul ROS Robot ARM/magician_ros2"
   ```
3. Beri izin eksekusi pada semua script:
   ```bash
   chmod +x scripts/*.sh
   ```
4. Jalankan instalasi:
   ```bash
   ./scripts/01_setup_install.sh
   ```
5. **RESTART komputer** setelah selesai

---

## MENGGUNAKAN ROBOT FISIK

### Terminal 1: Jalankan sistem kontrol
```bash
cd "/home/otomasi/Praktikum-Mekatronika-dan-Robotika/Modul ROS Robot ARM/magician_ros2"
./scripts/02_run_dobot.sh
```
Biarkan terminal ini tetap berjalan!

### Terminal 2: Homing (wajib pertama kali)
```bash
cd "/home/otomasi/Praktikum-Mekatronika-dan-Robotika/Modul ROS Robot ARM/magician_ros2"
./scripts/04_homing.sh
```

### Terminal 2: Gerakkan robot / demo
```bash
./scripts/05_demo.sh          # demo otomatis
./scripts/06_move_robot.sh    # gerak manual
./scripts/07_control_panel.sh # GUI
```

---

## VISUALISASI SAJA (Tanpa Robot)

Untuk latihan melihat model 3D robot di RViz:
```bash
cd "/home/otomasi/Praktikum-Mekatronika-dan-Robotika/Modul ROS Robot ARM/magician_ros2"
./scripts/03_run_rviz_only.sh
```
Geser slider untuk menggerakkan model robot.

---

## TIPS

- Selalu buka terminal baru dengan **Ctrl+Alt+T**
- Jika robot tidak terdeteksi, cek kabel USB dan pastikan robot menyala
- Jika error "permission denied /dev/ttyUSB0", pastikan sudah restart setelah install
- Untuk menghentikan program: tekan **Ctrl+C** di terminal

---

## KOORDINAT DOBOT MAGICIAN

- X: 150-300 mm (jarak ke depan dari base)
- Y: -150 sampai 150 mm (kiri-kanan)
- Z: -30 sampai 150 mm (atas-bawah)
- R: -90 sampai 90 derajat (rotasi end effector)
