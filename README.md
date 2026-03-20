# SmartAttend GUI - Project Baru

Proyek baru terpisah untuk Modul 1 dengan **GUI penuh** (tanpa menu terminal) sesuai `Project.md`.

## Fitur Wajib
- Load image dari file
- Capture image dari webcam
- Pipeline: resize (400x250), grayscale preview, brightness/contrast
- Annotation: timestamp, nomor urut, watermark, border
- Save output otomatis dengan format `YYYYMMDD_HHMMSS_kartu.jpg` dan folder tanggal
- Simpan hasil dalam grayscale (efisien)
- Collage harian 4x4 sebagai ringkasan presensi

## Fitur Inovasi
- **Smart Auto-Fix**:
  - Deteksi orientasi kartu otomatis (portrait/landscape)
  - Rotasi otomatis bila portrait
  - Quality score otomatis
  - Auto-enhance (histogram + CLAHE) jika quality rendah
  - Quality badge ditampilkan pada hasil

## Cara Menjalankan
```bash
pip install -r requirements.txt
python main.py
```

## Generate Sample KTM UNDIP
```bash
python generate_sample_ktm.py
```

Script ini membuat 5 contoh KTM UNDIP + `input/sample_kartu.jpg`.

## Kesesuaian dengan Project.md
- ✅ Capture/Load Image (file + webcam)
- ✅ Pipeline (resize 400x250, grayscale preview, brightness/contrast)
- ✅ Annotation (timestamp, border, watermark)
- ✅ Tambahan nomor urut otomatis pada kartu saat simpan
- ✅ Save Output (format `YYYYMMDD_HHMMSS_kartu.jpg`, folder tanggal)
- ✅ Collage Generator (4x4 daily summary)
- ✅ Bonus GUI Tkinter
- ✅ Bonus real-time webcam preview
- ✅ Bonus deteksi orientasi kartu
- ✅ Bonus histogram equalization untuk gambar gelap

## Struktur
- `main.py` : aplikasi GUI utama
- `capture_module.py` : load/capture image
- `process_module.py` : pipeline pemrosesan
- `utils.py` : save & helper file
- `innovation_module.py` : fitur inovasi quality-aware
- `config.py` : konfigurasi
- `input/` : gambar input
- `output/` : hasil proses + collage
