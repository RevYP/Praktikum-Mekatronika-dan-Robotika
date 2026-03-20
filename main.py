import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from config import INPUT_FOLDER, OUTPUT_FOLDER, validate_config
from capture_module import load_image, capture_from_webcam
from process_module import (
    resize_standard,
    convert_to_grayscale,
    adjust_brightness_contrast,
    auto_crop_card_roi,
    add_timestamp,
    add_profile_text,
    add_sequence_number,
    add_watermark,
    add_border,
    create_collage,
)
from innovation_module import detect_card_orientation, calculate_image_quality_score, auto_enhance_image
from utils import ensure_directories, save_with_organization, save_collage, get_files_in_folder, get_next_sequence_for_today


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartAttend GUI - Project Baru")
        self.root.geometry("1260x760")

        validate_config()
        ensure_directories()
        os.makedirs(INPUT_FOLDER, exist_ok=True)

        self.original = None
        self.gray = None
        self.processed = None

        self.img_original_tk = None
        self.img_gray_tk = None
        self.img_processed_tk = None

        self.brightness = tk.IntVar(value=0)
        self.contrast = tk.DoubleVar(value=1.0)
        self.keep_ratio = tk.BooleanVar(value=False)
        self.auto_crop = tk.BooleanVar(value=True)
        self.auto_capture_webcam = tk.BooleanVar(value=True)
        self.save_grayscale = tk.BooleanVar(value=False)
        self.collage_count = tk.IntVar(value=4)
        self.nama_var = tk.StringVar(value="Mahasiswa Contoh")
        self.nim_var = tk.StringVar(value="24010111129999")
        self.ttl_var = tk.StringVar(value="Semarang, 01-01-2000")

        self._build_ui()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(outer, text="Aksi", padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Load Image", command=self.on_load).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(top, text="Load Multiple for Collage", command=self.on_load_multiple).grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(top, text="Capture Webcam", command=self.on_capture).grid(row=0, column=2, padx=4, pady=4)
        ttk.Button(top, text="Capture Multiple for Collage", command=self.on_capture_multiple).grid(row=0, column=3, padx=4, pady=4)
        ttk.Button(top, text="Process Manual", command=self.on_process_manual).grid(row=0, column=4, padx=4, pady=4)
        ttk.Button(top, text="Smart Auto-Fix (Inovasi)", command=self.on_process_smart).grid(row=0, column=5, padx=4, pady=4)
        ttk.Button(top, text="Save Output", command=self.on_save).grid(row=0, column=6, padx=4, pady=4)
        ttk.Button(top, text="Create Daily Collage 4x4", command=self.on_collage).grid(row=0, column=7, padx=4, pady=4)

        setting = ttk.LabelFrame(outer, text="Pengaturan", padding=8)
        setting.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(setting, text="Brightness").grid(row=0, column=0, sticky="w")
        ttk.Scale(setting, from_=-100, to=100, variable=self.brightness, orient=tk.HORIZONTAL, length=220).grid(row=0, column=1, padx=4)
        ttk.Label(setting, textvariable=self.brightness, width=5).grid(row=0, column=2)

        ttk.Label(setting, text="Contrast").grid(row=0, column=3, sticky="w", padx=(14, 0))
        ttk.Scale(setting, from_=0.5, to=3.0, variable=self.contrast, orient=tk.HORIZONTAL, length=220).grid(row=0, column=4, padx=4)
        self.lbl_contrast = ttk.Label(setting, width=6)
        self.lbl_contrast.grid(row=0, column=5)
        self._sync_contrast_label()
        self.contrast.trace_add("write", lambda *_: self._sync_contrast_label())

        ttk.Checkbutton(setting, text="Keep aspect ratio", variable=self.keep_ratio).grid(row=0, column=6, padx=(16, 0))
        ttk.Checkbutton(setting, text="Auto Crop Kartu", variable=self.auto_crop).grid(row=0, column=7, padx=(8, 0))
        ttk.Checkbutton(setting, text="Auto Capture Webcam", variable=self.auto_capture_webcam).grid(row=0, column=8, padx=(8, 0))
        ttk.Checkbutton(setting, text="Save as Grayscale", variable=self.save_grayscale).grid(row=0, column=9, padx=(8, 0))

        ttk.Label(setting, text="Nama").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(setting, textvariable=self.nama_var, width=28).grid(row=1, column=1, columnspan=2, sticky="we", pady=(8, 0))

        ttk.Label(setting, text="NIM").grid(row=1, column=3, sticky="w", pady=(8, 0), padx=(14, 0))
        ttk.Entry(setting, textvariable=self.nim_var, width=24).grid(row=1, column=4, columnspan=2, sticky="we", pady=(8, 0))

        ttk.Label(setting, text="TTL").grid(row=1, column=6, sticky="w", pady=(8, 0), padx=(16, 0))
        ttk.Entry(setting, textvariable=self.ttl_var, width=32).grid(row=1, column=7, columnspan=2, sticky="we", pady=(8, 0), padx=(8, 0))

        ttk.Label(setting, text="Collage Images").grid(row=1, column=9, sticky="w", pady=(8, 0), padx=(16, 0))
        ttk.Spinbox(setting, from_=2, to=16, textvariable=self.collage_count, width=5).grid(row=1, column=10, sticky="w", pady=(8, 0))

        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.pnl1 = self._panel(body, "Original")
        self.pnl2 = self._panel(body, "Grayscale")
        self.pnl3 = self._panel(body, "Processed")

        self.pnl1["frame"].grid(row=0, column=0, sticky="nsew", padx=6)
        self.pnl2["frame"].grid(row=0, column=1, sticky="nsew", padx=6)
        self.pnl3["frame"].grid(row=0, column=2, sticky="nsew", padx=6)

        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=1)
        body.rowconfigure(0, weight=1)

        box = ttk.LabelFrame(outer, text="Log", padding=8)
        box.pack(fill=tk.X, pady=(10, 0))
        self.log = tk.Text(box, height=6)
        self.log.pack(fill=tk.X)
        self._write_log("GUI siap. Silakan load/capture gambar.")

    def _panel(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        label = ttk.Label(frame, text="Belum ada gambar")
        label.pack(fill=tk.BOTH, expand=True)
        return {"frame": frame, "label": label}

    def _write_log(self, text):
        now = datetime.now().strftime("%H:%M:%S")
        self.log.insert(tk.END, f"[{now}] {text}\n")
        self.log.see(tk.END)

    def _sync_contrast_label(self):
        self.lbl_contrast.config(text=f"{self.contrast.get():.2f}")

    def _to_photo(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((380, 260), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil)

    def _show(self):
        self._show_panel(self.pnl1, self.original, "o")
        self._show_panel(self.pnl2, self.gray, "g")
        self._show_panel(self.pnl3, self.processed, "p")

    def _show_panel(self, panel, image, kind):
        if image is None:
            panel["label"].config(image="", text="Belum ada gambar")
            return

        photo = self._to_photo(image)
        panel["label"].config(image=photo, text="")

        if kind == "o":
            self.img_original_tk = photo
        elif kind == "g":
            self.img_gray_tk = photo
        else:
            self.img_processed_tk = photo

    def on_load(self):
        path = filedialog.askopenfilename(
            title="Pilih gambar kartu",
            initialdir=INPUT_FOLDER,
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")],
        )
        if not path:
            return

        img = load_image(path)
        if img is None:
            messagebox.showerror("Error", "Gagal load gambar")
            return

        self.original = img
        gray2d = convert_to_grayscale(img)
        self.gray = cv2.cvtColor(gray2d, cv2.COLOR_GRAY2BGR)
        self.processed = None
        self._show()
        self._write_log(f"Loaded: {os.path.basename(path)}")

    def on_capture_multiple(self):
        count = self.collage_count.get()
        self._write_log(f"Membuka webcam untuk capture {count} gambar... SPACE: ambil | ESC: batal")
        
        profile = {
            "name": self.nama_var.get().strip() or "-",
            "nim": self.nim_var.get().strip() or "-",
            "ttl": self.ttl_var.get().strip() or "-",
        }
        
        imgs = []
        for i in range(count):
            self._write_log(f"Capture {i+1}/{count}...")
            img = capture_from_webcam(
                auto_capture=self.auto_capture_webcam.get(),
                profile=profile,
            )
            if img is None:
                self._write_log(f"Capture {i+1} dibatalkan")
                break
            imgs.append(img)
            self._write_log(f"Capture {i+1} berhasil")
        
        if len(imgs) < 2:
            messagebox.showwarning("Info", "Minimal 2 gambar untuk collage")
            return
        
        if len(imgs) >= 16:
            grid = (4, 4)
        elif len(imgs) >= 9:
            grid = (3, 3)
        elif len(imgs) >= 6:
            grid = (3, 2)
        else:
            grid = (2, 2)
        
        collage = create_collage(imgs, grid_size=grid, title=f"WEBCAM CAPTURE")
        if collage is not None:
            self.processed = collage
            self._show()
            self._write_log(f"Collage webcam dari {len(imgs)} images ({grid[0]}x{grid[1]})")
            messagebox.showinfo("Sukses", f"Collage dari {len(imgs)} capture webcam\nGrid: {grid[0]}x{grid[1]}")
        else:
            messagebox.showerror("Error", "Gagal membuat collage")

    def on_load_multiple(self):
        paths = filedialog.askopenfilenames(
            title="Pilih beberapa gambar untuk collage",
            initialdir=INPUT_FOLDER,
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")],
        )
        if not paths:
            return

        imgs = []
        for path in paths:
            img = load_image(path)
            if img is not None:
                imgs.append(img)

        if len(imgs) < 4:
            messagebox.showwarning("Info", f"Minimal 4 gambar untuk collage (loaded: {len(imgs)})")
            return

        # Tentukan grid size berdasarkan jumlah gambar
        count = len(imgs)
        if count >= 16:
            grid = (4, 4)
        elif count >= 9:
            grid = (3, 3)
        elif count >= 6:
            grid = (3, 2)
        else:
            grid = (2, 2)

        collage = create_collage(imgs, grid_size=grid, title=f"COLLAGE {count} IMAGES")
        if collage is not None:
            self.processed = collage
            self._show()
            self._write_log(f"Collage created from {len(imgs)} images ({grid[0]}x{grid[1]})")
            messagebox.showinfo("Sukses", f"Collage dibuat dari {len(imgs)} gambar\nGrid: {grid[0]}x{grid[1]}")
        else:
            messagebox.showerror("Error", "Gagal membuat collage")

    def on_capture(self):
        mode = "AUTO" if self.auto_capture_webcam.get() else "MANUAL"
        self._write_log(f"Membuka webcam mode {mode}... SPACE capture, ESC batal")
        profile = {
            "name": self.nama_var.get().strip() or "-",
            "nim": self.nim_var.get().strip() or "-",
            "ttl": self.ttl_var.get().strip() or "-",
        }
        img = capture_from_webcam(
            auto_capture=self.auto_capture_webcam.get(),
            profile=profile,
        )
        if img is None:
            self._write_log("Capture dibatalkan/gagal")
            return

        self.original = img
        gray2d = convert_to_grayscale(img)
        self.gray = cv2.cvtColor(gray2d, cv2.COLOR_GRAY2BGR)
        self.processed = None
        self._show()
        self._write_log("Capture berhasil")

    def _pipeline(self, image):
        working = image
        if self.auto_crop.get():
            cropped, found = auto_crop_card_roi(image)
            working = cropped
            if found:
                self._write_log("Auto-crop: kartu terdeteksi dan dipotong otomatis")
            else:
                self._write_log("Auto-crop: kontur kartu tidak terdeteksi, lanjut gambar asli")

        out = resize_standard(working, keep_aspect_ratio=self.keep_ratio.get())
        out = adjust_brightness_contrast(out, brightness=self.brightness.get(), contrast=self.contrast.get())
        out = add_timestamp(out)
        out = add_watermark(out)
        out = add_border(out)
        return out

    def on_process_manual(self):
        if self.original is None:
            messagebox.showwarning("Info", "Load/capture dulu")
            return

        self.processed = self._pipeline(self.original)
        self._show()
        self._write_log("Process manual selesai")

    def on_process_smart(self):
        if self.original is None:
            messagebox.showwarning("Info", "Load/capture dulu")
            return

        work = self.original.copy()
        orient = detect_card_orientation(work)
        if orient and orient.get("orientation") == "PORTRAIT":
            work = cv2.rotate(work, cv2.ROTATE_90_CLOCKWISE)
            self._write_log("Inovasi: portrait terdeteksi, rotate otomatis")

        quality = calculate_image_quality_score(work)
        score = quality.get("overall_score", 0)
        if score < 65:
            work = auto_enhance_image(work)
            self._write_log(f"Inovasi: auto-enhance aktif, quality {score}/100")
        else:
            self._write_log(f"Quality {score}/100, auto-enhance tidak dibutuhkan")

        out = self._pipeline(work)
        cv2.putText(out, f"Quality:{score}/100", (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self.processed = out
        self._show()
        self._write_log("Smart Auto-Fix selesai")

    def on_save(self):
        if self.processed is None:
            messagebox.showwarning("Info", "Belum ada hasil process")
            return

        seq = get_next_sequence_for_today()
        to_save = add_sequence_number(self.processed, seq)
        path = save_with_organization(to_save, grayscale=self.save_grayscale.get())
        if path:
            self.processed = to_save
            self._show()
            self._write_log(f"Saved: {path}")
            messagebox.showinfo("Sukses", f"Tersimpan:\n{path}")
        else:
            messagebox.showerror("Error", "Gagal menyimpan")

    def on_collage(self):
        today = datetime.now().strftime("%Y-%m-%d")
        folder = os.path.join(OUTPUT_FOLDER, today)
        files = sorted(get_files_in_folder(folder, extension=".jpg"))
        if len(files) < 16:
            messagebox.showwarning("Info", "Butuh minimal 16 gambar hasil hari ini untuk collage 4x4")
            return

        imgs = []
        for f in files[-16:]:
            img = load_image(f)
            if img is not None:
                imgs.append(img)

        if len(imgs) < 16:
            messagebox.showerror("Error", "Tidak cukup gambar valid untuk collage 4x4")
            return

        collage = create_collage(imgs, grid_size=(4, 4), title="DAILY SUMMARY UNDIP")
        out = save_collage(collage)
        if out:
            self.processed = collage
            self._show()
            self._write_log(f"Collage saved: {out}")
            messagebox.showinfo("Sukses", f"Collage tersimpan:\n{out}")


def main():
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
