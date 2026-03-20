import os
import cv2
import numpy as np

from config import INPUT_FOLDER


def create_ktm_template(name, nim, faculty, prodi, index):
    width, height = 980, 620
    card = np.full((height, width, 3), 245, dtype=np.uint8)

    blue = (180, 60, 20)
    dark_blue = (130, 30, 10)
    white = (255, 255, 255)
    black = (20, 20, 20)

    cv2.rectangle(card, (0, 0), (width - 1, height - 1), blue, 22)
    cv2.rectangle(card, (30, 30), (width - 31, height - 31), dark_blue, 3)

    cv2.rectangle(card, (0, 0), (width, 130), blue, -1)
    cv2.putText(card, "UNIVERSITAS DIPONEGORO", (170, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.1, white, 3, cv2.LINE_AA)
    cv2.putText(card, "KARTU TANDA MAHASISWA", (250, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.9, white, 2, cv2.LINE_AA)

    cv2.rectangle(card, (75, 185), (285, 455), (225, 225, 225), -1)
    cv2.rectangle(card, (75, 185), (285, 455), dark_blue, 2)
    cv2.putText(card, "PHOTO", (130, 325), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (140, 140, 140), 2, cv2.LINE_AA)

    cv2.putText(card, f"NAMA   : {name}", (335, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.82, black, 2, cv2.LINE_AA)
    cv2.putText(card, f"NIM    : {nim}", (335, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.82, black, 2, cv2.LINE_AA)
    cv2.putText(card, f"FAKULTAS: {faculty}", (335, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.78, black, 2, cv2.LINE_AA)
    cv2.putText(card, f"PRODI  : {prodi}", (335, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.78, black, 2, cv2.LINE_AA)

    cv2.rectangle(card, (335, 430), (890, 475), (235, 235, 235), -1)
    cv2.rectangle(card, (335, 430), (890, 475), dark_blue, 1)
    cv2.putText(card, "Berlaku selama menjadi mahasiswa aktif", (350, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (50, 50, 50), 1, cv2.LINE_AA)

    cv2.putText(card, f"No. KTM: UNDIP-{index:04d}", (75, 565), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2, cv2.LINE_AA)

    return card


def main():
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    samples = [
        ("Budi Santoso", "24010111120001", "Teknik", "Informatika"),
        ("Siti Aisyah", "24010111120002", "Sains", "Matematika"),
        ("Rizky Pratama", "24010111120003", "Ekonomika", "Manajemen"),
        ("Nadia Putri", "24010111120004", "Kedokteran", "Pendidikan Dokter"),
        ("Dimas Saputra", "24010111120005", "Hukum", "Ilmu Hukum"),
    ]

    for i, (name, nim, faculty, prodi) in enumerate(samples, start=1):
        img = create_ktm_template(name, nim, faculty, prodi, i)
        out = os.path.join(INPUT_FOLDER, f"sample_ktm_undip_{i}.jpg")
        cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Created: {out}")

    # Sesuai struktur Project.md: sediakan sample_kartu.jpg
    base_img = create_ktm_template(
        "Mahasiswa Contoh",
        "24010111129999",
        "Teknik",
        "Teknik Komputer",
        0,
    )
    sample_path = os.path.join(INPUT_FOLDER, "sample_kartu.jpg")
    cv2.imwrite(sample_path, base_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Created: {sample_path}")


if __name__ == "__main__":
    main()
