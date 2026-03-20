import os
from datetime import datetime
import cv2
from config import OUTPUT_FOLDER, COLLAGE_FOLDER, OUTPUT_QUALITY, OUTPUT_FORMAT


def ensure_directories():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(COLLAGE_FOLDER, exist_ok=True)


def get_files_in_folder(folder, extension=None):
    if not os.path.exists(folder):
        return []

    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            if extension is None or name.lower().endswith(extension.lower()):
                files.append(path)
    return files


def get_next_sequence_for_today():
    date_folder = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(OUTPUT_FOLDER, date_folder)
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for name in os.listdir(out_dir):
        lower = name.lower()
        if lower.endswith("_kartu.jpg") or lower.endswith("_kartu.jpeg") or lower.endswith("_kartu.png"):
            count += 1
    return count + 1


def save_with_organization(image, grayscale=False):
    if image is None:
        return None

    date_folder = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(OUTPUT_FOLDER, date_folder)
    os.makedirs(out_dir, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S_kartu") + f".{OUTPUT_FORMAT}"
    out_path = os.path.join(out_dir, filename)

    # Simpan grayscale jika opsi diaktifkan
    if grayscale and len(image.shape) == 3:
        image_to_save = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_to_save = image

    ok = cv2.imwrite(out_path, image_to_save, [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_QUALITY])
    return out_path if ok else None


def save_collage(collage):
    if collage is None:
        return None

    os.makedirs(COLLAGE_FOLDER, exist_ok=True)
    filename = datetime.now().strftime("summary_%Y-%m-%d") + f".{OUTPUT_FORMAT}"
    out_path = os.path.join(COLLAGE_FOLDER, filename)

    ok = cv2.imwrite(out_path, collage, [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_QUALITY])
    return out_path if ok else None
