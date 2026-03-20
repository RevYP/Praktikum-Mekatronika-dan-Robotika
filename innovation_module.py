import cv2
import numpy as np


def detect_card_orientation(image):
    if image is None:
        return None

    h, w = image.shape[:2]
    ratio = w / h
    if ratio > 1.2:
        orientation = "LANDSCAPE"
    elif ratio < 0.85:
        orientation = "PORTRAIT"
    else:
        orientation = "SQUARE"

    return {"orientation": orientation, "aspect_ratio": ratio, "width": w, "height": h}


def calculate_image_quality_score(image):
    if image is None:
        return {"overall_score": 0}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(100, int(lap_var / 3))

    brightness = np.mean(gray)
    if 70 <= brightness <= 180:
        bright_score = 100
    elif 50 <= brightness <= 200:
        bright_score = 80
    else:
        bright_score = 55

    contrast = np.std(gray)
    if contrast >= 50:
        contrast_score = 100
    elif contrast >= 25:
        contrast_score = 80
    else:
        contrast_score = 55

    overall = int(0.5 * blur_score + 0.25 * bright_score + 0.25 * contrast_score)
    overall = max(0, min(100, overall))

    return {
        "overall_score": overall,
        "blur_score": blur_score,
        "brightness_score": bright_score,
        "contrast_score": contrast_score,
    }


def auto_enhance_image(image):
    if image is None:
        return None

    if len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    v = cv2.equalizeHist(v)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(v)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
