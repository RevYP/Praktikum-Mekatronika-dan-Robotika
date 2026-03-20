from datetime import datetime
import cv2
import numpy as np
from config import (
    CARD_WIDTH,
    CARD_HEIGHT,
    TIMESTAMP_COLOR,
    TIMESTAMP_FORMAT,
    TIMESTAMP_POSITION,
    TIMESTAMP_SCALE,
    WATERMARK_TEXT,
    WATERMARK_POSITION,
    WATERMARK_COLOR,
    WATERMARK_SCALE,
    BORDER_COLOR,
    BORDER_THICKNESS,
)


def _order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def auto_crop_card_roi(image):
    if image is None:
        return None, False

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 60, 180)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image.copy(), False

    img_h, img_w = gray.shape[:2]
    img_area = img_h * img_w
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_quad = None
    best_area = 0

    for contour in contours[:20]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        area = cv2.contourArea(contour)

        if area < img_area * 0.12:
            continue
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        if area > best_area:
            best_area = area
            best_quad = approx.reshape(4, 2)

    if best_quad is None:
        return image.copy(), False

    rect = _order_points(best_quad)
    dst = np.array(
        [
            [0, 0],
            [CARD_WIDTH - 1, 0],
            [CARD_WIDTH - 1, CARD_HEIGHT - 1],
            [0, CARD_HEIGHT - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (CARD_WIDTH, CARD_HEIGHT))
    return warped, True


def resize_standard(image, width=CARD_WIDTH, height=CARD_HEIGHT, keep_aspect_ratio=False):
    if image is None:
        return None

    if not keep_aspect_ratio:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y_off = (height - new_h) // 2
    x_off = (width - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def convert_to_grayscale(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    if image is None:
        return None
    return cv2.convertScaleAbs(image, alpha=float(contrast), beta=int(brightness))


def add_timestamp(image):
    if image is None:
        return None
    result = image.copy()
    text = datetime.now().strftime(TIMESTAMP_FORMAT)
    cv2.putText(result, text, TIMESTAMP_POSITION, cv2.FONT_HERSHEY_SIMPLEX, TIMESTAMP_SCALE, TIMESTAMP_COLOR, 2, cv2.LINE_AA)
    return result


def add_sequence_number(image, sequence_number):
    if image is None:
        return None

    result = image.copy()
    label = f"No Urut: {int(sequence_number):03d}"
    h, _ = result.shape[:2]
    cv2.putText(
        result,
        label,
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def add_profile_text(image, name, nim, ttl):
    if image is None:
        return None

    result = image.copy()
    h, w = result.shape[:2]

    panel_h = 58
    y1 = max(0, h - panel_h - 34)
    y2 = min(h - 4, y1 + panel_h)
    x1 = max(0, int(w * 0.02))
    x2 = min(w - 4, int(w * 0.98))

    overlay = result.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.45, result, 0.55, 0)
    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), 1)

    safe_name = (name or "-").strip()
    safe_nim = (nim or "-").strip()
    safe_ttl = (ttl or "-").strip()

    cv2.putText(result, f"Nama: {safe_name}", (x1 + 8, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, f"NIM : {safe_nim}", (x1 + 8, y1 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, f"TTL : {safe_ttl}", (x1 + 8, y1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def add_watermark(image):
    if image is None:
        return None
    result = image.copy()
    cv2.putText(result, WATERMARK_TEXT, WATERMARK_POSITION, cv2.FONT_HERSHEY_SIMPLEX, WATERMARK_SCALE, WATERMARK_COLOR, 1, cv2.LINE_AA)
    return result


def add_border(image, color=BORDER_COLOR, thickness=BORDER_THICKNESS):
    if image is None:
        return None
    result = image.copy()
    h, w = result.shape[:2]
    cv2.rectangle(result, (0, 0), (w - 1, h - 1), color, thickness)
    return result


def create_collage(images, grid_size=(2, 2), padding=8, title="DAILY SUMMARY"):
    if not images:
        return None

    rows, cols = grid_size
    required = rows * cols
    normalized = []

    for img in images[:required]:
        normalized.append(resize_standard(img, CARD_WIDTH, CARD_HEIGHT))

    while len(normalized) < required:
        normalized.append(np.zeros((CARD_HEIGHT, CARD_WIDTH, 3), dtype=np.uint8))

    title_h = 50
    collage_h = rows * CARD_HEIGHT + (rows + 1) * padding + title_h
    collage_w = cols * CARD_WIDTH + (cols + 1) * padding
    canvas = np.full((collage_h, collage_w, 3), 235, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (collage_w, title_h), (40, 40, 40), -1)
    cv2.putText(canvas, f"{title}: {datetime.now().strftime('%Y-%m-%d')}", (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            y = title_h + padding + r * (CARD_HEIGHT + padding)
            x = padding + c * (CARD_WIDTH + padding)
            canvas[y:y + CARD_HEIGHT, x:x + CARD_WIDTH] = normalized[idx]
            idx += 1

    return canvas
