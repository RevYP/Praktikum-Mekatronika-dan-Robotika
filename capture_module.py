import os
import cv2
import numpy as np
from config import SUPPORTED_IMAGE_FORMATS, CARD_WIDTH, CARD_HEIGHT


def load_image(image_path):
    if not os.path.exists(image_path):
        return None

    _, ext = os.path.splitext(image_path)
    if ext.lower() not in SUPPORTED_IMAGE_FORMATS:
        return None

    return cv2.imread(image_path)


def _order_points(points):
    pts = np.array(points, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _detect_card_quad(roi):
    if roi is None or roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 60, 170)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    roi_h, roi_w = roi.shape[:2]
    roi_area = roi_h * roi_w
    best_quad = None
    best_area = 0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        area = cv2.contourArea(contour)
        if area < roi_area * 0.18:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        if area > best_area:
            best_area = area
            best_quad = approx.reshape(4, 2)

    return best_quad


def _warp_card_from_roi(roi, quad):
    if quad is None:
        return None

    src = _order_points(quad)
    dst = np.array(
        [
            [0, 0],
            [CARD_WIDTH - 1, 0],
            [CARD_WIDTH - 1, CARD_HEIGHT - 1],
            [0, CARD_HEIGHT - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(roi, matrix, (CARD_WIDTH, CARD_HEIGHT))


def _fit_frame_to_box(frame, box_w, box_h):
    if frame is None or frame.size == 0 or box_w <= 0 or box_h <= 0:
        return np.zeros((max(1, box_h), max(1, box_w), 3), dtype=np.uint8)

    src_h, src_w = frame.shape[:2]
    scale = min(box_w / src_w, box_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)

    x_off = (box_w - new_w) // 2
    y_off = (box_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _draw_undip_ktm_overlay(view, frame, x1, y1, x2, y2, profile=None):
    roi_w = x2 - x1
    roi_h = y2 - y1

    blue = (255, 0, 0)
    dark_blue = (210, 0, 0)
    white = (255, 255, 255)
    light_gray = (235, 235, 235)
    dark_text = (25, 25, 25)

    profile = profile or {}
    name = profile.get("name", "-")
    nim = profile.get("nim", "-")
    ttl = profile.get("ttl", "-")

    card_x1, card_y1 = x1 + 10, y1 + 10
    card_x2, card_y2 = x2 - 10, y2 - 10

    cv2.rectangle(view, (card_x1, card_y1), (card_x2, card_y2), light_gray, -1)
    cv2.rectangle(view, (x1, y1), (x2, y2), blue, 3)
    cv2.rectangle(view, (card_x1, card_y1), (card_x2, card_y2), dark_blue, 2)

    header_h = max(38, int(roi_h * 0.2))
    cv2.rectangle(view, (x1 + 12, y1 + 12), (x2 - 12, y1 + header_h), blue, -1)
    cv2.putText(
        view,
        "UNIVERSITAS DIPONEGORO",
        (x1 + 28, y1 + int(header_h * 0.5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        white,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        view,
        "KARTU TANDA MAHASISWA",
        (x1 + 44, y1 + int(header_h * 0.85)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        white,
        1,
        cv2.LINE_AA,
    )

    # Area PHOTO transparan (menampilkan frame kamera asli)
    photo_x1 = x1 + int(roi_w * 0.06)
    photo_y1 = y1 + int(roi_h * 0.28)
    photo_x2 = x1 + int(roi_w * 0.26)
    photo_y2 = y1 + int(roi_h * 0.78)

    photo_w = max(1, photo_x2 - photo_x1)
    photo_h = max(1, photo_y2 - photo_y1)
    live_patch = _fit_frame_to_box(frame, photo_w, photo_h)
    view[photo_y1:photo_y2, photo_x1:photo_x2] = live_patch
    cv2.rectangle(view, (photo_x1, photo_y1), (photo_x2, photo_y2), blue, 2)

    # Isian biodata di sisi kanan
    info_x1 = x1 + int(roi_w * 0.31)
    row1 = y1 + int(roi_h * 0.43)
    row2 = y1 + int(roi_h * 0.55)
    row3 = y1 + int(roi_h * 0.67)
    cv2.putText(view, f"NAMA : {name}", (info_x1, row1), cv2.FONT_HERSHEY_SIMPLEX, 0.58, dark_text, 2, cv2.LINE_AA)
    cv2.putText(view, f"NIM  : {nim}", (info_x1, row2), cv2.FONT_HERSHEY_SIMPLEX, 0.58, dark_text, 2, cv2.LINE_AA)
    cv2.putText(view, f"TTL  : {ttl}", (info_x1, row3), cv2.FONT_HERSHEY_SIMPLEX, 0.58, dark_text, 2, cv2.LINE_AA)

    return {
        "photo_x1": photo_x1,
        "photo_y1": photo_y1,
        "photo_x2": photo_x2,
        "photo_y2": photo_y2,
    }


def _create_clean_ktm_card(frame, profile=None):
    """Render clean KTM card tanpa instruction overlay untuk output final"""
    clean = np.full((CARD_HEIGHT, CARD_WIDTH, 3), 235, dtype=np.uint8)
    
    blue = (255, 0, 0)
    dark_blue = (210, 0, 0)
    white = (255, 255, 255)
    dark_text = (25, 25, 25)
    
    profile = profile or {}
    name = profile.get("name", "-")
    nim = profile.get("nim", "-")
    ttl = profile.get("ttl", "-")
    
    # Border
    cv2.rectangle(clean, (0, 0), (CARD_WIDTH - 1, CARD_HEIGHT - 1), blue, 3)
    cv2.rectangle(clean, (5, 5), (CARD_WIDTH - 6, CARD_HEIGHT - 6), dark_blue, 2)
    
    # Header UNDIP
    header_h = 50
    cv2.rectangle(clean, (7, 7), (CARD_WIDTH - 8, 7 + header_h), blue, -1)
    cv2.putText(clean, "UNIVERSITAS DIPONEGORO", (50, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, white, 2, cv2.LINE_AA)
    cv2.putText(clean, "KARTU TANDA MAHASISWA", (75, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.52, white, 1, cv2.LINE_AA)
    
    # Area PHOTO
    photo_x1 = 25
    photo_y1 = 70
    photo_x2 = 105
    photo_y2 = 195
    photo_w = photo_x2 - photo_x1
    photo_h = photo_y2 - photo_y1
    
    live_patch = _fit_frame_to_box(frame, photo_w, photo_h)
    clean[photo_y1:photo_y2, photo_x1:photo_x2] = live_patch
    cv2.rectangle(clean, (photo_x1, photo_y1), (photo_x2, photo_y2), blue, 2)
    
    # Biodata
    info_x1 = 125
    row1 = 110
    row2 = 138
    row3 = 166
    cv2.putText(clean, f"NAMA : {name}", (info_x1, row1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dark_text, 2, cv2.LINE_AA)
    cv2.putText(clean, f"NIM  : {nim}", (info_x1, row2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dark_text, 2, cv2.LINE_AA)
    cv2.putText(clean, f"TTL  : {ttl}", (info_x1, row3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dark_text, 2, cv2.LINE_AA)
    
    # No KTM bottom
    cv2.putText(clean, "No. KTM: UNDIP-0002", (25, CARD_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dark_text, 1, cv2.LINE_AA)
    
    return clean


def capture_from_webcam(camera_index=0, auto_capture=False, stable_seconds=3.0, profile=None):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None

    captured = None
    face_countdown_frames = 0
    target_frames = max(10, int(stable_seconds * 12))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    window_name = "SmartAttend Webcam"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_h, frame_w = frame.shape[:2]
        target_ratio = 400 / 250  # rasio kartu standar

        roi_w = int(frame_w * 0.72)
        roi_h = int(roi_w / target_ratio)

        if roi_h > int(frame_h * 0.78):
            roi_h = int(frame_h * 0.78)
            roi_w = int(roi_h * target_ratio)

        x1 = (frame_w - roi_w) // 2
        y1 = (frame_h - roi_h) // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h
        roi = frame[y1:y2, x1:x2]

        # Jendela hanya sesuai area KTM (rasio kartu), bukan full frame kamera
        view = np.full((roi_h, roi_w, 3), 220, dtype=np.uint8)
        layout = _draw_undip_ktm_overlay(view, frame, 0, 0, roi_w - 1, roi_h - 1, profile=profile)

        # Optional deteksi kontur kartu tetap dijalankan pada ROI untuk auto-cut
        detected_quad = _detect_card_quad(roi)

        # Area foto di dalam template KTM untuk deteksi wajah
        p_x1 = layout["photo_x1"]
        p_y1 = layout["photo_y1"]
        p_x2 = layout["photo_x2"]
        p_y2 = layout["photo_y2"]
        photo_roi = view[p_y1:p_y2, p_x1:p_x2]

        faces = ()
        if photo_roi.size > 0 and not face_cascade.empty():
            photo_gray = cv2.cvtColor(photo_roi, cv2.COLOR_BGR2GRAY)
            photo_gray = cv2.equalizeHist(photo_gray)
            faces = face_cascade.detectMultiScale(
                photo_gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(24, 24),
            )
        cv2.putText(
            view,
            "Posisikan kartu di dalam bingkai | SPACE: Foto | ESC: Batal",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if detected_quad is not None:
            cv2.polylines(view, [detected_quad.astype(np.int32)], True, (0, 255, 255), 2)

        if len(faces) > 0:
            for fx, fy, fw, fh in faces:
                g_x1 = p_x1 + fx
                g_y1 = p_y1 + fy
                g_x2 = g_x1 + fw
                g_y2 = g_y1 + fh
                cv2.rectangle(view, (g_x1, g_y1), (g_x2, g_y2), (0, 255, 0), 2)

        status_y1 = roi_h - 34
        status_y2 = roi_h - 8
        cv2.rectangle(view, (8, status_y1 - 18), (roi_w - 8, status_y2 + 8), (230, 230, 230), -1)

        if detected_quad is not None:
            cv2.putText(view, "Kartu terdeteksi", (16, status_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 120, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(view, "Mode KTM: kamera hanya area PHOTO", (16, status_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 50, 0), 2, cv2.LINE_AA)

        if auto_capture:
            if len(faces) > 0:
                face_countdown_frames += 1
                remaining = max(0, int((target_frames - face_countdown_frames) / 12) + 1)
                cv2.putText(
                    view,
                    f"Wajah terdeteksi - auto foto dalam: {remaining}",
                    (16, status_y2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                face_countdown_frames = 0
                cv2.putText(
                    view,
                    "Menunggu wajah pada area PHOTO...",
                    (16, status_y2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            if face_countdown_frames >= target_frames:
                captured = _create_clean_ktm_card(frame, profile=profile)
                break

        cv2.imshow(window_name, view)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        if key == 32:
            captured = _create_clean_ktm_card(frame, profile=profile)
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured
