import os

CARD_WIDTH = 400
CARD_HEIGHT = 250
OUTPUT_QUALITY = 95
OUTPUT_FORMAT = "jpg"

WATERMARK_TEXT = "SmartAttend v1.0"
WATERMARK_POSITION = (10, 230)
WATERMARK_COLOR = (255, 255, 255)
WATERMARK_SCALE = 0.5

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMESTAMP_POSITION = (10, 20)
TIMESTAMP_COLOR = (255, 255, 255)
TIMESTAMP_SCALE = 0.55

BORDER_COLOR = (255, 0, 0)
BORDER_THICKNESS = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "input")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
COLLAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "collages")

SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


def validate_config():
    assert CARD_WIDTH > 0 and CARD_HEIGHT > 0, "Ukuran kartu harus positif"
    assert 0 <= OUTPUT_QUALITY <= 100, "OUTPUT_QUALITY harus 0-100"
