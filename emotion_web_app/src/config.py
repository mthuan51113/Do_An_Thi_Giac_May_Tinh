"""Cấu hình dùng chung. Đổi checkpoint bằng EMOTION_MODEL_PATH."""

import os
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    Path(
        os.environ.get("EMOTION_MODEL_PATH", BASE_DIR / "models" / "best_efficientnet_b0_vgaf.pth")
    )
    .expanduser()
    .resolve()
)
CLASS_NAMES_PATH = MODEL_PATH.with_name("class_names.json")
OUTPUT_DIR = BASE_DIR / "runs" / "outputs"
FACE_DETECTOR_MODEL_PATH = BASE_DIR / "models" / "blaze_face_short_range.tflite"
if not FACE_DETECTOR_MODEL_PATH.exists():
    FACE_DETECTOR_MODEL_PATH = BASE_DIR.parent / "models" / FACE_DETECTOR_MODEL_PATH.name

IMG_SIZE = 64
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
MIN_DETECTION_CONFIDENCE = 0.5
FACE_BOX_PADDING = 0.18
DEVICE = os.environ.get("EMOTION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
if DEVICE not in {"cpu", "cuda"}:
    raise ValueError("EMOTION_DEVICE phải là cpu hoặc cuda.")
