"""MediaPipe detector khởi tạo khi cần và được bảo vệ khi chạy nhiều luồng."""

import atexit
import threading

import cv2
import numpy as np

from .config import FACE_BOX_PADDING, FACE_DETECTOR_MODEL_PATH, MIN_DETECTION_CONFIDENCE

_detector = None
_detector_lock = threading.Lock()


def close_detector():
    global _detector
    with _detector_lock:
        if _detector is not None:
            _detector.close()
            _detector = None


atexit.register(close_detector)


def detect_faces(image):
    """Trả tọa độ (x1,y1,x2,y2) để crop ảnh OpenCV BGR."""
    global _detector
    if not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("Ảnh đầu vào phải là mảng uint8 không rỗng.")
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB if image.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Ảnh phải có 1, 3 hoặc 4 kênh.")
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    with _detector_lock:
        if _detector is None:
            if not FACE_DETECTOR_MODEL_PATH.is_file():
                raise FileNotFoundError(
                    "Thiếu face detector. Chạy: python scripts/download_models.py"
                )
            options = vision.FaceDetectorOptions(
                base_options=python.BaseOptions(model_asset_path=str(FACE_DETECTOR_MODEL_PATH)),
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            )
            _detector = vision.FaceDetector.create_from_options(options)
        result = _detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        )
    height, width = rgb.shape[:2]
    boxes = []
    for detection in result.detections:
        box = detection.bounding_box
        padding = int(max(box.width, box.height) * FACE_BOX_PADDING)
        x1, y1 = max(0, box.origin_x - padding), max(0, box.origin_y - padding)
        x2 = min(width, box.origin_x + box.width + padding)
        y2 = min(height, box.origin_y + box.height + padding)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes
