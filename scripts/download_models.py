"""Tải face detector nhỏ từ Google; checkpoint cảm xúc đã có trong repository."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
DEFAULT_PATH = (
    Path(__file__).resolve().parents[1] / "emotion_web_app/models/blaze_face_short_range.tflite"
)
SHA256 = "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"


def download_detector(destination=DEFAULT_PATH):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "EmotionProject/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(5_000_001)
        if len(data) > 5_000_000 or len(data) < 8 or data[4:8] != b"TFL3":
            raise ValueError("Phản hồi không phải model TFLite hợp lệ.")
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("SHA256 face detector không khớp bản phát hành đã kiểm tra.")
        destination.write_bytes(data)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != SHA256:
        raise ValueError("File face detector có sẵn không khớp SHA256 được công bố trong script.")
    print(f"Ready: {destination}")
    print(f"SHA256: {hashlib.sha256(destination.read_bytes()).hexdigest()}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    download_detector(parser.parse_args().output)
