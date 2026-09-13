"""Small, framework-independent helpers for presentation and result identity."""

import hashlib
from pathlib import Path

EMOTION_NAMES = {
    "angry": "Tức giận", "anger": "Tức giận", "contempt": "Khinh miệt",
    "disgust": "Ghê tởm", "fear": "Sợ hãi", "happy": "Vui vẻ", "happiness": "Vui vẻ",
    "neutral": "Trung tính", "sad": "Buồn", "sadness": "Buồn", "surprise": "Ngạc nhiên",
    "uncertain": "Chưa chắc chắn", "unknown": "Chưa xác định",
}


def emotion_name(label):
    return EMOTION_NAMES.get(label.lower(), label)


def file_revision(path):
    """Use nanosecond timestamps and size so replacing a file invalidates caches."""
    path = Path(path)
    if not path.is_file():
        return None
    info = path.stat()
    return info.st_mtime_ns, info.st_size


def result_key(content, model_signature, threshold):
    digest = hashlib.sha256(content)
    digest.update(str(model_signature).encode("utf-8"))
    digest.update(float(threshold).hex().encode("ascii"))
    return digest.hexdigest()


def prediction_rows(results, threshold):
    return [
        {
            "Khuôn mặt": item["face_id"],
            "Biểu cảm gợi ý": emotion_name(item.get("predicted_label", item["label"])),
            "Mức tin cậy": f"{item['confidence']:.1%}",
            "Trạng thái": "Đủ tin cậy" if item["confidence"] >= threshold else "Chưa chắc chắn",
            "Vùng mặt (x1, y1, x2, y2)": ", ".join(str(value) for value in item["box"]),
        }
        for item in results
    ]
