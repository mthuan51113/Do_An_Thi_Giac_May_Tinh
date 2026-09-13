"""Nạp checkpoint và dự đoán nhiều khuôn mặt trong một batch."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import CLASS_NAMES_PATH, DEVICE, IMG_SIZE, MODEL_PATH, NORMALIZE_MEAN, NORMALIZE_STD
from .model import build_model


def _validate_classes(class_names):
    if (
        not isinstance(class_names, list)
        or not class_names
        or any(not isinstance(name, str) or not name.strip() for name in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError("Nhãn phải là danh sách tên lớp không rỗng, không trùng nhau.")
    return class_names


def load_class_names(class_names_path=CLASS_NAMES_PATH):
    with Path(class_names_path).open(encoding="utf-8") as file:
        return _validate_classes(json.load(file))


def load_emotion_model(model_path=MODEL_PATH, class_names_path=None, device=DEVICE):
    """Metadata trong checkpoint là nguồn nhãn và tiền xử lý chính thức."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Chưa tìm thấy checkpoint: {model_path.name}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint phải chứa state_dict và metadata, không phải model pickle.")
    class_names = checkpoint.get("class_names")
    if class_names is None:
        class_names = load_class_names(class_names_path or model_path.with_name("class_names.json"))
    _validate_classes(class_names)
    if class_names_path is not None and load_class_names(class_names_path) != class_names:
        raise ValueError("Thứ tự nhãn trong JSON không khớp checkpoint.")
    architecture = checkpoint.get("architecture", "custom_resnet")
    model = build_model(len(class_names), architecture=architecture)
    state_dict = checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(checkpoint.get(key), dict):
            state_dict = checkpoint[key]
            break
    model.load_state_dict(state_dict, strict=True)
    image_size = int(checkpoint.get("image_size", checkpoint.get("img_size", IMG_SIZE)))
    mean = tuple(checkpoint.get("normalization_mean", NORMALIZE_MEAN))
    std = tuple(checkpoint.get("normalization_std", NORMALIZE_STD))
    if image_size <= 0 or len(mean) != 3 or len(std) != 3:
        raise ValueError("Kích thước ảnh hoặc cấu hình chuẩn hóa không hợp lệ.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or min(std) <= 0:
        raise ValueError("Giá trị chuẩn hóa phải hữu hạn và std phải lớn hơn 0.")
    model.emotion_architecture = architecture
    model.emotion_class_names = class_names
    model.emotion_image_size = image_size
    model.emotion_normalization_mean = mean
    model.emotion_normalization_std = std
    model.emotion_use_grayscale = checkpoint.get(
        "use_grayscale", architecture in {"custom_resnet", "custom_resnet18", None}
    )
    model.emotion_metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model_state_dict", "state_dict", "model", "optimizer_state_dict"}
        and not isinstance(value, torch.Tensor)
    }
    model.emotion_transform = _make_transform(model)
    return model.to(device).eval()


def _opencv_to_pil_rgb(face_image):
    if isinstance(face_image, Image.Image):
        return face_image.convert("RGB")
    if not isinstance(face_image, np.ndarray):
        raise TypeError("Ảnh phải là numpy.ndarray BGR hoặc PIL.Image RGB.")
    if face_image.size == 0 or face_image.dtype != np.uint8:
        raise ValueError("Ảnh phải không rỗng và có kiểu uint8.")
    if face_image.ndim == 2:
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
    elif face_image.ndim == 3 and face_image.shape[2] in (3, 4):
        conversion = cv2.COLOR_BGRA2RGB if face_image.shape[2] == 4 else cv2.COLOR_BGR2RGB
        rgb_image = cv2.cvtColor(face_image, conversion)
    else:
        raise ValueError("Ảnh phải có 1, 3 hoặc 4 kênh.")
    return Image.fromarray(rgb_image)


def _make_transform(model):
    steps = []
    if getattr(model, "emotion_use_grayscale", True):
        steps.append(transforms.Grayscale(num_output_channels=3))
    size = getattr(model, "emotion_image_size", IMG_SIZE)
    steps.extend(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                getattr(model, "emotion_normalization_mean", NORMALIZE_MEAN),
                getattr(model, "emotion_normalization_std", NORMALIZE_STD),
            ),
        ]
    )
    return transforms.Compose(steps)


def preprocess_face(face_image, model=None):
    transform = getattr(model, "emotion_transform", None) or _make_transform(model)
    return transform(_opencv_to_pil_rgb(face_image)).unsqueeze(0)


def predict_emotions(face_images, model, class_names):
    """Giữ thứ tự đầu vào; chỉ thực hiện một lượt suy luận cho mỗi batch."""
    if not face_images:
        return []
    if hasattr(model, "emotion_class_names") and list(class_names) != model.emotion_class_names:
        raise ValueError("Nhãn dự đoán không khớp thứ tự nhãn của model.")
    device = next(model.parameters()).device
    batch = torch.cat([preprocess_face(face, model) for face in face_images]).to(device)
    with torch.inference_mode():
        logits = model(batch)
        if logits.ndim != 2 or logits.shape != (len(face_images), len(class_names)):
            raise ValueError("Số lớp đầu ra không khớp danh sách nhãn.")
        probabilities = logits.softmax(dim=1).cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("Model trả về xác suất không hợp lệ.")
    return [
        {
            "label": class_names[int(row.argmax())],
            "confidence": float(row.max()),
            "probabilities": dict(zip(class_names, map(float, row))),
        }
        for row in probabilities
    ]


def predict_emotion(face_image, model, class_names):
    return predict_emotions([face_image], model, class_names)[0]
