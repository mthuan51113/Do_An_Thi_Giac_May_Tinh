import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .data import canonical_label, split_directories


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class EmotionFolder(datasets.ImageFolder):
    """Resolve dataset label aliases into the checkpoint's explicit class order."""

    def __init__(self, root, class_names, transform, image_size=224, cache_in_memory=True):
        super().__init__(root, transform=transform)
        canonical = [canonical_label(name) for name in class_names]
        if len(canonical) != len(set(canonical)):
            raise ValueError("Checkpoint has duplicate emotion aliases.")
        found = {canonical_label(name) for name in self.classes}
        if found != set(canonical):
            raise ValueError(f"Dataset/checkpoint labels differ: {found} versus {canonical}")
        remap = {old: canonical.index(canonical_label(name))
                 for old, name in enumerate(self.classes)}
        self.samples = [(path, remap[label]) for path, label in self.samples]
        self.imgs = self.samples
        self.targets = [label for _, label in self.samples]
        self.classes = list(class_names)
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.cache_in_memory = cache_in_memory
        if cache_in_memory:
            print(f"Pre-resizing and caching {len(self.samples)} images in RAM ({image_size}x{image_size})...", flush=True)
            self.cached_images = []
            for path, _ in self.samples:
                with Image.open(path) as img:
                    img = img.convert("RGB").resize((image_size, image_size))
                    self.cached_images.append(img)

    def __getitem__(self, index):
        path, target = self.samples[index]
        if getattr(self, "cache_in_memory", False):
            sample = self.cached_images[index]
        else:
            sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def image_transform(metadata, augment=False):
    steps = []
    if metadata.get("use_grayscale", False):
        steps.append(transforms.Grayscale(num_output_channels=3))
    if augment:
        steps.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
        ])
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(metadata["normalization_mean"], metadata["normalization_std"]),
    ])
    return transforms.Compose(steps)


def make_loader(root, split, metadata, batch_size, workers, device, seed=42):
    dataset = EmotionFolder(split_directories(root)[split], metadata["class_names"],
                            image_transform(metadata, augment=split == "train"),
                            image_size=int(metadata.get("image_size", 224)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                      num_workers=workers, pin_memory=device.type == "cuda",
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed),
                      persistent_workers=workers > 0)


def load_model(checkpoint, device, class_names=None, pretrained=True, image_size=224, arch="convnext_tiny"):
    if checkpoint:
        from emotion_web_app.src.predict import load_emotion_model
        model = load_emotion_model(model_path=checkpoint, device=str(device))
        metadata = {"architecture": model.emotion_architecture,
                    "class_names": model.emotion_class_names,
                    "image_size": model.emotion_image_size,
                    "normalization_mean": list(model.emotion_normalization_mean),
                    "normalization_std": list(model.emotion_normalization_std),
                    "use_grayscale": model.emotion_use_grayscale}
    else:
        import timm
        model = timm.create_model(arch, pretrained=pretrained, num_classes=len(class_names))
        metadata = {"architecture": arch, "class_names": class_names,
                    "image_size": image_size, "normalization_mean": [0.485, 0.456, 0.406],
                    "normalization_std": [0.229, 0.224, 0.225], "use_grayscale": False}
    return model.to(device), metadata


@torch.inference_mode()
def predict_loader(model, loader, device):
    model.eval()
    labels, predictions = [], []
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        labels.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    if not labels:
        raise ValueError("Cannot evaluate an empty dataset.")
    accuracy = sum(a == b for a, b in zip(labels, predictions)) / len(labels)
    return {"loss": total_loss / len(labels), "accuracy": accuracy}, labels, predictions
