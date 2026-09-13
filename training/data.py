"""Dataset checks shared by preparation, training and evaluation."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

ALIASES = {
    "anger": "angry", "angry": "angry", "contempt": "contempt",
    "disgust": "disgust", "disgusted": "disgust", "fear": "fear",
    "fearful": "fear", "happiness": "happy", "happy": "happy",
    "neutral": "neutral", "sadness": "sad", "sad": "sad",
    "surprised": "surprise", "surprise": "surprise",
}
SPLIT_ALIASES = {"train": {"train", "training"},
                 "val": {"val", "valid", "validation"},
                 "test": {"test", "testing"}}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def canonical_label(label):
    try:
        return ALIASES[label.strip().lower()]
    except KeyError as error:
        raise ValueError(f"Unknown emotion label: {label!r}") from error


def split_directories(root):
    root = Path(root).resolve()
    result = {}
    for split, aliases in SPLIT_ALIASES.items():
        matches = [p for p in root.iterdir() if p.is_dir() and p.name.lower() in aliases]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {split} directory in {root}.")
        result[split] = matches[0]
    return result


def image_digest(path):
    """Hash decoded pixels, catching identical images with different file metadata."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def scan_folder(folder, split, source, subject_regex=None):
    folder = Path(folder).resolve()
    pattern = re.compile(subject_regex) if subject_regex else None
    records = []
    for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        label = canonical_label(class_dir.name)
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            relative = path.relative_to(folder).as_posix()
            subject = None
            if pattern:
                match = pattern.search(relative)
                if not match or pattern.groups != 1:
                    raise ValueError(f"Subject regex must capture one ID in every path: {relative}")
                subject = match.group(1)
            try:
                digest = image_digest(path)
            except Exception as error:
                raise ValueError(f"Unreadable image: {path}: {error}") from error
            records.append(dict(path=str(path), split=split, source=source,
                                label=label, sha256=digest, subject=subject))
    if not records:
        raise ValueError(f"No images found in {folder}")
    return records


def validate_records(records):
    hashes, subjects = {}, {}
    for row in records:
        prior = hashes.setdefault(row["sha256"], row)
        if prior["label"] != row["label"]:
            raise ValueError(f"Conflicting labels for duplicate images: {prior['path']} / {row['path']}")
        if prior["split"] != row["split"]:
            raise ValueError(f"Cross-split duplicate: {prior['path']} / {row['path']}")
        if row.get("subject"):
            previous = subjects.setdefault(row["subject"], row["split"])
            if previous != row["split"]:
                raise ValueError(f"Subject {row['subject']} occurs in {previous} and {row['split']}.")
    classes = {s: {r["label"] for r in records if r["split"] == s} for s in SPLIT_ALIASES}
    if not classes["train"] or any(v != classes["train"] for v in classes.values()):
        raise ValueError(f"Every split must contain the same nonempty emotion classes: {classes}")
    return sorted(classes["train"])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def audit_dataset(root):
    """Recheck current pixels; the manifest is provenance, never a substitute for checking."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    provenance = {r["prepared_path"]: r for r in manifest.get("records", [])}
    records = []
    for split, folder in split_directories(root).items():
        for row in scan_folder(folder, split, "dataset"):
            relative = Path(row["path"]).relative_to(root).as_posix()
            if provenance:
                prior = provenance.get(relative)
                if not prior or prior["sha256"] != row["sha256"] or prior["split"] != split:
                    raise ValueError(f"Image differs from prepared manifest: {relative}")
                row["subject"] = prior.get("subject")
            records.append(row)
    if provenance and len(records) != len(provenance):
        raise ValueError("Prepared images were removed after manifest creation.")
    classes = validate_records(records)
    signature = sorted(f"{r['split']}/{r['label']}/{r['sha256']}" for r in records)
    return {"class_names": classes, "counts": dict(Counter(r["split"] for r in records)),
            "fingerprint": hashlib.sha256("\n".join(signature).encode()).hexdigest(),
            "subject_checked": all(r.get("subject") for r in records),
            "duplicate_check": "decoded_rgb_sha256", "root": str(root)}
