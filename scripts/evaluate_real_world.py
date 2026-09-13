"""Evaluate labeled real photos through the same detector/crop/model as the web app.

CSV columns: path,label. Every image must show exactly one intended face.
Missing/multiple faces count as errors, so detection failures cannot inflate accuracy.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion_web_app.src.predict import load_emotion_model  # noqa: E402
from emotion_web_app.src.video_utils import process_frame  # noqa: E402
from training.data import canonical_label, image_digest  # noqa: E402


def evaluate_photos(labels_path, checkpoint, output, reference_root, target=0.9, device="cpu"):
    """CSV-relative paths; exact duplicates of development data are rejected."""
    labels_path = Path(labels_path).resolve()
    reference_root = Path(reference_root).resolve()
    if not reference_root.is_dir():
        raise ValueError("Reference dataset directory does not exist.")
    with labels_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires path,label columns.")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no labeled images.")
    model = load_emotion_model(checkpoint, device=device)
    classes = model.emotion_class_names
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    known = {
        image_digest(path)
        for path in reference_root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    }
    if not known:
        raise ValueError("Reference dataset contains no readable images.")
    seen, prepared = set(), []
    for row in rows:
        label = canonical_label(row["label"])
        if label not in classes:
            raise ValueError(f"Label is not supported by checkpoint: {label}")
        path = (labels_path.parent / row["path"]).resolve()
        digest = image_digest(path)
        if digest in known or digest in seen:
            raise ValueError(f"Repeated image or overlap with development dataset: {path.name}")
        seen.add(digest)
        prepared.append((path, label, digest))
    predictions = []
    for path, expected, digest in prepared:
        with Image.open(path) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        _, faces = process_frame(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), model, classes, True)
        predicted = (
            faces[0]["label"] if len(faces) == 1 else "no_face" if not faces else "multiple_faces"
        )
        predictions.append(
            {
                "file": path.name,
                "sha256": digest,
                "expected": expected,
                "predicted": predicted,
                "face_count": len(faces),
                "correct": predicted == expected,
            }
        )
    correct = sum(row["correct"] for row in predictions)
    from sklearn.metrics import classification_report, confusion_matrix

    expected = [row["expected"] for row in predictions]
    predicted = [row["predicted"] for row in predictions]
    output_labels = classes + ["no_face", "multiple_faces"]
    report = {
        "evaluation": "real_world_single_face_end_to_end",
        "samples": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "target_accuracy": target,
        "target_met": correct / len(predictions) >= target,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "class_names": classes,
        "unrepresented_classes": sorted(set(classes) - set(expected)),
        "classification_report": classification_report(
            expected, predicted, labels=classes, output_dict=True, zero_division=0
        ),
        "confusion_labels": output_labels,
        "confusion_matrix": confusion_matrix(expected, predicted, labels=output_labels).tolist(),
        "limitations": [
            "Exact pixel overlaps checked; near-duplicates and subject overlap require manual audit.",
            "Accuracy covers the provided labels and conditions only.",
        ],
        "predictions": predictions,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
        help="All development images (train/validation/test) to check exact overlaps",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/real_world/metrics.json"))
    parser.add_argument("--target-accuracy", type=float, default=0.9)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if not 0 <= args.target_accuracy <= 1:
        parser.error("Target accuracy must be between 0 and 1.")
    try:
        report = evaluate_photos(
            args.labels,
            args.checkpoint,
            args.output,
            args.reference_root,
            args.target_accuracy,
            args.device,
        )
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(
        f"Real-world accuracy: {report['accuracy']:.2%} ({report['correct']}/{report['samples']})"
    )
    raise SystemExit(0 if report["target_met"] else 2)


if __name__ == "__main__":
    main()
