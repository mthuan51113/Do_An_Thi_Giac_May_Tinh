"""Evaluate a fixed checkpoint on the complete test split and enforce an accuracy target."""

import argparse
import csv
import hashlib
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from .common import load_model, make_loader, predict_loader
from .data import audit_dataset, write_json


def classification_metrics(labels, predictions, class_names):
    if not labels or len(labels) != len(predictions):
        raise ValueError("Expected equally sized, nonempty labels and predictions.")
    indices = list(range(len(class_names)))
    report = classification_report(labels, predictions, labels=indices, target_names=class_names,
                                   output_dict=True, zero_division=0)
    return {"accuracy": sum(a == b for a, b in zip(labels, predictions)) / len(labels),
            "macro_f1": report["macro avg"]["f1-score"], "samples": len(labels),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(labels, predictions, labels=indices).tolist()}


def evaluate(args):
    if not 0 <= args.target_accuracy <= 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Target accuracy must be in [0,1], batch-size >= 1 and workers >= 0.")
    audit = audit_dataset(args.data)
    device = torch.device(args.device)
    model, metadata = load_model(args.checkpoint, device)
    training_audit = getattr(model, "emotion_metadata", {}).get("dataset_audit", {}) or {}
    test_count = (training_audit.get("counts") or {}).get("test")
    if test_count is not None and test_count != audit.get("counts", {}).get("test"):
        raise ValueError("Dataset test split count differs from training audit; evaluate the original held-out test split.")
    loader = make_loader(args.data, "test", metadata, args.batch_size, args.workers, device)
    values, labels, predictions = predict_loader(model, loader, device)
    report = {**classification_metrics(labels, predictions, metadata["class_names"]),
              "loss": values["loss"], "split": "test", "evaluation_mode": "standard",
              "class_names": metadata["class_names"], "architecture": metadata["architecture"],
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "dataset_audit": audit, "target_accuracy": args.target_accuracy,
              "target_met": values["accuracy"] >= args.target_accuracy,
              "pretraining_overlap_verified": False}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "test_metrics.json", report)
    with (args.output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "true_label", "predicted_label"])
        for (path, _), label, predicted in zip(loader.dataset.samples, labels, predictions):
            writer.writerow([Path(path).relative_to(args.data.resolve()).as_posix(),
                             metadata["class_names"][label], metadata["class_names"][predicted]])
    print(f"Test accuracy={report['accuracy']:.4%}, macro F1={report['macro_f1']:.4f}, "
          f"n={report['samples']}; target {args.target_accuracy:.0%}: "
          f"{'PASS' if report['target_met'] else 'NOT MET'}", flush=True)
    return 0 if report["target_met"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    try:
        status = evaluate(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
