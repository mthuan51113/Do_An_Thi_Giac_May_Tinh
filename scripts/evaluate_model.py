"""Script to evaluate a model checkpoint on test dataset and print detailed metrics."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from training.common import load_model, make_loader, predict_loader
from training.evaluate import classification_metrics


def main(args=None):
    parser = argparse.ArgumentParser(description="Evaluate emotion model checkpoint.")
    parser.add_argument("--data", type=Path, default=Path("data/prepared/local_v2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/exp_finetune_vgaf_v2/best.pth"))
    parsed_args = parser.parse_args(args=args)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_path = parsed_args.data.resolve()
        ckpt_path = parsed_args.checkpoint.resolve()

        model, metadata = load_model(ckpt_path, device)
        loader = make_loader(data_path, "test", metadata, batch_size=32, workers=0, device=device)
        values, labels, predictions = predict_loader(model, loader, device)
        metrics = classification_metrics(labels, predictions, metadata["class_names"])

        print("=" * 60)
        print("                MODEL EVALUATION REPORT                ")
        print("=" * 60)
        print(f"Checkpoint Architecture : {metadata.get('architecture', 'unknown')}")
        print(f"Test Set Image Count   : {metrics['samples']}")
        print(f"Test Accuracy          : {metrics['accuracy']:.4%} ({sum(a == b for a, b in zip(labels, predictions))}/{metrics['samples']})")
        print(f"Macro F1 Score         : {metrics['macro_f1']:.4f}")
        print("-" * 60)
        print(f"{'Class':12s} | {'Precision':10s} | {'Recall':10s} | {'F1-Score':10s}")
        print("-" * 60)
        report = metrics["classification_report"]
        for name in metadata["class_names"]:
            cls_m = report.get(name, {})
            print(f"{name:12s} | {cls_m.get('precision', 0.0):10.4f} | {cls_m.get('recall', 0.0):10.4f} | {cls_m.get('f1-score', 0.0):10.4f}")
        print("=" * 60)
    except Exception as err:
        import traceback
        print("EVALUATION ERROR:", err)
        traceback.print_exc()


if __name__ == "__main__":
    main()
