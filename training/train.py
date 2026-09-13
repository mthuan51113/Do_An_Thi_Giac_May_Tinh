"""Fine-tune an emotion model, selecting the best checkpoint using validation only."""

import argparse
import csv
import time
from pathlib import Path

import torch

from .common import load_model, make_loader, predict_loader, seed_everything
from .data import audit_dataset, write_json


def save_checkpoint(path, model, metadata, epoch, accuracy, audit, args):
    checkpoint = {**metadata, "model_state_dict": {
        key: value.detach().cpu() for key, value in model.state_dict().items()},
        "epoch": epoch, "best_val_accuracy": accuracy,
        "selection_metric": "validation_accuracy", "evaluation_mode": "standard",
        "dataset_audit": audit, "seed": args.seed,
        "training_config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train(args):
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0 or args.patience < 1 or args.lr <= 0:
        raise ValueError("Use epochs/patience >= 1, batch-size >= 2, workers >= 0 and lr > 0.")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new run directory.")
    seed_everything(args.seed)
    device = torch.device(args.device)
    print("Checking all split pixels and labels before training...", flush=True)
    audit = audit_dataset(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "dataset_audit.json", audit)
    arch = getattr(args, "arch", "convnext_tiny")
    model, metadata = load_model(args.checkpoint, device, audit["class_names"],
                                args.pretrained, args.image_size, arch=arch)
    train_loader = make_loader(args.data, "train", metadata, args.batch_size, args.workers, device, args.seed)
    val_loader = make_loader(args.data, "val", metadata, args.batch_size, args.workers, device, args.seed)
    classifier = getattr(model, "classifier", getattr(model, "fc", getattr(model, "head", None)))
    if classifier is None and hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
    if classifier is None:
        classifier = model  # Fallback to entire model if head is not separated
    head_ids = {id(p) for p in classifier.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    if args.freeze_backbone:
        for parameter in backbone:
            parameter.requires_grad = False
    parameters = [{"params": classifier.parameters(), "lr": args.lr * 10}]
    if not args.freeze_backbone:
        parameters.append({"params": backbone, "lr": args.lr})
    optimizer = torch.optim.AdamW(parameters, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    counts = torch.bincount(torch.tensor(train_loader.dataset.targets), minlength=len(metadata["class_names"]))
    weights = (counts.sum() / (len(counts) * counts)).to(device) if args.class_weights else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    baseline, _, _ = predict_loader(model, val_loader, device)
    best_accuracy, stale_epochs, history = baseline["accuracy"], 0, []
    save_checkpoint(args.output / "best.pth", model, metadata, 0, best_accuracy, audit, args)
    write_json(args.output / "baseline_validation.json", baseline)
    print(f"Initial validation accuracy: {best_accuracy:.4%}; device={device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        if args.freeze_backbone:
            model.eval()  # Keep frozen BatchNorm statistics fixed as well as its parameters.
            classifier.train()
        total_loss, correct, seen = 0.0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; check images, learning rate and AMP.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            seen += targets.numel()
            total_loss += loss.item() * targets.numel()
            correct += (logits.argmax(1) == targets).sum().item()
        validation, _, _ = predict_loader(model, val_loader, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": total_loss / seen, "train_accuracy": correct / seen,
               "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
               "seconds": round(time.monotonic() - started, 2)}
        history.append(row)
        with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch}/{args.epochs}: train={row['train_accuracy']:.4%}, "
              f"val={row['val_accuracy']:.4%}, {row['seconds']}s", flush=True)
        if validation["accuracy"] > best_accuracy:
            best_accuracy, stale_epochs = validation["accuracy"], 0
            save_checkpoint(args.output / "best.pth", model, metadata, epoch, best_accuracy, audit, args)
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping: validation did not improve for {args.patience} epochs.", flush=True)
            break
    write_json(args.output / "class_names.json", metadata["class_names"])
    print(f"Best validation accuracy: {best_accuracy:.4%}. Test has not been evaluated.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Fine-tune local checkpoint; optimizer starts fresh")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5, help="Backbone learning rate; classifier uses 10x")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224, help="New model only; existing metadata is preserved")
    parser.add_argument("--arch", default="convnext_tiny", help="Architecture name when creating a new model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
