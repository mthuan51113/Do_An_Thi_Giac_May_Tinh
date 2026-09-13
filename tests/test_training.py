"""Checks that guard labels, held-out boundaries, checkpoint selection and metrics."""

import argparse
import json

import pytest
import torch
from PIL import Image

from training.common import EmotionFolder, image_transform
from training.data import audit_dataset
from training.evaluate import classification_metrics, evaluate
from training.prepare import prepare
from training.train import train


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "source"
    for split_index, split in enumerate(["Train", "Val", "Test"]):
        for class_index, label in enumerate(["anger", "happiness"]):
            folder = root / split / label
            folder.mkdir(parents=True)
            for number in range(2):
                color = (split_index * 80 + number * 20, class_index * 120, 30)
                Image.new("RGB", (36, 36), color).save(folder / f"S{split_index}_{number}.png")
    return root


def test_prepare_preserves_splits_aliases_and_subjects(dataset, tmp_path):
    output = tmp_path / "prepared"
    assert prepare([("local", dataset)], output, subject_regex=r"(S\d+)") == 12
    audit = audit_dataset(output)
    assert audit["counts"] == {"train": 4, "val": 4, "test": 4}
    assert audit["class_names"] == ["angry", "happy"]
    assert audit["subject_checked"] is True
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["split_policy"] == "preserve_existing"
    assert len(list(dataset.rglob("*.png"))) == 12


def test_pixel_duplicate_blocks_training_even_different_encoding(dataset):
    source = dataset / "Train" / "anger" / "S0_0.png"
    with Image.open(source) as image:
        image.save(dataset / "Test" / "anger" / "same_pixels.bmp")
    with pytest.raises(ValueError, match="Cross-split duplicate"):
        audit_dataset(dataset)


def test_subject_leakage_and_ckplus_require_ids(dataset, tmp_path):
    with pytest.raises(ValueError, match="requires --subject-regex"):
        prepare([("ckplus", dataset)], tmp_path / "ck")
    with pytest.raises(ValueError, match="Subject.*occurs"):
        prepare([("local", dataset)], tmp_path / "leaky", subject_regex=r"S\d+_(\d+)")


def test_modified_prepared_pixels_rejected(dataset, tmp_path):
    output = tmp_path / "prepared"
    prepare([("local", dataset)], output)
    image = next(output.rglob("*.png"))
    Image.new("RGB", (36, 36), "white").save(image)
    with pytest.raises(ValueError, match="differs from prepared manifest"):
        audit_dataset(output)


def test_checkpoint_order_controls_targets(dataset):
    metadata = {"image_size": 32, "normalization_mean": [0.5] * 3,
                "normalization_std": [0.5] * 3}
    folder = EmotionFolder(dataset / "Train", ["happy", "angry"], image_transform(metadata))
    assert folder.targets == [1, 1, 0, 0]


def test_metrics_count_missing_predictions_and_classes():
    report = classification_metrics([0, 0, 1, 1], [0, 0, 0, 0], ["angry", "happy"])
    assert report["accuracy"] == 0.5
    assert report["macro_f1"] == pytest.approx(1 / 3)
    assert report["confusion_matrix"] == [[2, 0], [2, 0]]
    with pytest.raises(ValueError):
        classification_metrics([0], [], ["angry"])


def test_cpu_train_evaluate_checkpoint_roundtrip(dataset, tmp_path):
    torch.set_num_threads(1)
    args = argparse.Namespace(data=dataset, output=tmp_path / "run", checkpoint=None,
                              epochs=1, patience=1, batch_size=2, workers=0, lr=2e-5,
                              seed=42, image_size=32, device="cpu", freeze_backbone=True,
                              pretrained=False, class_weights=True)
    train(args)
    checkpoint = torch.load(args.output / "best.pth", weights_only=True, map_location="cpu")
    assert checkpoint["selection_metric"] == "validation_accuracy"
    assert checkpoint["evaluation_mode"] == "standard"
    assert "test_accuracy" not in checkpoint
    assert checkpoint["best_val_accuracy"] >= json.loads(
        (args.output / "baseline_validation.json").read_text())["accuracy"]
    eval_args = argparse.Namespace(data=dataset, checkpoint=args.output / "best.pth",
                                   output=tmp_path / "evaluation", target_accuracy=1.0,
                                   batch_size=2, workers=0, device="cpu")
    status = evaluate(eval_args)
    report = json.loads((eval_args.output / "test_metrics.json").read_text())
    assert report["samples"] == 4
    assert status == (0 if report["target_met"] else 2)
