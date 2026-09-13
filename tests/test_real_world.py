from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import evaluate_real_world as real_world


def test_real_world_detection_failures_count_as_incorrect(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    reference.mkdir()
    Image.new("RGB", (12, 12), "white").save(reference / "train.png")
    Image.new("RGB", (12, 12), "red").save(tmp_path / "one.png")
    Image.new("RGB", (12, 12), "blue").save(tmp_path / "two.png")
    labels = tmp_path / "labels.csv"
    labels.write_text("path,label\none.png,happy\ntwo.png,sad\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"mocked checkpoint")
    monkeypatch.setattr(
        real_world,
        "load_emotion_model",
        lambda *a, **kw: SimpleNamespace(emotion_class_names=["happy", "sad"]),
    )
    faces = iter([[{"label": "happy"}], []])
    monkeypatch.setattr(real_world, "process_frame", lambda *a: (None, next(faces)))
    report = real_world.evaluate_photos(labels, checkpoint, tmp_path / "report.json", reference)
    assert report["samples"] == 2
    assert report["accuracy"] == 0.5
    assert report["target_met"] is False
    assert report["predictions"][1]["predicted"] == "no_face"
    assert report["confusion_matrix"][1][2] == 1
    Image.new("RGB", (12, 12), "red").save(reference / "overlap.png")
    with pytest.raises(ValueError, match="overlap"):
        real_world.evaluate_photos(labels, checkpoint, tmp_path / "invalid.json", reference)
