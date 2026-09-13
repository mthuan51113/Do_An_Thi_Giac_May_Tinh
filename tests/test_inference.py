import json

import numpy as np
import pytest
import torch
from torch import nn

from emotion_web_app.src import predict, video_utils
from emotion_web_app.src.face_detector import detect_faces


class FixedClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.calls = 0
        self.emotion_class_names = ["happy", "sad"]

    def forward(self, batch):
        self.calls += 1
        return self.logits.repeat(batch.shape[0], 1)


def test_batch_predictions_and_uncertainty_preserve_face_count(monkeypatch):
    model = FixedClassifier()
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    monkeypatch.setattr(
        video_utils, "detect_faces", lambda image: [(0, 0, 30, 30), (40, 40, 80, 80)]
    )
    output, results = video_utils.process_frame(
        frame, model, model.emotion_class_names, True, confidence_threshold=0.9
    )
    assert model.calls == 1
    assert len(results) == 2
    assert all(row["label"] == "uncertain" and row["predicted_label"] == "happy" for row in results)
    assert all(sum(row["probabilities"].values()) == pytest.approx(1) for row in results)
    assert output.shape == frame.shape
    assert not frame.any()


def test_bgr_pil_conversion_and_invalid_inputs():
    face = np.zeros((8, 8, 3), dtype=np.uint8)
    face[:, :, 2] = 255
    assert predict._opencv_to_pil_rgb(face).getpixel((0, 0)) == (255, 0, 0)
    for invalid in (np.zeros((0, 8, 3), dtype=np.uint8), np.zeros((8, 8, 2), dtype=np.uint8)):
        with pytest.raises(ValueError):
            predict.preprocess_face(invalid)
        with pytest.raises(ValueError):
            detect_faces(invalid)


def test_label_mismatch_is_rejected():
    model = FixedClassifier()
    with pytest.raises(ValueError, match="thứ tự"):
        predict.predict_emotions([np.zeros((8, 8, 3), dtype=np.uint8)], model, ["sad", "happy"])


def test_checkpoint_embedded_labels_and_safe_loader(tmp_path, monkeypatch):
    model = FixedClassifier()
    path = tmp_path / "model.pth"
    torch.save({"model_state_dict": model.state_dict(), "class_names": ["happy", "sad"]}, path)
    monkeypatch.setattr(predict, "build_model", lambda *args, **kwargs: FixedClassifier())
    loaded = predict.load_emotion_model(path, device="cpu")
    assert loaded.emotion_class_names == ["happy", "sad"]
    assert not loaded.training
    labels = tmp_path / "class_names.json"
    labels.write_text(json.dumps(["sad", "happy"]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        predict.load_emotion_model(path, labels, device="cpu")


def test_video_encode_and_cleanup(tmp_path, monkeypatch):
    import av

    source = tmp_path / "input.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((64, 64, 3), dtype=np.uint8), format="bgr24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    monkeypatch.setattr(video_utils, "detect_faces", lambda image: [])
    output, summary = video_utils.process_video_file(source, tmp_path / "output.mp4", None, [])
    assert summary["processed_frames"] == 3
    assert summary["total_face_detections"] == 0
    with av.open(str(output)) as result:
        assert result.streams.video[0].codec_context.name == "h264"
        assert len(list(result.decode(video=0))) == 3

    def fail(*args):
        raise RuntimeError("callback failed")

    partial = tmp_path / "partial.mp4"
    with pytest.raises(RuntimeError, match="callback failed"):
        video_utils.process_video_file(source, partial, None, [], progress_callback=fail)
    assert not partial.exists()
    with pytest.raises(ValueError):
        video_utils.process_video_file(source, source, None, [])
    assert source.exists()
