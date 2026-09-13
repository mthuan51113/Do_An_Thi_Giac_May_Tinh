"""Exercise Streamlit reruns, downloads and failure recovery without model inference."""

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "emotion_web_app"))
import app
from ui_utils import file_revision, prediction_rows, result_key


def run_app():
    from app import main
    main()


@pytest.fixture
def ui(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"test checkpoint")
    model = SimpleNamespace(emotion_class_names=["happy", "sad"], emotion_architecture="test")
    loads = []

    def load_model(model_path):
        loads.append(model_path)
        return model

    app.load_resources.clear()
    monkeypatch.setattr(app, "available_models", lambda: [checkpoint])
    monkeypatch.setattr(app, "load_emotion_model", load_model)
    return AppTest.from_function(run_app, default_timeout=20), checkpoint, loads


def image_upload():
    content = io.BytesIO()
    Image.new("RGB", (48, 48), "white").save(content, format="PNG")
    content.name = "portrait.png"
    return content


def test_image_result_survives_rerun_and_threshold_invalidates_it(ui, monkeypatch):
    page, _, _ = ui
    monkeypatch.setattr(app.st, "file_uploader", lambda *args, **kwargs: image_upload())
    calls = []

    def predict(frame, model, class_names, **kwargs):
        calls.append(kwargs)
        return frame, [{"face_id": 1, "label": "happy", "predicted_label": "happy",
                        "confidence": 0.8, "box": (1, 1, 40, 40),
                        "probabilities": {"happy": 0.8, "sad": 0.2}}]

    monkeypatch.setattr(app, "process_frame", predict)
    page.run()
    page.button(key="analyze_image").click().run()
    assert not page.exception
    assert [metric.value for metric in page.metric] == ["1", "1"]
    assert len(page.get("download_button")) == 2
    page.run()
    assert len(page.get("download_button")) == 2
    assert len(calls) == 1
    page.slider[0].set_value(0.9).run()
    assert not page.exception
    assert not page.metric
    assert not page.get("download_button")
    assert len(calls) == 1


def test_video_download_survives_rerun_and_temp_files_are_removed(ui, monkeypatch):
    page, _, _ = ui
    upload = io.BytesIO(b"test input video")
    upload.name = "clip.mp4"
    monkeypatch.setattr(app.st, "file_uploader", lambda *args, **kwargs: upload)
    paths = []

    def process_video(input_path, output_path, model, class_names, **kwargs):
        paths.append(input_path)
        assert input_path.read_bytes() == b"test input video"
        output_path.write_bytes(b"test encoded video")
        kwargs["progress_callback"](3, 3)
        return output_path, {"processed_frames": 3, "frames_with_faces": 2,
                             "total_face_detections": 2, "duration_seconds": 0.1,
                             "emotion_counts": {"happy": 1, "uncertain": 1}}

    monkeypatch.setattr(app, "process_video_file", process_video)
    page.run()
    page.sidebar.radio[0].set_value("Video").run()
    page.button(key="analyze_video").click().run()
    assert not page.exception
    assert len(page.metric) == 4
    assert len(page.get("download_button")) == 2
    assert all(not path.parent.exists() for path in paths)
    page.run()
    assert len(page.metric) == 4
    assert len(page.get("download_button")) == 2
    assert len(paths) == 1
    page.slider[0].set_value(0.8).run()
    assert not page.get("download_button")


def test_bad_video_shows_error_and_cleans_temporary_directory(ui, monkeypatch):
    page, _, _ = ui
    upload = io.BytesIO(b"broken")
    upload.name = "broken.mp4"
    monkeypatch.setattr(app.st, "file_uploader", lambda *args, **kwargs: upload)
    paths = []

    def fail(input_path, *args, **kwargs):
        paths.append(input_path)
        raise RuntimeError("Video không hợp lệ")

    monkeypatch.setattr(app, "process_video_file", fail)
    page.run()
    page.sidebar.radio[0].set_value("Video").run()
    page.button(key="analyze_video").click().run()
    assert not page.exception
    assert any("Video không hợp lệ" in error.value for error in page.error)
    assert all(not path.parent.exists() for path in paths)


def test_invalid_image_reports_readable_error(ui, monkeypatch):
    page, _, _ = ui
    monkeypatch.setattr(app.st, "file_uploader", lambda *args, **kwargs: io.BytesIO(b"broken png"))
    page.run()
    assert not page.exception
    assert any("Không đọc được ảnh" in error.value for error in page.error)


def test_checkpoint_replacement_reloads_cached_model(ui):
    page, checkpoint, loads = ui
    page.run()
    page.run()
    assert not page.exception
    assert len(loads) == 1
    checkpoint.write_bytes(b"replacement checkpoint with changed size")
    page.run()
    assert not page.exception
    assert loads == [checkpoint, checkpoint]


def test_missing_checkpoint_has_recovery_instructions(ui, monkeypatch):
    page, _, _ = ui

    def missing(**kwargs):
        raise FileNotFoundError("Chưa tìm thấy checkpoint")

    monkeypatch.setattr(app, "load_emotion_model", missing)
    page.run()
    assert not page.exception
    assert any("Chưa tìm thấy checkpoint" in error.value for error in page.error)
    assert any("checkpoint .pth hợp lệ" in caption.value for caption in page.caption)


def test_uncertain_prediction_keeps_original_suggested_label():
    rows = prediction_rows([{"face_id": 1, "label": "uncertain", "predicted_label": "happy",
                             "confidence": 0.4, "box": (0, 0, 10, 10)}], 0.55)
    assert rows[0]["Biểu cảm gợi ý"] == "Vui vẻ"
    assert rows[0]["Trạng thái"] == "Chưa chắc chắn"


def test_result_identity_changes_with_input_model_and_threshold(tmp_path):
    initial = result_key(b"video a", "model a:revision 1", 0.55)
    assert initial != result_key(b"video b", "model a:revision 1", 0.55)
    assert initial != result_key(b"video a", "model a:revision 2", 0.55)
    assert initial != result_key(b"video a", "model a:revision 1", 0.8)
    assert file_revision(tmp_path / "missing.pth") is None
