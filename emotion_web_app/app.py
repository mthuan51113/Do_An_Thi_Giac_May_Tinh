"""Streamlit interface for image, video and live facial-expression recognition."""

import io
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from src.config import DEVICE, MODEL_PATH
from src.predict import load_emotion_model
from src.video_utils import process_frame, process_video_file
from ui_utils import emotion_name, file_revision, prediction_rows, result_key

APP_DIR = Path(__file__).resolve().parent
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_SIDE = 1600


@st.cache_resource(max_entries=2, show_spinner="Đang nạp mô hình…")
def load_resources(model_path, model_revision, labels_revision):
    """Reload when the selected checkpoint or its fallback labels change."""
    model = load_emotion_model(model_path=Path(model_path))
    return model, list(model.emotion_class_names)


def configure_page():
    st.set_page_config(
        page_title="Emotion Studio · Nhận diện biểu cảm",
        page_icon="◉",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1440px; padding-top: 2.5rem;}
        [data-testid="stSidebar"] {background: #f1f5f9;}
        [data-testid="stMetric"] {
            border: 1px solid #dce5ee; border-radius: 14px; padding: 16px;
        }
        .eyebrow {color: #087f8c; font-size: .8rem; font-weight: 700;
                  letter-spacing: .15em; margin-bottom: .4rem;}
        .intro {color: #526375; max-width: 780px; line-height: 1.7;}
        .welcome {border: 1px dashed #a9bbc9; border-radius: 18px;
                  padding: 2.5rem; margin-top: 1rem; background: #f8fafc;}
        .welcome h3 {margin-top: 0; color: #18364a;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def available_models():
    candidates = {Path(MODEL_PATH).resolve()}
    for directory in (APP_DIR / "models", APP_DIR.parent / "models"):
        if directory.exists():
            candidates.update(path.resolve() for path in directory.glob("*.pth"))
    return sorted(candidates, key=lambda path: (path != Path(MODEL_PATH).resolve(), path.name))


def sidebar_settings():
    with st.sidebar:
        st.markdown("### ◉ Emotion Studio")
        st.caption("ĐỒ ÁN THỊ GIÁC MÁY TÍNH")
        mode = st.radio("Nguồn đầu vào", ["Ảnh", "Video", "Webcam trực tiếp"])
        st.divider()
        st.markdown("#### Thiết lập nhận diện")
        threshold = st.slider(
            "Ngưỡng tin cậy", 0.0, 1.0, 0.55, 0.05,
            help="Dưới ngưỡng này, kết quả được đánh dấu chưa chắc chắn. "
                 "Tăng ngưỡng không làm tăng accuracy của mô hình.",
        )
        model_path = st.selectbox(
            "Mô hình", available_models(), format_func=lambda path: path.name,
        )
        st.caption(f"Thiết bị xử lý: {'GPU · CUDA' if DEVICE == 'cuda' else 'CPU'}")
        st.divider()
        st.caption(
            "Ảnh/video được xử lý trên máy chạy ứng dụng. Webcam cần quyền camera "
            "và địa chỉ localhost hoặc HTTPS."
        )
    return mode, threshold, model_path


def empty_state(title, description):
    # Both arguments are static interface copy, never uploaded content.
    st.markdown(
        f'<div class="welcome"><h3>{title}</h3><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def render_prediction_table(results, threshold):
    if not results:
        st.warning("Chưa phát hiện khuôn mặt. Hãy thử ảnh rõ, đủ sáng và nhìn gần chính diện.")
        return
    st.dataframe(prediction_rows(results, threshold), hide_index=True, width="stretch")
    if any(item["confidence"] < threshold for item in results):
        st.info("Một số khuôn mặt chưa đủ tin cậy. Nhãn gợi ý vẫn được giữ để bạn đối chiếu.")
    with st.expander("Xem phân bố xác suất theo khuôn mặt"):
        face_index = st.selectbox(
            "Khuôn mặt", range(len(results)),
            format_func=lambda index: f"Khuôn mặt {results[index]['face_id']}",
        )
        probabilities = results[face_index].get("probabilities", {})
        st.bar_chart(
            [{"Biểu cảm": emotion_name(label), "Xác suất": probability}
             for label, probability in probabilities.items()],
            x="Biểu cảm", y="Xác suất", horizontal=True,
        )


def read_image(image_bytes):
    """Respect phone orientation and bound inference work for large photographs."""
    with Image.open(io.BytesIO(image_bytes)) as source:
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise ValueError("Ảnh quá lớn. Hãy chọn ảnh có tối đa 20 triệu điểm ảnh.")
        image = ImageOps.exif_transpose(source).convert("RGB")
    original_size = image.size
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return np.asarray(image), original_size


def image_mode(model, class_names, threshold, model_signature):
    st.subheader("Khám phá biểu cảm từ một bức ảnh")
    source = st.radio("Chọn nguồn ảnh", ["Tải ảnh lên", "Chụp từ camera"], horizontal=True)
    if source == "Tải ảnh lên":
        uploaded = st.file_uploader(
            "Kéo thả ảnh hoặc chọn tệp", type=["jpg", "jpeg", "png", "webp"], key="image_upload",
        )
    else:
        uploaded = st.camera_input("Chụp ảnh khuôn mặt", key="camera_snapshot")
    if uploaded is None:
        empty_state("Bắt đầu với một khuôn mặt", "Chọn ảnh chân dung hoặc ảnh nhóm. "
                    "Ứng dụng sẽ tìm từng khuôn mặt và hiển thị biểu cảm cùng mức tin cậy.")
        return
    try:
        image_bytes = uploaded.getvalue()
        rgb_image, original_size = read_image(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        st.error(f"Không đọc được ảnh: {error}")
        return
    key = result_key(image_bytes, model_signature, threshold)
    if original_size != (rgb_image.shape[1], rgb_image.shape[0]):
        st.caption(f"Ảnh được thu nhỏ về {rgb_image.shape[1]} × {rgb_image.shape[0]} để xử lý nhanh hơn.")
    if st.button("Phân tích ảnh", type="primary", key="analyze_image"):
        try:
            with st.spinner("Đang tìm khuôn mặt và phân tích biểu cảm…"):
                annotated, predictions = process_frame(
                    cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR), model, class_names,
                    return_results=True, confidence_threshold=threshold,
                )
                output = io.BytesIO()
                Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(output, format="PNG")
                st.session_state["image_result"] = {
                    "key": key, "image": output.getvalue(), "predictions": predictions,
                }
        except Exception as error:
            st.error(f"Không thể phân tích ảnh: {error}")
    result = st.session_state.get("image_result", {})
    original_column, result_column = st.columns(2, gap="large")
    with original_column:
        st.markdown("#### Ảnh đầu vào")
        st.image(rgb_image, width="stretch")
    with result_column:
        st.markdown("#### Kết quả nhận diện")
        if result.get("key") == key:
            st.image(result["image"], width="stretch")
        else:
            st.info("Bấm **Phân tích ảnh** để xem kết quả với thiết lập hiện tại.")
    if result.get("key") != key:
        return
    predictions = result["predictions"]
    total_column, accepted_column = st.columns(2)
    total_column.metric("Khuôn mặt phát hiện", len(predictions))
    accepted_column.metric("Đủ ngưỡng tin cậy", sum(item["confidence"] >= threshold for item in predictions))
    render_prediction_table(predictions, threshold)
    image_download, json_download = st.columns(2)
    image_download.download_button("Tải ảnh kết quả", result["image"], "emotion_result.png", "image/png")
    json_download.download_button(
        "Tải dữ liệu JSON", json.dumps({"confidence_threshold": threshold, "predictions": predictions},
                                      ensure_ascii=False, indent=2),
        "emotion_predictions.json", "application/json",
    )


def render_video_result(result):
    summary = result["summary"]
    st.markdown("#### Video kết quả")
    _, video_column, _ = st.columns([1, 4, 1])
    video_column.video(result["video"], format="video/mp4")
    columns = st.columns(4)
    columns[0].metric("Khung hình đã xử lý", summary["processed_frames"])
    columns[1].metric("Khung hình có mặt", summary["frames_with_faces"])
    columns[2].metric("Lượt phát hiện", summary["total_face_detections"])
    columns[3].metric("Thời lượng", f"{summary['duration_seconds']:.1f} giây")
    st.caption("Lượt phát hiện được tính theo từng khung hình; một người có thể được đếm nhiều lần.")
    counts = summary.get("emotion_counts", {})
    if counts:
        st.bar_chart(
            [{"Biểu cảm": emotion_name(label), "Lượt phát hiện": count}
             for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)],
            x="Biểu cảm", y="Lượt phát hiện",
        )
    else:
        st.info("Video chưa có dự đoán biểu cảm đủ tin cậy. Kiểm tra khuôn mặt và ngưỡng đang chọn.")
    video_download, summary_download = st.columns(2)
    video_download.download_button("Tải video kết quả", result["video"], result["filename"], "video/mp4")
    summary_download.download_button(
        "Tải thống kê JSON", json.dumps(summary, ensure_ascii=False, indent=2),
        "emotion_video_summary.json", "application/json",
    )


def video_mode(model, class_names, threshold, model_signature):
    st.subheader("Theo dõi biểu cảm trong video")
    st.caption("Video xuất ra định dạng MP4/H.264, không có âm thanh. Nên bắt đầu bằng clip ngắn.")
    uploaded = st.file_uploader("Chọn video", type=["mp4", "avi", "mov", "mkv", "webm"], key="video_upload")
    if uploaded is None:
        empty_state("Phân tích một đoạn video", "Xem tiến độ và khung hình nhận diện trong lúc xử lý. "
                    "Kết quả gồm video gắn nhãn và thống kê có thể tải xuống.")
        return
    video_bytes = uploaded.getvalue()
    key = result_key(video_bytes, model_signature, threshold)
    with st.expander("Xem video đầu vào"):
        st.video(video_bytes)
    if st.button("Phân tích video", type="primary", key="analyze_video"):
        progress = st.progress(0.0, text="Chuẩn bị video…")
        _, preview_column, _ = st.columns([1, 4, 1])
        preview = preview_column.empty()
        previous_update = 0.0

        def update_progress(done, total):
            nonlocal previous_update
            now = time.monotonic()
            if now - previous_update >= 0.2 or done == total:
                progress.progress(min(done / total, 1.0) if total else 0.0,
                                  text=f"Đã xử lý {done:,}" + (f" / {total:,}" if total else "") + " khung hình")
                previous_update = now

        def update_preview(frame, done, total):
            preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                          caption=f"Đang phân tích · Khung hình {done:,}", width="stretch")

        try:
            # Session bytes survive Streamlit reruns; disk files are removed even on failure.
            with tempfile.TemporaryDirectory(prefix="emotion_video_") as directory:
                input_path = Path(directory) / f"input{Path(uploaded.name).suffix.lower()}"
                input_path.write_bytes(video_bytes)
                output_path, summary = process_video_file(
                    input_path, Path(directory) / "result.mp4", model, class_names,
                    confidence_threshold=threshold, progress_callback=update_progress,
                    preview_callback=update_preview,
                )
                st.session_state["video_result"] = {
                    "key": key, "video": Path(output_path).read_bytes(), "summary": summary,
                    "filename": f"{Path(uploaded.name).stem}_emotion.mp4",
                }
            st.success("Đã phân tích xong. Bạn có thể xem và tải kết quả bên dưới.")
        except Exception as error:
            st.error(f"Không thể xử lý video: {error}")
        finally:
            preview.empty()
            progress.empty()
    result = st.session_state.get("video_result", {})
    if result.get("key") == key:
        render_video_result(result)
    else:
        st.info("Bấm **Phân tích video** để tạo kết quả với video và thiết lập hiện tại.")


def webcam_mode(model, class_names, threshold, model_signature):
    # Optional browser/video dependencies should not prevent image mode from starting.
    try:
        import av
        from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
    except ImportError:
        st.error("Thiếu thư viện webcam. Chạy: pip install -r emotion_web_app/requirements.txt")
        return
    st.subheader("Nhận diện trực tiếp từ webcam")
    st.caption("Bấm START và cho phép trình duyệt dùng camera. Bấm STOP để kết thúc; webcam không được ghi lại.")

    class EmotionVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.previous_time = time.monotonic()
            self.confidence_threshold = threshold
            self.error = None

        def recv(self, frame):
            bgr = frame.to_ndarray(format="bgr24")
            try:
                annotated = process_frame(bgr, model, class_names,
                                          confidence_threshold=self.confidence_threshold)
                self.error = None
            except Exception as error:
                self.error = str(error)
                annotated = bgr.copy()
                cv2.putText(annotated, "Inference error - restart webcam", (12, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 240), 2)
            now = time.monotonic()
            fps = 1.0 / max(now - self.previous_time, 1e-6)
            self.previous_time = now
            cv2.putText(annotated, f"FPS: {fps:.1f}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 120), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    _, webcam_column, _ = st.columns([1, 4, 1])
    with webcam_column:
        context = webrtc_streamer(
            key=f"emotion-webcam-{model_signature}", mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionVideoProcessor,
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
            async_processing=True,
        )
        if context.video_processor:
            context.video_processor.confidence_threshold = threshold

        @st.fragment(run_every=1)
        def show_camera_error():
            if context.video_processor and context.video_processor.error:
                st.error(f"Lỗi nhận diện: {context.video_processor.error}")

        show_camera_error()
    with st.expander("Camera chưa kết nối?"):
        st.write("Cho phép truy cập camera trên trình duyệt, đóng ứng dụng khác đang dùng camera, "
                 "rồi bấm STOP / START. Trên máy cá nhân, dùng địa chỉ localhost. "
                 "Triển khai qua mạng cần HTTPS và có thể cần cấu hình TURN cho WebRTC.")


def main():
    configure_page()
    mode, threshold, model_path = sidebar_settings()
    st.markdown('<div class="eyebrow">COMPUTER VISION / FACIAL EXPRESSION</div>', unsafe_allow_html=True)
    st.title("Đọc biểu cảm. Khám phá cảm xúc.")
    st.markdown('<p class="intro">Phân tích biểu cảm khuôn mặt từ ảnh, video và camera. '
                'Quan sát kết quả, xem mức tin cậy và tải dữ liệu chỉ trong một nơi.</p>', unsafe_allow_html=True)
    st.info("Confidence là mức tin cậy cho từng dự đoán, không phải độ chính xác trên tập kiểm thử. "
            "Mục tiêu accuracy ≥ 90% cần được xác nhận bằng đánh giá độc lập.")
    try:
        model_revision = file_revision(model_path)
        labels_revision = file_revision(model_path.with_name("class_names.json"))
        model, class_names = load_resources(str(model_path.resolve()), model_revision, labels_revision)
    except Exception as error:
        st.error(f"Chưa thể nạp mô hình: {error}")
        st.caption("Đặt checkpoint .pth hợp lệ vào emotion_web_app/models/ hoặc models/, "
                   "sau đó tải lại trang và chọn mô hình ở thanh bên. Xem README để cài đặt.")
        st.stop()
    model_signature = f"{model_path.resolve()}:{model_revision}:{labels_revision}"
    with st.sidebar.expander(f"{len(class_names)} nhóm biểu cảm"):
        st.write(" · ".join(emotion_name(name) for name in class_names))
        st.caption(f"Kiến trúc: {getattr(model, 'emotion_architecture', 'Theo checkpoint')}")
    st.divider()
    if mode == "Ảnh":
        image_mode(model, class_names, threshold, model_signature)
    elif mode == "Video":
        video_mode(model, class_names, threshold, model_signature)
    else:
        webcam_mode(model, class_names, threshold, model_signature)
    st.divider()
    st.caption("Kết quả mô tả biểu cảm quan sát được trên khuôn mặt; không xác định chắc chắn cảm xúc bên trong.")


if __name__ == "__main__":
    main()
