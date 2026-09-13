# Emotion Studio

Ứng dụng Streamlit phân tích **biểu cảm khuôn mặt** bằng MediaPipe và PyTorch. Giao diện tiếng Việt hỗ trợ ảnh, video và webcam; kết quả gồm nhãn gợi ý, xác suất theo lớp và trạng thái chưa chắc chắn khi confidence thấp.

## Cài đặt và chạy

Chạy các lệnh sau **tại thư mục gốc repository** (thư mục chứa README chính):

```powershell
python -m pip install -r emotion_web_app/requirements.txt
python scripts/download_models.py
python -m streamlit run emotion_web_app/app.py
```

Mở địa chỉ Streamlit in ra, thường là <http://localhost:8501>. Script `download_models.py` chỉ tải face detector MediaPipe và kiểm tra SHA256; checkpoint nhận diện biểu cảm được đặt trong `emotion_web_app/models/`.

Để dùng CPU khi GPU đang huấn luyện:

```powershell
$env:EMOTION_DEVICE = "cpu"
python -m streamlit run emotion_web_app/app.py
```

Cấu hình giao diện và giới hạn upload 200 MB nằm tại [../.streamlit/config.toml](../.streamlit/config.toml). Chạy từ thư mục gốc để Streamlit đọc đúng cấu hình này.

## Sử dụng

| Chế độ | Thao tác | Kết quả |
| --- | --- | --- |
| Ảnh | Tải JPG, PNG, WEBP hoặc chụp camera; bấm **Phân tích ảnh** | Ảnh trước/sau, số khuôn mặt, nhãn tiếng Việt, phân bố xác suất; tải PNG và JSON |
| Video | Tải MP4, AVI, MOV, MKV, WEBM; bấm **Phân tích video** | Tiến độ, xem trước khung hình, video H.264, thống kê; tải MP4 và JSON |
| Webcam trực tiếp | Bấm START, cấp quyền camera; STOP để dừng | Khung hình gắn nhãn và FPS; không ghi lại webcam |

Ảnh từ điện thoại được chỉnh hướng theo EXIF; ảnh trên 20 triệu điểm ảnh bị từ chối, ảnh lớn hơn 1600 px mỗi cạnh được thu nhỏ trước khi xử lý. Tọa độ khuôn mặt trong JSON là tọa độ trên ảnh đã xử lý.

Video kết quả **không có âm thanh**. Lượt phát hiện trong thống kê là số khuôn mặt qua từng khung hình, không phải số người khác nhau. Tệp tạm được xóa khi xử lý hoàn tất hoặc gặp lỗi; video kết quả nằm trong bộ nhớ của phiên và vẫn có thể tải xuống sau khi Streamlit chạy lại giao diện. Tải kết quả trước khi đóng phiên. Đổi tệp, checkpoint hoặc ngưỡng tin cậy cần bấm phân tích lại.

## Mô hình và độ tin cậy

Mặc định ứng dụng dùng `emotion_web_app/models/best_efficientnet_b0_vgaf.pth`. Chọn checkpoint khác trong thanh bên: ứng dụng tìm tệp `.pth` ở `emotion_web_app/models/` và `models/`. Có thể đặt đường dẫn bên ngoài bằng biến môi trường `EMOTION_MODEL_PATH` trước khi chạy app.

Nhãn và cấu hình tiền xử lý được lấy từ metadata checkpoint. Chỉ checkpoint không có nhãn tích hợp mới cần `class_names.json` nằm cạnh tệp `.pth`; thứ tự nhãn phải khớp lúc huấn luyện. Mô hình được cache theo đường dẫn, thời gian sửa tệp và kích thước; thay checkpoint hoặc JSON nhãn sẽ làm mới cache.

**Confidence không phải accuracy.** Thanh ngưỡng tin cậy chỉ quyết định khi nào kết quả được đánh dấu `Chưa chắc chắn`; tăng ngưỡng không chứng minh mô hình đạt accuracy 90%. Mục tiêu ≥ 90% cần kết quả đánh giá trên tập kiểm thử độc lập, có nhãn, không tham gia lựa chọn mô hình. Biểu cảm nhìn thấy cũng không khẳng định cảm xúc bên trong của người trong ảnh.

## Xử lý sự cố

- **Không nạp được model:** kiểm tra checkpoint đã được đặt đúng thư mục, kiến trúc được hỗ trợ và nhãn metadata hợp lệ. Chỉ dùng checkpoint từ nguồn tin cậy.
- **Thiếu face detector:** chạy `python scripts/download_models.py` từ thư mục gốc.
- **Không phát hiện mặt:** thử ảnh rõ, đủ sáng và khuôn mặt nhìn gần chính diện; mặt quá nhỏ hoặc bị che có thể bị bỏ sót.
- **Video không mở được:** thử xuất lại bằng MP4/H.264 và kiểm tra tệp có khung hình hợp lệ.
- **Webcam không kết nối:** cấp quyền camera cho trình duyệt; dùng localhost hoặc HTTPS, đóng ứng dụng khác đang chiếm camera. Khi triển khai qua mạng, một số mạng cần thêm cấu hình TURN cho WebRTC.
- **Xử lý chậm:** bắt đầu bằng video ngắn hoặc ảnh nhỏ hơn; GPU CUDA được dùng tự động nếu khả dụng, trừ khi đặt `EMOTION_DEVICE=cpu`.

Ảnh/video được xử lý trên máy chạy Streamlit; nếu triển khai lên máy chủ, dữ liệu tải lên sẽ được gửi đến máy chủ đó. Ứng dụng không chủ động lưu bản sao media sau khi phiên xử lý kết thúc.

## Cấu trúc

```text
emotion_web_app/
├── app.py              # Giao diện, luồng nhập/xuất, bộ nhớ phiên
├── ui_utils.py         # Nhãn tiếng Việt, khóa cache, dữ liệu bảng
├── models/             # Checkpoint và face detector
├── src/
│   ├── config.py       # Đường dẫn, thiết bị, cấu hình detector
│   ├── model.py        # Kiến trúc mô hình
│   ├── predict.py      # Nạp model và suy luận theo batch
│   ├── face_detector.py
│   └── video_utils.py  # Nhận diện, gắn nhãn, xuất video
└── requirements.txt
```

Kiểm thử giao diện: `python -m pytest tests/test_app.py -q` từ thư mục gốc. Các bài kiểm tra dùng Streamlit AppTest để kiểm tra kết quả qua rerun, vô hiệu hóa cache khi thay checkpoint/ngưỡng, xử lý tệp lỗi và dọn thư mục tạm.
