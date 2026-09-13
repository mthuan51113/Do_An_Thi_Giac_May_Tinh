# Kiểm tra thực tế

Mục tiêu: **Accuracy ≥ 90% trên tập test và trên bộ ảnh thực tế có nhãn**.
Độ tin cậy softmax của một lần dự đoán không phải Accuracy.

## Bộ ảnh thực tế có nhãn

1. Thu ảnh từ những người/phiên chụp không dùng trong train/validation/test;
   đa dạng ánh sáng, góc mặt, kính và khoảng cách. Giữ đủ 8 lớp nếu đánh giá model 8 lớp.
2. Gán nhãn trước khi xem dự đoán. Mỗi ảnh có một khuôn mặt cần đánh giá.
   Nên có hai người kiểm tra các nhãn khó; biểu cảm nhìn thấy không xác định cảm xúc nội tâm.
3. Tạo CSV UTF-8 với đường dẫn ảnh tính từ vị trí CSV:

   ```csv
   path,label
   photos/001.jpg,happy
   photos/002.jpg,sad
   ```

4. Chạy từ thư mục repository:

   ```powershell
   python scripts/evaluate_real_world.py --labels data/real_world/labels.csv --checkpoint emotion_web_app/models/best_efficientnet_b0_vgaf.pth --reference-root ../dataset --output runs/real_world/metrics.json --target-accuracy 0.90
   ```

Script dùng đúng detector, vùng crop, tiền xử lý và model của web. Không phát hiện mặt
hoặc phát hiện nhiều mặt đều tính là lỗi trên tổng số ảnh, không bị loại khỏi mẫu số.
Ảnh trùng pixel với dữ liệu phát triển bị từ chối. Báo cáo có Accuracy, kết quả từng ảnh,
ma trận nhầm lẫn, SHA256 checkpoint và trạng thái đạt mục tiêu; mã thoát `2` nghĩa là chưa đạt.
Script chưa tự phát hiện cùng người hoặc ảnh gần trùng: cần kiểm tra bằng danh tính/phiên chụp.

## Kiểm tra chức năng web

- Ảnh: ảnh dọc có EXIF, nhiều mặt, không có mặt, ảnh hỏng; tải PNG/JSON.
- Video: upload MP4, xem tiến độ, phát H.264, tải kết quả sau khi tương tác lại giao diện.
  Video xuất chỉ chứa hình ảnh, không giữ âm thanh. Số lượt nhận diện không phải số người.
- Webcam: mở trên localhost hoặc HTTPS, cho phép camera trong trình duyệt; kiểm tra Start/Stop.
  Kết nối từ mạng ngoài có thể cần cấu hình TURN theo môi trường triển khai.
- Đo độ trễ/FPS riêng với Accuracy. Không dùng bộ lọc confidence để loại dự đoán sai khi báo Accuracy.

Ảnh/video có sẵn nhưng chưa gán nhãn chỉ chứng minh đường xử lý hoạt động;
không đủ cơ sở xác nhận Accuracy thực tế 90%.
