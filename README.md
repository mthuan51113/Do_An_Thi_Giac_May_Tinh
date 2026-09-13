# Nhận diện biểu cảm khuôn mặt

Đồ án thị giác máy tính dùng **PyTorch + EfficientNet-B0 + MediaPipe**, với ứng dụng
Streamlit nhận diện biểu cảm từ ảnh, video và webcam. Hỗ trợ 8 lớp:
`angry`, `contempt`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

**Tiêu chí nghiệm thu: Accuracy tối thiểu 90% trên tập test và bộ ảnh thực tế có nhãn.**
Đây là mục tiêu cần đo, không phải cam kết từ kiến trúc hay phần trăm confidence trên giao diện.
Checkpoint ban đầu lưu test Accuracy **71,01%**; số này là metadata từ lần huấn luyện cũ.
Xem [kết quả và giới hạn đánh giá](docs/RESULTS.md) trước khi sử dụng trong báo cáo.

## Chạy web

Python 3.10 trở lên; chạy các lệnh tại thư mục gốc repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_models.py
python -m streamlit run emotion_web_app/app.py
```

Mở `http://localhost:8501`. Trên máy hiện tại đã có môi trường ở `../.venv`:

```powershell
..\.venv\Scripts\python.exe -m streamlit run emotion_web_app/app.py
```

Checkpoint mẫu đã có trong `emotion_web_app/models/`. Face detector được tải từ Google
và kiểm tra SHA256. Không cần `hsemotion`, không có bước tải model ngầm khi chạy web.
Muốn dùng CPU: đặt `$env:EMOTION_DEVICE="cpu"`.
Muốn dùng checkpoint mới: chọn trong sidebar hoặc đặt `$env:EMOTION_MODEL_PATH="runs/experiment/best.pth"`.
Xem [hướng dẫn ứng dụng](emotion_web_app/README.md).

## Dữ liệu và huấn luyện

[Danh sách dataset bổ sung](docs/DATASETS.md) ghi nguồn chính thức, quyền truy cập,
nhãn và cách giữ tập test độc lập. Không đồng nhất `contempt` với `disgust`.
Dữ liệu ảnh, video cá nhân và kết quả tạm được giữ ngoài Git.

Chuẩn bị dữ liệu từ các thư mục `Train/Val/Test` hoặc `train/val/test` có sẵn:

```powershell
python -m training.prepare --source local=../dataset --output data/prepared/local
```

Script chuẩn hóa tên lớp, kiểm tra ảnh hỏng, nhãn xung đột và ảnh trùng pixel giữa các split.
Nếu bị từ chối, sửa nguồn dữ liệu hoặc tạo một bộ dữ liệu mới đã kiểm toán;
không tắt kiểm tra để nâng điểm. Với chuỗi ảnh cùng người, cần chia theo người/phiên chụp.
Thêm ảnh huấn luyện từ nguồn khác bằng `--extra-train ten_nguon=duong_dan_imagefolder`.

```powershell
python -m training.train --data data/prepared/local --checkpoint emotion_web_app/models/best_efficientnet_b0_vgaf.pth --output runs/experiment --epochs 12 --batch-size 16
python -m training.evaluate --data data/prepared/local --checkpoint runs/experiment/best.pth --output runs/experiment/evaluation --target-accuracy 0.90
```

Chọn epoch tốt nhất bằng validation; cấu hình suy luận được cố định trước khi chấm test.
Không chọn TTA, threshold hay model dựa trên test. Huấn luyện cần bản PyTorch tương thích GPU;
CPU vẫn dùng được nhưng chậm hơn. Dùng `--help` để xem cấu hình của từng lệnh.

[Đánh giá thực tế](docs/REAL_WORLD_TEST.md) dùng CSV nhãn và toàn bộ chuỗi phát hiện mặt →
crop → phân loại giống web; ảnh không phát hiện mặt được tính là lỗi.

## Cấu trúc

```text
emotion_web_app/   Giao diện, model, tiền xử lý và xử lý video
training/          Chuẩn bị dữ liệu, huấn luyện, đánh giá
scripts/           Tải tài nguyên và đánh giá ảnh thực tế
tests/             Kiểm tra dữ liệu, inference, video và giao diện
docs/              Dataset, kết quả, hướng dẫn kiểm thử
runs/              Kết quả chạy cục bộ, được Git bỏ qua
```

Notebook thử nghiệm cũ được thay bằng các module và lệnh tái lập ở trên; lịch sử Git vẫn lưu
phiên bản cũ để đối chiếu. Đã loại cache Python và video đầu ra cũ khỏi repository.

## Kiểm tra code

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
```

GitHub Actions chạy kiểm tra trên CPU. Kiểm thử code không thay thế phép đo Accuracy của model.
