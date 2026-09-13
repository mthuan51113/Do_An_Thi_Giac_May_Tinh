# Dữ liệu và mục tiêu accuracy tối thiểu 90%

Nguồn được kiểm tra ngày 13/09/2026. **90% là accuracy trên tập kiểm thử độc lập**, không phải độ tin cậy của một dự đoán hoặc accuracy huấn luyện. Thêm dữ liệu không bảo đảm đạt ngưỡng này; kết quả phải được đo lại sau khi chốt mô hình bằng validation.

## Dữ liệu đang có

Đã tìm thấy `../dataset` trên máy phát triển, gồm 5.715 ảnh JPG:

| Nhãn | Train | Val | Test |
|---|---:|---:|---:|
| angry | 533 | 85 | 83 |
| contempt | 513 | 82 | 84 |
| disgust | 531 | 82 | 85 |
| fear | 532 | 81 | 89 |
| happy | 549 | 86 | 87 |
| neutral | 574 | 86 | 94 |
| sad | 547 | 81 | 93 |
| surprise | 551 | 88 | 99 |
| **Tổng** | **4.330** | **671** | **714** |

Kiểm tra toàn bộ ảnh bằng SHA-256 của file và SHA-256 của kích thước + pixel RGB đã giải mã: **0 nhóm ảnh trùng chính xác**, cả trong và giữa các split; không có ảnh lỗi giải mã. Điều này chưa loại trừ ảnh gần trùng hoặc cùng người xuất hiện ở nhiều split. Dataset chưa có tài liệu nguồn, giấy phép hoặc định danh người; một phần tên file mang thời gian chụp. Cần bổ sung `subject_id`, `session_id`, nguồn và điều kiện sử dụng trước khi khẳng định test độc lập theo người.

File `class_names.json` cũ từng thiếu `disgust`, trong khi ảnh thực tế có đủ tám lớp. Thứ tự nhãn phải lấy từ checkpoint và kiểm tra với dataset; không lấy từ một danh sách bảy nhãn cũ.

## Các nguồn bổ sung

| Dataset | Dữ liệu / nhãn | Truy cập và giá trị sử dụng |
|---|---|---|
| **JAFFE v2** | 213 ảnh, 10 người, 7 nhãn, không có `contempt`. | Đã tải bản chính thức 12,3 MB, kiểm tra checksum, chuyển thành ảnh RGB dùng bổ sung Train. Chỉ dùng nghiên cứu khoa học phi thương mại; cấm phân phối lại hoặc đăng ảnh lên GitHub/web. [Tác giả trên Zenodo](https://zenodo.org/records/14974867). |
| **FER2013 + FER+** | FER+ bổ sung phiếu nhãn cho cùng 35.887 ảnh FER2013; 8 cảm xúc và phiếu `unknown`/`NF`. | Đã tải **nhãn**, chưa có ảnh FER2013. Nhãn/code của Microsoft có giấy phép riêng; không suy ra quyền phân phối ảnh từ giấy phép repository. Giữ split chính thức: Training 28.709, PublicTest 3.589, PrivateTest 3.589. [Microsoft FER+](https://github.com/microsoft/FERPlus), [giấy phép](https://github.com/microsoft/FERPlus/blob/master/LICENSE.md), [ảnh gốc](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). |
| **RAF-DB Basic** | 7 lớp biểu cảm cơ bản; có `disgust`, không có `contempt`. | Phù hợp nghiên cứu ảnh ngoài thực tế. Lấy dữ liệu và split từ chủ sở hữu; trang tải bị timeout trong lần kiểm tra này nên chưa xác minh được điều khoản truy cập hiện tại. Không dùng bản mirror không rõ quyền. [Trang chính thức](https://www.whdeng.cn/RAF/model1.html), [bài báo gốc của tác giả](https://www.whdeng.cn/RAF/li_RAFDB_2017_CVPR.pdf). |
| **AffectNet / AffectNet+** | AffectNet có hơn 1 triệu ảnh thu thập, khoảng 440 nghìn ảnh gán nhãn thủ công; hỗ trợ bài toán 7/8 lớp. AffectNet+ thêm nhãn mềm và metadata. | Phù hợp nhất để tăng độ đa dạng cho hệ thống 8 lớp, nhưng cần quyền truy cập nghiên cứu. Trang hiện tại chuyển yêu cầu AffectNet sang AffectNet+; chỉ PI, quản lý phòng lab hoặc giáo sư được nộp thỏa thuận. Chưa tải, chưa gửi yêu cầu thay người dùng. [AffectNet](https://www.mohammadmahoor.com/pages/databases/affectnet/), [AffectNet+ và mẫu yêu cầu](https://www.mohammadmahoor.com/pages/databases/affectnetplus/). |
| **CK+** | 593 chuỗi từ 123 người; 327 chuỗi có nhãn biểu cảm đỉnh thuộc 7 lớp, có `contempt`. | Phù hợp nghiên cứu biểu cảm có chủ đích trong phòng thí nghiệm. Nếu lấy nhiều frame, phải chia theo **người**, không chia ngẫu nhiên frame. Neutral ở đầu chuỗi cần quy tắc gán nhãn riêng. Chưa tải; xác minh quyền với chủ sở hữu. [Bài báo gốc tại University of Pittsburgh](https://www.pitt.edu/~jeffcohn/CVPR2010_CK%2B2.pdf). |
| **SFEW** | Ảnh biểu cảm trích từ phim, 7 lớp, nhiều điều kiện ánh sáng và góc mặt. | Hữu ích làm đánh giá khác nguồn khi có quyền; giữ protocol của đúng phiên bản và tránh đưa frame cùng người/phim vào cả train và test. Chưa tải. [Danh mục dataset của đồng tác giả](https://rolandgoecke.net/research/datasets/). |

JAFFE là nguồn nhỏ với biểu cảm được yêu cầu tạo dáng. Cần trích dẫn [Lyons, Kamachi & Gyoba (2020)](https://arxiv.org/abs/2009.05938) và [Lyons (2021)](https://arxiv.org/abs/2107.13998) khi dùng. Không thay tập kiểm thử hiện tại bằng JAFFE rồi coi kết quả là mức cải thiện trên cùng bài toán.

## Tải lại dữ liệu đã chuẩn bị

Chạy từ thư mục gốc repository với môi trường Python của dự án:

```powershell
python scripts/download_ferplus_labels.py
python scripts/download_jaffe.py
```

Script xác minh SHA-256 trước khi dùng file. Lệnh JAFFE yêu cầu thư mục output trống để không trộn dữ liệu âm thầm. Có thể chỉ định `--output` khác cho một bản chuẩn bị mới.

| Vị trí local | Trạng thái |
|---|---|
| `data/raw/ferplus/fer2013new.csv` | 1.602.762 byte, 35.887 dòng nhãn; chưa có pixel ảnh. |
| `data/raw/jaffe/jaffe.zip` | 12.290.558 byte; MD5 khớp bản công bố `fe13f3302eb9968ef04367456f665436`. |
| `data/supplemental/jaffe/Train/` | 213 PNG: angry 30, disgust 29, fear 32, happy 31, neutral 30, sad 31, surprise 30. |
| `data/supplemental/jaffe/manifest.csv` | Giữ tên ảnh gốc, mã người, nguồn và vai trò chỉ bổ sung Train. |

Các thư mục ảnh và archive được `.gitignore` loại khỏi Git. Script tải chỉ chuẩn bị dữ liệu; **không tự gộp** vào lần huấn luyện hoặc thay các split hiện tại. Quy trình thí nghiệm phải ghi rõ nguồn nào thật sự được sử dụng.

## Ghép nhãn và giữ đánh giá hợp lệ

Các tên đồng nghĩa có thể chuẩn hóa: `anger → angry`, `happiness → happy`, `sadness → sad`. **`disgust` và `contempt` là hai lớp khác nhau**, không gộp hoặc đổi nhãn để khớp số đầu ra.

FER+ cần ghép ảnh theo đúng thứ tự hàng gốc; không ghép theo thứ tự file sau khi đổi tên. Các cột là số phiếu, không phải label ID. Phải công bố quy tắc xử lý phiếu hòa, `unknown` và `NF`; loại mẫu làm thay đổi số lượng cần ghi vào báo cáo. FER+ dùng lại ảnh FER2013 nên hai nguồn này không phải hai bộ ảnh độc lập.

Nếu dùng ảnh từ benchmark khác để bổ sung Train, chỉ dùng phần train của nguồn đó. Mọi mẫu bổ sung cần đối chiếu trùng chính xác/gần trùng và nguồn người với Val/Test. Không đưa dữ liệu giữ lại để đánh giá vào augmentation hoặc bước chọn siêu tham số.

## Lộ trình thực tế cho mốc 90%

1. Giữ nguyên Test 714 ảnh hiện tại làm đối chiếu lịch sử; sửa quy trình chọn checkpoint và TTA chỉ bằng Val. Test đã được notebook cũ dùng chọn TTA nên cần thu thêm tập test cuối cùng chưa từng tham gia quyết định.
2. Ưu tiên ảnh đúng nguồn sử dụng web app: nhiều người, phiên chụp, camera, góc mặt, ánh sáng; tăng chất lượng nhãn cho các cặp dễ nhầm. Chia theo người/phiên trước khi augment. JAFFE chỉ là bổ sung nhỏ; AffectNet/FER+ cần hoàn tất quyền truy cập và nguồn ảnh.
3. Thử transfer learning, cân bằng Train và augmentation vừa phải; so sánh cấu hình trên Val. Chỉ dùng cấu hình đã chốt để đo test cuối cùng.
4. Báo cáo accuracy, macro-F1, recall từng lớp, confusion matrix và số mẫu thực sự được đánh giá. Không loại dự đoán khó khỏi mẫu số để làm tăng accuracy. Nếu chưa đạt 90%, ghi rõ chưa đạt và lưu kết quả đo thật.

Trên 714 ảnh, cần đúng ít nhất **643 ảnh** để accuracy điểm đạt 90%. Đây là phép tính ngưỡng, không phải kết quả thực nghiệm. Các đề xuất huấn luyện ở trên là hướng triển khai của dự án, không phải cam kết hiệu năng từ tác giả dataset.
