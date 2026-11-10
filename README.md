# 🚦 Red-Light Violation Detection with YOLO & OpenCV

Hệ thống tự động phát hiện lỗi vượt đèn đỏ từ video giao thông sử dụng mô hình **YOLOv8** kết hợp **ByteTrack** để theo dõi phương tiện, cùng với **OpenCV** để xử lý hình ảnh, nhận diện trạng thái đèn giao thông theo không gian màu HSV và kiểm tra sự kiện cắt vạch dừng (Stop Line).

---

## 📋 Tính năng chính
* **Tự động nhận diện phương tiện**: Theo dõi ô tô, xe máy, xe đạp, xe buýt và xe tải thông qua YOLOv8.
* **Tương tác trực quan**: Chọn vạch dừng (`Stop Line`) và vùng đèn giao thông (`Traffic Light ROI`) trực tiếp trên khung hình đầu tiên.
* **Phát hiện vượt đèn đỏ**: Ghi lại sự kiện phương tiện cắt qua vạch khi đèn đang chuyển sang màu ĐỎ.
* **Xuất báo cáo & Dữ liệu**: Lưu trữ video kết quả kèm theo các file thống kê sự kiện định dạng CSV (`_events.csv`).
* **Hỗ trợ co giãn màn hình**: Tự động điều chỉnh kích thước hiển thị cửa sổ OpenCV phù hợp với mọi độ phân giải màn hình mà không làm thay đổi tọa độ gốc hay chất lượng video xuất ra.

---

## 🛠️ Yêu cầu hệ thống & Cài đặt

1. **Python**: Khuyên dùng phiên bản Python 3.10 trở lên.
2. **Cài đặt thư viện**:
   Mở terminal (hoặc cửa sổ lệnh trong IDE của bạn) và chạy lệnh sau:
   ```bash
   pip install -r requirements.txt
## Cấu trúc thư mục
IoT_RedLightViolation/
│
├── main.py                  # File mã nguồn chính của chương trình
├── yolov8n.pt               # Model YOLOv8 (hoặc yolov8s.pt)
├── 1108.mov                 # Video đầu vào (được cấu hình trong code hoặc truyền vào tham số)
├── .gitignore               # Cấu hình bỏ qua các file không cần thiết lên Git
└── output/                  # Thư mục lưu kết quả (video xử lý và file CSV)
🚀 Hướng dẫn chạy chương trình
1. Chạy với tham số mặc định
   ```bash
   python main.py
2. Chạy với tham số tùy chỉnh
   ```bash
   python main.py --video path/to/video.mov --model yolov8s.pt --output output/result.mp4 --conf 0.3 --show
🖱️ Hướng dẫn thao tác lần đầu tiên chạy
Khi khởi chạy chương trình, người dùng sẽ thực hiện 2 bước cấu hình tương tác trên khung hình đầu tiên:
1. Chọn vạch dừng (Stop Line):Cửa sổ "Select Stop Line" bật lên. Click chuột trái 2 điểm để định nghĩa vạch dừng (Điểm P1 và P2).Nhấn Enter để xác nhận (hoặc nhấn R để chọn lại từ đầu).
2. Chọn vùng đèn giao thông (Traffic Light ROI):Cửa sổ "Select Traffic Light ROI" bật lên. Dùng chuột kéo khoanh vùng khu vực có đèn giao thông, sau đó nhấn Enter hoặc Space để hoàn tất.Tọa độ vạch dừng và ROI sẽ được lưu tự động để sử dụng cho các lần chạy sau.
