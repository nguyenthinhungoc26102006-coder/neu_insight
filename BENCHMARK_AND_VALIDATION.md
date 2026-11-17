# Kiểm chứng thực thi (môi trường kiểm thử)

Đã kiểm thử pipeline Python trên bộ dữ liệu BTC được cung cấp **trong môi trường làm việc riêng**, không sao chép điểm BTC vào gói mã nguồn này. Các con số đo thực tế phụ thuộc CPU/RAM và có thể thay đổi giữa các lần chạy.

## Kiểm tra chức năng

- Chạy `python -m pytest tests -q`: **4 kiểm thử đạt**.
- Kiểm thử REST API: POST upload cả bảng điểm và thời khóa biểu, xem `/api/summary`, `/api/anomalies`, `/api/sections/{id}`, `/api/students/{id}`, tải `/api/download/*`: tất cả trả HTTP 200 khi hoàn tất.
- Bộ giả lập: nhận diện lớp được tiêm lệch cao (`HIGH_INFLATION`), lệch thấp (`UNUSUALLY_STEEP`), và nén phân phối (`SUSPICIOUS_DISTRIBUTION`). Lớp nén phân phối không bị tự động dịch điểm.
- Trường hợp L0186 của dữ liệu BTC có thiếu điểm quá trình: `DATA_QUALITY_REVIEW`, không tự động hiệu chỉnh.

## Kiểm tra hiệu năng

- Dữ liệu BTC thực: 38.342 bản ghi, 877 lớp, thời gian pipeline khoảng **11–12 giây**.
- Dữ liệu **giả lập** tăng quy mô bằng 18 bản sao độc lập có mã lớp/sinh viên riêng (để tránh đếm trùng), có nhiễu nhỏ: **103.680 bản ghi, 3.456 lớp, 4.320 mã sinh viên**; pipeline chạy trong **18,15 giây** ở môi trường kiểm thử. Thời gian đo bao gồm làm sạch, tạo đặc trưng, huấn luyện theo năm, phát hiện, hiệu chỉnh và TCI; không bao gồm thời gian upload tệp qua mạng.
- Test này chỉ cho thấy hệ thống xử lý được trên 100.000 bản ghi mô phỏng, **chưa chứng minh** hiệu suất production hoặc tính đúng đắn trên dữ liệu ngoài phân phối đã kiểm thử.

## Hạn chế phải trình bày với BTC

- Chưa có nhãn lạm phát điểm/điều chỉnh được giảng viên thẩm định, nên không công bố Precision/F1 trên dữ liệu thật.
- Z và q-value dùng xấp xỉ cần hiệu chỉnh giả thuyết; chưa được chứng minh là phép kiểm định chính xác cho mọi nhóm môn.
- Trọng số Anomaly Score, giới hạn dịch 1,5, bảng quy đổi GPA 4 và ngưỡng PROVISIONAL là **cấu hình nghiên cứu**, không phải quy chế chính thức.
- TCI chỉ tính trên điểm thuộc các học phần quan sát, không đại diện GPA toàn trường.
- Mặc định ứng dụng chỉ phục vụ demo cục bộ, không có phân quyền/rà soát con người để triển khai học vụ thật.
