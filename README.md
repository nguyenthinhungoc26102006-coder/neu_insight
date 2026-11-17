# NEU Grade Intelligence · FIT Innovation 2026

**Web chạy thật (FastAPI + giao diện SPA + Plotly offline)** cho ba giải pháp:

1. **Grade Anomaly Detection** — nhóm tham chiếu phân tầng, thống kê phân phối, dự báo điểm ngoài mẫu (Ridge và CatBoost), Z-score điều chỉnh có xét độ lệch giữa các lớp, Beta–Binomial, Spearman, điểm KTTX–Thi, Anomaly Score 0–1.
2. **Grade Remediation** — ước lượng dịch điểm bằng Empirical Bayes và độ bất định, tự từ chối hiệu chỉnh khi bằng chứng yếu, khoảng dự báo 95% kiểu conformal cho từng sinh viên.
3. **True Competency Index** — GPA đã hiệu chỉnh theo trọng số tín chỉ và bảng quy đổi thang 4 **giả định**, bốn nhóm đề bài, cờ `PROVISIONAL`, phân rã đóng góp vào ΔTCI từng môn.

## 1. Cài đặt trên Windows (PowerShell)

Cần cài Python 3.11+ (khuyến nghị 3.12) và bật Add Python to PATH.

```powershell
cd NEU_FIT_Web
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Nếu PowerShell chặn kích hoạt venv, **không cần chỉnh chính sách bảo mật**. Chạy trực tiếp:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Mở trình duyệt tại **http://127.0.0.1:8000**. Bấm **Chạy dữ liệu giả lập** để xem ngay ba giải pháp.

Nếu muốn dùng file BTC: vào **Quản lý dữ liệu** → chọn `Du_lieu.csv`, có thể chọn thêm `Thoi_khoa_bieu.csv` → **Bắt đầu phân tích 3 giải pháp**. Dữ liệu gốc BTC *không* nằm trong gói mã nguồn. File `.xlsx` chỉ là phiên bản cùng dữ liệu được BTC cung cấp; web nhập từ CSV để tránh nhập trùng.

## 2. Chạy bằng Docker

```bash
docker compose up --build
```

Mở `http://127.0.0.1:8000`. Docker chỉ lắng nghe loopback trên máy người dùng.

## 3. Giải thích thuật toán thực tế trong code

### 3.1. Dự báo ngoài mẫu

`app/engine.py`: tạo `prior_mean`, `prior_count`, `prior_group_mean` từ **những học kỳ trước**, không dùng điểm cùng kỳ hoặc tương lai.

`app/modeling.py`: mỗi năm học Y chỉ huấn luyện trên các năm `<Y`. Huấn luyện Ridge và CatBoost độc lập; quyết định mô hình dự báo cho năm Y dùng MAE ở các năm đã kiểm thử trước Y, không nhìn điểm của năm Y. Năm đầu tiên thiếu dữ liệu huấn luyện được đánh dấu `NOT_AVAILABLE` (không tự tạo dự báo).

### 3.2. Phát hiện lớp bất thường

`app/detection.py`: so sánh theo ưu tiên (1) cùng môn/kỳ/EP/đồ án, (2) cùng môn gần năm, (3) cùng nhóm môn/cùng năm, (4) cùng nhóm môn. **Loại chính lớp đang xét** khỏi mốc. Phần dư sau dự báo: `r = score - expected`. Độ lệch lớp `delta = mean(r) - median(mean_r_peers)`, hoặc chỉ so điểm thô khi thiếu dự báo (khi đó không tự hiệu chỉnh).

Z-score điều chỉnh: `z = delta / sqrt(tau² + v)`, với `tau` = 1.4826×MAD trung bình phần dư giữa lớp tham chiếu (sàn 0.4), `v = sd_residual²/n_predicted`. BH-FDR tính q-value tham khảo dưới giả định xấp xỉ; chưa phải kiểm định đã được hiệu chuẩn trên dữ liệu gán nhãn.

Tỷ lệ A và điểm không đạt được kiểm tra bằng **phân phối dự báo Beta–Binomial**, không chỉ dùng ngưỡng A>60%. Spearman đo tương quan điểm với nền tảng, KTTX–Thi tính trên các bản ghi có đủ cả hai thành phần. Gộp 4 tín hiệu location/tail/distribution/assessment bằng trọng số 0.40/0.25/0.20/0.15, tự chuẩn hóa lại khi thiếu tín hiệu. Đây là trọng số **khởi tạo**, cần thử nghiệm và đánh giá.

`INSUFFICIENT_EVIDENCE` và `DATA_QUALITY_REVIEW` có thể có score cao nhưng **không được gán nhãn chấm sai**. Dữ liệu thiếu điểm quá trình khiến kết quả không nhất quán với điểm thi cũng được rà soát trước khi hiệu chỉnh.

### 3.3. Hiệu chỉnh điểm

`app/remediation.py`: EB prior `delta_class ~ N(0, tau²)` và observed mean residual `Delta ~ N(delta_class, v)`. Shrinkage `lambda = tau²/(tau²+v)`, shift `lambda*Delta`. Chỉ đề xuất `clip(score - shift,0,10)` nếu lớp có hiệu ứng lệch có hướng, phần dư ngoài mẫu, đủ đối chứng, hiệu ứng đủ lớn so với độ bất định. Giới hạn dịch tối đa 1.5 điểm là **rào bảo vệ thử nghiệm**, không phải quy định học vụ.

Giữ nguyên mọi điểm 0 vì chúng có thể là vắng thi/bỏ học. Thiếu điểm không được điền 0. `individual_review` sử dụng khoảng dự báo thực nghiệm 95% từ phần dư dự báo ngoài mẫu, với nhóm môn đủ lớn thì hiệu chỉnh riêng theo nhóm. Khoảng bao phủ 95% là mục tiêu xấp xỉ, không bảo đảm tuyệt đối trên từng sinh viên hoặc mọi nhóm.

### 3.4. TCI

`app/config.py`: bảng quy đổi thang 4 là **quy ước minh họa**. Để dùng trong trường, phải đổi sang quy chế GPA được BTC/nhà trường xác nhận.

`app/tci.py`: TCI là trung bình `sum(credits*GPA_adj)/sum(credits)` trên đúng những học phần có điểm trong dữ liệu. Cùng học phần tính `GPA_before` để so sánh. Có TCI không tính đồ án để xem độ nhạy, nhưng TCI chính vẫn tính đầy đủ theo tín chỉ. `PROVISIONAL` nếu có dưới 4 học phần; ngưỡng 4 cũng là tham số demo.

**Không gọi TCI là năng lực tuyệt đối toàn trường**: dữ liệu do một khoa phụ trách và không có bảng điểm từ tất cả các khoa, không có điểm đầu vào hoặc đề thi chuẩn hóa.

## 4. API

- `GET /api/health`, `GET /api/job` — kiểm tra tình trạng.
- `POST /api/demo` — chạy dữ liệu giả lập.
- `POST /api/analyze` — multipart: `grades` bắt buộc, `schedule` tùy chọn.
- `GET /api/summary` — thống kê và mô hình.
- `GET /api/anomalies?limit=100&label=HIGH_INFLATION` — danh sách lớp.
- `GET /api/sections/{section_id}` — chỉ số và điểm một lớp.
- `GET /api/students?query=S00001` — danh sách năng lực.
- `GET /api/students/{student_id}` — TCI và giải thích cá nhân.
- `GET /api/transitions` — ma trận chuyển nhóm.
- `GET /api/schedule` — phân tích lịch độc lập.
- `GET /api/download/anomalies`, `/adjustments`, `/tci`, `/summary` — xuất kết quả tại máy cục bộ.

Interactive API docs: `http://127.0.0.1:8000/docs`.

## 5. Kiểm thử

```bash
python -m pytest tests -q
```

Bộ dữ liệu giả lập và nhãn mô phỏng tương ứng: `sample/demo_synthetic.csv`, `sample/demo_truth.json`.

Khi chạy với dữ liệu BTC thật, không có nhãn lạm phát điểm được thẩm định, do đó **không thể tuyên bố Precision/Recall/F1 trên dữ liệu BTC**. Để đánh giá công bằng, cần tiêm bất thường có kiểm soát và báo cáo Precision, Recall, MAE hiệu chỉnh, False Positive theo nhóm.

## 6. An toàn dữ liệu và giới hạn

- Tài liệu BTC: **không đưa dữ liệu ra dịch vụ ngoài cuộc thi**, không xác định danh tính sinh viên/giảng viên. Không commit, tải GitHub, host public hoặc gửi sang API AI đối với dữ liệu BTC.
- Ứng dụng này thiết kế **cho một máy local**, không có đăng nhập/RBAC; `app.main` giữ lượt phân tích hiện tại trong RAM và lưu kết quả ở `output/`. Mặc định chỉ truy cập `127.0.0.1`. Không dùng phiên bản này làm hệ thống học vụ production.
- Thời khóa biểu có ID học phần/lớp khác hệ với bảng điểm. Không join hai file theo tên/mã đã mã hóa; chỉ thống kê độc lập.
- Bốn tín hiệu và ngưỡng gắn nhãn còn phải hiệu chuẩn với các lớp được chuyên gia kiểm định. Độ lệch thống kê có thể do năng lực thật, độ khó, biến đổi đánh giá hoặc dữ liệu thiếu.
- Anomaly Score không phải xác suất có gian lận. EB-proposed score chỉ là gợi ý; không ghi đè điểm chính thức.
- Thiếu năm học đầu, số mẫu ít hoặc biến đặc trưng thiếu có thể làm mô hình dự báo không ổn định. Mục `Algorithm Diagnostics` thể hiện MAE trên dữ liệu ngoài mẫu theo năm.

## 7. Cấu trúc

```text
app/
  engine.py        Import/clean/history
  modeling.py      Ridge/CatBoost outside-time
  detection.py     Peer stats, Beta–Binomial, Spearman, FDR, multi-signal
  remediation.py   Empirical Bayes, conformal residual bands
  tci.py           Grade mapping and TCI decomposition
  pipeline.py      Orchestrator
  main.py          FastAPI and CSV exports
web/               Offline browser UI + bundled plotly.min.js
sample/            ONLY synthetic example + generator + truth
output/            Local reports; excluded from zip and git
```

### Chạy nhanh bằng một cú nhấp đúp

Trên Windows, mở thư mục và chạy **`START_WINDOWS.bat`** (máy đã cài Python 3.11+ và có `py` launcher). Tệp tự tạo môi trường, cài thư viện và mở web. Với môi trường không có quyền cài đặt, dùng các lệnh PowerShell phía trên.
