# Phát hiện SMS spam với PCA và SVD

Đồ án Machine Learning nhỏ bằng Python, dành cho sinh viên mới học. Project so sánh **TF-IDF + Logistic Regression** với hai phương án giảm chiều bằng **PCA** và **Truncated SVD**. Dữ liệu là SMS tiếng Anh có nhãn `ham` (bình thường) hoặc `spam` (tin nhắn rác). Chương trình chạy trên CPU; không cần GPU.

## 1. Bắt đầu từ đâu?

| **Bạn muốn xem** | **File cần mở** | **Vai trò** |
|---|---|---|
| **Source code để học và thuyết trình** | **[svd_pca_spam.ipynb](svd_pca_spam.ipynb)** | Giải thích → code → output → biểu đồ → nhận xét; có 16 ô code chạy theo thứ tự. |
| **Giải thích từng phần code** | **[GIAI_THICH_CODE.md](GIAI_THICH_CODE.md)** | Bản đồ 16 ô notebook và các hàm Python được gọi. |
| **Tài liệu dành cho người đọc** | **[README.md](README.md)** | Tổng quan, cấu trúc, cách chạy và cách đọc kết quả. |
| **Output thực tế của mô hình** | **[KET_QUA_MO_HINH.md](KET_QUA_MO_HINH.md)** | Bản xuất notebook có code, bảng số liệu, hình và output đã chạy. |

Nếu chỉ cần xem bài làm, đọc file này rồi mở **KET_QUA_MO_HINH.md**. Nếu cần hiểu cách triển khai, đọc **GIAI_THICH_CODE.md** song song với notebook. Dữ liệu chi tiết làm minh chứng nằm trong [outputs/results/](outputs/results/).

## 2. Bài toán và hướng thực hiện

Input là một tin nhắn tiếng Anh. Output gồm nhãn dự đoán và xác suất spam do mô hình ước lượng.

```text
SMS → TF-IDF (tối đa 1.000 đặc trưng)
          ├─ Logistic Regression
          ├─ PCA → Logistic Regression
          └─ Truncated SVD → Logistic Regression
```

TF-IDF đổi văn bản thành số. PCA/SVD giữ lại `k` tọa độ để khảo sát ảnh hưởng của giảm chiều. Logistic Regression là classifier dùng chung, giúp so sánh ba biểu diễn.

Dữ liệu lấy từ [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection). Bản chạy kèm theo đọc **5.574 dòng**, loại **415 bản trùng**, còn **5.159 SMS**: 4.517 ham và 642 spam. Dữ liệu được chia khoảng **70%/15%/15%**, tương ứng **3.611 train / 774 validation / 774 test**. Xem [thống kê dữ liệu](outputs/results/data_summary.json).

Chỉ train được dùng để học vocabulary, IDF và hướng chiếu. Thử `k = 10, 50, 100`, chọn bằng macro-F1 validation, rồi fit lại trên train + validation và đánh giá test. Test không dùng để chọn `k`.

## 3. Cấu trúc thư mục

```text
.
├── README.md                 # Hướng dẫn người đọc
├── GIAI_THICH_CODE.md         # Giải thích các phần code
├── KET_QUA_MO_HINH.md         # Notebook xuất ra Markdown, kèm output
├── svd_pca_spam.ipynb         # Source dạng notebook
├── requirements.txt          # Thư viện đã dùng
├── src/                      # Source Python: dữ liệu, toán, train, predict, vẽ hình
├── scripts/                  # Chạy notebook và xuất Markdown
├── tests/                    # Kiểm tra toán, pipeline và xuất notebook
├── data/raw/                 # Dataset UCI và thông tin nguồn
└── outputs/
    ├── models/               # Ba pipeline đã huấn luyện (.joblib)
    ├── results/              # CSV, JSON và log kết quả thật
    └── figures/              # Biểu đồ PNG
```

`src/` và notebook chứa **source code**. File `.joblib` là **mô hình đã học**, gồm cả TF-IDF, reducer nếu có và classifier. CSV/PNG là **kết quả thực nghiệm**, không phải source code.

## 4. Cài đặt trong VS Code

Mở thư mục project bằng **File → Open Folder**, sau đó **Terminal → New Terminal**. Các lệnh sau chạy tại thư mục có `README.md`, `src/` và notebook, không nhập vào file `.py`.

Bản kết quả kèm theo dùng Python **3.14.7**; phiên bản thư viện được ghim trong `requirements.txt` và ghi tại [run_config.json](outputs/results/run_config.json).

Nếu chưa có `.venv`:

```powershell
python -m venv .venv
```

Cài thư viện bằng đúng Python của môi trường này:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Trong VS Code, cài extension **Python** và **Jupyter** của Microsoft. Mở notebook → **Select Kernel** → chọn Python trong `.venv` → **Run All**. Output xuất hiện dưới mỗi ô code. Chạy từ đầu theo thứ tự để các biến được tạo đầy đủ.

Ô đầu notebook kiểm tra các thư viện và tự cài từ `requirements.txt` bằng đúng Python của kernel nếu còn thiếu, gồm cả `joblib`. Bước này cần Internet khi phải tải thư viện. Nếu cài đặt thất bại, ô sẽ báo lệnh cài kèm đường dẫn Python đang chạy; sau khi xử lý, chạy lại từ ô đầu. Khi đủ thư viện, bước kiểm tra không cài lại. Hướng dẫn thao tác được giữ trong README; notebook tập trung vào nội dung bài toán, code và kết quả.

## 5. Chạy bằng terminal

**Dự đoán ngay với mô hình có sẵn:**

```powershell
.\.venv\Scripts\python.exe -m src.predict --model outputs/models/svd.joblib --text "Can we meet at the library tomorrow?"
```

Thay nội dung sau `--text` để thử SMS khác. Đổi tên model thành `baseline.joblib` hoặc `pca.joblib` để thử hai pipeline còn lại. Lệnh dự đoán chỉ tải model, không huấn luyện lại.

**Chạy toàn bộ notebook và cập nhật file output:**

```powershell
.\.venv\Scripts\python.exe -m scripts.run_notebook
```

**Các lệnh riêng khi cần:**

```powershell
# Huấn luyện và tạo kết quả trong outputs/.
.\.venv\Scripts\python.exe -m src.train

# Kiểm chứng toán trên ma trận nhỏ.
.\.venv\Scripts\python.exe -m src.math_demo

# Chạy bộ kiểm tra.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Chỉ xuất notebook đã lưu; không chạy lại model.
.\.venv\Scripts\python.exe -m scripts.ipynb_to_markdown svd_pca_spam.ipynb --output KET_QUA_MO_HINH.md
```

Sau khi train riêng, chạy notebook để cập nhật cả nội dung và output của báo cáo. Chạy lại sẽ ghi đè các artifact tương ứng.

## 6. Đọc kết quả

Snapshot test kèm project:

| Pipeline | Số chiều | Accuracy | Macro-F1 | Spam recall |
|---|---:|---:|---:|---:|
| Baseline | 1.000 | 97,29% | 0,9310 | 78,13% |
| PCA | 100 | 95,48% | 0,8760 | 63,54% |
| Truncated SVD | 100 | 95,35% | 0,8717 | 62,50% |

Xem số chính xác trong [test_metrics.csv](outputs/results/test_metrics.csv), các ứng viên trong [validation_metrics.csv](outputs/results/validation_metrics.csv) và [confusion matrix](outputs/figures/test_confusion.png).

PCA/SVD giảm **90% số chiều** nhưng macro-F1 giảm và bỏ sót nhiều spam hơn baseline. Biểu diễn giảm chiều là dense nên có thể tốn dung lượng hơn sparse TF-IDF. Đây là kết quả cần phân tích khi bảo vệ: giữ cấu trúc ma trận chưa chắc giữ được các dấu hiệu phân loại.

PCA variance và SVD energy có định nghĩa khác nhau. Accuracy cũng chưa đủ vì ham chiếm đa số. Kết quả chỉ dựa trên một split của SMS tiếng Anh; demo không xác nhận khả năng lọc spam tiếng Việt hoặc spam mới. Thời gian chạy phụ thuộc máy.

## 7. Mang project sang máy khác hoặc đưa lên GitHub

Trên GitHub, chọn **Code → Download ZIP**, giải nén và mở thư mục chứa README trong VS Code. Nếu đã cài Git, có thể chọn **Code**, sao chép URL HTTPS rồi dùng `git clone <URL>` để tải repository. Sau đó tạo môi trường và cài thư viện như mục 4. Model đã lưu nên nạp bằng các phiên bản thư viện đã ghi.

Repository gồm source trong `src/`, các script trong `scripts/`, notebook có output, tài liệu hướng dẫn, bộ kiểm tra, dữ liệu UCI và các artifact minh chứng. File `data/raw/UCI_README.txt` giữ thông tin nguồn và điều kiện sử dụng dataset.

Môi trường `.venv`, cache và thư mục `dist/` được bỏ qua theo `.gitignore`; người đọc tạo lại môi trường trên máy mình. Gói `dist/SVD_PCA_Spam.zip` là bản chia sẻ cục bộ, không nằm trong repository GitHub.
