# Phát hiện SMS spam với PCA và SVD

Đồ án Machine Learning nhỏ bằng Python, dành cho sinh viên mới học. Project so sánh **TF-IDF + Logistic Regression** với hai phương án giảm chiều bằng **PCA** và **Truncated SVD**. Dữ liệu là SMS tiếng Anh có nhãn `ham` (bình thường) hoặc `spam` (tin nhắn rác). Chương trình chạy trên CPU; không cần GPU.

## 1. Bắt đầu từ đâu?

| **Bạn muốn xem** | **File cần mở** | **Vai trò** |
|---|---|---|
| **Source code để học và thuyết trình** | **[svd_pca_spam.ipynb](svd_pca_spam.ipynb)** | Giải thích → code → output → biểu đồ → nhận xét; có 16 ô code chạy theo thứ tự. |
| **Giải thích từng phần code** | **[GIAI_THICH_CODE.md](GIAI_THICH_CODE.md)** | Bản đồ 16 ô notebook và các hàm Python được gọi. |
| **Tài liệu dành cho người đọc** | **[README.md](README.md)** | Tổng quan, cấu trúc, cách chạy và cách đọc kết quả. |
| **Output thực tế của mô hình** | **[KET_QUA_MO_HINH.md](KET_QUA_MO_HINH.md)** | Bản xuất notebook có code, bảng số liệu, hình và output đã chạy. |

Nếu chỉ cần xem bài làm, đọc file này rồi mở **KET_QUA_MO_HINH.md**. Nếu cần hiểu cách triển khai, đọc **GIAI_THICH_CODE.md** song song với notebook. Các bảng kết quả làm minh chứng nằm trong [outputs/results/](outputs/results/).

## 2. Bài toán và hướng thực hiện

Input là một tin nhắn tiếng Anh. Output gồm nhãn dự đoán và xác suất spam do mô hình ước lượng.

```text
SMS → TF-IDF (tối đa 1.000 đặc trưng)
          ├─ Logistic Regression
          ├─ PCA → Logistic Regression
          └─ Truncated SVD → Logistic Regression
```

TF-IDF đổi văn bản thành số. PCA/SVD giữ lại `k` tọa độ để khảo sát ảnh hưởng của giảm chiều. Logistic Regression là classifier dùng chung, giúp so sánh ba biểu diễn.

Dữ liệu lấy từ [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection), do Tiago Almeida và José María Gómez Hidalgo cung cấp theo giấy phép **CC BY 4.0**. Dataset không được đóng gói trong repository; khi chạy, người đọc tải từ nguồn UCI như mục 5. Bản kết quả kèm theo đọc **5.574 dòng**, loại **415 bản trùng**, còn **5.159 SMS**: 4.517 ham và 642 spam. Dữ liệu được chia khoảng **70%/15%/15%**, tương ứng **3.611 train / 774 validation / 774 test**. Xem [thống kê dữ liệu](outputs/results/data_summary.json).

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
├── data/raw/                 # Tạo khi tải UCI trên máy, không theo dõi trong Git
└── outputs/
    ├── models/               # Tạo khi train trên máy, không theo dõi trong Git
    ├── results/              # Bảng số liệu, JSON và log minh chứng
    └── figures/              # Biểu đồ PNG kèm tài liệu
```

`src/` và notebook chứa **source code**. CSV/PNG được giữ làm **kết quả thực nghiệm** để người đọc xem tài liệu. Thư mục `data/raw/` và `outputs/models/` được tạo khi chạy trên máy và không nằm trong bản GitHub. Sau khi huấn luyện, mỗi file `.joblib` chứa TF-IDF, reducer nếu có và classifier đã học.

Các file theo từng bản ghi `outputs/results/split_assignments.csv` và `outputs/results/test_predictions.csv` cũng chỉ được sinh trên máy, không đăng kèm GitHub. Repository giữ các bảng tổng hợp, cấu hình và biểu đồ để đọc kết quả.

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

Ô dữ liệu của notebook tự tải dataset từ UCI vào `data/raw/` nếu chưa có file; lần đầu cần Internet. Chạy tiếp các ô huấn luyện sẽ tạo ba model trong `outputs/models/` để ô dự đoán sử dụng.

Ô đầu notebook kiểm tra các thư viện và tự cài từ `requirements.txt` bằng đúng Python của kernel nếu còn thiếu, gồm cả `joblib`. Bước này cần Internet khi phải tải thư viện. Nếu cài đặt thất bại, ô sẽ báo lệnh cài kèm đường dẫn Python đang chạy; sau khi xử lý, chạy lại từ ô đầu. Khi đủ thư viện, bước kiểm tra không cài lại. Hướng dẫn thao tác được giữ trong README; notebook tập trung vào nội dung bài toán, code và kết quả.

## 5. Chạy bằng terminal

**Tải dữ liệu và huấn luyện trước lần dự đoán đầu tiên:**

```powershell
.\.venv\Scripts\python.exe -m src.data --download
.\.venv\Scripts\python.exe -m src.train
```

Lệnh đầu tải dữ liệu chính thức vào `data/raw/`, kèm README nguồn và thông tin gói tải; nếu đã có dataset thì không tải lại. `src.train` đọc file đã tải và tạo `baseline.joblib`, `pca.joblib`, `svd.joblib` trong `outputs/models/`. Cần tải dữ liệu trước khi chạy `src.train` trực tiếp.

**Dự đoán sau khi huấn luyện:**

```powershell
.\.venv\Scripts\python.exe -m src.predict --model outputs/models/svd.joblib --text "Can we meet at the library tomorrow?"
```

Thay nội dung sau `--text` để thử SMS khác. Đổi tên model thành `baseline.joblib` hoặc `pca.joblib` để thử hai pipeline còn lại. Những lần dự đoán tiếp theo chỉ nạp model đã tạo trên máy, không cần huấn luyện lại.

**Chạy toàn bộ notebook và cập nhật file output:**

```powershell
.\.venv\Scripts\python.exe -m scripts.run_notebook
```

Lệnh này chạy notebook từ đầu, tự tải UCI nếu thiếu rồi huấn luyện và tạo các model trên máy.

**Các lệnh riêng khi cần:**

```powershell
# Huấn luyện lại từ dữ liệu đã tải và tạo kết quả trong outputs/.
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

Trên GitHub, chọn **Code → Download ZIP**, giải nén và mở thư mục chứa README trong VS Code. Nếu đã cài Git, có thể chọn **Code**, sao chép URL HTTPS rồi dùng `git clone <URL>` để tải repository. Sau đó tạo môi trường và cài thư viện như mục 4, rồi chạy notebook hoặc tải dữ liệu và huấn luyện như mục 5.

Repository gồm source trong `src/`, các script trong `scripts/`, notebook có output, tài liệu hướng dẫn, bộ kiểm tra, hình và bảng kết quả minh chứng. Dataset UCI và model `.joblib` không được đăng kèm. Nguồn và giấy phép dataset được dẫn ở mục 2; khi tải dữ liệu, script lưu thêm `data/raw/UCI_README.txt` trên máy.

Môi trường `.venv`, cache, `data/raw/`, `outputs/models/`, hai CSV theo từng bản ghi nêu ở mục 3 và thư mục `dist/` được bỏ qua theo `.gitignore`; người đọc tạo lại môi trường, tải dữ liệu và huấn luyện trên máy mình. Gói `dist/SVD_PCA_Spam.zip` là bản chia sẻ cục bộ, không nằm trong repository GitHub.
