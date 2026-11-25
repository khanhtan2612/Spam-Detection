# Phát hiện SMS spam bằng SVD và PCA

Đồ án môn Toán cho Khoa học Máy tính. Project so sánh **TF-IDF + Logistic Regression** với hai cách giảm chiều trước khi phân loại: **SVD** và **PCA**. Mọi thuật toán (TF-IDF, SVD, PCA, Logistic Regression, các độ đo) đều được **tự cài bằng NumPy**, không dùng thư viện học máy. pandas chỉ dùng để đọc và ghi bảng, matplotlib chỉ dùng để vẽ.

## 1. Bắt đầu từ đâu?

| Muốn xem | Mở file |
|---|---|
| Cơ sở toán học: định nghĩa, định lý, chứng minh, công thức (1)–(39) và bảng tra công thức → code | [script.md](script.md) |
| Bài làm chính: công thức → code → kết quả → nhận xét | [svd_pca_spam.ipynb](svd_pca_spam.ipynb) |
| Bản đã chạy, đọc được không cần Jupyter | [KET_QUA_MO_HINH.md](KET_QUA_MO_HINH.md) |
| Mỗi file và mỗi hàm làm gì | [GIAI_THICH_CODE.md](GIAI_THICH_CODE.md) |
| Kịch bản thuyết trình 8–10 phút | [outputs/KICH_BAN_THUYET_TRINH_SVD_PCA.md](outputs/KICH_BAN_THUYET_TRINH_SVD_PCA.md) |

Thứ tự đọc gợi ý: đọc [script.md](script.md) để nắm phần toán, rồi đọc notebook (hoặc KET_QUA_MO_HINH.md) để xem từng công thức được cài và kiểm chứng ra sao. Khi cần sửa code, dùng bảng tra ở cuối script.md cùng với GIAI_THICH_CODE.md.

## 2. Bài toán và mạch trình bày

Input là một tin nhắn SMS tiếng Anh. Output là nhãn `ham` (tin thường) hoặc `spam` (tin rác), kèm xác suất P(spam).

```text
SMS → TF-IDF (1000 chiều) ─┬──────────────────→ Logistic Regression   (baseline)
                           ├─ SVD (k chiều) ──→ Logistic Regression
                           └─ PCA (k chiều) ──→ Logistic Regression
```

Mọi bước học đều đi theo cùng một mạch **mô hình → loss → tối ưu → score**:

| | Mô hình | Loss | Tối ưu | Score |
|---|---|---|---|---|
| SVD / PCA | $X \approx X_k = Z V_k^T$ | $\Vert X - X_k \Vert_F^2 = \sum_{j \gt k} \sigma_j^2$ | orthogonal iteration (nhân ma trận + QR) | tỷ lệ năng lượng (SVD) / phương sai (PCA) giữ lại |
| Logistic Regression | $p = \sigma(Xw + b)$ | cross-entropy + $\frac{\lambda}{2N}\Vert w \Vert^2$ | gradient descent | accuracy, precision, recall, F1, macro-F1 |

Phần Logistic Regression đi đủ 5 bước, mỗi bước là một hoặc hai hàm trong [src/logistic.py](src/logistic.py): mô hình (`sigmoid`, `predict_proba`) → loss (`loss`) → gradient (`gradient`, có gradient check) → tối ưu bằng gradient descent (`fit`) → dự đoán và score (`predict`, `score`).

## 3. Dữ liệu

Dữ liệu lấy từ [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) của Tiago Almeida và José María Gómez Hidalgo, giấy phép **CC BY 4.0**. Repository không chứa dataset.

- Đọc **5.574** dòng, bỏ **415** tin trùng nội dung, còn **5.159** tin: 4.517 ham và 642 spam (12,4%).
- `stratified_split(df, (0.7, 0.15, 0.15), seed=42)` chia theo từng lớp thành **3.611 train / 774 validation / 774 test**. Tỷ lệ spam của ba tập lần lượt là 12,4% / 12,4% / 12,5%.
- Train dùng để học vocabulary, idf, $V_k$, $\mu$, $w$, $b$. Validation dùng để chọn k ∈ {10, 50, 100}. Test chỉ được dùng **một lần** sau khi đã khóa k và fit lại trên train + validation.

## 4. Kết quả

**Chọn k trên validation** (fit trên 3.611 tin train):

| Pipeline | k | Score SVD/PCA (giữ lại) | Recall spam | Macro-F1 |
|---|---:|---:|---:|---:|
| Baseline | — | — | 0,7708 | 0,9215 |
| SVD | 10 / 50 / **100** | 0,106 / 0,289 / 0,431 | 0,552 / 0,656 / 0,729 | 0,8349 / 0,8759 / **0,9065** |
| PCA | 10 / 50 / **100** | 0,082 / 0,269 / 0,415 | 0,573 / 0,656 / 0,719 | 0,8441 / 0,8759 / **0,9027** |

Cả SVD và PCA đều chọn **k = 100**.

**Đánh giá test một lần** (fit lại trên 4.385 tin train + validation, test 774 tin, trong đó 97 spam):

| Pipeline | Số chiều | Accuracy | Precision spam | Recall spam | F1 spam | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1.000 | 0,9780 | 0,9762 | 0,8454 | 0,9061 | **0,9468** |
| SVD | 100 | 0,9703 | 0,9744 | 0,7835 | 0,8686 | 0,9259 |
| PCA | 100 | 0,9703 | 0,9744 | 0,7835 | 0,8686 | 0,9259 |

Số chính xác nằm trong [validation_metrics.csv](outputs/results/validation_metrics.csv) và [test_metrics.csv](outputs/results/test_metrics.csv). Các hình đi kèm: [loss theo epoch](outputs/figures/loss_curve.png), [confusion matrix](outputs/figures/test_confusion.png), [chọn k](outputs/figures/validation_f1.png), [score SVD/PCA theo k](outputs/figures/retention.png).

**Đọc kết quả**

- Giảm từ 1.000 xuống 100 chiều làm macro-F1 test giảm 0,021. Ba mô hình đều chỉ báo nhầm 2/677 tin ham, nhưng SVD/PCA bỏ sót 21/97 tin spam, so với 15/97 của baseline.
- Khi fit trên 3.611 tin train, 100 chiều chỉ giữ 43,1% năng lượng (SVD) và 41,5% phương sai (PCA); khi fit lại trên train + validation, các con số này là 42,6% và 41,0%. Giả thuyết của em: SVD/PCA tìm hướng có loss tái tạo nhỏ nhất mà không nhìn nhãn, nên phần bị bỏ đi có thể vẫn chứa tín hiệu phân biệt spam. Đồ án chưa có thí nghiệm riêng để kiểm chứng điều này.
- Gradient descent hội tụ ở cả ba pipeline: loss giảm sau mọi epoch, và vòng lặp dừng vì $|\Delta J| \lt 10^{-10}$ sau 7.273–9.765 epoch, chưa chạm giới hạn 20.000.

**So với bản trước.** Baseline bản trước đạt macro-F1 test 0,9310, bản này đạt 0,9468. Chênh lệch đến từ cách chia tập mới (`stratified_split` tự viết cho ra một tập test khác), không phải từ thuật toán. Khi chạy code mới (cùng hàm `fit_and_score`) trên đúng cách chia của bản trước, baseline cho macro-F1 test **0,9310**, trùng với bản trước. Đây là kiểm tra làm riêng một lần, không nằm trong notebook; file chia tập của bản trước (`outputs/results/split_assignments.csv`) chỉ có trên máy, không đưa lên Git. Cùng lần kiểm tra đó, SVD và PCA với k = 100 cho 0,8802, nhỉnh hơn bản trước (0,872 và 0,876), có thể vì bản này lặp đến khi phân rã hội tụ thay vì dùng phép xấp xỉ ngẫu nhiên.

## 5. Cấu trúc thư mục

```text
.
├── README.md                  # Tài liệu này
├── script.md                  # Cơ sở toán học: định lý, chứng minh, bảng tra công thức → code
├── GIAI_THICH_CODE.md          # Giải thích từng file, từng hàm
├── KET_QUA_MO_HINH.md          # Notebook đã chạy, xuất ra Markdown
├── svd_pca_spam.ipynb          # Bài làm chính
├── requirements.txt           # numpy, pandas, matplotlib + công cụ chạy notebook
├── slide/                     # Sơ đồ pipeline (PNG) dùng khi thuyết trình
├── src/
│   ├── data.py                # Đọc, làm sạch, stratified_split
│   ├── tfidf.py               # TF-IDF tự cài
│   ├── decomposition.py       # Orthogonal iteration, SVD, PCA (loss, score)
│   ├── logistic.py            # Logistic Regression: mô hình → loss → gradient → GD → score
│   ├── metrics.py             # Confusion matrix, precision, recall, F1, macro-F1
│   ├── pipeline.py            # Ghép TF-IDF → SVD/PCA → Logistic Regression; lưu/đọc model
│   ├── train.py               # Chọn k → refit → test
│   ├── plots.py               # Vẽ biểu đồ
│   └── predict.py             # Dự đoán một SMS bằng model đã lưu
├── scripts/                   # Chạy notebook và xuất Markdown
├── tests/                     # Kiểm thử (unittest)
├── data/raw/                  # Dataset UCI (không đưa lên Git)
└── outputs/
    ├── results/               # CSV/JSON kết quả
    ├── figures/               # Biểu đồ PNG
    ├── models/                # Model .npz + .json (không đưa lên Git)
    └── KICH_BAN_THUYET_TRINH_SVD_PCA.md
```

## 6. Kiểm chứng

Thư mục `tests/` kiểm tra: SVD/PCA tự viết khớp `np.linalg.svd`/`np.linalg.eigh` (σ và projector của không gian con), gradient check, ví dụ confusion matrix/F1 tính tay, ba tập không trùng nhau và giữ tỷ lệ lớp, TF-IDF không học từ validation. `np.linalg.svd`/`eigh` chỉ dùng trong test để đối chiếu, không dùng trong pipeline.

## 7. Giới hạn

Kết quả dựa trên một lần chia dữ liệu, ba giá trị k và vocabulary 1.000 từ của SMS tiếng Anh. Chênh lệch giữa các mô hình chưa được kiểm định thống kê. Xác suất spam là ước lượng của mô hình, chưa được hiệu chỉnh.
