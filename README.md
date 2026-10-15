# Ứng dụng SVD và PCA trong phát hiện tin nhắn rác (Spam Detection)

Đồ án môn Toán cho Khoa học Máy tính. Đồ án cài đặt từ đầu Singular Value Decomposition (SVD) và Principal Component Analysis (PCA) bằng NumPy (SVD qua trị riêng của $A^\mathsf{T}A$, randomized SVD, PCA tâm hóa ngầm trên ma trận thưa), kiểm chứng với scikit-learn, và dùng chúng để giảm chiều biểu diễn TF-IDF của tin nhắn SMS trước một bộ phân loại logistic regression. Ba kết quả lý thuyết (tồn tại SVD, PCA là SVD của dữ liệu tâm hóa, Eckart–Young–Mirsky) được chứng minh trong `slide/script.md` và kiểm tra bằng số trong notebook.

## Cấu trúc thư mục

```
Spam-Detection/
├── README.md                # tài liệu này
├── requirements.txt         # numpy, scipy, pandas, matplotlib, scikit-learn>=1.4, jupyter
├── spam_svd_pca.ipynb       # notebook chính, đã chạy sẵn và lưu output
├── spam_svd_pca.md          # bản xuất Markdown của notebook (nbconvert)
├── spam_svd_pca_files/      # 5 hình do nbconvert xuất ra
├── data/SMSSpamCollection   # bộ dữ liệu UCI (5.574 dòng)
└── slide/script.md          # cơ sở toán học: định nghĩa, định lý, chứng minh, công thức (1)–(30)
```

## Cài đặt và chạy

```powershell
git clone https://github.com/khanhtan2612/Spam-Detection.git
cd Spam-Detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Mở `spam_svd_pca.ipynb` trong VS Code (extension Python và Jupyter), chọn kernel `.venv` bằng nút **Select Kernel**, rồi bấm **Run All**. Notebook tự tải dữ liệu nếu `data/SMSSpamCollection` chưa có. Chạy từ dòng lệnh:

```powershell
jupyter nbconvert --to notebook --execute --inplace spam_svd_pca.ipynb
jupyter nbconvert --to markdown spam_svd_pca.ipynb
```

Thời gian chạy toàn bộ đo bằng lệnh trên là **61 giây** (51 giây trong kernel; các lần chạy khác trên cùng máy từ 55 đến 61 giây), trên Windows 11, CPU AMD 16 luồng logic, Python 3.14.7, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1. Mọi phép ngẫu nhiên dùng `RANDOM_STATE = 42`; chỉ các số đo thời gian thay đổi theo máy.

## Quy trình

```
SMSSpamCollection: 5.574 tin (4.827 ham, 747 spam)
  │ normalize_text: html.unescape → chữ thường → email → URL → tiền → số điện thoại → số → bỏ ký tự lạ
  │ bỏ 2 dòng có clean rỗng
  │ loại nhóm clean mang nhiều nhãn: 0 nhóm
  │ loại trùng theo clean: bỏ 476 dòng
  ▼
5.096 tin (4.500 ham, 596 spam)
  │ chia 80/20, stratify, seed 42
  ├──► test: 1.020 tin (119 spam), chỉ dùng ở bước đánh giá cuối
  ▼
train: 4.076 tin (477 spam) ── chia 75/25 ──► sub-train 3.057 + validation 1.019
  │                                            │ TF-IDF riêng: X_sub 3057 × 4204
  │ TF-IDF (1–2 gram, min_df=2,                │ bộ đệm F_sub, F_sub_c (2 lần eigh)
  │   sublinear_tf, stop words)                │ quét k ∈ {2, 10, 50, 100, 300, 1000}
  ▼                                            ▼ chọn k theo F1 validation
X_train 4076 × 5445 (thưa, 0,16% khác 0)
  │ bộ đệm F_train, F_train_c (2 lần eigh của Gram 4076 × 4076)
  ├── kiểm chứng: svd_via_eig, randomized SVD, scikit-learn, Eckart–Young, PCA = SVD(X_c), góc chính
  └── LSA / PCA với k đã chọn → LogisticRegression(C=1, balanced) → đánh giá trên test
```

Loại trùng làm trên văn bản đã chuẩn hóa, sau khi loại nhãn mâu thuẫn: nhiều tin spam cùng mẫu chỉ khác số điện thoại trở thành cùng một chuỗi `clean`. Notebook `assert` rằng tập `clean` của test không giao train và của validation không giao sub-train.

## Hướng dẫn đọc notebook

| Mục trong notebook | Làm gì | Công thức / định lý trong `slide/script.md` |
|---|---|---|
| Bộ dữ liệu sử dụng | Giới thiệu bộ dữ liệu và mục tiêu | |
| Tải và đọc dữ liệu | Tải từ UCI (có mirror), đọc với `quoting=3` | |
| Mô tả dữ liệu | Phân bố nhãn, độ dài tin theo lớp, 2 tin mẫu mỗi lớp | |
| Tiền xử lý | `normalize_text` với 12 `assert`; làm sạch, loại trùng theo `clean`, chia tập; TF-IDF | (25) |
| Cài đặt SVD và PCA từ đầu | `svd_via_eig`; toán tử thưa và tâm hóa ngầm; randomized SVD; bộ đệm `exact_factors` (một lần `eigh` Gram cho mỗi ma trận và chế độ tâm hóa); `TruncatedSVDScratch`, `PCAScratch`; kiểm chứng tại $k = 50$ với ARPACK | Định lý 1, (3)–(5), (8), (14), (18), (27)–(29), mục 4.5–4.6 |
| Phổ giá trị kỳ dị | Toàn phổ, năng lượng tích lũy, $\sum\sigma_i^2 = \Vert X\Vert_F^2$ | (8) |
| Trực quan hóa tin nhắn trong không gian 2 chiều | Scatter hai thành phần đầu của LSA và PCA | mục 2.6 |
| Diễn giải các thành phần | 3 PC đầu, từ có trọng số lớn nhất, tương quan với nhãn | mục 4.2 |
| Xây dựng mô hình | `ReducedClassifier`, đo riêng phân rã, tạo đặc trưng và classifier | (26) |
| Chọn số thành phần $k$ | Quét $k$ trên bộ đệm sub-train, quy tắc chọn cố định trước | mục 3.4 |
| Kiểm chứng lý thuyết bằng thực nghiệm | Eckart–Young qua đồng nhất thức phần dư; PCA = SVD($X_c$); tâm hóa, góc chính LSA–PCA | Định lý 5, (22); Định lý 3, Mệnh đề 4, (19) |
| Đánh giá cuối cùng trên tập test | Bảng metric, confusion matrix + ROC, lỗi của mô hình được chọn | |
| Thử mô hình với tin nhắn mới | `predict_messages` trên 4 tin tự viết | (26) |
| Kết luận | Tóm tắt từ số liệu của lần chạy này | PHẦN 5 |

## Kết quả trên tập test

Quét $k$ trên validation: LSA và PCA cùng đạt F1 cao nhất 0,9205 tại $k = 300$ (baseline 0,9283). Theo quy tắc chọn (hòa thì LSA), mô hình giảm chiều được chọn là LSA với $k = 300$. Tập test có 1.020 tin, trong đó 119 spam.

| Mô hình | Số đặc trưng | Accuracy | Precision | Recall | F1 (spam) | ROC-AUC | Thời gian huấn luyện, gồm chuẩn bị cache $k_{\max} = 1000$ (s) |
|---|---|---|---|---|---|---|---|
| Baseline (TF-IDF) | 5445 | 0,9814 | 0,9237 | 0,9160 | 0,9198 | 0,9917 | 0,02 |
| LSA ($k = 300$) | 300 | 0,9784 | 0,8943 | 0,9244 | 0,9091 | 0,9864 | 5,56 |
| PCA ($k = 300$) | 300 | 0,9784 | 0,8943 | 0,9244 | 0,9091 | 0,9862 | 6,03 |

Cột thời gian bằng tổng: phân rã chính xác (5,53 s với LSA, 6,00 s với PCA, cho toàn phổ và 1000 vectơ) + tạo đặc trưng $XV_k$ tại $k = 300$ (khoảng 0,01 s) + huấn luyện classifier (khoảng 0,02 s). Đây không phải thời gian huấn luyện độc lập tại $k = 300$.

Nhận xét:

- Mô hình giảm chiều thấp hơn baseline một chút: F1 0,909 so với 0,920. LSA có 22 lỗi, 16 trong đó trùng với lỗi của baseline (baseline có 19 lỗi). Chênh lệch này chưa được kiểm định thống kê.
- Định lý Eckart–Young chỉ nói về sai số tái tạo nên không giải thích trực tiếp chênh lệch trên. Một giả thuyết phù hợp với dữ liệu là các từ hiếm nhưng có tính phân biệt nằm ở đuôi phổ. Phổ suy giảm chậm: 300 thành phần chỉ giữ 43,9% năng lượng.
- LSA và PCA cho cùng metric tại $k = 300$, nhưng góc chính lớn nhất giữa hai không gian con là 59,3°. Sự giống nhau về phân loại là quan sát thực nghiệm, không suy ra được từ việc hai ma trận Gram khác nhau một số hạng hạng một.
- Phần lý thuyết được tái hiện bằng số. `svd_via_eig` và nghiệm Gram khớp tham chiếu đến $10^{-15}$. Randomized SVD ($p = k$, $q = 10$) lệch dưới $3 \cdot 10^{-4}$. Sai số chiếu khớp $\sum_{i\gt k}\sigma_i^2$ đến $10^{-12}$. Ba cách tính phương sai PCA khớp đến $10^{-17}$.
- So với phiên bản trước của đồ án (cả ba mô hình F1 0,9389 với $k = 1000$), kết quả thay đổi do đồng thời việc loại rò rỉ dữ liệu và việc đổi từ randomized SVD sang nghiệm chính xác. Không thể quy chênh lệch cho riêng một nguyên nhân.

## Hạn chế

- Mô hình tuyến tính trên bag-of-words, không nắm thứ tự từ hay ngữ nghĩa ngoài đồng xuất hiện.
- Từ điển, idf, $\mu$ và $V_k$ gắn với tập huấn luyện; khi từ vựng thay đổi phải tính lại TF-IDF và phân rã.
- Bộ dữ liệu nhỏ, một ngôn ngữ, tập test chỉ có 119 tin spam.
- Nghiệm chính xác qua Gram $n \times n$ chỉ khả thi khi $n$ vừa phải; với $n$ lớn hơn nhiều phải dùng randomized SVD.

## Tài liệu tham khảo

1. T. A. Almeida, J. M. Gómez Hidalgo, A. Yamakami. *SMS Spam Collection*. UCI Machine Learning Repository, 2012. https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. N. Halko, P. G. Martinsson, J. A. Tropp. *Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions*. SIAM Review 53(2), 2011, 217–288.
3. C. Eckart, G. Young. *The approximation of one matrix by another of lower rank*. Psychometrika 1(3), 1936, 211–218.
4. L. Mirsky. *Symmetric gauge functions and unitarily invariant norms*. Quarterly Journal of Mathematics 11, 1960, 50–59.
5. G. Strang. *Linear Algebra and Learning from Data*. Wellesley-Cambridge Press, 2019.
6. scikit-learn: `TruncatedSVD`, `PCA`, `TfidfVectorizer`, `LogisticRegression`. https://scikit-learn.org/stable/
