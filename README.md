# Ứng dụng SVD và PCA trong phát hiện tin nhắn rác (Spam Detection)

Đồ án môn Toán cho Khoa học Máy tính. Đồ án cài đặt từ đầu Singular Value Decomposition (SVD) và Principal Component Analysis (PCA) bằng NumPy, dùng chúng để giảm chiều biểu diễn TF-IDF của tin nhắn SMS trước một bộ phân loại logistic regression, và kiểm chứng bằng số các định lý nền tảng (tồn tại SVD, PCA = SVD của dữ liệu tâm hóa, Eckart–Young–Mirsky) ngay trên ma trận dữ liệu thực.

## 1. Giới thiệu và mục tiêu

Bài toán: cho một tin nhắn SMS, dự đoán nó là `ham` (tin thường) hay `spam` (tin rác). Biểu diễn bag-of-words TF-IDF cho ma trận $X$ kích thước $4136 \times 5649$ với hơn 99,8% phần tử bằng 0. Câu hỏi của đồ án là: có thể thay $X$ bằng phép chiếu $XV_k$ lên $k \ll 5649$ chiều mà vẫn giữ được khả năng phân loại hay không, và các tính chất toán học nào của SVD/PCA giải thích kết quả đó.

Mục tiêu cụ thể:

1. Cài đặt SVD qua trị riêng của $A^\mathsf{T}A$, randomized SVD (Halko–Martinsson–Tropp) và PCA với tâm hóa ngầm trên ma trận thưa; đối chiếu với `scikit-learn`.
2. So sánh logistic regression trên TF-IDF đầy đủ, trên đặc trưng LSA (truncated SVD) và trên đặc trưng PCA với $k \in\lbrace 2, \dots, 1000\rbrace$.
3. Kiểm chứng thực nghiệm định lý Eckart–Young, quan hệ PCA–SVD, tính không tương quan của các thành phần chính và vai trò của tâm hóa.

## 2. Cấu trúc thư mục

```
spam-svd-pca/
├── README.md                # tài liệu này
├── requirements.txt         # numpy, scipy, pandas, matplotlib, scikit-learn>=1.4, jupyter
├── spam_svd_pca.ipynb       # notebook chính, đã chạy sẵn và lưu output
├── spam_svd_pca.md          # bản xuất markdown của notebook (nbconvert)
├── spam_svd_pca_files/      # hình ảnh do nbconvert xuất ra, được spam_svd_pca.md tham chiếu
├── data/SMSSpamCollection   # bộ dữ liệu UCI (tab-separated, 5.574 dòng)
└── slide/script.md          # cơ sở toán học: định nghĩa, định lý và chứng minh (tiếng Việt)
```

Vai trò từng file:

| File | Nội dung |
|---|---|
| `spam_svd_pca.ipynb` | Toàn bộ code và diễn giải, chia làm 14 mục (xem mục 5 dưới đây). Có thể Run All để tái tạo mọi số liệu. |
| `spam_svd_pca.md` | Cùng nội dung ở dạng markdown để đọc không cần Jupyter. |
| `slide/script.md` | PHẦN 1 (SVD), PHẦN 2 (PCA), PHẦN 3 (Eckart–Young–Mirsky), PHẦN 4 (áp dụng cho văn bản: TF-IDF, LSA, thủ thuật Gram, tâm hóa ngầm, randomized SVD), PHẦN 5 (kết luận). Công thức đánh số (1)–(30). |
| `data/SMSSpamCollection` | Dữ liệu gốc. Notebook tự tải nếu file chưa có. |

## 3. Cài đặt và chạy trên VS Code

Yêu cầu: Python 3.10 trở lên, VS Code với extension Python và Jupyter.

```powershell
cd spam-svd-pca
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows; trên macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Mở `spam_svd_pca.ipynb` trong VS Code, chọn kernel là interpreter trong `.venv` (nút **Select Kernel** ở góc trên bên phải), rồi bấm **Run All**. Notebook tự tải dữ liệu từ UCI nếu `data/SMSSpamCollection` chưa tồn tại (có mirror dự phòng trên GitHub). Thời gian chạy toàn bộ khoảng 10–12 phút trên máy tính cá nhân; phần tốn thời gian nhất là quét $k$ đến 1000 (mục 10) và fit lại mô hình cuối (mục 12).

Chạy từ dòng lệnh thay cho VS Code:

```powershell
jupyter nbconvert --to notebook --execute --inplace spam_svd_pca.ipynb
jupyter nbconvert --to markdown spam_svd_pca.ipynb
```

Mọi phép ngẫu nhiên (chia tập, ma trận Gauss của randomized SVD, không gian con ngẫu nhiên) dùng `RANDOM_STATE = 42`, nên số liệu tái tạo được; chỉ các cột thời gian thay đổi theo máy.

## 4. Quy trình thực hiện

```
SMSSpamCollection (5.574 tin)
        │  loại 403 tin trùng → 5.171 tin
        ▼
normalize_text: html.unescape → lowercase → URL/email/tiền/số điện thoại/số → token → bỏ ký tự lạ
        │
        ▼
chia 80/20 (stratify, seed 42) ──► test (1.035 tin, chỉ dùng ở mục 12)
        │
   train (4.136 tin) ──► chia 75/25 ──► sub-train (3.102) + validation (1.034)
        │                                        │
        ▼                                        ▼
TfidfVectorizer(1-2 gram, min_df=2,      vectorizer riêng fit trên sub-train
  sublinear_tf, stop_words) fit trên train       │
        │                                        ▼
        │                              quét k ∈ {2,…,1000} cho LSA và PCA,
        │                              chọn k theo F1 trên validation
        ▼
X_train (4136 × 5649, thưa)
        ├── svd_via_eig / randomized_svd / gram_svd  ──► đối chiếu sklearn
        ├── phổ σ_i qua eigh(X Xᵀ)                     ──► năng lượng tích lũy, V_k chính xác
        ├── Eckart–Young, PCA = SVD(X_c), Cov(Z) chéo, cos(v₁, μ)
        └── Reducer(k*) → LogisticRegression(C=1, balanced) ──► đánh giá trên test
```

## 5. Diễn giải từng mục của notebook

Các tiêu đề dưới đây trùng với tiêu đề cell trong notebook. Ký hiệu (PHẦN x, công thức (y)) trỏ đến `slide/script.md`.

### 5.1. Bộ dữ liệu sử dụng, tải và đọc dữ liệu

Đọc bằng `pd.read_csv(sep="\t", header=None, quoting=3, encoding="utf-8")`. `quoting=3` (QUOTE_NONE) là bắt buộc: nhiều tin có dấu nháy không khép, nếu để pandas xử lý dấu nháy thì nhiều dòng sẽ bị gộp làm một. Hàm `download_dataset` tải file zip từ UCI, giải nén trong bộ nhớ; nếu lỗi thì dùng mirror.

### 5.2. Mô tả dữ liệu

5.574 tin, 4.827 ham (86,6%) và 747 spam (13,4%); 403 dòng trùng. Tin spam dài hơn hẳn (trung vị 149 ký tự so với 52 của ham, tập trung sát giới hạn 160 ký tự của SMS). Top 20 từ mỗi lớp cho thấy spam bị chi phối bởi *free, call, txt, claim, prize*, còn ham là từ hội thoại.

### 5.3. Tiền xử lý

- `normalize_text`: `html.unescape` (dữ liệu có `&lt;#&gt;`, `&amp;`), chữ thường, rồi thay lần lượt URL → `urltoken`, email → `emailtoken`, tiền (`£1000`, `150p`, `20,000 pounds`) → `moneytoken`, chuỗi ≥ 5 chữ số (số điện thoại, mã ngắn) → `phonetoken`, số còn lại → `numtoken`, cuối cùng bỏ mọi ký tự không phải chữ cái. Lý do: số điện thoại hay số tiền cụ thể không lặp lại nên không học được, nhưng *sự có mặt* của chúng là tín hiệu spam mạnh.
- Loại trùng trước khi chia tập (5.574 → 5.171) để không có tin nào xuất hiện ở cả train và test.
- Chia train/test 80/20 có phân tầng, rồi train thành sub-train/validation 75/25. Tập test chỉ dùng một lần ở mục 12.
- `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, stop_words="english")`, công thức (25) trong PHẦN 4. Fit trên train cho $X_\text{train}$ ($4136 \times 5649$, 37.349 phần tử khác 0, mật độ 0,16%, trung bình 9 từ/tin). Dạng đặc chiếm 178 MiB, dạng CSR 0,44 MiB (gấp 402 lần). Một vectorizer thứ hai fit trên sub-train ($3102 \times 4216$) dùng cho bước chọn $k$, để validation không ảnh hưởng đến từ điển và idf.

### 5.4. Cài đặt SVD và PCA từ đầu

- `svd_via_eig(A)`: thuật toán rút ra từ chứng minh Định lý 1 (PHẦN 1, công thức (3)–(5)): `eigh(AᵀA)` → $\sigma_i = \sqrt{\lambda_i}$, $u_i = Av_i/\sigma_i$. Khi $n \gt m$ thì áp dụng cho $A^\mathsf{T}$. Ngưỡng hạng đặt trên $\lambda_i \gt\lambda_1 \cdot\max(m,n)\thinspace\varepsilon$ (nhận xét số học trong PHẦN 1.3). Đối chiếu với `np.linalg.svd` trên hai ma trận Gauss, một ma trận hạng 5 và một khối 300 hàng TF-IDF: sai số $\sigma$ cỡ $10^{-16}$ đến $10^{-15}$, sai số tái tạo cỡ $10^{-15}$, $\lVert U^\mathsf{T}U - I\rVert$ và $\lVert V^\mathsf{T}V - I\rVert$ cỡ $10^{-14}$; hạng của khối TF-IDF được xác định đúng là 296.
- `SparseOperator`, `CenteredOperator`: hai lớp cung cấp `matmat` ($AM$), `rmatmat` ($A^\mathsf{T}M$), `gram` ($AA^\mathsf{T}$), `gram_t` ($A^\mathsf{T}A$). `CenteredOperator` cài công thức tâm hóa ngầm (28) và ma trận Gram tâm hóa (29); ma trận $X_c$ đặc không bao giờ được tạo.
- `randomized_svd(A, k, n_oversamples=k, n_iter=10)`: thuật toán HMT (PHẦN 4.5): $\Omega$ Gauss → $Y = A\Omega$ → QR → power iteration có QR → $B = Q^\mathsf{T}A$ → SVD của $B$. Bảng hội tụ tại $k = 50$ so với ARPACK:

  | $p$ | $q$ | sai số tương đối lớn nhất của $\sigma$ |
  |---|---|---|
  | 10 | 4 | $4{,}7 \cdot 10^{-2}$ |
  | 50 | 7 | $1{,}8 \cdot 10^{-3}$ |
  | 50 | 10 | $2{,}6 \cdot 10^{-4}$ |
  | 100 | 10 | $2{,}4 \cdot 10^{-6}$ |

  Phổ TF-IDF suy giảm chậm nên cần $q$ lớn hơn thường lệ; mặc định của đồ án là $p = k$, $q = 10$ (dưới ngưỡng $10^{-3}$ với chi phí khoảng 1 giây).
- `gram_svd(A, k)`: thuật toán chính xác qua `eigh` của ma trận Gram nhỏ (PHẦN 4.3).
- `TruncatedSVDScratch` (LSA, không tâm hóa) và `PCAScratch` (tâm hóa): cùng lớp cơ sở `_SpectralReducer` với `fit / transform / fit_transform / inverse_transform`, `components_`, `singular_values_`, `explained_variance_` ($= \sigma_i^2/(n-1)$ với PCA, công thức (15)), `explained_variance_ratio_` (mẫu số tính theo (19) mà không tâm hóa). Đối chiếu với `TruncatedSVD(algorithm="arpack")` và `PCA(svd_solver="arpack")` tại $k = 50$:

  | Mô hình | sai số $\sigma$ lớn nhất | min $\lvert\cos\rvert$ | tổng EVR (ta / sklearn) |
  |---|---|---|---|
  | LSA (randomized) | $2{,}6 \cdot 10^{-4}$ | 0,9983 | 0,14742 / 0,14742 |
  | LSA (exact) | $2{,}4 \cdot 10^{-15}$ | 1,0000 | 0,14742 / 0,14742 |
  | PCA (randomized) | $2{,}0 \cdot 10^{-4}$ | 0,9988 | 0,14826 / 0,14826 |
  | PCA (exact) | $2{,}5 \cdot 10^{-15}$ | 1,0000 | 0,14826 / 0,14826 |

### 5.5. Phổ giá trị kỳ dị

Vì $n \lt d$, toàn bộ phổ lấy từ `eigh(X Xᵀ)` ($4136 \times 4136$, khoảng 5 giây), giữ lại $U$; $V_k = X^\mathsf{T}U_k\Sigma_k^{-1}$ cho 1000 thành phần đầu (PHẦN 4.3, công thức (27)). Kiểm tra (8): $\sum\sigma_i^2 = 4090{,}000000 = \lVert X\rVert_F^2$ = số hàng khác rỗng (46 tin mất hết từ sau tiền xử lý). Hạng số học 3694. $\sigma_1 = 6{,}659$, $\sigma_2 = 5{,}230$. Năng lượng tích lũy: 50% cần $k = 390$, 70% → 829, 80% → 1174, 90% → 1702, 95% → 2131. Phổ không có khuỷu, đặc trưng của văn bản thưa.

### 5.6. Trực quan hóa 2D và diễn giải các thành phần

Scatter LSA-2 và PCA-2 tô màu theo nhãn: spam tụ về một vùng nhưng hai chiều chưa đủ tách. Tọa độ LSA thứ nhất không âm với mọi tài liệu (Perron–Frobenius, PHẦN 2.6). Sáu thành phần PCA đầu (đổi dấu theo tương quan với nhãn): PC1 có tương quan $+0{,}68$ với nhãn spam, phía dương là `numtoken`, `numtoken numtoken`, `phonetoken`, `moneytoken`, *free, ur, txt, prize*, phía âm là *ok, come, good, know*. PC2–PC6 gần như không tương quan với nhãn (0,02–0,18) và mô tả các chủ đề khác (*ok/lor*, *come/home*, *like/know*).

### 5.7. Xây dựng mô hình, huấn luyện, vẽ biểu đồ

`ReducedClassifier` = reducer (LSA / PCA / không) + `LogisticRegression(C=1, class_weight="balanced", max_iter=5000)`. Không dùng StandardScaler sau phép chiếu: $V_k$ có cột trực chuẩn nên $z = V_k^\mathsf{T}x$ là phép chiếu trực giao, tọa độ thứ $i$ có độ lệch chuẩn tỉ lệ $\sigma_i$ (công thức (17)); chuẩn hóa sẽ khuếch đại các hướng phương sai nhỏ, chủ yếu là nhiễu, và trong thử nghiệm sơ bộ làm giảm F1 (PHẦN 4.2).

Quét $k$ trên validation (vectorizer fit trên sub-train, baseline F1 = 0,9732):

| $k$ | 2 | 5 | 10 | 20 | 50 | 100 | 200 | 300 | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 LSA | 0,702 | 0,787 | 0,860 | 0,895 | 0,917 | 0,928 | 0,958 | 0,962 | 0,962 | 0,970 |
| F1 PCA | 0,680 | 0,811 | 0,886 | 0,888 | 0,917 | 0,928 | 0,958 | 0,962 | 0,966 | 0,970 |
| fit LSA (s) | 0,0 | 0,0 | 0,1 | 0,2 | 0,7 | 2,3 | 13,1 | 15,7 | 26,1 | 53,5 |
| fit PCA (s) | 0,0 | 0,0 | 0,1 | 0,2 | 0,8 | 2,5 | 13,6 | 21,0 | 31,8 | 58,9 |

$k$ tốt nhất: 1000 cho cả hai. Hai đường F1 gần trùng nhau từ $k \ge 50$ (hai ma trận Gram chỉ khác một số hạng hạng một, công thức (20)). Thời gian fit tăng gần bậc hai theo $k$ vì randomized SVD làm việc với $\ell= 2k$ cột (PHẦN 4.5).

### 5.8. Kiểm chứng lý thuyết bằng thực nghiệm

**Eckart–Young** (PHẦN 3). Sai số $\lVert X - XV V^\mathsf{T}\rVert_F^2$ đo trực tiếp theo khối 512 hàng:

| $k$ | SVD (đo) | $\sum_{i\gt k}\sigma_i^2$ | ngẫu nhiên | $k$ cột có chuẩn lớn nhất | chênh lệch SVD và lý thuyết |
|---|---|---|---|---|---|
| 10 | 3883,13 | 3883,13 | 4082,91 | 3924,10 | $\lt 10^{-9}$ |
| 50 | 3463,01 | 3463,01 | 4053,81 | 3559,64 | $\lt 10^{-9}$ |
| 100 | 3117,62 | 3117,62 | 4017,42 | 3264,36 | $\lt 10^{-9}$ |
| 200 | 2638,60 | 2638,60 | 3945,52 | 2870,86 | $\lt 10^{-9}$ |
| 500 | 1789,94 | 1789,94 | 3728,16 | 2197,87 | $\lt 10^{-9}$ |
| 1000 | 1005,97 | 1005,97 | 3366,37 | 1575,93 | $\lt 10^{-9}$ |

Không gian con SVD trùng lý thuyết và luôn tốt hơn hai cách còn lại, đúng như Định lý 5.

**PCA = SVD của dữ liệu tâm hóa** (PHẦN 2, Định lý 3 và Mệnh đề 4). Trị riêng của $C$ (qua Gram tâm hóa, công thức (29)), $\sigma_i^2/(n-1)$ của `PCAScratch` và $\operatorname{Var}(Z_i)$ trùng nhau đến $10^{-12}$ cho 10 thành phần đầu ($\lambda_1 = 0{,}00796$, $\lambda_2 = 0{,}00596$, ...); $\operatorname{tr}(C) = 0{,}982128$ = `total_variance_`; phần tử ngoài đường chéo lớn nhất của $\operatorname{Cov}(Z)$ là $5 \cdot 10^{-18}$. Các trị riêng tâm hóa và không tâm hóa đan xen (Cauchy), kiểm tra in ra `True`. Heatmap tương quan giữa 10 từ phổ biến nhất và 10 PC cho thấy mỗi từ gắn với một hoặc hai thành phần.

**Vai trò của tâm hóa** (PHẦN 2.6). $|\cos(v_1^{\text{SVD}}, \mu)| = 0{,}867$, $|\cos(v_1^{\text{PCA}}, \mu)| = 0{,}340$, $|\cos(v_1^{\text{SVD}}, v_1^{\text{PCA}})| = 0{,}744$; tỉ lệ tài liệu cùng dấu trên thành phần đầu: SVD 100%, PCA 73%; $\sigma_1$ giảm từ 6,659 xuống 5,736 sau tâm hóa.

### 5.9. Đánh giá cuối trên tập test, thử tin nhắn mới, kết luận

Fit lại trên toàn bộ train với $k = 1000$; kết quả ở mục 6. `predict_messages` nhận danh sách chuỗi, chạy toàn bộ pipeline và trả xác suất spam của ba mô hình; 5 tin tự viết đều được phân loại đúng (xác suất spam 0,07–0,09 cho hai tin thường, 0,94–0,98 cho ba tin rác).

## 6. Kết quả chính

Tập test (1.035 tin, 131 spam), mỗi mô hình chạy đúng một lần:

| Mô hình | số đặc trưng | accuracy | precision | recall | F1 (spam) | ROC-AUC | fit (s) | predict (s) |
|---|---|---|---|---|---|---|---|---|
| Baseline (TF-IDF) | 5649 | 0,9845 | 0,9389 | 0,9389 | 0,9389 | 0,9964 | 0,02 | 0,0005 |
| LSA ($k = 1000$) | 1000 | 0,9845 | 0,9389 | 0,9389 | 0,9389 | 0,9963 | 88,9 | 0,025 |
| PCA ($k = 1000$) | 1000 | 0,9845 | 0,9389 | 0,9389 | 0,9389 | 0,9963 | 62,0 | 0,030 |

Ma trận nhầm lẫn của cả ba mô hình: 896 ham đúng, 8 ham bị gán spam, 8 spam bị bỏ sót, 123 spam đúng. Các tin bị phân loại sai giống nhau ở cả ba mô hình: tin thường viết kiểu mệnh lệnh có từ *free/call* ("Are you free now? can i call now?") và tin rác dùng từ vựng hiếm.

Nhận xét:

- Giảm chiều **không cải thiện** độ chính xác trên bộ dữ liệu này; ở $k = 1000$ nó bắt kịp baseline với cùng 16 lỗi, ở $k$ nhỏ hơn thì thua (F1 validation 0,93 tại $k = 100$ so với 0,97). Điều này nhất quán với Eckart–Young: phép chiếu tối ưu theo năng lượng, trong khi các từ hiếm nhưng mang tính phân biệt nằm ở đuôi phổ.
- Chi phí của SVD (khoảng 1–2 phút cho $k = 1000$) lớn hơn nhiều so với huấn luyện baseline (0,02 s). Lợi ích của bước giảm chiều nằm ở biểu diễn đặc 1000 chiều, khả năng diễn giải các trục và trực quan hóa, không nằm ở độ chính xác.
- Toàn bộ phần lý thuyết được tái hiện bằng số với sai số ở mức làm tròn: SVD tự cài đặt ($10^{-15}$), randomized SVD ($\lt 3 \cdot 10^{-4}$), Eckart–Young ($\lt 10^{-9}$), PCA–SVD ($10^{-12}$), $\operatorname{Cov}(Z)$ chéo ($10^{-18}$).

## 7. Hạn chế và hướng phát triển

- Mô hình tuyến tính trên bag-of-words: không nắm thứ tự từ hay ngữ nghĩa ngoài đồng xuất hiện.
- Từ điển, idf, $\mu$ và $V_k$ gắn với tập huấn luyện; khi từ vựng thay đổi phải tính lại toàn bộ (có thể thay bằng incremental SVD).
- Bộ dữ liệu nhỏ, một ngôn ngữ, test chỉ có 131 spam: chênh lệch dưới 1% giữa các mô hình không có ý nghĩa thống kê.
- Randomized SVD với $p = k$, $q = 10$ chính xác nhưng chậm ở $k$ lớn; với $k \ge 300$ thủ thuật Gram (`algorithm="exact"`) nhanh hơn trên dữ liệu này.
- Hướng phát triển: embedding học sẵn thay cho TF-IDF, mô hình phi tuyến trên không gian PCA, kernel PCA, hoặc so sánh với NMF (phân tích không âm, dễ diễn giải hơn cho văn bản).

## Tài liệu tham khảo

1. T. A. Almeida, J. M. Gómez Hidalgo, A. Yamakami. *SMS Spam Collection*. UCI Machine Learning Repository, 2012. https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. N. Halko, P. G. Martinsson, J. A. Tropp. *Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions*. SIAM Review 53(2), 2011, 217–288.
3. C. Eckart, G. Young. *The approximation of one matrix by another of lower rank*. Psychometrika 1(3), 1936, 211–218.
4. L. Mirsky. *Symmetric gauge functions and unitarily invariant norms*. Quarterly Journal of Mathematics 11, 1960, 50–59.
5. G. Strang. *Linear Algebra and Learning from Data*. Wellesley-Cambridge Press, 2019.
6. scikit-learn: `TruncatedSVD`, `PCA`, `TfidfVectorizer`, `LogisticRegression`. https://scikit-learn.org/stable/
