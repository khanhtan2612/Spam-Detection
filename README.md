# Ứng dụng SVD và PCA trong phát hiện tin nhắn rác (Spam Detection)

Đồ án môn Toán cho Khoa học Máy tính. Đồ án nghiên cứu Singular Value Decomposition (SVD) và Principal Component Analysis (PCA) ở hai mặt: chứng minh các kết quả lý thuyết nền tảng, và cài đặt từ đầu bằng NumPy để giảm chiều biểu diễn TF-IDF của tin nhắn SMS trước một bộ phân loại logistic regression. Các kết quả lý thuyết chính được kiểm tra bằng số, phần lớn trên chính ma trận TF-IDF của bài toán.

## Nội dung chính

- **Lý thuyết** (`slide/script.md`): chứng minh xây dựng sự tồn tại của SVD qua chéo hóa $A^\mathsf{T}A$; PCA là bài toán cực đại phương sai và bằng SVD của dữ liệu đã tâm hóa; định lý Eckart–Young–Mirsky về xấp xỉ hạng thấp tối ưu theo chuẩn Frobenius; áp dụng cho văn bản (TF-IDF, LSA, thủ thuật Gram khi $n \lt d$, tâm hóa ngầm cho ma trận thưa, randomized SVD).
- **Cài đặt** (`spam_svd_pca.ipynb`): SVD qua trị riêng của $A^\mathsf{T}A$ theo đúng chứng minh; randomized SVD theo Halko–Martinsson–Tropp; PCA trên ma trận thưa không tạo ma trận tâm hóa đặc; bộ đệm phân rã chính xác (toàn phổ và 1000 vectơ kỳ dị phải đầu) dùng chung cho mọi số chiều $k \le 1000$.
- **Kiểm chứng**: đối chiếu với `numpy.linalg.svd` và với `TruncatedSVD`, `PCA` của scikit-learn; kiểm tra Eckart–Young, PCA = SVD của dữ liệu tâm hóa, tính không tương quan của các thành phần chính, vai trò của tâm hóa.
- **Ứng dụng**: so sánh logistic regression trên TF-IDF đầy đủ, trên đặc trưng LSA và trên đặc trưng PCA, với số chiều $k$ chọn trên tập validation.

## Cấu trúc bài nộp

```
Spam-Detection/
├── README.md                # tài liệu này
├── requirements.txt         # thư viện sử dụng
├── spam_svd_pca.ipynb       # notebook chính, đã chạy sẵn và lưu toàn bộ output
├── spam_svd_pca.md          # bản Markdown của notebook, đọc được không cần Jupyter
├── spam_svd_pca_files/      # 5 hình của notebook
├── data/SMSSpamCollection   # bộ dữ liệu SMS Spam Collection (UCI), 5.574 tin
└── slide/script.md          # cơ sở toán học: định nghĩa, định lý, chứng minh, công thức (1)–(30)
```

## Thứ tự đọc

1. `slide/script.md`, PHẦN 1–3: ba chứng minh chính (Định lý 1, Định lý 3 cùng Mệnh đề 4, Định lý 5).
2. `slide/script.md`, PHẦN 4: cách các kết quả trên được dùng cho dữ liệu văn bản, cũng là cơ sở của từng thuật toán trong notebook.
3. `spam_svd_pca.ipynb` (hoặc `spam_svd_pca.md`): cài đặt, kiểm chứng và kết quả. Bảng dưới đây ghi mục tương ứng trong `slide/script.md` cho từng phần của notebook.
4. `slide/script.md`, PHẦN 5: tổng hợp, liên hệ lý thuyết với số liệu thực nghiệm.

## Đối chiếu notebook với phần lý thuyết

| Mục trong notebook | Nội dung | Công thức / định lý trong `slide/script.md` |
|---|---|---|
| Bộ dữ liệu sử dụng | Giới thiệu bộ dữ liệu và mục tiêu | |
| Tải và đọc dữ liệu | Đọc file với `quoting=3` vì nhiều tin có dấu nháy không khép | |
| Mô tả dữ liệu | Phân bố nhãn, độ dài tin theo lớp, tin mẫu | |
| Tiền xử lý | Chuẩn hóa văn bản (kèm 12 phép kiểm tra), loại trùng theo văn bản đã chuẩn hóa, chia tập, TF-IDF | (25) |
| Cài đặt SVD và PCA từ đầu | `svd_via_eig`; toán tử thưa và tâm hóa ngầm; randomized SVD; bộ đệm `exact_factors`; `TruncatedSVDScratch`, `PCAScratch`; đối chiếu với scikit-learn tại $k = 50$ | Định lý 1, (3)–(5), (8), (14), (18), (27)–(29), mục 4.5–4.6 |
| Phổ giá trị kỳ dị | Toàn phổ, năng lượng tích lũy, kiểm tra $\sum\sigma_i^2 = \Vert X\Vert_F^2$ | (8) |
| Trực quan hóa tin nhắn trong không gian 2 chiều | Hai thành phần đầu của LSA và PCA | mục 2.6 |
| Diễn giải các thành phần | Ba thành phần chính đầu, từ có trọng số lớn nhất, tương quan với nhãn | mục 4.2 |
| Xây dựng mô hình | Reducer + logistic regression, không chuẩn hóa lại sau phép chiếu | (26), mục 4.2 |
| Chọn số thành phần $k$ | Quét $k$ trên tập validation với quy tắc chọn cố định trước | mục 3.4 |
| Kiểm chứng lý thuyết bằng thực nghiệm | Eckart–Young qua đồng nhất thức phần dư; PCA = SVD($X_c$); tâm hóa; góc chính giữa không gian con LSA và PCA | Định lý 5, (22); Định lý 3, Mệnh đề 4, (19) |
| Đánh giá cuối cùng trên tập test | Bảng metric, confusion matrix, ROC, các tin bị phân loại sai | |
| Thử mô hình với tin nhắn mới | Dự đoán cho 4 tin tự viết | (26) |
| Kết luận | Tổng hợp từ số liệu của lần chạy | PHẦN 5 |

## Một số lựa chọn thiết kế

- **Loại trùng sau khi chuẩn hóa.** Nhiều tin spam cùng mẫu, chỉ khác số điện thoại hay số tiền, trở thành cùng một chuỗi sau khi chuẩn hóa. Nếu loại trùng trên văn bản gốc, các bản gần giống nhau có thể rơi vào cả tập huấn luyện lẫn tập kiểm tra và làm kết quả lạc quan. Vì vậy dữ liệu được chuẩn hóa trước, loại các nhóm mang nhãn mâu thuẫn, rồi mới loại trùng (476 dòng). Notebook kiểm tra bằng `assert` rằng tập test không có tin nào trùng với tập train sau chuẩn hóa.
- **Nghiệm chính xác qua ma trận Gram.** Vì số tài liệu nhỏ hơn số từ ($4076 \lt 5445$ với tập huấn luyện, $3057 \lt 4204$ với sub-train), SVD chính xác được tính bằng một lần chéo hóa ma trận Gram $n \times n$ cho mỗi cặp (ma trận, chế độ tâm hóa): $4076 \times 4076$ cho tập huấn luyện và $3057 \times 3057$ cho sub-train dùng khi chọn $k$. Toàn notebook chỉ có bốn lần chéo hóa ma trận Gram của dữ liệu; mọi giá trị $k \le 1000$ đều cắt từ kết quả đó, vì nghiệm $k$ thành phần đầu của SVD chính xác lồng nhau theo $k$. Randomized SVD được giữ lại như thuật toán cho trường hợp ma trận lớn hơn và được kiểm chứng riêng.
- **Tách tập chọn mô hình và tập đánh giá.** Số chiều $k$ được chọn trong lưới $\lbrace 2, 10, 50, 100, 300, 1000\rbrace$ trên tập validation theo quy tắc cố định trước (F1 cao nhất, hòa thì chọn $k$ nhỏ hơn; giữa LSA và PCA, hòa thì chọn LSA). Tập test chỉ được dùng một lần ở bước cuối.
- **Không chuẩn hóa lại đặc trưng sau phép chiếu.** $V_k$ có cột trực chuẩn nên phép chiếu giữ nguyên khoảng cách và góc trong không gian con; chuẩn hóa lại sẽ khuếch đại các hướng có phương sai nhỏ.

## Dữ liệu

| Bước | Số tin |
|---|---|
| Dữ liệu gốc | 5.574 (4.827 ham, 747 spam) |
| Bỏ tin rỗng sau chuẩn hóa | 2 |
| Bỏ nhóm mang nhãn mâu thuẫn | 0 |
| Bỏ trùng sau chuẩn hóa | 476 |
| Còn lại | 5.096 (4.500 ham, 596 spam) |
| Tập huấn luyện / tập test | 4.076 (477 spam) / 1.020 (119 spam) |
| Trong tập huấn luyện: sub-train / validation | 3.057 / 1.019 |

Ma trận TF-IDF của tập huấn luyện có kích thước $4076 \times 5445$, chỉ 0,16% phần tử khác 0; 45 hàng rỗng vì mọi từ của tin là stop word hoặc xuất hiện ở ít hơn hai tài liệu.

## Kết quả kiểm chứng lý thuyết

| Kiểm tra | Kết quả |
|---|---|
| `svd_via_eig` so với `numpy.linalg.svd` | sai số tương đối của $\sigma_i$ cỡ $10^{-15}$, các nhân tử trực chuẩn |
| Nghiệm chính xác so với ARPACK của scikit-learn ($k = 50$) | sai số cỡ $10^{-15}$, $\vert\cos\vert = 1$ giữa các thành phần |
| Randomized SVD, $p = k$, $q = 10$ ($k = 50$) | sai số tương đối lớn nhất của $\sigma_i$: $2{,}2 \cdot 10^{-4}$ (LSA), $2{,}6 \cdot 10^{-4}$ (PCA); với $p = 10$, $q = 4$ là $4{,}4 \cdot 10^{-2}$ |
| Eckart–Young: sai số chiếu đo được so với $\sum_{i\gt k}\sigma_i^2$ | khớp đến cỡ $10^{-12}$ tại $k = 10, 50, 100, 300, 1000$; luôn nhỏ hơn không gian con ngẫu nhiên và việc giữ $k$ từ có chuẩn lớn nhất |
| PCA = SVD($X_c$): ba cách tính phương sai | chênh lệch tuyệt đối cỡ $10^{-17}$ trên các giá trị cỡ $10^{-3}$; phần tử ngoài đường chéo của $\mathrm{Cov}(Z)$ cỡ $10^{-18}$ |
| Tâm hóa | $\vert\cos(v_1, \mu)\vert$ là 0,883 với LSA và 0,327 với PCA |
| Góc chính lớn nhất giữa không gian con LSA và PCA | 72,4° tại $k = 50$, 59,3° tại $k = 300$ |

Phổ giá trị kỳ dị suy giảm chậm: 300 thành phần giữ 43,9% năng lượng, 1000 thành phần giữ 75,4%.

## Kết quả phân loại

Trên tập validation, F1 cao nhất của LSA là 0,9205, đạt tại cả $k = 300$ và $k = 1000$; F1 cao nhất của PCA cũng là 0,9205, tại $k = 300$; baseline đạt 0,9283. Theo quy tắc chọn, mô hình giảm chiều được chọn là LSA với $k = 300$. Kết quả trên tập test (1.020 tin, 119 spam):

| Mô hình | Số đặc trưng | Accuracy | Precision | Recall | F1 (spam) | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline (TF-IDF đầy đủ) | 5445 | 0,9814 | 0,9237 | 0,9160 | 0,9198 | 0,9917 |
| LSA ($k = 300$) | 300 | 0,9784 | 0,8943 | 0,9244 | 0,9091 | 0,9864 |
| PCA ($k = 300$) | 300 | 0,9784 | 0,8943 | 0,9244 | 0,9091 | 0,9862 |

Nhận xét:

- Mô hình giảm chiều thấp hơn baseline một chút (F1 0,909 so với 0,920). LSA có 22 lỗi, 16 trong đó trùng với lỗi của baseline (19 lỗi). Chênh lệch chưa được kiểm định thống kê.
- Định lý Eckart–Young nói về sai số tái tạo, không nói về khả năng phân loại, nên không giải thích trực tiếp chênh lệch này. Một giả thuyết, chưa được kiểm định trong notebook, là thông tin phân biệt của các từ hiếm nằm chủ yếu ở các hướng kỳ dị có $\sigma_i$ nhỏ, vốn bị cắt khi chỉ giữ 300 thành phần.
- Tại $k = 300$, LSA và PCA có cùng accuracy, precision, recall và F1 trên tập test (ROC-AUC 0,9864 so với 0,9862), dù hai không gian con khác nhau rõ (góc chính lớn nhất 59,3°). Đây là quan sát thực nghiệm: việc $X^\mathsf{T}X$ và $X_c^\mathsf{T}X_c$ khác nhau một ma trận hạng một $n\mu\mu^\mathsf{T}$ (công thức (19)) chỉ suy ra tính đan xen của trị riêng, không suy ra hai không gian con gần nhau.
- Trên bộ dữ liệu này, giảm chiều không cải thiện độ chính xác. Điều thu được là số đặc trưng giảm từ 5445 xuống 300, thành phần chính thứ nhất diễn giải được và tương quan 0,68 với nhãn spam dù PCA không dùng nhãn (PC2, PC3 chủ yếu ứng với các từ *ok*, *come* và gần như không tương quan với nhãn), cùng khả năng trực quan hóa trong 2 chiều.

## Hạn chế và hướng phát triển

- Mô hình tuyến tính trên bag-of-words, không nắm thứ tự từ hay ngữ nghĩa ngoài đồng xuất hiện.
- Từ điển, trọng số idf, vectơ trung bình và các thành phần chính gắn với tập huấn luyện; khi từ vựng thay đổi phải tính lại toàn bộ.
- Bộ dữ liệu nhỏ, một ngôn ngữ, tập test chỉ có 119 tin spam.
- Nghiệm chính xác qua ma trận Gram chỉ khả thi khi số tài liệu vừa phải; với dữ liệu lớn hơn nhiều cần dùng randomized SVD hoặc SVD gia tăng.
- Hướng phát triển: embedding học sẵn thay cho TF-IDF, mô hình phi tuyến trên không gian PCA, so sánh với NMF.

## Tài liệu tham khảo

1. T. A. Almeida, J. M. Gómez Hidalgo, A. Yamakami. *SMS Spam Collection*. UCI Machine Learning Repository, 2012. https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. N. Halko, P. G. Martinsson, J. A. Tropp. *Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions*. SIAM Review 53(2), 2011, 217–288.
3. C. Eckart, G. Young. *The approximation of one matrix by another of lower rank*. Psychometrika 1(3), 1936, 211–218.
4. L. Mirsky. *Symmetric gauge functions and unitarily invariant norms*. Quarterly Journal of Mathematics 11, 1960, 50–59.
5. G. Strang. *Linear Algebra and Learning from Data*. Wellesley-Cambridge Press, 2019.
6. scikit-learn: `TruncatedSVD`, `PCA`, `TfidfVectorizer`, `LogisticRegression`. https://scikit-learn.org/stable/
