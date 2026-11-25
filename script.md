# SVD, PCA VÀ LOGISTIC REGRESSION: CƠ SỞ TOÁN HỌC CHO BÀI TOÁN PHÁT HIỆN SMS SPAM

Tài liệu này trình bày phần toán đứng sau notebook [svd_pca_spam.ipynb](svd_pca_spam.ipynb) và code trong [src/](src/). Mỗi kết quả gồm phát biểu, chứng minh ngắn, rồi chỉ ra chỗ nó được dùng trong code. Số liệu thực nghiệm lấy từ lần chạy hiện tại, xem [KET_QUA_MO_HINH.md](KET_QUA_MO_HINH.md) và [outputs/results/](outputs/results/) (phổ $\sigma_j$, $\Vert X \Vert_F^2$ và $\Vert X_c \Vert_F^2$ nằm trong `spectrum.json`).

Sơ đồ tổng quan: [pipeline_tuan_tu.png](slide/pipeline_tuan_tu.png) (14 bước theo thứ tự), [pipeline_model.png](slide/pipeline_model.png) (công thức của từng khối) và [pipeline_code_cong_thuc.png](slide/pipeline_code_cong_thuc.png) (mỗi bước ứng với file, hàm, công thức và kết quả nào).

## Cách dùng tài liệu

Mọi bước học trong đồ án đi theo cùng một mạch: **mô hình → loss → tối ưu → score**.

| | Mô hình | Loss | Tối ưu | Score |
|---|---|---|---|---|
| SVD / PCA | SVD: $X \approx ZV_k^\mathsf{T}$; PCA: $X \approx ZV_k^\mathsf{T} + \mathbf{1}\mu^\mathsf{T}$ (Phần 1–2) | $\sum_{j \gt k}\sigma_j^2$, với $\sigma_j$ của $X$ (SVD) hoặc của $X_c$ (PCA) (Phần 3) | orthogonal iteration (Phần 4) | tỉ lệ năng lượng / phương sai giữ lại (mục 2.5, 3.4) |
| Logistic Regression | $p = \sigma(Xw + b)$ (mục 6.1) | cross-entropy + L2 (mục 6.2) | gradient descent (mục 6.3–6.6) | accuracy, precision, recall, F1, macro-F1 (Phần 7) |

Muốn đọc để hiểu: đi lần lượt từ Phần 1 đến Phần 8. Muốn sửa hoặc kiểm tra code: dùng [bảng tra công thức → code](#bảng-tra-công-thức--code--notebook) ở cuối tài liệu.

**Ký hiệu.** $X \in \mathbb{R}^{N \times D}$ là ma trận TF-IDF: $N$ hàng (tin nhắn), $D$ cột (từ); trong đồ án $N = 3.611$ tin train và $D = 1.000$ từ. $x_i^\mathsf{T}$ là hàng thứ $i$. $\mathbf{1}$ là vectơ toàn số 1. $\Vert A \Vert_F^2 = \sum_{i,j} a_{ij}^2 = \mathrm{tr}(A^\mathsf{T}A)$ là bình phương chuẩn Frobenius. Nhãn $y_i = 1$ nếu tin là spam, $y_i = 0$ nếu là ham. Với ma trận đối xứng, "chéo hóa trực giao" nghĩa là viết $S = Q\Lambda Q^\mathsf{T}$ với $Q$ trực giao và $\Lambda$ chéo. $U_k$, $\Sigma_k$, $V_k$ là $k$ cột (hoặc $k$ giá trị) đầu của $U$, $\Sigma$, $V$. $\Vert X \Vert_F^2 = \sum_i\sigma_i^2$ được gọi là **năng lượng** của $X$.

**Lưu ý hai ký hiệu có hai nghĩa.** $\lambda_i$ (có chỉ số) là trị riêng, còn $\lambda$ (không chỉ số, chỉ ở Phần 6) là hệ số phạt L2, tham số `lam` trong code; trị riêng lớn nhất luôn viết $\lambda_{\max}(\cdot)$. Tương tự, $\sigma_i$ (có chỉ số) là giá trị kỳ dị, còn $\sigma(\cdot)$ là hàm sigmoid.

---

## PHẦN 1: SINGULAR VALUE DECOMPOSITION (SVD)

### 1.1. Định nghĩa

Cho $A \in \mathbb{R}^{m \times n}$ bất kỳ. Một **phân tích giá trị kỳ dị** của $A$ là cách viết

$$
A = U \Sigma V^\mathsf{T}, \qquad (1)
$$

trong đó $U \in \mathbb{R}^{m \times m}$ và $V \in \mathbb{R}^{n \times n}$ trực giao ($U^\mathsf{T}U = I_m$, $V^\mathsf{T}V = I_n$), còn $\Sigma \in \mathbb{R}^{m \times n}$ chỉ có phần tử khác 0 trên đường chéo chính, $\Sigma_{ii} = \sigma_i$, với $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_p \ge 0$, $p = \min(m, n)$.

Các số $\sigma_i$ là **giá trị kỳ dị**; các cột $u_i$ của $U$ và $v_i$ của $V$ là vectơ kỳ dị trái và phải. Từ (1) suy ra

$$
A v_i = \sigma_i u_i, \qquad A^\mathsf{T} u_i = \sigma_i v_i, \qquad i = 1, \dots, p. \qquad (2)
$$

### 1.2. Hai bổ đề về $A^\mathsf{T}A$

**Bổ đề 1.** $A^\mathsf{T}A$ là ma trận đối xứng và nửa xác định dương.

*Chứng minh.* $(A^\mathsf{T}A)^\mathsf{T} = A^\mathsf{T}A$. Với mọi $x$: $x^\mathsf{T}A^\mathsf{T}Ax = \Vert Ax \Vert^2 \ge 0$. $∎$

**Bổ đề 2 (định lý phổ).** Mọi ma trận đối xứng thực $S$ chéo hóa trực giao được: $S = Q\Lambda Q^\mathsf{T}$ với $Q$ trực giao và $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_n)$ thực. Nếu $S$ nửa xác định dương thì mọi $\lambda_i \ge 0$.

*Chứng minh phần cuối.* Với $q_i$ là cột thứ $i$ của $Q$: $\lambda_i = q_i^\mathsf{T} S q_i \ge 0$. Phần đầu là định lý phổ, được coi là đã biết. $∎$

### 1.3. Định lý tồn tại SVD (chứng minh xây dựng)

**Định lý 1.** Mọi ma trận $A \in \mathbb{R}^{m \times n}$ đều có một phân tích dạng (1).

*Chứng minh.* Theo Bổ đề 1 và 2,

$$
A^\mathsf{T}A = V \Lambda V^\mathsf{T}, \qquad \lambda_1 \ge \dots \ge \lambda_n \ge 0, \qquad (3)
$$

với $V = [v_1 \cdots v_n]$ trực giao. Đặt $\sigma_i = \sqrt{\lambda_i}$ và gọi $r$ là số chỉ số có $\sigma_i \gt 0$ (tức $r = \mathrm{rank}(A)$).

**Bước 1** ($Av_i$ đôi một trực giao). Với mọi $i, j$:

$$
(Av_i)^\mathsf{T}(Av_j) = v_i^\mathsf{T}(A^\mathsf{T}A)v_j = \lambda_j v_i^\mathsf{T}v_j = \lambda_j \delta_{ij}. \qquad (4)
$$

Vậy $\Vert Av_i \Vert = \sigma_i$ và $Av_i = 0$ khi $i \gt r$.

**Bước 2** (xây dựng $u_i$). Với $i \le r$ đặt

$$
u_i = \frac{A v_i}{\sigma_i}. \qquad (5)
$$

Theo (4), $u_i^\mathsf{T}u_j = \delta_{ij}$. Bổ sung (Gram–Schmidt) thành cơ sở trực chuẩn $u_1, \dots, u_m$ và đặt $U = [u_1 \cdots u_m]$.

**Bước 3** (kiểm tra). Phần tử $(i, j)$ của $U^\mathsf{T}AV$ là $u_i^\mathsf{T}Av_j$. Nếu $j \le r$, $Av_j = \sigma_j u_j$ nên phần tử bằng $\sigma_j\delta_{ij}$. Nếu $j \gt r$, $Av_j = 0$. Vậy $U^\mathsf{T}AV = \Sigma$, tức $A = U\Sigma V^\mathsf{T}$. $∎$

**Trong code.** Áp dụng cho $A = X$ (tức $m = N$, $n = D$): chỉ cần các vectơ riêng của $X^\mathsf{T}X$ là đủ để có $V_k$. Ma trận này cỡ $D \times D = 1.000 \times 1.000$, nhỏ hơn $XX^\mathsf{T}$ cỡ $N \times N$. Code không tính $U$: tọa độ mới của các tin nhắn là $Z = XV_k = U_k\Sigma_k$ theo (2). Vectơ riêng được tìm bằng orthogonal iteration (Phần 4), không dùng hàm SVD hay hàm trị riêng có sẵn; bước QR dùng `np.linalg.qr`. `np.linalg.svd` và `np.linalg.eigh` chỉ xuất hiện trong `tests/test_decomposition.py` để đối chiếu.

**Nhận xét số học.** Vì đi qua $\lambda_i = \sigma_i^2$, sai số làm tròn cỡ $\lambda_1\varepsilon$ trên $\lambda_i$ (với $\varepsilon \approx 2{,}2 \cdot 10^{-16}$ là độ chính xác máy) trở thành sai số cỡ $\sigma_1\sqrt{\varepsilon}$ trên các $\sigma_i$ nhỏ. Với các $\sigma_i$ lớn được giữ lại trong đồ án ($\sigma_{100} \approx 2{,}9$ so với $\sigma_1 \approx 11{,}2$), ảnh hưởng này không đáng kể.

### 1.4. Dạng compact và dạng tổng các ma trận hạng một

Vì $\sigma_i = 0$ khi $i \gt r$:

$$
A = U_r\Sigma_rV_r^\mathsf{T} = \sum_{i=1}^{r}\sigma_i u_i v_i^\mathsf{T}. \qquad (6)
$$

Các ma trận $u_iv_i^\mathsf{T}$ có chuẩn Frobenius bằng 1 và đôi một trực giao: $\langle u_iv_i^\mathsf{T}, u_jv_j^\mathsf{T}\rangle_F = (u_i^\mathsf{T}u_j)(v_i^\mathsf{T}v_j) = \delta_{ij}$. Do đó

$$
\Vert A \Vert_F^2 = \sum_{i=1}^{r}\sigma_i^2. \qquad (7)
$$

**Trong code.** Nhờ (7), score của SVD $\sum_{i \le k}\sigma_i^2 / \Vert X \Vert_F^2$ luôn nằm trong $[0, 1]$, và mẫu số được tính trực tiếp bằng $\Vert X \Vert_F^2$ mà không cần toàn bộ phổ (`total_energy_`). Mỗi hàng TF-IDF khác 0 có độ dài 1 (mục 5.1), nên $\Vert X \Vert_F^2$ bằng số hàng khác 0: $3.591$, vì có 20 tin train không chứa từ nào trong vocabulary.

---

## PHẦN 2: PRINCIPAL COMPONENT ANALYSIS (PCA)

### 2.1. Tâm hóa và ma trận hiệp phương sai

Cho $X$ với $N \ge 2$ hàng. Vectơ trung bình và ma trận đã tâm hóa:

$$
\mu = \frac{1}{N}X^\mathsf{T}\mathbf{1} \in \mathbb{R}^D, \qquad X_c = X - \mathbf{1}\mu^\mathsf{T}. \qquad (8)
$$

Ma trận hiệp phương sai mẫu

$$
C = \frac{1}{N-1}X_c^\mathsf{T}X_c \qquad (9)
$$

đối xứng, nửa xác định dương (Bổ đề 1). Với hướng đơn vị $w$, tọa độ chiếu $z_i = w^\mathsf{T}(x_i - \mu)$ có trung bình 0 và phương sai mẫu

$$
\mathrm{Var}(z) = \frac{1}{N-1}\Vert X_cw \Vert^2 = w^\mathsf{T}Cw. \qquad (10)
$$

### 2.2. Thành phần chính thứ nhất: bài toán cực đại phương sai

**Định lý 2.** Nghiệm của bài toán

$$
\max_{\Vert w \Vert = 1} w^\mathsf{T}Cw \qquad (11)
$$

là vectơ riêng $v_1$ ứng với trị riêng lớn nhất $\lambda_1$ của $C$, và giá trị cực đại bằng $\lambda_1$.

*Chứng minh.* Chéo hóa $C = V\Lambda V^\mathsf{T}$ và đặt $y = V^\mathsf{T}w$, thì $\Vert y \Vert = 1$ và $w^\mathsf{T}Cw = \sum_i \lambda_i y_i^2 \le \lambda_1\sum_i y_i^2 = \lambda_1$, dấu bằng khi $w = v_1$. $∎$

Thành phần thứ $j$ là nghiệm của (11) với thêm ràng buộc $w \perp v_1, \dots, v_{j-1}$; lập luận tương tự (cho $y_1 = \dots = y_{j-1} = 0$) cho nghiệm $v_j$. Tổng quát, $\mathrm{span}(v_1, \dots, v_k)$ cực đại tổng phương sai $\mathrm{tr}(W^\mathsf{T}CW)$ trên mọi $W \in \mathbb{R}^{D \times k}$ có cột trực chuẩn; đây cũng là hệ quả của Định lý 5 ở Phần 3 áp dụng cho $X_c$.

### 2.3. PCA là SVD của ma trận đã tâm hóa

**Định lý 3.** Gọi $X_c = U\Sigma V^\mathsf{T}$ là SVD của ma trận đã tâm hóa. Khi đó

$$
C = \frac{1}{N-1}V\Sigma^\mathsf{T}\Sigma V^\mathsf{T}, \qquad (12)
$$

tức các vectơ kỳ dị phải $v_i$ của $X_c$ là vectơ riêng của $C$, với trị riêng

$$
\lambda_i = \frac{\sigma_i^2}{N-1}. \qquad (13)
$$

*Chứng minh.* $X_c^\mathsf{T}X_c = V\Sigma^\mathsf{T}U^\mathsf{T}U\Sigma V^\mathsf{T} = V(\Sigma^\mathsf{T}\Sigma)V^\mathsf{T}$, chia cho $N - 1$. Đây là một phép chéo hóa trực giao của $C$. $∎$

**Trong code.** Lớp `PCA` kế thừa `TruncatedSVD`: học $\mu$ trên tập fit, lập $X_c^\mathsf{T}X_c = (N-1)C$ (không chia cho $N - 1$) rồi chạy đúng orthogonal iteration như nhánh SVD. `explained_variance_` được đặt bằng $\sigma_i^2/(N-1)$ theo (13). Dữ liệu mới trừ đúng $\mu$ đã học, không tự tính lại mean. ("Tập fit" là train khi chọn $k$, và train + validation ở lần fit cuối trước khi đánh giá test; quy ước này áp dụng cho mọi tham số học được trong tài liệu.)

### 2.4. Các điểm PCA không tương quan

Điểm PCA là $Z = X_cV_k$ với $V_k = [v_1 \cdots v_k]$.

**Mệnh đề 4.** Các cột của $Z$ có trung bình 0 và

$$
\frac{1}{N-1}Z^\mathsf{T}Z = \mathrm{diag}(\lambda_1, \dots, \lambda_k). \qquad (14)
$$

*Chứng minh.* $\mathbf{1}^\mathsf{T}Z = (\mathbf{1}^\mathsf{T}X - N\mu^\mathsf{T})V_k = 0$ theo (8). Dùng $X_cV_k = U_k\Sigma_k$ (từ (2)): $Z^\mathsf{T}Z = \Sigma_kU_k^\mathsf{T}U_k\Sigma_k = \Sigma_k^2$, rồi áp dụng (13). $∎$

Vậy PCA không chỉ đổi hệ trục mà còn **khử tương quan**: hai tọa độ bất kỳ của $Z$ có hiệp phương sai mẫu bằng 0.

### 2.5. Tỉ lệ phương sai giữ lại (score của PCA)

Tổng phương sai là $\mathrm{tr}(C) = \frac{1}{N-1}\Vert X_c \Vert_F^2$ (theo (7)). Khi tổng này khác 0, tỉ lệ phương sai giữ lại bởi $k$ thành phần đầu là

$$
\mathrm{score}_{\mathrm{PCA}}(k) = \frac{\sum_{i \le k}\lambda_i}{\sum_i \lambda_i} = \frac{\sum_{i \le k}\sigma_i^2}{\Vert X_c \Vert_F^2}. \qquad (15)
$$

Mẫu số có thể tính từ $X$ mà không lập $X_c$:

$$
\Vert X_c \Vert_F^2 = \Vert X \Vert_F^2 - N\Vert \mu \Vert^2, \qquad (16)
$$

vì $X^\mathsf{T}\mathbf{1} = N\mu$. Trên train: $\Vert X_c \Vert_F^2 = 3.591 - 103{,}2 = 3.487{,}8$. Code tính trực tiếp tổng bình phương của $X - \mu$ và cho cùng kết quả.

### 2.6. Vì sao phải tâm hóa

Nếu làm SVD thẳng trên $X$ (cách làm này còn gọi là LSA), hướng $v_1$ cực đại $\Vert Xw \Vert^2$ chứ không phải phương sai. Hai đại lượng liên hệ qua

$$
\frac{1}{N}X^\mathsf{T}X = \frac{1}{N}X_c^\mathsf{T}X_c + \mu\mu^\mathsf{T}. \qquad (17)
$$

*Chứng minh.* $X = X_c + \mathbf{1}\mu^\mathsf{T}$, và các số hạng chéo bằng 0 vì $X_c^\mathsf{T}\mathbf{1} = N\mu - N\mu = 0$. $∎$

Số hạng $\mu\mu^\mathsf{T}$ có hạng không quá một và tác động theo hướng trung bình $\mu$. Với mọi hướng đơn vị $w$, công thức (17) cho

$$
\frac{1}{N}\Vert Xw \Vert^2 = \frac{N-1}{N}w^\mathsf{T}Cw + (w^\mathsf{T}\mu)^2. \qquad (18)
$$

Vì vậy SVD không tâm hóa tối ưu moment bậc hai quanh gốc, còn PCA tối ưu phương sai quanh trung bình. Nếu số hạng trung bình đủ lớn, hướng kỳ dị đầu có thể nghiêng mạnh về phía $\mu$; tính không âm của TF-IDF không bảo đảm điều này luôn xảy ra. Với $X$ không âm và khác 0, có thể chọn $u_1, v_1$ không âm, nên các điểm chiếu đầu tiên đều không âm; nhưng độ lớn của chúng vẫn có thể giúp phân biệt các nhóm tin nhắn.

Tâm hóa là trừ vị trí trung bình, không phải chiếu bỏ hướng $\mu$, nên các thành phần PCA không nhất thiết trực giao với $\mu$. Trên dữ liệu của đồ án, notebook đo được $\vert \cos(v_1^{\mathrm{SVD}}, \mu) \vert = 0{,}989$: hướng mạnh nhất của SVD gần như chính là hướng của "tin nhắn trung bình".

---

## PHẦN 3: ĐỊNH LÝ ECKART–YOUNG–MIRSKY

### 3.1. Phát biểu

Cho $A = \sum_{i=1}^{r}\sigma_iu_iv_i^\mathsf{T}$ và $k \lt r$. Đặt

$$
A_k = \sum_{i=1}^{k}\sigma_iu_iv_i^\mathsf{T} = U_k\Sigma_kV_k^\mathsf{T}. \qquad (19)
$$

**Định lý 5 (Eckart–Young 1936, Mirsky 1960).** Với mọi $B$ có $\mathrm{rank}(B) \le k$,

$$
\Vert A - B \Vert_F^2 \ \ge \ \Vert A - A_k \Vert_F^2 \ = \ \sum_{i=k+1}^{r}\sigma_i^2. \qquad (20)
$$

Đây chính là **loss** của SVD trong đồ án: sai số tái tạo khi chỉ giữ $k$ chiều. Định lý nói rằng không có ma trận hạng $k$ nào có loss nhỏ hơn $A_k$.

### 3.2. Chứng minh đẳng thức

$A - A_k = \sum_{i \gt k}\sigma_iu_iv_i^\mathsf{T}$ là tổng các ma trận hạng một trực giao, nên theo lập luận dẫn đến (7), $\Vert A - A_k \Vert_F^2 = \sum_{i \gt k}\sigma_i^2$. $∎$

Ngoài ra $A_k = AV_kV_k^\mathsf{T}$, vì $AV_k = U_k\Sigma_k$ theo (2). Tức là $A_k$ là phép chiếu trực giao các hàng của $A$ lên $\mathrm{span}(v_1, \dots, v_k)$.

**Đồng nhất thức phần dư.** Với mọi $V \in \mathbb{R}^{n \times k}$ có cột trực chuẩn,

$$
\Vert A - AVV^\mathsf{T} \Vert_F^2 = \Vert A \Vert_F^2 - \Vert AV \Vert_F^2. \qquad (21)
$$

*Chứng minh.* $P = VV^\mathsf{T}$ là phép chiếu trực giao. Với mỗi hàng $a^\mathsf{T}$ của $A$, theo Pythagoras $\Vert a \Vert^2 = \Vert Pa \Vert^2 + \Vert a - Pa \Vert^2$ và $\Vert Pa \Vert = \Vert V^\mathsf{T}a \Vert$; cộng theo mọi hàng. $∎$

**Trong code.** `loss()` tính sai số tái tạo bằng (21): $\Vert X \Vert_F^2 - \sum_{j \le k}\sigma_j^2$, không cần lập ma trận phần dư. Lưu ý (21) đúng với **mọi** $V$ trực chuẩn, nên so (21) với cách tính trực tiếp không kiểm tra được SVD có đúng hay không. Việc kiểm tra SVD đúng dựa vào phần dư $\Vert Av_j - \hat\lambda_jv_j \Vert$ (mục 4.4) và các test đối chiếu với `np.linalg.svd`/`eigh`. Phép so với 100 hướng ngẫu nhiên trong notebook chỉ minh họa (20): không gian SVD cho loss nhỏ hơn hẳn (2.043 so với 3.232).

### 3.3. Chứng minh tính tối ưu

**Bổ đề 3 (thương Rayleigh trên không gian con).** Cho $S$ đối xứng cỡ $n \times n$ với trị riêng $\lambda_1 \ge \dots \ge \lambda_n$ và vectơ riêng trực chuẩn $q_1, \dots, q_n$. Nếu $W$ là không gian con có $\dim W = n - k$, thì $W$ chứa một hệ trực chuẩn $w_1, \dots, w_{n-k}$ với

$$
\sum_{l=1}^{n-k}w_l^\mathsf{T}Sw_l \ \ge \ \sum_{l=1}^{n-k}\lambda_{k+l}.
$$

*Chứng minh.* Quy nạp theo $l$. Giả sử đã chọn được $w_1, \dots, w_{l-1}$ trực chuẩn trong $W$. Không gian $W_l = W \cap \lbrace w_1, \dots, w_{l-1} \rbrace^\perp$ có số chiều $\ge n - k - l + 1$, còn $E_l = \mathrm{span}(q_1, \dots, q_{k+l})$ có số chiều $k + l$. Tổng hai số chiều lớn hơn $n$, nên $W_l \cap E_l$ chứa một vectơ đơn vị $w_l$. Viết $w_l = \sum_{i \le k+l}c_iq_i$ với $\sum c_i^2 = 1$, ta có $w_l^\mathsf{T}Sw_l = \sum_{i \le k+l}\lambda_ic_i^2 \ge \lambda_{k+l}$. Cộng các bất đẳng thức với $l = 1, \dots, n - k$. $∎$

*Chứng minh Định lý 5.* Lấy $B$ có $\mathrm{rank}(B) \le k$. Hạt nhân $\ker B$ có số chiều $\ge n - k$; chọn $W \subset \ker B$ với $\dim W = n - k$. Với $w \in W$ thì $Bw = 0$ nên $(A - B)w = Aw$. Gọi $w_1, \dots, w_{n-k}$ là hệ trong Bổ đề 3 (với $S = A^\mathsf{T}A$, trị riêng $\sigma_1^2 \ge \dots \ge \sigma_n^2$) và bổ sung thành cơ sở trực chuẩn $w_1, \dots, w_n$ của $\mathbb{R}^n$. Chuẩn Frobenius không đổi khi đổi cơ sở trực chuẩn, nên

$$
\Vert A - B \Vert_F^2 = \sum_{l=1}^{n}\Vert (A - B)w_l \Vert^2 \ \ge \ \sum_{l=1}^{n-k}\Vert Aw_l \Vert^2 = \sum_{l=1}^{n-k}w_l^\mathsf{T}A^\mathsf{T}Aw_l \ \ge \ \sum_{i=k+1}^{n}\sigma_i^2 = \sum_{i=k+1}^{r}\sigma_i^2. \qquad (22)
$$

$∎$

**Nhận xét.** Chứng minh chỉ dùng hai sự kiện: $B$ hạng thấp nên có hạt nhân lớn, và trên hạt nhân đó $A - B$ trùng với $A$. Vì vậy không ma trận hạng $k$ nào tránh được phần năng lượng $\sum_{i \gt k}\sigma_i^2$.

### 3.4. Ý nghĩa đối với đồ án

- Áp dụng cho $X$: trong mọi không gian con $k$ chiều, phép chiếu lên $\mathrm{span}(v_1, \dots, v_k)$ giữ được nhiều năng lượng nhất. **Score** của SVD là phần năng lượng đó, $\sum_{i \le k}\sigma_i^2 / \Vert X \Vert_F^2$.
- Áp dụng cho $X_c$: PCA giữ nhiều phương sai nhất (score (15)).
- Số liệu trên train: score của SVD là 10,6% / 28,9% / 43,1% và của PCA là 8,2% / 26,9% / 41,5% với k = 10 / 50 / 100. Ở k = 100, loss của SVD là 2.043, trong khi chiếu lên 100 hướng ngẫu nhiên cho loss 3.232. Trong các phép chiếu trực giao cùng số chiều, không gian SVD cho loss nhỏ nhất; nếu có các $\sigma$ bằng nhau ở ranh giới $k$ thì không gian tối ưu không duy nhất.
- Định lý chỉ nói về sai số tái tạo, **không nói gì về phân loại**. Một hướng mang ít năng lượng vẫn có thể phân biệt spam tốt. Vì vậy $k$ phải được chọn bằng score của bộ phân loại trên validation (Phần 7), không phải bằng năng lượng.

---

## PHẦN 4: TÍNH $V_k$ BẰNG ORTHOGONAL ITERATION

Phần này giải thích hàm `top_eigenvectors` trong [src/decomposition.py](src/decomposition.py): tìm $k$ vectơ riêng lớn nhất của $A = X^\mathsf{T}X$ (hoặc $X_c^\mathsf{T}X_c$) chỉ bằng phép nhân ma trận và phân tích QR.

### 4.1. Power iteration (k = 1)

Viết vectơ khởi tạo ngẫu nhiên $q_0 = \sum_j c_jv_j$ theo các vectơ riêng của $A$. Nhân $t$ lần với $A$:

$$
A^tq_0 = \sum_j c_j\lambda_j^tv_j = \lambda_1^t\Big(c_1v_1 + \sum_{j \ge 2}c_j\Big(\frac{\lambda_j}{\lambda_1}\Big)^tv_j\Big). \qquad (23)
$$

Nếu $\lambda_1 \gt \lambda_2$ và $c_1 \ne 0$ (xảy ra với xác suất 1 khi $q_0$ ngẫu nhiên), sau khi chuẩn hóa, $A^tq_0$ tiến về $\pm v_1$ với tốc độ $(\lambda_2/\lambda_1)^t$.

### 4.2. Orthogonal iteration (k vectơ)

Bắt đầu từ $Q_0 \in \mathbb{R}^{D \times k}$ có cột trực chuẩn ngẫu nhiên, lặp lại phép phân tích QR của $AQ_t$:

$$
Q_{t+1}R_{t+1} = AQ_t. \qquad (24)
$$

Giả sử $\lambda_k \gt \lambda_{k+1}$ và $V_k^\mathsf{T}Q_0$ khả nghịch (xảy ra với xác suất 1 khi $Q_0$ ngẫu nhiên). Khi đó mọi $R_t$ khả nghịch nên $\mathrm{span}(Q_t) = \mathrm{span}(A^tQ_0)$, và theo lập luận (23), không gian con này tiến về $\mathrm{span}(v_1, \dots, v_k)$ với tốc độ $(\lambda_{k+1}/\lambda_k)^t$. Hơn nữa, $j$ cột đầu của $Q_t$ chính là kết quả của cùng phép lặp với $j$ cột (vì $R_t$ tam giác trên), nên khi các trị riêng phân biệt, cột thứ $j$ tiến về $\pm v_j$. QR có hai vai trò: giữ các cột trực chuẩn, và ngăn mọi cột cùng đổ về $v_1$ như khi chỉ nhân $A$ liên tiếp. Code còn đổi dấu cột theo dấu đường chéo của $R$ để các cột không đổi dấu qua các vòng.

### 4.3. Ước lượng trị riêng và điều kiện dừng

Sau mỗi vòng, trị riêng được ước lượng bằng thương Rayleigh $\hat\lambda_j = q_j^\mathsf{T}Aq_j$. Vòng lặp dừng khi

$$
\max_j \big\vert \hat\lambda_j^{(t)} - \hat\lambda_j^{(t-1)} \big\vert \lt 10^{-8} \hat\lambda_1, \qquad (25)
$$

hoặc khi đã chạy đủ 5.000 vòng; trường hợp sau sẽ in cảnh báo. Khi đó $\sigma_j = \sqrt{\hat\lambda_j}$ và $V_k = Q$.

Thương Rayleigh chính xác hơn vectơ: nếu $q = (\cos\theta) v_j + (\sin\theta) v_\perp$ với $v_\perp$ là vectơ riêng khác ứng với $\lambda_\perp$, thì

$$
q^\mathsf{T}Aq - \lambda_j = \sin^2\theta \cdot (\lambda_\perp - \lambda_j), \qquad (26)
$$

tức sai số trị riêng tỉ lệ với **bình phương** sai số góc. Vì vậy bộ test (chạy với ngưỡng chặt $10^{-14}$ trên ma trận nhỏ $40 \times 8$) so $\sigma$ với NumPy tới $10^{-8}$, còn không gian con chỉ so tới $10^{-6}$.

### 4.4. Áp dụng trên dữ liệu SMS

- Số vòng lặp trên train: SVD cần 263 / 1.352 / 2.414 vòng và PCA cần 341 / 893 / 2.405 vòng với k = 10 / 50 / 100. Ở k = 100 các giá trị kỳ dị đứng cạnh nhau rất sát (chẳng hạn $\sigma_{99} \approx 2{,}888$ và $\sigma_{100} \approx 2{,}875$, nên $\lambda_{100}/\lambda_{99} \approx 0{,}991$). Các tỉ số $\lambda_{j+1}/\lambda_j$ gần 1 và điều kiện dừng (25) theo dõi mọi $\hat\lambda_j$, nên (24) hội tụ chậm.
- Notebook kiểm tra phần dư $\max_j \Vert Av_j - \hat\lambda_jv_j \Vert / \hat\lambda_1$ với $A = X^\mathsf{T}X$ (SVD) hoặc $A = X_c^\mathsf{T}X_c$ (PCA): cỡ $2 \cdot 10^{-5}$ và $3{,}5 \cdot 10^{-5}$.
- Bộ phân loại phía sau chỉ phụ thuộc vào **không gian con** $\mathrm{span}(V_k)$, không phụ thuộc cách chọn cơ sở trực chuẩn trong đó. Nếu $Z' = ZO$ với $O$ là ma trận trực giao, đặt $w' = O^\mathsf{T}w$ thì $Z'w' = Zw$ và $\Vert w' \Vert = \Vert w \Vert$, nên mô hình (29) và loss (30) ở Phần 6 đều không đổi. Do đó các $v_j$ có trị riêng gần bằng nhau có thể xoay lẫn nhau mà không ảnh hưởng kết quả phân loại.

---

## PHẦN 5: ÁP DỤNG CHO DỮ LIỆU VĂN BẢN

### 5.1. Biểu diễn TF-IDF

Tin nhắn được chuyển về chữ thường rồi tách từ bằng regex `\b\w\w+\b`. Từ điển gồm các từ (unigram) xuất hiện trong ít nhất 2 tin train, giữ 1.000 từ có tổng số lần xuất hiện lớn nhất. Với $\mathrm{tf}_{it}$ là số lần từ $t$ xuất hiện trong tin $i$, $\mathrm{df}_t$ là số tin train chứa $t$:

$$
\tilde{x}_{it} = (1 + \log\mathrm{tf}_{it})\Big(\log\frac{1 + N}{1 + \mathrm{df}_t} + 1\Big), \qquad x_i = \frac{\tilde{x}_i}{\Vert \tilde{x}_i \Vert}. \qquad (27)
$$

Công thức của $\tilde{x}_{it}$ dùng khi $\mathrm{tf}_{it} \gt 0$; nếu từ $t$ không có trong tin thì $\tilde{x}_{it} = 0$. Nếu $\tilde{x}_i = 0$ (tin không chứa từ nào trong từ điển) thì giữ $x_i = 0$.

Thừa số $1 + \log\mathrm{tf}$ làm giảm ảnh hưởng của từ lặp nhiều lần; thừa số idf hạ trọng số các từ có trong rất nhiều tin; chuẩn hóa hàng khiến $x_i^\mathsf{T}x_j$ là cosin góc giữa hai tin khác 0. $N$ và $\mathrm{df}_t$ chỉ được tính trên tập fit (mục 2.3). Tin không chứa từ nào trong từ điển thành vectơ 0, nên $\Vert X \Vert_F^2$ bằng số hàng khác 0 chứ không nhất thiết bằng số tin. Trên train, mỗi tin trung bình chỉ dùng khoảng 10 trong 1.000 từ (1,04% phần tử khác 0).

### 5.2. Chiếu tin nhắn mới

Với tin mới $x$ (đã qua cùng bộ TF-IDF), biểu diễn $k$ chiều là

$$
z = V_k^\mathsf{T}x \quad (\mathrm{SVD}), \qquad z = V_k^\mathsf{T}(x - \mu) \quad (\mathrm{PCA}). \qquad (28)
$$

Biểu diễn SVD là ánh xạ tuyến tính theo $x$; biểu diễn PCA là ánh xạ affine theo $x$, tuyến tính theo $x - \mu$. Cả hai chỉ dùng tham số học từ tập fit. Đặt $y = x$ với SVD và $y = x - \mu$ với PCA. Vì $V_k$ có cột trực chuẩn: $\Vert z \Vert \le \Vert y \Vert$ và $z_i^\mathsf{T}z_j = y_i^\mathsf{T}V_kV_k^\mathsf{T}y_j$. Phép đổi tọa độ bảo toàn tích vô hướng của phần đã chiếu vào không gian con, nhưng nói chung không bảo toàn khoảng cách hay góc giữa các vectơ gốc.

Code giữ nguyên thang tọa độ sau khi chiếu (không chuẩn hóa hay "whitening" từng tọa độ) để ba pipeline dùng cùng một mức phạt L2. Logistic Regression trên $z$ học một siêu phẳng trong $\mathbb{R}^k$, tương đương với vectơ trọng số $V_k\beta$ nằm trong $\mathrm{span}(V_k)$ của không gian từ gốc; với PCA, phần $-\beta^\mathsf{T}V_k^\mathsf{T}\mu$ được gộp vào hệ số chặn $b$.

---

## PHẦN 6: LOGISTIC REGRESSION — TỪ LOSS FUNCTION ĐẾN GRADIENT DESCENT

Phần này giải thích lớp `LogisticRegressionGD` trong [src/logistic.py](src/logistic.py). Ở đây $X$ là ma trận đặc trưng đưa vào bộ phân loại: TF-IDF (baseline) hoặc $Z$ (SVD/PCA).

### 6.1. Mô hình

$$
z_i = w^\mathsf{T}x_i + b, \qquad p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}} = P(y_i = 1 \mid x_i). \qquad (29)
$$

Lấy nghịch đảo: $\log\frac{p_i}{1 - p_i} = z_i$, tức mô hình giả sử log-odds của spam là hàm tuyến tính của đặc trưng. Code: `sigmoid`, `predict_proba`.

### 6.2. Loss function: Binary Cross-Entropy + L2

Mô hình cho xác suất của nhãn quan sát được là $P(y_i \mid x_i) = p_i^{y_i}(1 - p_i)^{1 - y_i}$. Cực đại likelihood $\prod_i P(y_i \mid x_i)$ tương đương cực tiểu trung bình của $-\log$ likelihood. Thêm phạt L2 cho $w$ (không phạt $b$):

$$
J(w, b) = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i\log p_i + (1 - y_i)\log(1 - p_i)\Big] + \frac{\lambda}{2N}\Vert w \Vert^2. \qquad (30)
$$

Đồ án dùng $\lambda = 1$, cho cùng hàm mục tiêu với bản trước (bản trước viết mức phạt dưới dạng tham số nghịch đảo $1/\lambda = 1$). Khi tính, $p$ được cắt vào $[10^{-12}, 1 - 10^{-12}]$ để tránh $\log 0$. Tại $w = 0, b = 0$, mọi $p_i = 0{,}5$ nên $J = \log 2 \approx 0{,}6931$. Code: `loss`.

### 6.3. Gradient

Đạo hàm sigmoid: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. Với $\ell_i = -[y_i\log p_i + (1 - y_i)\log(1 - p_i)]$:

$$
\frac{\partial\ell_i}{\partial z_i} = -y_i\frac{p_i(1 - p_i)}{p_i} + (1 - y_i)\frac{p_i(1 - p_i)}{1 - p_i} = p_i - y_i.
$$

Theo quy tắc dây chuyền với $\partial z_i/\partial w = x_i$ và $\partial z_i/\partial b = 1$:

$$
\frac{\partial J}{\partial w} = \frac{1}{N}X^\mathsf{T}(p - y) + \frac{\lambda}{N}w, \qquad \frac{\partial J}{\partial b} = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i). \qquad (31)
$$

Code: `gradient`.

### 6.4. Tính lồi

Gọi $d$ là số cột của $X$ ($d = D = 1.000$ cho baseline, $d = k$ cho SVD/PCA). Đặt $M = [X \ \mathbf{1}]$ và $S = \mathrm{diag}\big(p_i(1 - p_i)\big)$. Ma trận Hessian của $J$ theo $\theta = (w, b)$ là

$$
H = \frac{1}{N}M^\mathsf{T}SM + \frac{\lambda}{N}E, \qquad E = \mathrm{diag}(1, \dots, 1, 0). \qquad (32)
$$

Ở đây $E$ có $d$ số 1 trên đường chéo (ứng với $w$) và một số 0 (ứng với $b$). Cả hai số hạng đều nửa xác định dương ($S$ có đường chéo không âm), nên $J$ là hàm lồi: mọi điểm có gradient bằng 0 đều là cực tiểu toàn cục. Đây là lý do gradient descent tìm được nghiệm tốt nhất, không bị kẹt ở cực tiểu địa phương.

### 6.5. Gradient descent và cách chọn learning rate

Cập nhật

$$
w \leftarrow w - \eta\frac{\partial J}{\partial w}, \qquad b \leftarrow b - \eta\frac{\partial J}{\partial b}, \qquad (33)
$$

bắt đầu từ $w = 0$, $b = 0$. Mỗi lần cập nhật (33) trên toàn bộ tập fit gọi là một **epoch**. Code: `fit`, lưu $J$ sau mỗi epoch vào `loss_history_`.

**Khi nào $J$ chắc chắn giảm?** Gọi $L$ là hằng số Lipschitz của gradient, tức $\lambda_{\max}(H) \le L$ tại mọi điểm. Khai triển Taylor bậc hai cho

$$
J(\theta - \eta\nabla J) \le J(\theta) - \eta\Big(1 - \frac{L\eta}{2}\Big)\Vert\nabla J\Vert^2, \qquad (34)
$$

nên $J$ giảm sau mỗi bước khi $0 \lt \eta \lt 2/L$ (trừ khi đã ở cực tiểu).

**Chặn $L$.** Vì $p(1 - p) \le 1/4$ nên ma trận $\frac{1}{4}I - S$ nửa xác định dương; ma trận $I - E$ cũng vậy. Từ (32):

$$
L \le \frac{1}{4N}\lambda_{\max}(M^\mathsf{T}M) + \frac{\lambda}{N} \le \frac{1}{4N}\mathrm{tr}(M^\mathsf{T}M) + \frac{\lambda}{N} = \frac{1}{4N}\sum_{i=1}^{N}\big(\Vert x_i \Vert^2 + 1\big) + \frac{\lambda}{N} \le \frac{1}{2} + \frac{\lambda}{N}. \qquad (35)
$$

Bước cuối dùng $\Vert x_i \Vert \le 1$ với TF-IDF. Với đặc trưng của SVD và PCA, (35) vẫn đúng vì chỉ cần $\sum_i \Vert z_i \Vert^2 \le N$: với SVD, $\Vert XV_k \Vert_F^2 \le \Vert X \Vert_F^2 \le N$; với PCA, $\Vert X_cV_k \Vert_F^2 \le \Vert X_c \Vert_F^2 \le \Vert X \Vert_F^2 \le N$ theo (16). Vậy $2/L \ge 4/(1 + 2\lambda/N) \approx 4$ cho cả ba pipeline, và đồ án chọn **$\eta = 2$**.

Trên dữ liệu thật, notebook tính cận $\frac{1}{4N}\lambda_{\max}(M^\mathsf{T}M) + \frac{\lambda}{N} = 0{,}2575$ bằng chính `top_eigenvectors` với $k = 1$ (power iteration), tức $J$ chắc chắn giảm khi $\eta \lt 7{,}77$. Thử nghiệm cho thấy $\eta = 50$ làm loss dao động, $\eta = 0{,}5$ giảm chậm, còn $\eta = 2$ giảm đều và nhanh.

### 6.6. Điều kiện dừng và gradient check

Vòng lặp dừng khi $|J^{(t)} - J^{(t-1)}| \lt 10^{-10}$ hoặc sau 20.000 epoch. Với baseline trên train, GD dừng sau 8.298 epoch, $J$ giảm đều từ 0,6931 xuống 0,1524, và $\Vert\partial J/\partial w\Vert \approx 7 \cdot 10^{-6}$ tại điểm dừng. Ba mô hình cuối (fit trên train + validation) dừng sau 7.273–9.765 epoch.

Để chắc công thức (31) đúng, `gradient_check` so nó với sai phân trung tâm

$$
\frac{\partial J}{\partial\theta_j} \approx \frac{J(\theta + \varepsilon e_j) - J(\theta - \varepsilon e_j)}{2\varepsilon}, \qquad \varepsilon = 10^{-6}, \qquad (36)
$$

có sai số $O(\varepsilon^2)$. Trên 50 tin nhắn và 1.001 tham số, sai số lớn nhất là $2{,}1 \cdot 10^{-9}$, nhỏ hơn ngưỡng $10^{-6}$.

---

## PHẦN 7: SCORE FUNCTION

Loss dùng để **học**; score dùng để **đánh giá**. Code: `predict`, `score` trong `src/logistic.py` và [src/metrics.py](src/metrics.py).

### 7.1. Từ xác suất đến nhãn

Dự đoán $\hat y_i = 1$ (spam) nếu $p_i \ge 0{,}5$, tức $z_i \ge 0$; ngược lại $\hat y_i = 0$. Đếm bốn loại kết quả:

| | Dự đoán ham | Dự đoán spam |
|---|---|---|
| **Thật là ham** | TN | FP (báo nhầm) |
| **Thật là spam** | FN (bỏ sót) | TP |

### 7.2. Các độ đo

$$
\mathrm{accuracy} = \frac{TP + TN}{N}, \qquad P = \frac{TP}{TP + FP}, \qquad R = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2PR}{P + R} = \frac{2TP}{2TP + FP + FN}. \qquad (37)
$$

$F_1$ là trung bình điều hòa của precision và recall, nên chỉ cao khi cả hai cùng cao. $F_1^{\mathrm{ham}}$ tính bằng cùng công thức khi coi ham là lớp dương (đổi vai TP ↔ TN, FP ↔ FN). Khi mẫu số bằng 0, code quy ước độ đo bằng 0.

$$
F_1^{\mathrm{macro}} = \frac{F_1^{\mathrm{spam}} + F_1^{\mathrm{ham}}}{2}. \qquad (38)
$$

**Vì sao không dùng accuracy để chọn mô hình?** Spam chỉ chiếm 12,4–12,5% mỗi tập, nên mô hình luôn đoán ham đã đạt accuracy khoảng 87,5% mà $F_1^{\mathrm{spam}} = 0$. Macro-F1 cho hai lớp trọng số bằng nhau nên không che được việc bỏ sót spam.

### 7.3. Ví dụ tính tay

Sáu tin nhắn, 2 spam và 4 ham. Giả sử mô hình bắt được 1 spam, bỏ sót 1 spam và báo nhầm 1 ham: TP = 1, FN = 1, FP = 1, TN = 3.

- accuracy = 4/6 ≈ 0,667;
- spam: P = 1/2, R = 1/2, nên $F_1^{\mathrm{spam}} = 0{,}5$;
- ham: P = 3/4, R = 3/4, nên $F_1^{\mathrm{ham}} = 0{,}75$;
- macro-F1 = (0,5 + 0,75)/2 = 0,625.

Ví dụ này có trong notebook (mục 6.5) và trong `tests/test_metrics.py`.

### 7.4. Chọn k và đánh giá test một lần

$$
k^* = \arg\max_{k \in \lbrace 10, 50, 100 \rbrace} F_1^{\mathrm{macro}}(k). \qquad (39)
$$

Ở đây $F_1^{\mathrm{macro}}(k)$ là macro-F1 trên validation; nếu bằng nhau thì lấy $k$ nhỏ hơn. Sau khi khóa $k^*$, mô hình được fit lại trên train + validation và đánh giá test **đúng một lần**. Test không được dùng để chọn bất kỳ tham số nào, nên điểm test là ước lượng không bị "nhìn trộm" của chất lượng trên dữ liệu mới.

---

## PHẦN 8: KẾT LUẬN VÀ LIÊN HỆ THỰC NGHIỆM

### 8.1. Những gì đã chứng minh

1. **SVD tồn tại với mọi ma trận** (Định lý 1), và chứng minh mang tính xây dựng: vectơ kỳ dị phải là vectơ riêng của $A^\mathsf{T}A$, $\sigma_i = \sqrt{\lambda_i}$.
2. **PCA là SVD của dữ liệu đã tâm hóa** (Định lý 3): trị riêng của hiệp phương sai là $\sigma_i^2/(N-1)$, thành phần đầu cực đại phương sai (Định lý 2), các điểm PCA không tương quan (Mệnh đề 4).
3. **Eckart–Young–Mirsky** (Định lý 5): $A_k$ là xấp xỉ hạng $k$ tốt nhất, với loss $\sum_{i \gt k}\sigma_i^2$.
4. **Orthogonal iteration** (phác thảo, với giả thiết $\lambda_k \gt \lambda_{k+1}$) tiến về không gian con của $k$ vectơ riêng lớn nhất, với tốc độ $(\lambda_{k+1}/\lambda_k)^t$.
5. **Logistic Regression** có loss lồi, gradient dạng $\frac{1}{N}X^\mathsf{T}(p - y) + \frac{\lambda}{N}w$, và gradient descent với $\eta = 2$ chắc chắn làm loss giảm cho cả ba pipeline.
6. **Tâm hóa thay đổi mục tiêu tối ưu** (mục 2.6): SVD tối ưu moment bậc hai quanh gốc, PCA tối ưu phương sai quanh trung bình.

### 8.2. Liên hệ với kết quả thực nghiệm

| Pipeline | Số chiều | Macro-F1 validation | Macro-F1 test | Spam bị bỏ sót (test) |
|---|---:|---:|---:|---:|
| Baseline (TF-IDF) | 1.000 | 0,9215 | 0,9468 | 15/97 |
| SVD | 100 | 0,9065 | 0,9259 | 21/97 |
| PCA | 100 | 0,9027 | 0,9259 | 21/97 |

- **Phổ suy giảm chậm trên dữ liệu này.** 100 chiều (10% số cột) chỉ giữ 43,1% năng lượng (SVD) và 41,5% phương sai (PCA). Đây là quan sát trên ma trận cụ thể; độ thưa tự nó không bảo đảm mọi ma trận văn bản đều như vậy.
- **Tối ưu tái tạo không bảo đảm tối ưu phân loại.** Định lý 5 bảo đảm loss tái tạo nhỏ nhất ở hạng cho trước, nhưng không bảo đảm F1 cao nhất. Trên test, giảm còn 100 chiều làm macro-F1 giảm 0,021, chủ yếu do bỏ sót thêm 6 tin spam. Một giả thuyết, chưa được kiểm định riêng, là một phần tín hiệu phân biệt spam nằm ở các hướng có $\sigma$ nhỏ đã bị cắt.
- **SVD và PCA cho kết quả gần nhau.** Theo (17), $X^\mathsf{T}X$ và $X_c^\mathsf{T}X_c$ khác nhau một hiệu chỉnh hạng không quá một; điều này chỉ ràng buộc các trị riêng đan xen nhau, không bảo đảm hai không gian con hay chất lượng phân loại gần nhau. Việc hai mô hình có cùng confusion matrix trên test là quan sát thực nghiệm. Một lý do có thể là hướng đầu của SVD gần trùng $\mu$ ($\cos = 0{,}989$), còn Logistic Regression đã có hệ số chặn $b$ để xử lý phần trung bình.

### 8.3. Ý nghĩa hình học

SVD cho một cách nhìn thống nhất: mọi ánh xạ tuyến tính $x \mapsto Ax$ là một phép trực giao ($V^\mathsf{T}$, có thể gồm xoay hoặc đối xứng), rồi co giãn theo các trục ($\Sigma$), rồi một phép trực giao nữa ($U$). Phổ $\lbrace\sigma_i\rbrace$ cho biết dữ liệu "dày" theo những hướng nào; PCA chọn các hướng dày nhất quanh tâm, SVD không tâm hóa chọn quanh gốc tọa độ, và Định lý Eckart–Young bảo đảm không có không gian con cùng số chiều nào giữ được nhiều hơn. Phần thực nghiệm minh họa và đối chiếu các quan hệ này trên dữ liệu thật, trong giới hạn sai số số học và điều kiện dừng của vòng lặp.

---

## Bảng tra: công thức → code → notebook

| Công thức | Nội dung | Code | Notebook |
|---|---|---|---|
| (27) | TF-IDF | `TfidfVectorizer.fit`, `transform` — [src/tfidf.py](src/tfidf.py) | mục 3 |
| (3), (24)–(25) | Vectơ riêng của $X^\mathsf{T}X$ bằng orthogonal iteration | `top_eigenvectors` — [src/decomposition.py](src/decomposition.py) | mục 4 |
| (2), (19) | $Z = XV_k$, $X_k = ZV_k^\mathsf{T}$ | `TruncatedSVD.transform`, `inverse_transform` | mục 4 |
| (20)–(21) | Loss tái tạo | `TruncatedSVD.loss` | mục 4–5 |
| (7), (15) | Score năng lượng / phương sai | `TruncatedSVD.score` (PCA kế thừa) | mục 4–5 |
| (8), (13), (16) | Tâm hóa, phương sai thành phần | `PCA.fit`, `explained_variance_` | mục 5 |
| (28) | Chiếu tin nhắn mới | `SpamPipeline.features` — [src/pipeline.py](src/pipeline.py) | mục 8 (demo) |
| (29) | Mô hình | `sigmoid`, `predict_proba` — [src/logistic.py](src/logistic.py) | mục 6.1 |
| (30) | Loss | `loss` | mục 6.2 |
| (31), (36) | Gradient, gradient check | `gradient`, `gradient_check` | mục 6.3 |
| (33)–(35) | Gradient descent, chọn $\eta$ | `fit` | mục 6.4 |
| (37)–(38) | Score function | `predict`, `score`; `confusion_matrix`, `classification_scores` — [src/metrics.py](src/metrics.py) | mục 6.5 |
| (39) | Chọn $k$, test một lần | `run_validation_sweep`, `choose_k`, `refit_and_test` — [src/train.py](src/train.py) | mục 7–8 |

Kiểm thử tương ứng nằm trong [tests/](tests/): `test_decomposition.py` (Phần 1–4), `test_tfidf.py` (5.1), `test_logistic.py` (Phần 6), `test_metrics.py` (Phần 7), `test_data.py` và `test_pipeline.py` (chia tập, chọn $k$, lưu và đọc model).

---

## TÀI LIỆU THAM KHẢO

1. C. Eckart, G. Young. *The approximation of one matrix by another of lower rank.* Psychometrika 1(3), 1936, 211–218.
2. L. Mirsky. *Symmetric gauge functions and unitarily invariant norms.* Quarterly Journal of Mathematics 11, 1960, 50–59.
3. L. N. Trefethen, D. Bau. *Numerical Linear Algebra.* SIAM, 1997: SVD, QR, power iteration và orthogonal (simultaneous) iteration.
4. G. Strang. *Linear Algebra and Learning from Data.* Wellesley-Cambridge Press, 2019, chương I.8–I.9.
5. C. M. Bishop. *Pattern Recognition and Machine Learning.* Springer, 2006: mục 4.3 (Logistic Regression), chương 12 (PCA).
6. S. Boyd, L. Vandenberghe. *Convex Optimization.* Cambridge University Press, 2004, chương 9 (gradient descent, điều kiện giảm (34)).
7. C. D. Manning, P. Raghavan, H. Schütze. *Introduction to Information Retrieval.* Cambridge University Press, 2008, chương 6 (TF-IDF).
8. T. A. Almeida, J. M. Gómez Hidalgo, A. Yamakami. *Contributions to the study of SMS spam filtering: new collection and results.* ACM DocEng 2011. Dữ liệu: UCI Machine Learning Repository, SMS Spam Collection.
