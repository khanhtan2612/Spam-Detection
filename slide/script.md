# SINGULAR VALUE DECOMPOSITION (SVD) VÀ PRINCIPAL COMPONENT ANALYSIS (PCA): CƠ SỞ TOÁN HỌC VÀ ỨNG DỤNG CHO BÀI TOÁN PHÁT HIỆN TIN NHẮN RÁC

*Chứng minh dựa trên chéo hóa trực giao ma trận đối xứng và định lý Eckart–Young–Mirsky*

Tài liệu này trình bày phần lý thuyết đứng sau notebook `spam_svd_pca.ipynb`. Mỗi kết quả được chứng minh đầy đủ, sau đó chỉ rõ nó được kiểm chứng ở đâu trong notebook.

Ký hiệu chung: ma trận dữ liệu $X \in\mathbb{R}^{n \times d}$ có $n$ hàng (tài liệu) và $d$ cột (từ). $\mathbf{1} \in\mathbb{R}^{n}$ là vectơ toàn số 1. $\Vert A \Vert_F^2 = \sum_{i,j} a_{ij}^2 = \mathrm{tr}(A^\mathsf{T}A)$ là bình phương chuẩn Frobenius. Với ma trận đối xứng, "chéo hóa trực giao" nghĩa là viết $S = Q\Lambda Q^\mathsf{T}$ với $Q$ trực giao và $\Lambda$ chéo.

---

## PHẦN 1: SINGULAR VALUE DECOMPOSITION (SVD)

### 1.1. Định nghĩa

Cho $A \in\mathbb{R}^{m \times n}$ bất kỳ. Một **phân tích giá trị kỳ dị** của $A$ là cách viết

$$
A = U \Sigma V^\mathsf{T}, \qquad (1) $$

trong đó $U \in\mathbb{R}^{m \times m}$ và $V \in\mathbb{R}^{n \times n}$ là các ma trận trực giao ($U^\mathsf{T}U = I_m$, $V^\mathsf{T}V = I_n$), còn $\Sigma\in\mathbb{R}^{m \times n}$ chỉ có các phần tử khác 0 trên đường chéo chính, $\Sigma_{ii} = \sigma_i$, với

$$
\sigma_1 \ge\sigma_2 \ge\dots\ge\sigma_{p} \ge 0, \qquad p = \min(m, n).
$$

Các số $\sigma_i$ gọi là **giá trị kỳ dị**; các cột $u_i$ của $U$ và $v_i$ của $V$ gọi là vectơ kỳ dị trái và phải. Từ (1) suy ra ngay hai hệ thức đặc trưng

$$
A v_i = \sigma_i u_i, \qquad A^\mathsf{T} u_i = \sigma_i v_i, \qquad i = 1, \dots, p. \qquad (2) $$

### 1.2. Hai bổ đề về $A^\mathsf{T}A$

**Bổ đề 1.** $A^\mathsf{T}A \in\mathbb{R}^{n \times n}$ là ma trận đối xứng và nửa xác định dương.

*Chứng minh.* Đối xứng: $(A^\mathsf{T}A)^\mathsf{T} = A^\mathsf{T}(A^\mathsf{T})^\mathsf{T} = A^\mathsf{T}A$. Nửa xác định dương: với mọi $x \in\mathbb{R}^n$,

$$
x^\mathsf{T}(A^\mathsf{T}A)x = (Ax)^\mathsf{T}(Ax) = \Vert Ax \Vert^2 \ge 0. \qquad∎
$$

**Bổ đề 2 (Spectral theorem).** Mọi ma trận đối xứng thực $S \in\mathbb{R}^{n \times n}$ đều chéo hóa được bằng ma trận trực giao: tồn tại $Q$ trực giao và $\Lambda= \mathrm{diag}(\lambda_1, \dots, \lambda_n)$ với $\lambda_i \in\mathbb{R}$ sao cho $S = Q\Lambda Q^\mathsf{T}$. Nếu $S$ nửa xác định dương thì $\lambda_i \ge 0$ với mọi $i$.

*Chứng minh phần cuối.* Với $q_i$ là cột thứ $i$ của $Q$: $\lambda_i = q_i^\mathsf{T} S q_i \ge 0$ theo định nghĩa nửa xác định dương. (Phần đầu là định lý phổ cho ma trận đối xứng thực, được coi là đã biết.) $∎$

### 1.3. Định lý tồn tại SVD (chứng minh xây dựng)

**Định lý 1.** Mọi ma trận $A \in\mathbb{R}^{m \times n}$ đều có một phân tích dạng (1).

*Chứng minh.* Theo Bổ đề 1 và 2, chéo hóa trực giao

$$
A^\mathsf{T}A = V \Lambda V^\mathsf{T}, \qquad\Lambda= \mathrm{diag}(\lambda_1, \dots, \lambda_n), \quad\lambda_1 \ge\dots\ge\lambda_n \ge 0, \qquad (3) $$

với $V = [v_1 \cdots v_n]$ trực giao. Đặt $\sigma_i = \sqrt{\lambda_i}$ và gọi $r$ là số chỉ số có $\sigma_i \gt 0$ (tức $r = \mathrm{rank}(A^\mathsf{T}A) = \mathrm{rank} A$).

**Bước 1** (các vectơ $Av_i$ trực giao với nhau). Với mọi $i, j$,

$$
(Av_i)^\mathsf{T}(Av_j) = v_i^\mathsf{T} (A^\mathsf{T}A) v_j = v_i^\mathsf{T} (\lambda_j v_j) = \lambda_j v_i^\mathsf{T} v_j = \lambda_j \delta_{ij}. \qquad (4) $$

Như vậy $\Vert Av_i \Vert= \sigma_i$, và $Av_i = 0$ khi $i \gt r$.

**Bước 2** (xây dựng $u_i$). Với $i \le r$ đặt

$$
u_i = \frac{A v_i}{\sigma_i}. \qquad (5) $$

Theo (4), $u_i^\mathsf{T} u_j = \frac{\lambda_j \delta_{ij}}{\sigma_i \sigma_j} = \delta_{ij}$, nên $u_1, \dots, u_r$ là hệ trực chuẩn trong $\mathbb{R}^m$. Bổ sung (Gram–Schmidt) thành cơ sở trực chuẩn $u_1, \dots, u_m$ của $\mathbb{R}^m$ và đặt $U = [u_1 \cdots u_m]$.

**Bước 3** (kiểm tra (1)). Ta chứng minh $U^\mathsf{T} A V = \Sigma$. Phần tử $(i, j)$ của vế trái là $u_i^\mathsf{T} A v_j$.

- Nếu $j \le r$: $A v_j = \sigma_j u_j$ theo (5), nên $u_i^\mathsf{T} A v_j = \sigma_j u_i^\mathsf{T} u_j = \sigma_j \delta_{ij}$.
- Nếu $j \gt r$: $A v_j = 0$ theo Bước 1, nên phần tử bằng 0.

Vậy $U^\mathsf{T}AV = \Sigma$, tức $A = U\Sigma V^\mathsf{T}$ vì $U, V$ trực giao. $∎$

Chứng minh trên chính là thuật toán `svd_via_eig` trong notebook: chéo hóa $A^\mathsf{T}A$ bằng `numpy.linalg.eigh`, lấy căn các trị riêng, rồi tính $u_i$ theo (5). Khi $n \gt m$, áp dụng cùng lập luận cho $A^\mathsf{T}$ để chỉ phải chéo hóa ma trận $m \times m$ nhỏ hơn.

**Nhận xét về số học.** Vì tính $\lambda_i = \sigma_i^2$ rồi lấy căn, sai số làm tròn trên $\lambda_i$ (cỡ $\lambda_1 \varepsilon$) trở thành sai số $\sigma_1\sqrt{\varepsilon}$ trên các giá trị kỳ dị nhỏ. Do đó hạng số học được xác định bằng ngưỡng $\lambda_i \gt\lambda_1 \cdot\max(m,n) \varepsilon$ chứ không bằng ngưỡng trên $\sigma_i$. Đây cũng là lý do các thư viện số không dùng cách này cho ma trận điều kiện xấu.

### 1.4. Dạng compact và dạng tổng các ma trận hạng một

Vì $\sigma_i = 0$ khi $i \gt r$, chỉ $r$ cột đầu của $U$ và $V$ đóng góp vào tích (1):

$$
A = U_r \Sigma_r V_r^\mathsf{T}, \qquad U_r \in\mathbb{R}^{m \times r},\ \Sigma_r = \mathrm{diag}(\sigma_1, \dots, \sigma_r),\ V_r \in\mathbb{R}^{n \times r}. \qquad (6) $$

Khai triển tích theo cột cho

$$
A = \sum_{i=1}^{r} \sigma_i u_i v_i^\mathsf{T}. \qquad (7) $$

Mỗi số hạng $u_i v_i^\mathsf{T}$ là một ma trận hạng một có chuẩn Frobenius bằng 1, và các số hạng đôi một trực giao theo tích vô hướng Frobenius: $\langle u_i v_i^\mathsf{T}, u_j v_j^\mathsf{T} \rangle_F = (u_i^\mathsf{T}u_j)(v_i^\mathsf{T}v_j) = \delta_{ij}$. Do đó

$$
\Vert A \Vert_F^2 = \sum_{i=1}^{r} \sigma_i^2. \qquad (8) $$

Notebook kiểm tra (8) trực tiếp trên ma trận TF-IDF ở mục *Phổ giá trị kỳ dị*: tổng bình phương các giá trị kỳ dị đúng bằng $\Vert X\Vert_F^2$, và vì mỗi hàng TF-IDF được chuẩn hóa về độ dài 1, giá trị này bằng số hàng khác 0 của $X$.

---

## PHẦN 2: PRINCIPAL COMPONENT ANALYSIS (PCA)

### 2.1. Tâm hóa và ma trận hiệp phương sai (covariance matrix)

Cho $X \in\mathbb{R}^{n \times d}$, mỗi hàng $x_i^\mathsf{T}$ là một quan sát. Vectơ trung bình và ma trận đã tâm hóa là

$$
\mu= \frac{1}{n} X^\mathsf{T}\mathbf{1} \in\mathbb{R}^d, \qquad X_c = X - \mathbf{1}\mu^\mathsf{T}. \qquad (9) $$

Ma trận hiệp phương sai mẫu

$$
C = \frac{1}{n-1} X_c^\mathsf{T} X_c \in\mathbb{R}^{d \times d} \qquad (10) $$

là đối xứng, nửa xác định dương (Bổ đề 1 áp dụng cho $X_c$). Với một hướng đơn vị $w \in\mathbb{R}^d$, các tọa độ chiếu $z_i = w^\mathsf{T}(x_i - \mu)$ có trung bình 0 và phương sai mẫu

$$
\mathrm{Var}(z) = \frac{1}{n-1}\sum_{i=1}^n \big(w^\mathsf{T}(x_i - \mu)\big)^2 = \frac{1}{n-1}\Vert X_c w\Vert^2 = w^\mathsf{T} C w. \qquad (11) $$

### 2.2. Thành phần chính thứ nhất: bài toán cực đại phương sai

**Định lý 2.** Nghiệm của bài toán

$$
\max_{w \in\mathbb{R}^d} w^\mathsf{T} C w \quad\mathrm{với điều kiện} \quad\Vert w\Vert^2 = 1 \qquad (12) $$

là vectơ riêng $v_1$ ứng với trị riêng lớn nhất $\lambda_1$ của $C$, và giá trị cực đại bằng $\lambda_1$.

*Chứng minh.* Chéo hóa $C = V\Lambda V^\mathsf{T}$ với $\lambda_1 \ge\dots\ge\lambda_d \ge 0$. Đặt $y = V^\mathsf{T}w$, thì $\Vert y\Vert= 1$ và

$$
w^\mathsf{T}Cw = y^\mathsf{T}\Lambda y = \sum_{i=1}^d \lambda_i y_i^2 \le\lambda_1 \sum_i y_i^2 = \lambda_1,
$$

đẳng thức khi $y = e_1$, tức $w = v_1$. $∎$

**Các thành phần tiếp theo.** Thành phần chính thứ $j$ là nghiệm của (12) với ràng buộc bổ sung $w \perp v_1, \dots, v_{j-1}$. Lặp lại lập luận trực tiếp với $y_1 = \dots= y_{j-1} = 0$ cho $w^\mathsf{T}Cw \le\lambda_j$, đạt được tại $w = v_j$. Tổng quát, không gian con $k$ chiều $\mathrm{span}(v_1, \dots, v_k)$ cực đại tổng phương sai $\mathrm{tr}(W^\mathsf{T}CW)$ trên mọi $W \in\mathbb{R}^{d \times k}$ có cột trực chuẩn (định lý Ky Fan; cũng là hệ quả của Định lý 5 ở PHẦN 3 áp dụng cho $X_c$).

### 2.3. PCA là SVD của ma trận đã tâm hóa

**Định lý 3.** Gọi $X_c = U\Sigma V^\mathsf{T}$ là SVD của ma trận đã tâm hóa, với các giá trị kỳ dị $\sigma_1 \ge\dots\ge\sigma_r \gt 0$. Khi đó

$$
C = \frac{1}{n-1} V \Sigma^\mathsf{T}\Sigma V^\mathsf{T}, \qquad (13) $$

nghĩa là các vectơ kỳ dị phải $v_i$ của $X_c$ là các vectơ riêng của $C$ với trị riêng

$$
\lambda_i = \frac{\sigma_i^2}{n-1}. \qquad (14) $$

*Chứng minh.* Thay SVD vào (10): $X_c^\mathsf{T}X_c = V\Sigma^\mathsf{T}U^\mathsf{T}U\Sigma V^\mathsf{T} = V(\Sigma^\mathsf{T}\Sigma)V^\mathsf{T}$ vì $U^\mathsf{T}U = I$. Ma trận $\Sigma^\mathsf{T}\Sigma$ là chéo với các phần tử $\sigma_i^2$ (và 0 khi $i \gt r$). Chia cho $n - 1$ được (13); đây chính là chéo hóa trực giao của $C$ nên các cột của $V$ là vectơ riêng với trị riêng (14). $∎$

Hệ quả thực hành: để làm PCA không cần lập ma trận $C$ kích thước $d \times d$; chỉ cần SVD (hoặc truncated SVD) của $X_c$. Trong notebook, `PCAScratch` làm đúng như vậy và `explained_variance_` được đặt bằng $\sigma_i^2/(n-1)$; mục *Kiểm chứng lý thuyết bằng thực nghiệm* đối chiếu ba đại lượng tính độc lập: trị riêng của $C$ do `PCA(svd_solver="arpack")` của scikit-learn trả về, $\sigma_i^2/(n-1)$ từ ma trận Gram đã tâm hóa, và phương sai mẫu của các điểm chiếu.

### 2.4. Các điểm PCA không tương quan

Điểm PCA (tọa độ của dữ liệu trong cơ sở $k$ thành phần đầu) là

$$
Z = X_c V_k \in\mathbb{R}^{n \times k}, \qquad V_k = [v_1 \cdots v_k]. \qquad (15) $$

**Mệnh đề 4.** Các cột của $Z$ có trung bình 0 và ma trận hiệp phương sai của $Z$ là ma trận chéo

$$
\frac{1}{n-1} Z^\mathsf{T}Z = \frac{1}{n-1}\Sigma_k^2 = \mathrm{diag}(\lambda_1, \dots, \lambda_k). \qquad (16) $$

*Chứng minh.* Trung bình: $\mathbf{1}^\mathsf{T}Z = \mathbf{1}^\mathsf{T}X_cV_k = (\mathbf{1}^\mathsf{T}X - n\mu^\mathsf{T})V_k = 0$ vì $\mathbf{1}^\mathsf{T}X = n\mu^\mathsf{T}$ theo (9). Hiệp phương sai: dùng $X_cV_k = U_k\Sigma_k$ (từ (2)),

$$
Z^\mathsf{T}Z = \Sigma_k U_k^\mathsf{T}U_k\Sigma_k = \Sigma_k^2. \qquad∎
$$

Vậy PCA không chỉ xoay dữ liệu mà còn **khử tương quan**: hai tọa độ bất kỳ trong $Z$ có hiệp phương sai mẫu bằng 0. Notebook kiểm tra phần tử ngoài đường chéo lớn nhất của $\mathrm{Cov}(Z)$ cho 10 thành phần đầu, kết quả cỡ $10^{-18}$.

### 2.5. Tỉ lệ phương sai giải thích

Tổng phương sai của dữ liệu là $\mathrm{tr}(C) = \sum_{i=1}^d \lambda_i = \frac{1}{n-1}\Vert X_c\Vert_F^2$ (dùng (8)). Tỉ lệ phương sai giải thích bởi $k$ thành phần đầu:

$$
\mathrm{EVR}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i} = \frac{\sum_{i=1}^k \sigma_i^2}{\Vert X_c\Vert_F^2}. \qquad (17) $$

Để tính mẫu số mà không tâm hóa tường minh, khai triển

$$
\Vert X_c\Vert_F^2 = \mathrm{tr}\big((X - \mathbf{1}\mu^\mathsf{T})^\mathsf{T}(X - \mathbf{1}\mu^\mathsf{T})\big) = \Vert X\Vert_F^2 - 2n\Vert\mu\Vert^2 + n\Vert\mu\Vert^2 = \Vert X\Vert_F^2 - n\Vert\mu\Vert^2, \qquad (18) $$

trong đó đã dùng $X^\mathsf{T}\mathbf{1} = n\mu$ và $\mathbf{1}^\mathsf{T}\mathbf{1} = n$. Đây là công thức của `total_variance_` trong notebook.

### 2.6. Vì sao phải tâm hóa

Nếu bỏ qua tâm hóa và làm SVD trực tiếp trên $X$ (như LSA), hướng $v_1$ cực đại $\Vert Xw\Vert^2 = \sum_i (w^\mathsf{T}x_i)^2$, tức **moment bậc hai không tâm hóa** $w^\mathsf{T}(\frac{1}{n}X^\mathsf{T}X)w$, chứ không phải phương sai. Hai đại lượng này liên hệ bởi

$$
\frac{1}{n}X^\mathsf{T}X = \frac{1}{n}X_c^\mathsf{T}X_c + \mu\mu^\mathsf{T}. \qquad (19) $$

*Chứng minh.* $X = X_c + \mathbf{1}\mu^\mathsf{T}$ nên $X^\mathsf{T}X = X_c^\mathsf{T}X_c + X_c^\mathsf{T}\mathbf{1}\mu^\mathsf{T} + \mu\mathbf{1}^\mathsf{T}X_c + n\mu\mu^\mathsf{T}$, và hai số hạng giữa bằng 0 vì $X_c^\mathsf{T}\mathbf{1} = X^\mathsf{T}\mathbf{1} - \mu\mathbf{1}^\mathsf{T}\mathbf{1} = n\mu- n\mu= 0$. $∎$

Nhân hai vế của (19) với $w$ đơn vị: $\frac{1}{n}\Vert Xw\Vert^2 = \frac{n-1}{n} w^\mathsf{T}Cw + (w^\mathsf{T}\mu)^2$. Hàm mục tiêu của SVD không tâm hóa vì vậy cộng thêm bình phương tọa độ của trung bình trên hướng $w$. Khi $\Vert\mu\Vert^2$ lớn so với các trị riêng của $C$, số hạng này có thể chi phối và $v_1$ nghiêng về phía $\mu$, khi đó tọa độ thứ nhất chủ yếu đo "độ điển hình" của tài liệu. Đây là một khả năng phụ thuộc dữ liệu chứ không phải hệ quả tất yếu. Riêng với ma trận không âm như TF-IDF, định lý Perron–Frobenius cho phép chọn $u_1, v_1$ không âm, nên mọi tài liệu có cùng dấu trên tọa độ thứ nhất.

Sau tâm hóa, PC1 cực đại phương sai $w^\mathsf{T}Cw$ quanh trung bình: hàm mục tiêu không còn số hạng $(w^\mathsf{T}\mu)^2$. Điều này **không** bảo đảm PC1 trực giao với $\mu$ hay có cosin nhỏ với $\mu$; nó chỉ bỏ $\mu$ ra khỏi tiêu chí chọn hướng. Trên dữ liệu của notebook đo được $|\cos(v_1^{\mathrm{SVD}}, \mu)| = 0{,}883$ với 100% tài liệu cùng dấu trên tọa độ thứ nhất, còn $|\cos(v_1^{\mathrm{PCA}}, \mu)| = 0{,}327$ với 74% tài liệu cùng dấu.

---

## PHẦN 3: ĐỊNH LÝ ECKART–YOUNG–MIRSKY

### 3.1. Phát biểu

Cho $A = \sum_{i=1}^r \sigma_i u_i v_i^\mathsf{T}$ là SVD của $A \in\mathbb{R}^{m \times n}$ và $k \lt r$. Đặt

$$
A_k = \sum_{i=1}^k \sigma_i u_i v_i^\mathsf{T} = U_k\Sigma_kV_k^\mathsf{T}. \qquad (20) $$

**Định lý 5 (Eckart–Young 1936, Mirsky 1960).** Với mọi ma trận $B \in\mathbb{R}^{m \times n}$ có $\mathrm{rank} B \le k$,

$$
\Vert A - B\Vert_F^2 \ \ge\ \Vert A - A_k\Vert_F^2 \ =\ \sum_{i=k+1}^{r}\sigma_i^2. \qquad (21) $$

### 3.2. Chứng minh đẳng thức

$A - A_k = \sum_{i=k+1}^r \sigma_i u_i v_i^\mathsf{T}$ là tổng các ma trận hạng một trực giao, nên theo lập luận dẫn đến (8), $\Vert A - A_k\Vert_F^2 = \sum_{i\gt k}\sigma_i^2$. $∎$

Ngoài ra, $A_k = AV_kV_k^\mathsf{T}$: thật vậy $AV_k = U_k\Sigma_k$ theo (2), nên $AV_kV_k^\mathsf{T} = U_k\Sigma_kV_k^\mathsf{T} = A_k$. Do đó $A_k$ là **phép chiếu trực giao các hàng của $A$** lên không gian con $\mathrm{span}(v_1, \dots, v_k)$.

**Đồng nhất thức phần dư.** Với mọi $V \in \mathbb{R}^{n \times k}$ có cột trực chuẩn,

$$
\Vert A - AVV^\mathsf{T}\Vert_F^2 = \Vert A\Vert_F^2 - \Vert AV\Vert_F^2. \qquad (22) $$

*Chứng minh.* $P = VV^\mathsf{T}$ là phép chiếu trực giao ($P^\mathsf{T} = P$, $P^2 = P$). Với mỗi hàng $a^\mathsf{T}$ của $A$, hai vectơ $Pa$ và $a - Pa$ trực giao nên theo Pythagoras $\Vert a\Vert^2 = \Vert Pa\Vert^2 + \Vert a - Pa\Vert^2$, và $\Vert Pa\Vert^2 = a^\mathsf{T}VV^\mathsf{T}a = \Vert V^\mathsf{T}a\Vert^2$; cộng theo mọi hàng được (22). $∎$

Notebook đo sai số chiếu bằng (22): chỉ cần $\Vert X\Vert_F^2$ và tích thưa–đặc $XV$, không phải tạo ma trận phần dư đặc; kết quả được đối chiếu với cách tính trực tiếp trên một khối 300 hàng rồi so với $\sum_{i\gt k}\sigma_i^2$.

### 3.3. Chứng minh tính tối ưu

Trước hết là một bổ đề về trị riêng.

**Bổ đề 3 (bất đẳng thức giao thoa Weyl, dạng cần dùng).** Cho $S \in\mathbb{R}^{n \times n}$ đối xứng với trị riêng $\lambda_1(S) \ge\dots\ge\lambda_n(S)$, và $W \subset\mathbb{R}^n$ là không gian con có số chiều $n - k$. Khi đó tồn tại vectơ đơn vị $w \in W$ với $w^\mathsf{T}Sw \ge\lambda_{k+1}(S)$. Tổng quát hơn, với $j \ge 1$ bất kỳ, $W$ chứa một hệ trực chuẩn $w_1, \dots, w_j$ sao cho $\sum_{l=1}^{j} w_l^\mathsf{T}Sw_l \ge\sum_{l=1}^{j}\lambda_{k+l}(S)$ (khi $j \le n - k$).

*Chứng minh.* Gọi $q_1, \dots, q_n$ là cơ sở trực chuẩn các vectơ riêng của $S$. Không gian $E = \mathrm{span}(q_1, \dots, q_{k+1})$ có số chiều $k + 1$, còn $\dim W = n - k$, nên $\dim(W \cap E) \ge (k+1) + (n-k) - n = 1$. Lấy $w \in W \cap E$ đơn vị, viết $w = \sum_{i \le k+1} c_i q_i$ với $\sum c_i^2 = 1$; khi đó $w^\mathsf{T}Sw = \sum_{i \le k+1}\lambda_i c_i^2 \ge\lambda_{k+1}$. Phát biểu tổng quát được chứng minh bằng quy nạp theo $j$: sau khi chọn $w_1$, lặp lại lập luận trong $W \cap w_1^\perp$ (số chiều $n - k - 1$) với $E' = \mathrm{span}(q_1, \dots, q_{k+2})$, v.v. $∎$

*Chứng minh Định lý 5.* Lấy $B$ bất kỳ với $\mathrm{rank} B \le k$. Không gian nghiệm $N = \ker B \subset\mathbb{R}^n$ có $\dim N = n - \mathrm{rank}B \ge n - k$. Chọn một không gian con $W \subset N$ với $\dim W = n - k$ đúng. Với mọi $w \in W$ ta có $Bw = 0$, do đó $(A - B)w = Aw$.

Gọi $w_1, \dots, w_{n-k}$ là cơ sở trực chuẩn của $W$ và bổ sung thành cơ sở trực chuẩn $w_1, \dots, w_n$ của $\mathbb{R}^n$. Vì chuẩn Frobenius bất biến qua phép đổi cơ sở trực chuẩn,

$$
\Vert A - B\Vert_F^2 = \sum_{l=1}^{n}\Vert (A-B)w_l\Vert^2 \ \ge\ \sum_{l=1}^{n-k}\Vert (A-B)w_l\Vert^2 = \sum_{l=1}^{n-k}\Vert Aw_l\Vert^2 = \sum_{l=1}^{n-k} w_l^\mathsf{T}(A^\mathsf{T}A)w_l. \qquad (23) $$

Áp dụng Bổ đề 3 cho $S = A^\mathsf{T}A$ (trị riêng $\sigma_1^2 \ge\dots\ge\sigma_n^2$, với $\sigma_i = 0$ khi $i \gt r$) và không gian $W$ số chiều $n - k$, với $j = n - k$: có thể chọn cơ sở trực chuẩn $w_l$ của $W$ sao cho

$$
\sum_{l=1}^{n-k} w_l^\mathsf{T}(A^\mathsf{T}A)w_l \ \ge\ \sum_{l=1}^{n-k}\sigma_{k+l}^2 = \sum_{i=k+1}^{n}\sigma_i^2 = \sum_{i=k+1}^{r}\sigma_i^2. \qquad (24) $$

Ghép (23) với (24) được $\Vert A - B\Vert_F^2 \ge\sum_{i\gt k}\sigma_i^2 = \Vert A - A_k\Vert_F^2$. $∎$

**Nhận xét.** Chứng minh chỉ dùng hai sự kiện: $B$ có hạt nhân lớn (vì hạng thấp), và trên hạt nhân đó $A - B$ trùng với $A$. Vì vậy không ma trận hạng $k$ nào "tránh" được phần năng lượng $\sum_{i\gt k}\sigma_i^2$ nằm ở các hướng kỳ dị sau. Định lý cũng đúng với chuẩn phổ $\Vert\cdot\Vert_2$ (khi đó sai số tối ưu là $\sigma_{k+1}$), nhưng notebook chỉ dùng chuẩn Frobenius.

### 3.4. Ý nghĩa đối với PCA và LSA

- Áp dụng Định lý 5 cho $X_c$: trong mọi không gian con $k$ chiều, phép chiếu lên $\mathrm{span}(v_1, \dots, v_k)$ của PCA giữ lại nhiều "năng lượng" $\Vert X_cV_kV_k^\mathsf{T}\Vert_F^2 = \sum_{i\le k}\sigma_i^2$ nhất, tức là giữ lại nhiều phương sai nhất (sau khi nhân với $\frac{1}{n-1}$). Đây chính là phát biểu "PCA tối ưu theo nghĩa tái tạo".
- Áp dụng cho $X$ không tâm hóa: LSA với $k$ chủ đề là xấp xỉ hạng $k$ tốt nhất của ma trận TF-IDF.
- Notebook so sánh sai số của ba không gian con cùng số chiều: không gian SVD, không gian ngẫu nhiên, và không gian tọa độ gồm $k$ từ có chuẩn cột lớn nhất (tức là *chọn* đặc trưng thay vì *tổ hợp* đặc trưng). Định lý bảo đảm không gian SVD luôn thắng. Trong thí nghiệm, khoảng cách với không gian ngẫu nhiên lớn ở mọi $k$; khoảng cách với phép chọn từ nhỏ ở $k$ nhỏ (khoảng 1% tại $k = 10$) và tăng dần theo $k$ (990 so với 1530 tại $k = 1000$).
- Định lý nói về sai số tái tạo theo chuẩn Frobenius, không nói gì về khả năng phân loại: một hướng mang ít năng lượng vẫn có thể phân biệt hai lớp tốt. Vì vậy không thể suy ra từ Định lý 5 rằng $k$ tối ưu cho tái tạo cũng tối ưu cho phân loại.

---

## PHẦN 4: ÁP DỤNG CHO DỮ LIỆU VĂN BẢN

### 4.1. Biểu diễn TF-IDF

Mỗi tin nhắn $i$ được biểu diễn bởi vectơ $x_i \in\mathbb{R}^d$ trên từ điển gồm $d$ unigram và bigram. Với $\mathrm{tf}_{it}$ là số lần từ $t$ xuất hiện trong tin $i$ và $\mathrm{df}_t$ là số tin chứa $t$,

$$
\tilde{x}_{it} = \big(1 + \log\mathrm{tf}_{it}\big)\cdot\Big(\log\frac{1+n}{1+\mathrm{df}_t} + 1\Big), \qquad x_i = \frac{\tilde{x}_i}{\Vert\tilde{x}_i\Vert}. \qquad (25) $$

Thừa số $1 + \log\mathrm{tf}$ (sublinear tf) làm giảm ảnh hưởng của từ lặp nhiều lần; thừa số idf hạ trọng số các từ xuất hiện ở hầu hết tài liệu; chuẩn hóa hàng khiến $\Vert x_i\Vert= 1$ nên tích vô hướng $x_i^\mathsf{T}x_j$ là cosin góc giữa hai tin. Hệ quả: $\Vert X\Vert_F^2$ bằng số tài liệu khác rỗng, và theo (8) đó cũng là $\sum_i\sigma_i^2$.

Một tài liệu có thể thành hàng rỗng sau TF-IDF nếu mọi từ của nó là stop word hoặc xuất hiện ở ít hơn `min_df` tài liệu; trong notebook có 45 hàng như vậy trong $X_\mathrm{train}$, nên $\Vert X\Vert_F^2 = 4031$ thay vì $4076$.

Ma trận $X$ rất thưa (chỉ khoảng 0,16% phần tử khác 0) nên được lưu ở dạng CSR; mọi thuật toán phía sau chỉ dùng các tích $XM$ và $X^\mathsf{T}M$.

### 4.2. LSA và phép chiếu dữ liệu mới

Latent Semantic Analysis (LSA) là truncated SVD của $X$: $X \approx U_k\Sigma_kV_k^\mathsf{T}$. Các cột của $V_k$ là $k$ hướng trong không gian từ ("chủ đề"), và tài liệu $i$ được biểu diễn bởi hàng thứ $i$ của $U_k\Sigma_k = XV_k$.

Với một tin nhắn mới $x \in\mathbb{R}^d$ (đã qua cùng bộ TF-IDF), biểu diễn thấp chiều là phép chiếu

$$
z = V_k^\mathsf{T}x \in\mathbb{R}^k \qquad\mathrm{(LSA)}, \qquad z = V_k^\mathsf{T}(x - \mu) \in\mathbb{R}^k \qquad\mathrm{(PCA)}. \qquad (26) $$

LSA là ánh xạ **tuyến tính** theo $x$; PCA là ánh xạ **affine** theo $x$ (tuyến tính theo $x - \mu$). Cả hai chỉ cần $V_k$ (và $\mu$) học từ tập huấn luyện. Vì các cột của $V_k$ trực chuẩn, $V_kV_k^\mathsf{T}$ là phép chiếu trực giao, nên

- với LSA: $\Vert z\Vert \le \Vert x\Vert$ và $z_i^\mathsf{T}z_j = x_i^\mathsf{T}V_kV_k^\mathsf{T}x_j$;
- với PCA: $\Vert z\Vert \le \Vert x - \mu\Vert$ và $z_i^\mathsf{T}z_j = (x_i - \mu)^\mathsf{T}V_kV_k^\mathsf{T}(x_j - \mu)$.

Trên tập huấn luyện, cột thứ $i$ của ma trận điểm chiếu có chuẩn $\sigma_i$ (vì $XV_k = U_k\Sigma_k$, và với PCA độ lệch chuẩn là $\sigma_i/\sqrt{n-1}$ theo (16)). Notebook **không** chuẩn hóa lại các tọa độ sau khi chiếu: làm vậy sẽ đưa các hướng có $\sigma_i$ nhỏ lên cùng thang với các hướng chính và thay đổi hình học mà phép chiếu trực giao giữ lại.

Bộ phân loại phía sau là logistic regression trên $z$; nó học một siêu phẳng trong $\mathbb{R}^k$, tương đương với siêu phẳng $w = V_k\beta$ nằm trong $\mathrm{span}(V_k)$ của không gian từ gốc. Giảm $k$ do đó là một dạng điều chuẩn: mô hình bị ép chỉ dùng các tổ hợp từ mang nhiều năng lượng. Với PCA, phần $-\beta^\mathsf{T}V_k^\mathsf{T}\mu$ được gộp vào hệ số chặn.

### 4.3. Thủ thuật Gram khi $n \lt d$

Khi số tài liệu $n$ nhỏ hơn số từ $d$ (trong notebook $n = 4076 \lt d = 5445$), $X$ có không quá $n$ giá trị kỳ dị khác 0 và chúng là căn bậc hai các trị riêng của ma trận Gram $G = XX^\mathsf{T} \in\mathbb{R}^{n\times n}$:

$$
XX^\mathsf{T} = U\Sigma V^\mathsf{T}V\Sigma^\mathsf{T}U^\mathsf{T} = U(\Sigma\Sigma^\mathsf{T})U^\mathsf{T}. \qquad (27) $$

Chéo hóa $G$ cho ngay $U$ và $\sigma_i$; các vectơ kỳ dị phải thu được từ (2): $v_i = X^\mathsf{T}u_i/\sigma_i$. Chi phí là $O(n^2 \cdot\mathrm{nnz}/n)$ để lập $G$ (tích hai ma trận thưa) cộng $O(n^3)$ cho `eigh`, không phụ thuộc $d$. Cách này cho **toàn bộ phổ chính xác**. Với PCA, ma trận Gram đã tâm hóa có công thức đóng (xem 4.4), nên cùng thủ thuật áp dụng được mà không cần lập $X_c$.

Notebook dùng thủ thuật này làm nguồn nghiệm chính xác duy nhất: hàm `exact_factors` chéo hóa Gram đúng một lần cho mỗi cặp (ma trận, chế độ tâm hóa), giữ toàn bộ phổ và 1000 vectơ $v_i$ đầu, nên cả notebook chỉ có bốn lần chéo hóa ma trận Gram của dữ liệu bằng `eigh` (train và sub-train, mỗi ma trận có và không tâm hóa). Vì nghiệm $k$ thành phần đầu của SVD chính xác lồng nhau theo $k$, một lần phân rã cho nghiệm tối ưu tái tạo ở mọi $k \le 1000$; phổ, năng lượng tích lũy, thí nghiệm Eckart–Young và bước quét $k$ đều cắt từ đó.

### 4.4. Tâm hóa ngầm cho ma trận thưa

Ma trận $X_c = X - \mathbf{1}\mu^\mathsf{T}$ là **đặc** (hầu như mọi phần tử bằng $-\mu_t \ne 0$) nên không bao giờ nên lập tường minh: với $n = 4076$, $d = 5445$ nó chiếm khoảng 169 MiB thay vì 0,43 MiB của $X$ ở dạng CSR. May mắn là mọi thuật toán cần đến $X_c$ chỉ dùng nó qua hai phép nhân, và cả hai đều tính được từ $X$ thưa cộng một hiệu chỉnh hạng một:

$$
(X - \mathbf{1}\mu^\mathsf{T})M = XM - \mathbf{1}(\mu^\mathsf{T}M), \qquad (X - \mathbf{1}\mu^\mathsf{T})^\mathsf{T}M = X^\mathsf{T}M - \mu(\mathbf{1}^\mathsf{T}M). \qquad (28) $$

Ở vế phải, $XM$ và $X^\mathsf{T}M$ là tích thưa–đặc, $\mu^\mathsf{T}M$ là một vectơ hàng, $\mathbf{1}^\mathsf{T}M$ là tổng các hàng của $M$. Chi phí thêm là $O((n + d) \ell)$ với $\ell$ là số cột của $M$, không đáng kể. Lớp `CenteredOperator` trong notebook cài đúng hai công thức này; randomized SVD của `PCAScratch` chạy trên toán tử đó, còn nghiệm chính xác dùng ma trận Gram tâm hóa dưới đây.

Ma trận Gram đã tâm hóa cũng có dạng đóng:

$$
X_cX_c^\mathsf{T} = XX^\mathsf{T} - (X\mu)\mathbf{1}^\mathsf{T} - \mathbf{1}(X\mu)^\mathsf{T} + \Vert\mu\Vert^2 \mathbf{1}\mathbf{1}^\mathsf{T}, \qquad X_c^\mathsf{T}X_c = X^\mathsf{T}X - n \mu\mu^\mathsf{T}. \qquad (29) $$

*Chứng minh.* Khai triển $(X - \mathbf{1}\mu^\mathsf{T})(X - \mathbf{1}\mu^\mathsf{T})^\mathsf{T} = XX^\mathsf{T} - X\mu\mathbf{1}^\mathsf{T} - \mathbf{1}\mu^\mathsf{T}X^\mathsf{T} + \mathbf{1}\mu^\mathsf{T}\mu\mathbf{1}^\mathsf{T}$; công thức thứ hai là (19) nhân với $n$. $∎$

### 4.5. Randomized SVD

Khi chỉ cần $k \ll\min(n, d)$ thành phần, thuật toán của Halko, Martinsson và Tropp (2011) tránh cả `eigh` cỡ $n \times n$:

1. Lấy $\Omega\in\mathbb{R}^{d\times\ell}$ Gauss với $\ell= k + p$ ($p$ là oversampling). Tính $Y = X\Omega$ và phân tích $QR$: $Y = QR$. Các cột của $Q$ là cơ sở trực chuẩn của $\mathrm{range}(Y)$, một không gian con ngẫu nhiên $\ell$ chiều "nghiêng" về phía các hướng kỳ dị lớn, vì $X\Omega= U\Sigma(V^\mathsf{T}\Omega)$ nhân mỗi hướng $u_i$ với $\sigma_i$.
2. **Power iteration.** Lặp $q$ lần: $Q \leftarrow\mathrm{qr}\big(X \mathrm{qr}(X^\mathsf{T}Q)\big)$. Về mặt toán học điều này thay $X\Omega$ bằng $(XX^\mathsf{T})^qX\Omega= U\Sigma^{2q+1}V^\mathsf{T}\Omega$, tức là thay tỉ số $\sigma_{k+1}/\sigma_k$ bằng $(\sigma_{k+1}/\sigma_k)^{2q+1}$ nên không gian con hội tụ nhanh hơn nhiều về $\mathrm{span}(u_1, \dots, u_k)$. Phân tích $QR$ sau mỗi phép nhân chỉ để giữ ổn định số (không có nó, các cột sẽ nhanh chóng đồng hướng về $u_1$).
3. Chiếu: $B = Q^\mathsf{T}X \in\mathbb{R}^{\ell\times d}$ (nhỏ: $\ell$ hàng). SVD của $B = \tilde{U}\Sigma V^\mathsf{T}$, rồi $U = Q\tilde{U}$. Khi đó $X \approx QQ^\mathsf{T}X = U\Sigma V^\mathsf{T}$.

**Vai trò trong notebook.** Randomized SVD là thuật toán cho trường hợp $n$ quá lớn để chéo hóa ma trận Gram $n \times n$. Notebook kiểm chứng nó tại $k = 50$: với $p = 10$, $q = 4$ sai số tương đối lớn nhất của $\sigma_i$ là $4{,}4 \cdot 10^{-2}$; với $p = 50$, $q = 4$ là $8{,}2 \cdot 10^{-3}$; với $p = k$, $q = 10$ (mặc định của đồ án) là $2{,}2 \cdot 10^{-4}$. Phổ suy giảm chậm của dữ liệu văn bản là lý do cần nhiều vòng power iteration hơn thường lệ (xem (30) ở mục 4.6).

Bước quét $k$ dùng một phân tích chính xác cố định để lấy nghiệm tối ưu tái tạo ở mọi $k$ (nghiệm $k$ thành phần đầu của SVD chính xác lồng nhau theo $k$). Cắt một kết quả randomized tại $k_{\max}$ cũng cho các phép chiếu xấp xỉ hợp lệ và lồng nhau, nhưng đó là một thiết kế thí nghiệm khác, không tương đương với các lần randomized độc lập theo từng $k$ (oversampling $p = k$ phụ thuộc $k$). Trên bộ dữ liệu này $n \approx 4.000$ nên dùng nghiệm chính xác.

### 4.6. Ghi chú bổ sung về randomized SVD

**Độ phức tạp.** Mỗi vòng cần hai tích thưa–đặc $O(\mathrm{nnz}(X)\cdot\ell)$ và hai phân tích $QR$ cỡ $O((n + d)\ell^2)$, tổng cộng $O\big((q+1)(\mathrm{nnz}(X)\ell + (n+d)\ell^2)\big)$, cộng SVD của $B$ cỡ $O(d\ell^2)$. Với $\ell = 2k$ và $k$ cỡ vài trăm, phần $(n+d)\ell^2$ chiếm ưu thế, nên chi phí tăng gần bậc hai theo $k$.

**Sai số.** Kết quả của Halko và cộng sự cho kỳ vọng sai số (chuẩn phổ) sau $q$ vòng power iteration là

$$
\mathbb{E}\Vert X - QQ^\mathsf{T}X\Vert_2 \le\Big[\Big(1 + \sqrt{\frac{k}{p-1}}\Big)\sigma_{k+1}^{2q+1} + \frac{e\sqrt{k+p}}{p}\Big(\sum_{j\gt k}\sigma_j^{2(2q+1)}\Big)^{1/2}\Big]^{1/(2q+1)}. \qquad (30) $$

Với phổ suy giảm chậm, tổng $\sum_{j\gt k}\sigma_j^{2(2q+1)}$ chỉ nhỏ khi $q$ đủ lớn, phù hợp với bảng hội tụ trong notebook.

---

## PHẦN 5: KẾT LUẬN VÀ Ý NGHĨA

### 5.1. Những gì đã chứng minh

1. **SVD tồn tại với mọi ma trận** (Định lý 1), và chứng minh là xây dựng: chéo hóa $A^\mathsf{T}A$, lấy căn trị riêng, $u_i = Av_i/\sigma_i$. Notebook cài đúng thuật toán này (`svd_via_eig`) và đối chiếu với `numpy.linalg.svd`: sai số cỡ $10^{-15}$, các nhân tử trực chuẩn, hạng số học xác định đúng.
2. **PCA là SVD của dữ liệu đã tâm hóa** (Định lý 3): trị riêng của ma trận hiệp phương sai là $\sigma_i^2/(n-1)$, các điểm PCA không tương quan (Mệnh đề 4), và thành phần thứ nhất là hướng cực đại phương sai (Định lý 2). Notebook đối chiếu ba cách tính độc lập (ARPACK của scikit-learn, Gram tâm hóa, phương sai của điểm chiếu), khớp đến $10^{-17}$, và kiểm tra $\mathrm{Cov}(Z)$ chéo.
3. **Eckart–Young–Mirsky** (Định lý 5): $A_k = AV_kV_k^\mathsf{T}$ là xấp xỉ hạng $k$ tốt nhất và bình phương sai số Frobenius bằng $\Vert A - A_k\Vert_F^2 = \sum_{i\gt k}\sigma_i^2$. Notebook đo sai số này qua đồng nhất thức (22) và thấy trùng với lý thuyết đến cỡ $10^{-12}$, đồng thời nhỏ hơn hẳn sai số của không gian ngẫu nhiên và của phép chọn từ.
4. **Tâm hóa đổi tiêu chí chọn thành phần đầu** (mục 2.6): không tâm hóa, hàm mục tiêu có thêm số hạng $(w^\mathsf{T}\mu)^2$; tâm hóa bỏ số hạng đó nhưng không bảo đảm PC1 trực giao với $\mu$. Số đo: $|\cos(v_1, \mu)|$ là 0,883 không tâm hóa và 0,327 có tâm hóa.

### 5.2. Liên hệ với kết quả thực nghiệm

- **Phổ suy giảm chậm.** Ma trận TF-IDF ($4076 \times 5445$) cần 390 thành phần để đạt một nửa năng lượng; 300 thành phần giữ 43,9% và 1000 thành phần giữ 75,4%. Đây là tính chất của dữ liệu thưa: mỗi tài liệu dùng một tập từ nhỏ, gần như trực giao với các tài liệu khác. Hệ quả là randomized SVD cần nhiều vòng power iteration hơn thường lệ (mục 4.5).
- **Năng lượng và khả năng phân loại là hai chuyện khác nhau.** Trên tập validation, LSA và PCA đạt F1 0,9205 tại $k = 300$, so với 0,9283 của baseline dùng toàn bộ TF-IDF. Trên tập test, mô hình LSA ($k = 300$) đạt F1 0,909 và ROC-AUC 0,986, baseline đạt 0,920 và 0,992. Định lý 5 chỉ nói về sai số tái tạo nên không giải thích trực tiếp chênh lệch này. Một giả thuyết, chưa được kiểm định trong notebook, là thông tin phân biệt của các từ hiếm nằm chủ yếu ở các hướng kỳ dị có $\sigma_i$ nhỏ, vốn bị cắt khi chỉ giữ 300 thành phần. Thành phần chính thứ nhất tương quan 0,68 với nhãn spam dù PCA không nhìn vào nhãn.
- **LSA và PCA: phân loại giống nhau, không gian con khác nhau.** Theo (19), $X^\mathsf{T}X$ và $X_c^\mathsf{T}X_c$ khác nhau một ma trận hạng một, và điều này chỉ suy ra được tính đan xen của các trị riêng (định lý Cauchy interlacing), notebook kiểm tra đúng. Nó không suy ra hai không gian con gần nhau: góc chính lớn nhất giữa $\mathrm{span}(V_k)$ của hai phương pháp đo được là 72,4° tại $k = 50$ và 59,3° tại $k = 300$. Việc hai mô hình cho cùng F1 tại $k = 300$ là quan sát thực nghiệm trên bộ dữ liệu này.

### 5.3. Ý nghĩa hình học

SVD cho một cách nhìn thống nhất: mọi ánh xạ tuyến tính $x \mapsto Ax$ là *xoay* ($V^\mathsf{T}$), *co giãn theo các trục* ($\Sigma$), rồi *xoay* lần nữa ($U$). Phổ $\lbrace\sigma_i\rbrace$ cho biết dữ liệu "dày" theo những hướng nào; PCA chọn các hướng dày nhất quanh tâm, LSA chọn quanh gốc tọa độ, và Định lý Eckart–Young bảo đảm rằng không có không gian con nào cùng số chiều giữ được nhiều hơn. Phần thực nghiệm của đồ án kiểm tra bằng số các khẳng định đó trên một ma trận dữ liệu thực.

---

## TÀI LIỆU THAM KHẢO

1. C. Eckart, G. Young. *The approximation of one matrix by another of lower rank.* Psychometrika 1(3), 1936, 211–218.
2. L. Mirsky. *Symmetric gauge functions and unitarily invariant norms.* Quarterly Journal of Mathematics 11, 1960, 50–59.
3. N. Halko, P. G. Martinsson, J. A. Tropp. *Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions.* SIAM Review 53(2), 2011, 217–288.
4. G. Strang. *Linear Algebra and Learning from Data.* Wellesley-Cambridge Press, 2019, chương I.8–I.9.
5. T. A. Almeida, J. M. Gómez Hidalgo, A. Yamakami. *Contributions to the study of SMS spam filtering: new collection and results.* ACM DocEng 2011. Bộ dữ liệu: UCI Machine Learning Repository, SMS Spam Collection.
6. Tài liệu scikit-learn: `sklearn.decomposition.TruncatedSVD`, `sklearn.decomposition.PCA`, `sklearn.feature_extraction.text.TfidfVectorizer`.
