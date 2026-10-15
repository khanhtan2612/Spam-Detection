# SINGULAR VALUE DECOMPOSITION (SVD) VÀ PRINCIPAL COMPONENT ANALYSIS (PCA): CƠ SỞ TOÁN HỌC VÀ ỨNG DỤNG CHO BÀI TOÁN PHÁT HIỆN TIN NHẮN RÁC

*Chứng minh dựa trên chéo hóa trực giao ma trận đối xứng và định lý Eckart–Young–Mirsky*

Tài liệu này trình bày phần lý thuyết đứng sau notebook `spam_svd_pca.ipynb`. Mỗi kết quả được chứng minh đầy đủ, sau đó chỉ rõ nó được kiểm chứng ở đâu trong notebook.

Ký hiệu chung: ma trận dữ liệu $X \in \mathbb{R}^{n \times d}$ có $n$ hàng (tài liệu) và $d$ cột (từ). $\mathbf{1} \in \mathbb{R}^{n}$ là vectơ toàn số 1. $\lVert A \rVert_F^2 = \sum_{i,j} a_{ij}^2 = \operatorname{tr}(A^\mathsf{T}A)$ là bình phương chuẩn Frobenius. Với ma trận đối xứng, "chéo hóa trực giao" nghĩa là viết $S = Q\Lambda Q^\mathsf{T}$ với $Q$ trực giao và $\Lambda$ chéo.

---

## PHẦN 1: SINGULAR VALUE DECOMPOSITION (SVD)

### 1.1. Định nghĩa

Cho $A \in \mathbb{R}^{m \times n}$ bất kỳ. Một **phân tích giá trị kỳ dị** của $A$ là cách viết

$$
A = U \Sigma V^\mathsf{T}, \tag{1}
$$

trong đó $U \in \mathbb{R}^{m \times m}$ và $V \in \mathbb{R}^{n \times n}$ là các ma trận trực giao ($U^\mathsf{T}U = I_m$, $V^\mathsf{T}V = I_n$), còn $\Sigma \in \mathbb{R}^{m \times n}$ chỉ có các phần tử khác 0 trên đường chéo chính, $\Sigma_{ii} = \sigma_i$, với

$$
\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{p} \ge 0, \qquad p = \min(m, n).
$$

Các số $\sigma_i$ gọi là **giá trị kỳ dị**; các cột $u_i$ của $U$ và $v_i$ của $V$ gọi là vectơ kỳ dị trái và phải. Từ (1) suy ra ngay hai hệ thức đặc trưng

$$
A v_i = \sigma_i u_i, \qquad A^\mathsf{T} u_i = \sigma_i v_i, \qquad i = 1, \dots, p. \tag{2}
$$

### 1.2. Hai bổ đề về $A^\mathsf{T}A$

**Bổ đề 1.** $A^\mathsf{T}A \in \mathbb{R}^{n \times n}$ là ma trận đối xứng và nửa xác định dương.

*Chứng minh.* Đối xứng: $(A^\mathsf{T}A)^\mathsf{T} = A^\mathsf{T}(A^\mathsf{T})^\mathsf{T} = A^\mathsf{T}A$. Nửa xác định dương: với mọi $x \in \mathbb{R}^n$,

$$
x^\mathsf{T}(A^\mathsf{T}A)x = (Ax)^\mathsf{T}(Ax) = \lVert Ax \rVert^2 \ge 0. \qquad\blacksquare
$$

**Bổ đề 2 (Spectral theorem).** Mọi ma trận đối xứng thực $S \in \mathbb{R}^{n \times n}$ đều chéo hóa được bằng ma trận trực giao: tồn tại $Q$ trực giao và $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ với $\lambda_i \in \mathbb{R}$ sao cho $S = Q\Lambda Q^\mathsf{T}$. Nếu $S$ nửa xác định dương thì $\lambda_i \ge 0$ với mọi $i$.

*Chứng minh phần cuối.* Với $q_i$ là cột thứ $i$ của $Q$: $\lambda_i = q_i^\mathsf{T} S q_i \ge 0$ theo định nghĩa nửa xác định dương. (Phần đầu là định lý phổ cho ma trận đối xứng thực, được coi là đã biết.) $\blacksquare$

### 1.3. Định lý tồn tại SVD (chứng minh xây dựng)

**Định lý 1.** Mọi ma trận $A \in \mathbb{R}^{m \times n}$ đều có một phân tích dạng (1).

*Chứng minh.* Theo Bổ đề 1 và 2, chéo hóa trực giao

$$
A^\mathsf{T}A = V \Lambda V^\mathsf{T}, \qquad \Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n), \quad \lambda_1 \ge \dots \ge \lambda_n \ge 0, \tag{3}
$$

với $V = [v_1 \,\cdots\, v_n]$ trực giao. Đặt $\sigma_i = \sqrt{\lambda_i}$ và gọi $r$ là số chỉ số có $\sigma_i > 0$ (tức $r = \operatorname{rank}(A^\mathsf{T}A) = \operatorname{rank} A$).

*Bước 1: các vectơ $Av_i$ trực giao với nhau.* Với mọi $i, j$,

$$
(Av_i)^\mathsf{T}(Av_j) = v_i^\mathsf{T} (A^\mathsf{T}A) v_j = v_i^\mathsf{T} (\lambda_j v_j) = \lambda_j\, v_i^\mathsf{T} v_j = \lambda_j \delta_{ij}. \tag{4}
$$

Như vậy $\lVert Av_i \rVert = \sigma_i$, và $Av_i = 0$ khi $i > r$.

*Bước 2: xây dựng $u_i$.* Với $i \le r$ đặt

$$
u_i = \frac{A v_i}{\sigma_i}. \tag{5}
$$

Theo (4), $u_i^\mathsf{T} u_j = \dfrac{\lambda_j \delta_{ij}}{\sigma_i \sigma_j} = \delta_{ij}$, nên $u_1, \dots, u_r$ là hệ trực chuẩn trong $\mathbb{R}^m$. Bổ sung (Gram–Schmidt) thành cơ sở trực chuẩn $u_1, \dots, u_m$ của $\mathbb{R}^m$ và đặt $U = [u_1 \,\cdots\, u_m]$.

*Bước 3: kiểm tra (1).* Ta chứng minh $U^\mathsf{T} A V = \Sigma$. Phần tử $(i, j)$ của vế trái là $u_i^\mathsf{T} A v_j$.

- Nếu $j \le r$: $A v_j = \sigma_j u_j$ theo (5), nên $u_i^\mathsf{T} A v_j = \sigma_j u_i^\mathsf{T} u_j = \sigma_j \delta_{ij}$.
- Nếu $j > r$: $A v_j = 0$ theo Bước 1, nên phần tử bằng 0.

Vậy $U^\mathsf{T}AV = \Sigma$, tức $A = U\Sigma V^\mathsf{T}$ vì $U, V$ trực giao. $\blacksquare$

Chứng minh trên chính là thuật toán `svd_via_eig` trong notebook: chéo hóa $A^\mathsf{T}A$ bằng `numpy.linalg.eigh`, lấy căn các trị riêng, rồi tính $u_i$ theo (5). Khi $n > m$, áp dụng cùng lập luận cho $A^\mathsf{T}$ để chỉ phải chéo hóa ma trận $m \times m$ nhỏ hơn.

**Nhận xét về số học.** Vì tính $\lambda_i = \sigma_i^2$ rồi lấy căn, sai số làm tròn trên $\lambda_i$ (cỡ $\lambda_1 \varepsilon$) trở thành sai số $\sigma_1\sqrt{\varepsilon}$ trên các giá trị kỳ dị nhỏ. Do đó hạng số học được xác định bằng ngưỡng $\lambda_i > \lambda_1 \cdot \max(m,n)\, \varepsilon$ chứ không bằng ngưỡng trên $\sigma_i$. Đây cũng là lý do các thư viện số không dùng cách này cho ma trận điều kiện xấu.

### 1.4. Dạng compact và dạng tổng các ma trận hạng một

Vì $\sigma_i = 0$ khi $i > r$, chỉ $r$ cột đầu của $U$ và $V$ đóng góp vào tích (1):

$$
A = U_r \Sigma_r V_r^\mathsf{T}, \qquad U_r \in \mathbb{R}^{m \times r},\ \Sigma_r = \operatorname{diag}(\sigma_1, \dots, \sigma_r),\ V_r \in \mathbb{R}^{n \times r}. \tag{6}
$$

Khai triển tích theo cột cho

$$
A = \sum_{i=1}^{r} \sigma_i\, u_i v_i^\mathsf{T}. \tag{7}
$$

Mỗi số hạng $u_i v_i^\mathsf{T}$ là một ma trận hạng một có chuẩn Frobenius bằng 1, và các số hạng đôi một trực giao theo tích vô hướng Frobenius: $\langle u_i v_i^\mathsf{T}, u_j v_j^\mathsf{T} \rangle_F = (u_i^\mathsf{T}u_j)(v_i^\mathsf{T}v_j) = \delta_{ij}$. Do đó

$$
\lVert A \rVert_F^2 = \sum_{i=1}^{r} \sigma_i^2. \tag{8}
$$

Notebook kiểm tra (8) trực tiếp trên ma trận TF-IDF ở mục *Phổ giá trị kỳ dị*: tổng bình phương các giá trị kỳ dị đúng bằng $\lVert X\rVert_F^2$, và vì mỗi hàng TF-IDF được chuẩn hóa về độ dài 1, giá trị này bằng số hàng khác 0 của $X$.

---

## PHẦN 2: PRINCIPAL COMPONENT ANALYSIS (PCA)

### 2.1. Tâm hóa và ma trận hiệp phương sai (covariance matrix)

Cho $X \in \mathbb{R}^{n \times d}$, mỗi hàng $x_i^\mathsf{T}$ là một quan sát. Vectơ trung bình và ma trận đã tâm hóa là

$$
\mu = \frac{1}{n} X^\mathsf{T}\mathbf{1} \in \mathbb{R}^d, \qquad X_c = X - \mathbf{1}\mu^\mathsf{T}. \tag{9}
$$

Ma trận hiệp phương sai mẫu

$$
C = \frac{1}{n-1} X_c^\mathsf{T} X_c \in \mathbb{R}^{d \times d} \tag{10}
$$

là đối xứng, nửa xác định dương (Bổ đề 1 áp dụng cho $X_c$). Với một hướng đơn vị $w \in \mathbb{R}^d$, các tọa độ chiếu $z_i = w^\mathsf{T}(x_i - \mu)$ có trung bình 0 và phương sai mẫu

$$
\operatorname{Var}(z) = \frac{1}{n-1}\sum_{i=1}^n \big(w^\mathsf{T}(x_i - \mu)\big)^2 = \frac{1}{n-1}\lVert X_c w\rVert^2 = w^\mathsf{T} C w. \tag{11}
$$

### 2.2. Thành phần chính thứ nhất: bài toán cực đại phương sai

**Định lý 2.** Nghiệm của bài toán

$$
\max_{w \in \mathbb{R}^d} w^\mathsf{T} C w \quad \text{với điều kiện} \quad \lVert w\rVert^2 = 1 \tag{12}
$$

là vectơ riêng $v_1$ ứng với trị riêng lớn nhất $\lambda_1$ của $C$, và giá trị cực đại bằng $\lambda_1$.

*Chứng minh bằng Lagrange multiplier.* Hàm Lagrange $\mathcal{L}(w, \alpha) = w^\mathsf{T}Cw - \alpha(w^\mathsf{T}w - 1)$. Điều kiện dừng

$$
\nabla_w \mathcal{L} = 2Cw - 2\alpha w = 0 \iff Cw = \alpha w. \tag{13}
$$

Vậy mọi điểm dừng là một vectơ riêng đơn vị của $C$, và tại đó $w^\mathsf{T}Cw = \alpha w^\mathsf{T}w = \alpha$. Giá trị mục tiêu lớn nhất đạt được khi $\alpha$ là trị riêng lớn nhất, tức $w = v_1$. (Cực đại tồn tại vì mặt cầu đơn vị compact và hàm mục tiêu liên tục.) $\blacksquare$

*Chứng minh trực tiếp không dùng Lagrange.* Chéo hóa $C = V\Lambda V^\mathsf{T}$ với $\lambda_1 \ge \dots \ge \lambda_d \ge 0$. Đặt $y = V^\mathsf{T}w$, thì $\lVert y\rVert = 1$ và

$$
w^\mathsf{T}Cw = y^\mathsf{T}\Lambda y = \sum_{i=1}^d \lambda_i y_i^2 \le \lambda_1 \sum_i y_i^2 = \lambda_1,
$$

đẳng thức khi $y = e_1$, tức $w = v_1$. $\blacksquare$

**Các thành phần tiếp theo.** Thành phần chính thứ $j$ là nghiệm của (12) với ràng buộc bổ sung $w \perp v_1, \dots, v_{j-1}$. Lặp lại lập luận trực tiếp với $y_1 = \dots = y_{j-1} = 0$ cho $w^\mathsf{T}Cw \le \lambda_j$, đạt được tại $w = v_j$. Tổng quát, không gian con $k$ chiều $\operatorname{span}(v_1, \dots, v_k)$ cực đại tổng phương sai $\operatorname{tr}(W^\mathsf{T}CW)$ trên mọi $W \in \mathbb{R}^{d \times k}$ có cột trực chuẩn (định lý Ky Fan; cũng là hệ quả của Định lý 3 ở PHẦN 3 áp dụng cho $X_c$).

### 2.3. PCA là SVD của ma trận đã tâm hóa

**Định lý 3.** Gọi $X_c = U\Sigma V^\mathsf{T}$ là SVD của ma trận đã tâm hóa, với các giá trị kỳ dị $\sigma_1 \ge \dots \ge \sigma_r > 0$. Khi đó

$$
C = \frac{1}{n-1} V \Sigma^\mathsf{T}\Sigma V^\mathsf{T}, \tag{14}
$$

nghĩa là các vectơ kỳ dị phải $v_i$ của $X_c$ là các vectơ riêng của $C$ với trị riêng

$$
\lambda_i = \frac{\sigma_i^2}{n-1}. \tag{15}
$$

*Chứng minh.* Thay SVD vào (10): $X_c^\mathsf{T}X_c = V\Sigma^\mathsf{T}U^\mathsf{T}U\Sigma V^\mathsf{T} = V(\Sigma^\mathsf{T}\Sigma)V^\mathsf{T}$ vì $U^\mathsf{T}U = I$. Ma trận $\Sigma^\mathsf{T}\Sigma$ là chéo với các phần tử $\sigma_i^2$ (và 0 khi $i > r$). Chia cho $n - 1$ được (14); đây chính là chéo hóa trực giao của $C$ nên các cột của $V$ là vectơ riêng với trị riêng (15). $\blacksquare$

Hệ quả thực hành: để làm PCA không cần lập ma trận $C$ kích thước $d \times d$; chỉ cần SVD (hoặc truncated SVD) của $X_c$. Trong notebook, `PCAScratch` làm đúng như vậy và `explained_variance_` được đặt bằng $\sigma_i^2/(n-1)$; mục *Kiểm chứng lý thuyết bằng thực nghiệm* đối chiếu ba đại lượng: trị riêng của $C$ (tính qua ma trận Gram đã tâm hóa), $\sigma_i^2/(n-1)$, và phương sai mẫu của các điểm chiếu.

### 2.4. Các điểm PCA không tương quan

Điểm PCA (tọa độ của dữ liệu trong cơ sở $k$ thành phần đầu) là

$$
Z = X_c V_k \in \mathbb{R}^{n \times k}, \qquad V_k = [v_1 \,\cdots\, v_k]. \tag{16}
$$

**Mệnh đề 4.** Các cột của $Z$ có trung bình 0 và ma trận hiệp phương sai của $Z$ là ma trận chéo

$$
\frac{1}{n-1} Z^\mathsf{T}Z = \frac{1}{n-1}\Sigma_k^2 = \operatorname{diag}(\lambda_1, \dots, \lambda_k). \tag{17}
$$

*Chứng minh.* Trung bình: $\mathbf{1}^\mathsf{T}Z = \mathbf{1}^\mathsf{T}X_cV_k = (\mathbf{1}^\mathsf{T}X - n\mu^\mathsf{T})V_k = 0$ vì $\mathbf{1}^\mathsf{T}X = n\mu^\mathsf{T}$ theo (9). Hiệp phương sai: dùng $X_cV_k = U_k\Sigma_k$ (từ (2)),

$$
Z^\mathsf{T}Z = \Sigma_k U_k^\mathsf{T}U_k\Sigma_k = \Sigma_k^2. \qquad\blacksquare
$$

Vậy PCA không chỉ xoay dữ liệu mà còn **khử tương quan**: hai tọa độ bất kỳ trong $Z$ có hiệp phương sai mẫu bằng 0. Notebook kiểm tra phần tử ngoài đường chéo lớn nhất của $\operatorname{Cov}(Z)$, kết quả cỡ $10^{-16}$.

### 2.5. Tỉ lệ phương sai giải thích

Tổng phương sai của dữ liệu là $\operatorname{tr}(C) = \sum_{i=1}^d \lambda_i = \dfrac{1}{n-1}\lVert X_c\rVert_F^2$ (dùng (8)). Tỉ lệ phương sai giải thích bởi $k$ thành phần đầu:

$$
\mathrm{EVR}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i} = \frac{\sum_{i=1}^k \sigma_i^2}{\lVert X_c\rVert_F^2}. \tag{18}
$$

Để tính mẫu số mà không tâm hóa tường minh, khai triển

$$
\lVert X_c\rVert_F^2 = \operatorname{tr}\big((X - \mathbf{1}\mu^\mathsf{T})^\mathsf{T}(X - \mathbf{1}\mu^\mathsf{T})\big) = \lVert X\rVert_F^2 - 2n\lVert\mu\rVert^2 + n\lVert\mu\rVert^2 = \lVert X\rVert_F^2 - n\lVert\mu\rVert^2, \tag{19}
$$

trong đó đã dùng $X^\mathsf{T}\mathbf{1} = n\mu$ và $\mathbf{1}^\mathsf{T}\mathbf{1} = n$. Đây là công thức của `total_variance_` trong notebook.

### 2.6. Vì sao phải tâm hóa

Nếu bỏ qua tâm hóa và làm SVD trực tiếp trên $X$ (như LSA), hướng $v_1$ cực đại $\lVert Xw\rVert^2 = \sum_i (w^\mathsf{T}x_i)^2$, tức **moment bậc hai không tâm hóa** $w^\mathsf{T}(\tfrac{1}{n}X^\mathsf{T}X)w$, chứ không phải phương sai. Hai đại lượng này liên hệ bởi

$$
\frac{1}{n}X^\mathsf{T}X = \frac{1}{n}X_c^\mathsf{T}X_c + \mu\mu^\mathsf{T}. \tag{20}
$$

*Chứng minh.* $X = X_c + \mathbf{1}\mu^\mathsf{T}$ nên $X^\mathsf{T}X = X_c^\mathsf{T}X_c + X_c^\mathsf{T}\mathbf{1}\mu^\mathsf{T} + \mu\mathbf{1}^\mathsf{T}X_c + n\mu\mu^\mathsf{T}$, và hai số hạng giữa bằng 0 vì $X_c^\mathsf{T}\mathbf{1} = X^\mathsf{T}\mathbf{1} - \mu\mathbf{1}^\mathsf{T}\mathbf{1} = n\mu - n\mu = 0$. $\blacksquare$

Số hạng $\mu\mu^\mathsf{T}$ là ma trận hạng một với trị riêng $\lVert\mu\rVert^2$ theo hướng $\mu$. Khi $\lVert\mu\rVert^2$ lớn so với các trị riêng của $C$ (điều luôn đúng với dữ liệu không âm như TF-IDF, nơi mọi tài liệu nằm trong cùng một góc phần tư), hướng kỳ dị đầu tiên của $X$ bị kéo về phía $\mu$ và chủ yếu đo "độ điển hình" của một tài liệu thay vì sự khác biệt giữa các tài liệu. Hơn nữa, vì $X$ có mọi phần tử không âm, định lý Perron–Frobenius cho phép chọn $u_1, v_1$ không âm: mọi tài liệu có cùng dấu trên tọa độ thứ nhất, tức tọa độ đó không tách được nhóm nào. Notebook đo $|\cos(v_1^{\text{SVD}}, \mu)|$ và $|\cos(v_1^{\text{PCA}}, \mu)|$ cùng tỉ lệ tài liệu có cùng dấu trên thành phần đầu để minh họa.

---

## PHẦN 3: ĐỊNH LÝ ECKART–YOUNG–MIRSKY

### 3.1. Phát biểu

Cho $A = \sum_{i=1}^r \sigma_i u_i v_i^\mathsf{T}$ là SVD của $A \in \mathbb{R}^{m \times n}$ và $k < r$. Đặt

$$
A_k = \sum_{i=1}^k \sigma_i u_i v_i^\mathsf{T} = U_k\Sigma_kV_k^\mathsf{T}. \tag{21}
$$

**Định lý 5 (Eckart–Young 1936, Mirsky 1960).** Với mọi ma trận $B \in \mathbb{R}^{m \times n}$ có $\operatorname{rank} B \le k$,

$$
\lVert A - B\rVert_F^2 \ \ge\ \lVert A - A_k\rVert_F^2 \ =\ \sum_{i=k+1}^{r}\sigma_i^2. \tag{22}
$$

### 3.2. Chứng minh đẳng thức

$A - A_k = \sum_{i=k+1}^r \sigma_i u_i v_i^\mathsf{T}$ là tổng các ma trận hạng một trực giao, nên theo lập luận dẫn đến (8), $\lVert A - A_k\rVert_F^2 = \sum_{i>k}\sigma_i^2$. $\blacksquare$

Ngoài ra, $A_k = AV_kV_k^\mathsf{T}$: thật vậy $AV_k = U_k\Sigma_k$ theo (2), nên $AV_kV_k^\mathsf{T} = U_k\Sigma_kV_k^\mathsf{T} = A_k$. Do đó $A_k$ là **phép chiếu trực giao các hàng của $A$** lên không gian con $\operatorname{span}(v_1, \dots, v_k)$. Dạng này là cách notebook đo sai số: tính trực tiếp $\lVert X - XV_kV_k^\mathsf{T}\rVert_F^2$ theo từng khối hàng rồi so với $\sum_{i>k}\sigma_i^2$.

### 3.3. Chứng minh tính tối ưu

Trước hết là một bổ đề về trị riêng.

**Bổ đề 3 (bất đẳng thức giao thoa Weyl, dạng cần dùng).** Cho $S \in \mathbb{R}^{n \times n}$ đối xứng với trị riêng $\lambda_1(S) \ge \dots \ge \lambda_n(S)$, và $W \subset \mathbb{R}^n$ là không gian con có số chiều $n - k$. Khi đó tồn tại vectơ đơn vị $w \in W$ với $w^\mathsf{T}Sw \ge \lambda_{k+1}(S)$. Tổng quát hơn, với $j \ge 1$ bất kỳ, $W$ chứa một hệ trực chuẩn $w_1, \dots, w_j$ sao cho $\sum_{l=1}^{j} w_l^\mathsf{T}Sw_l \ge \sum_{l=1}^{j}\lambda_{k+l}(S)$ (khi $j \le n - k$).

*Chứng minh.* Gọi $q_1, \dots, q_n$ là cơ sở trực chuẩn các vectơ riêng của $S$. Không gian $E = \operatorname{span}(q_1, \dots, q_{k+1})$ có số chiều $k + 1$, còn $\dim W = n - k$, nên $\dim(W \cap E) \ge (k+1) + (n-k) - n = 1$. Lấy $w \in W \cap E$ đơn vị, viết $w = \sum_{i \le k+1} c_i q_i$ với $\sum c_i^2 = 1$; khi đó $w^\mathsf{T}Sw = \sum_{i \le k+1}\lambda_i c_i^2 \ge \lambda_{k+1}$. Phát biểu tổng quát được chứng minh bằng quy nạp theo $j$: sau khi chọn $w_1$, lặp lại lập luận trong $W \cap w_1^\perp$ (số chiều $n - k - 1$) với $E' = \operatorname{span}(q_1, \dots, q_{k+2})$, v.v. $\blacksquare$

*Chứng minh Định lý 5.* Lấy $B$ bất kỳ với $\operatorname{rank} B \le k$. Không gian nghiệm $N = \ker B \subset \mathbb{R}^n$ có $\dim N = n - \operatorname{rank}B \ge n - k$. Chọn một không gian con $W \subset N$ với $\dim W = n - k$ đúng. Với mọi $w \in W$ ta có $Bw = 0$, do đó $(A - B)w = Aw$.

Gọi $w_1, \dots, w_{n-k}$ là cơ sở trực chuẩn của $W$ và bổ sung thành cơ sở trực chuẩn $w_1, \dots, w_n$ của $\mathbb{R}^n$. Vì chuẩn Frobenius bất biến qua phép đổi cơ sở trực chuẩn,

$$
\lVert A - B\rVert_F^2 = \sum_{l=1}^{n}\lVert (A-B)w_l\rVert^2 \ \ge\ \sum_{l=1}^{n-k}\lVert (A-B)w_l\rVert^2 = \sum_{l=1}^{n-k}\lVert Aw_l\rVert^2 = \sum_{l=1}^{n-k} w_l^\mathsf{T}(A^\mathsf{T}A)w_l. \tag{23}
$$

Áp dụng Bổ đề 3 cho $S = A^\mathsf{T}A$ (trị riêng $\sigma_1^2 \ge \dots \ge \sigma_n^2$, với $\sigma_i = 0$ khi $i > r$) và không gian $W$ số chiều $n - k$, với $j = n - k$: có thể chọn cơ sở trực chuẩn $w_l$ của $W$ sao cho

$$
\sum_{l=1}^{n-k} w_l^\mathsf{T}(A^\mathsf{T}A)w_l \ \ge\ \sum_{l=1}^{n-k}\sigma_{k+l}^2 = \sum_{i=k+1}^{n}\sigma_i^2 = \sum_{i=k+1}^{r}\sigma_i^2. \tag{24}
$$

Ghép (23) với (24) được $\lVert A - B\rVert_F^2 \ge \sum_{i>k}\sigma_i^2 = \lVert A - A_k\rVert_F^2$. $\blacksquare$

**Nhận xét.** Chứng minh chỉ dùng hai sự kiện: $B$ có hạt nhân lớn (vì hạng thấp), và trên hạt nhân đó $A - B$ trùng với $A$. Vì vậy không ma trận hạng $k$ nào "tránh" được phần năng lượng $\sum_{i>k}\sigma_i^2$ nằm ở các hướng kỳ dị sau. Định lý cũng đúng với chuẩn phổ $\lVert\cdot\rVert_2$ (khi đó sai số tối ưu là $\sigma_{k+1}$), nhưng notebook chỉ dùng chuẩn Frobenius.

### 3.4. Ý nghĩa đối với PCA và LSA

- Áp dụng Định lý 5 cho $X_c$: trong mọi không gian con $k$ chiều, phép chiếu lên $\operatorname{span}(v_1, \dots, v_k)$ của PCA giữ lại nhiều "năng lượng" $\lVert X_cV_kV_k^\mathsf{T}\rVert_F^2 = \sum_{i\le k}\sigma_i^2$ nhất, tức là giữ lại nhiều phương sai nhất (nhân $1/(n-1)$). Đây chính là phát biểu "PCA tối ưu theo nghĩa tái tạo".
- Áp dụng cho $X$ không tâm hóa: LSA với $k$ chủ đề là xấp xỉ hạng $k$ tốt nhất của ma trận TF-IDF.
- Notebook so sánh sai số của ba không gian con cùng số chiều: không gian SVD, không gian ngẫu nhiên, và không gian tọa độ gồm $k$ từ có chuẩn cột lớn nhất (tức là *chọn* đặc trưng thay vì *tổ hợp* đặc trưng). Định lý bảo đảm không gian SVD luôn thắng; thí nghiệm cho thấy khoảng cách là rất lớn.

---

## PHẦN 4: ÁP DỤNG CHO DỮ LIỆU VĂN BẢN

### 4.1. Biểu diễn TF-IDF

Mỗi tin nhắn $i$ được biểu diễn bởi vectơ $x_i \in \mathbb{R}^d$ trên từ điển gồm $d$ unigram và bigram. Với $\mathrm{tf}_{it}$ là số lần từ $t$ xuất hiện trong tin $i$ và $\mathrm{df}_t$ là số tin chứa $t$,

$$
\tilde{x}_{it} = \big(1 + \log \mathrm{tf}_{it}\big)\cdot\Big(\log\frac{1+n}{1+\mathrm{df}_t} + 1\Big), \qquad x_i = \frac{\tilde{x}_i}{\lVert\tilde{x}_i\rVert}. \tag{25}
$$

Thừa số $1 + \log\mathrm{tf}$ (sublinear tf) làm giảm ảnh hưởng của từ lặp nhiều lần; thừa số idf hạ trọng số các từ xuất hiện ở hầu hết tài liệu; chuẩn hóa hàng khiến $\lVert x_i\rVert = 1$ nên tích vô hướng $x_i^\mathsf{T}x_j$ là cosin góc giữa hai tin. Hệ quả: $\lVert X\rVert_F^2$ bằng số tài liệu khác rỗng, và theo (8) đó cũng là $\sum_i\sigma_i^2$.

Ma trận $X$ rất thưa (chỉ khoảng 0,16% phần tử khác 0) nên được lưu ở dạng CSR; mọi thuật toán phía sau chỉ dùng các tích $XM$ và $X^\mathsf{T}M$.

### 4.2. LSA và phép chiếu dữ liệu mới

Latent Semantic Analysis (LSA) là truncated SVD của $X$: $X \approx U_k\Sigma_kV_k^\mathsf{T}$. Các cột của $V_k$ là $k$ hướng trong không gian từ ("chủ đề"), và tài liệu $i$ được biểu diễn bởi hàng thứ $i$ của $U_k\Sigma_k = XV_k$.

Với một tin nhắn mới $x \in \mathbb{R}^d$ (đã qua cùng bộ TF-IDF), biểu diễn thấp chiều là phép chiếu

$$
z = V_k^\mathsf{T}x \in \mathbb{R}^k \qquad\text{(LSA)}, \qquad z = V_k^\mathsf{T}(x - \mu) \in \mathbb{R}^k \qquad\text{(PCA)}. \tag{26}
$$

Cả hai đều tuyến tính và chỉ cần $V_k$ (và $\mu$) học từ tập huấn luyện. Vì các cột của $V_k$ trực chuẩn, phép chiếu bảo toàn tích vô hướng trong không gian con: $\lVert z\rVert \le \lVert x\rVert$ và $z_i^\mathsf{T}z_j = x_i^\mathsf{T}V_kV_k^\mathsf{T}x_j$. Hơn nữa tọa độ thứ $i$ của $z$ trên tập huấn luyện có độ lệch chuẩn tỉ lệ với $\sigma_i$ (từ (17)); đây là lý do notebook **không** chuẩn hóa lại các tọa độ sau khi chiếu — làm vậy sẽ khuếch đại các hướng có $\sigma_i$ nhỏ, vốn chủ yếu là nhiễu, lên cùng thang với các hướng chính.

Bộ phân loại phía sau là logistic regression trên $z$; nó học một siêu phẳng trong $\mathbb{R}^k$, tương đương với siêu phẳng $w = V_k\beta$ nằm trong $\operatorname{span}(V_k)$ của không gian từ gốc. Giảm $k$ do đó là một dạng điều chuẩn: mô hình bị ép chỉ dùng các tổ hợp từ mang nhiều năng lượng.

### 4.3. Thủ thuật Gram khi $n < d$

Khi số tài liệu $n$ nhỏ hơn số từ $d$ (trong notebook $n = 4136 < d = 5649$), $X$ có không quá $n$ giá trị kỳ dị khác 0 và chúng là căn bậc hai các trị riêng của ma trận Gram $G = XX^\mathsf{T} \in \mathbb{R}^{n\times n}$:

$$
XX^\mathsf{T} = U\Sigma V^\mathsf{T}V\Sigma^\mathsf{T}U^\mathsf{T} = U(\Sigma\Sigma^\mathsf{T})U^\mathsf{T}. \tag{27}
$$

Chéo hóa $G$ cho ngay $U$ và $\sigma_i$; các vectơ kỳ dị phải thu được từ (2): $v_i = X^\mathsf{T}u_i/\sigma_i$. Chi phí là $O(n^2 \cdot \mathrm{nnz}/n)$ để lập $G$ (tích hai ma trận thưa) cộng $O(n^3)$ cho `eigh`, không phụ thuộc $d$. Cách này cho **toàn bộ phổ chính xác**, được notebook dùng để vẽ phổ, tính năng lượng tích lũy và lấy $V_k$ chính xác cho thí nghiệm Eckart–Young. Với PCA, ma trận Gram đã tâm hóa có công thức đóng (xem 4.4), nên cùng thủ thuật áp dụng được mà không cần lập $X_c$.

### 4.4. Tâm hóa ngầm cho ma trận thưa

Ma trận $X_c = X - \mathbf{1}\mu^\mathsf{T}$ là **đặc** (hầu như mọi phần tử bằng $-\mu_t \ne 0$) nên không bao giờ nên lập tường minh: với $n = 4136$, $d = 5649$ nó chiếm 187 MB thay vì 0,45 MB của $X$ ở dạng CSR. May mắn là mọi thuật toán cần đến $X_c$ chỉ dùng nó qua hai phép nhân, và cả hai đều tính được từ $X$ thưa cộng một hiệu chỉnh hạng một:

$$
(X - \mathbf{1}\mu^\mathsf{T})M = XM - \mathbf{1}(\mu^\mathsf{T}M), \qquad (X - \mathbf{1}\mu^\mathsf{T})^\mathsf{T}M = X^\mathsf{T}M - \mu(\mathbf{1}^\mathsf{T}M). \tag{28}
$$

Ở vế phải, $XM$ và $X^\mathsf{T}M$ là tích thưa–đặc, $\mu^\mathsf{T}M$ là một vectơ hàng, $\mathbf{1}^\mathsf{T}M$ là tổng các hàng của $M$. Chi phí thêm là $O((n + d)\,\ell)$ với $\ell$ là số cột của $M$, không đáng kể. Lớp `CenteredOperator` trong notebook cài đúng hai công thức này, và `PCAScratch` chạy randomized SVD trên toán tử đó.

Ma trận Gram đã tâm hóa cũng có dạng đóng:

$$
X_cX_c^\mathsf{T} = XX^\mathsf{T} - (X\mu)\mathbf{1}^\mathsf{T} - \mathbf{1}(X\mu)^\mathsf{T} + \lVert\mu\rVert^2\,\mathbf{1}\mathbf{1}^\mathsf{T}, \qquad X_c^\mathsf{T}X_c = X^\mathsf{T}X - n\,\mu\mu^\mathsf{T}. \tag{29}
$$

*Chứng minh.* Khai triển $(X - \mathbf{1}\mu^\mathsf{T})(X - \mathbf{1}\mu^\mathsf{T})^\mathsf{T} = XX^\mathsf{T} - X\mu\mathbf{1}^\mathsf{T} - \mathbf{1}\mu^\mathsf{T}X^\mathsf{T} + \mathbf{1}\mu^\mathsf{T}\mu\mathbf{1}^\mathsf{T}$; công thức thứ hai là (20) nhân với $n$. $\blacksquare$

### 4.5. Randomized SVD

Khi chỉ cần $k \ll \min(n, d)$ thành phần, thuật toán của Halko, Martinsson và Tropp (2011) tránh cả `eigh` cỡ $n \times n$:

1. Lấy $\Omega \in \mathbb{R}^{d\times\ell}$ Gauss với $\ell = k + p$ ($p$ là oversampling). Tính $Y = X\Omega$ và phân tích $QR$: $Y = QR$. Các cột của $Q$ là cơ sở trực chuẩn của $\operatorname{range}(Y)$, một không gian con ngẫu nhiên $\ell$ chiều "nghiêng" về phía các hướng kỳ dị lớn, vì $X\Omega = U\Sigma(V^\mathsf{T}\Omega)$ nhân mỗi hướng $u_i$ với $\sigma_i$.
2. **Power iteration.** Lặp $q$ lần: $Q \leftarrow \mathrm{qr}\big(X\,\mathrm{qr}(X^\mathsf{T}Q)\big)$. Về mặt toán học điều này thay $X\Omega$ bằng $(XX^\mathsf{T})^qX\Omega = U\Sigma^{2q+1}V^\mathsf{T}\Omega$, tức là thay tỉ số $\sigma_{k+1}/\sigma_k$ bằng $(\sigma_{k+1}/\sigma_k)^{2q+1}$ nên không gian con hội tụ nhanh hơn nhiều về $\operatorname{span}(u_1, \dots, u_k)$. Phân tích $QR$ sau mỗi phép nhân chỉ để giữ ổn định số (không có nó, các cột sẽ nhanh chóng đồng hướng về $u_1$).
3. Chiếu: $B = Q^\mathsf{T}X \in \mathbb{R}^{\ell\times d}$ (nhỏ: $\ell$ hàng). SVD của $B = \tilde{U}\Sigma V^\mathsf{T}$, rồi $U = Q\tilde{U}$. Khi đó $X \approx QQ^\mathsf{T}X = U\Sigma V^\mathsf{T}$.

**Độ phức tạp.** Mỗi vòng cần hai tích thưa–đặc $O(\mathrm{nnz}(X)\cdot\ell)$ và hai phân tích $QR$ cỡ $O((n + d)\ell^2)$, tổng cộng $O\big((q+1)(\mathrm{nnz}(X)\,\ell + (n+d)\ell^2)\big)$, cộng SVD của $B$ cỡ $O(d\ell^2)$. Với $\ell = 2k$ và $k$ cỡ vài trăm, phần $(n+d)\ell^2$ chiếm ưu thế, nên thời gian tăng gần bậc hai theo $k$; đó là dạng của đường thời gian fit trong notebook.

**Sai số.** Kết quả của Halko và cộng sự cho kỳ vọng sai số (chuẩn phổ) sau $q$ vòng power iteration là

$$
\mathbb{E}\lVert X - QQ^\mathsf{T}X\rVert_2 \le \Big[\Big(1 + \sqrt{\tfrac{k}{p-1}}\Big)\sigma_{k+1}^{2q+1} + \tfrac{e\sqrt{k+p}}{p}\Big(\sum_{j>k}\sigma_j^{2(2q+1)}\Big)^{1/2}\Big]^{1/(2q+1)}. \tag{30}
$$

Với phổ suy giảm chậm như dữ liệu văn bản, tổng $\sum_{j>k}\sigma_j^{2(2q+1)}$ chỉ nhỏ khi $q$ đủ lớn; đó là lý do giá trị mặc định $p = 10$, $q = 4$ (phù hợp với ảnh hoặc ma trận đánh giá) cho sai số cỡ 1% trên ma trận TF-IDF, còn $p = k$, $q = 10$ cho sai số dưới $10^{-3}$ (bảng hội tụ trong notebook).

---

## PHẦN 5: KẾT LUẬN VÀ Ý NGHĨA

### 5.1. Những gì đã chứng minh

1. **SVD tồn tại với mọi ma trận** (Định lý 1), và chứng minh là xây dựng: chéo hóa $A^\mathsf{T}A$, lấy căn trị riêng, $u_i = Av_i/\sigma_i$. Notebook cài đúng thuật toán này (`svd_via_eig`) và đối chiếu với `numpy.linalg.svd`: sai số cỡ $10^{-15}$, các nhân tử trực chuẩn, hạng số học xác định đúng.
2. **PCA là SVD của dữ liệu đã tâm hóa** (Định lý 3): trị riêng của ma trận hiệp phương sai là $\sigma_i^2/(n-1)$, các điểm PCA không tương quan (Mệnh đề 4), và thành phần thứ nhất là hướng cực đại phương sai (Định lý 2). Notebook đối chiếu ba cách tính cùng một đại lượng và kiểm tra $\operatorname{Cov}(Z)$ chéo.
3. **Eckart–Young–Mirsky** (Định lý 5): $A_k = AV_kV_k^\mathsf{T}$ là xấp xỉ hạng $k$ tốt nhất và sai số bằng $\sum_{i>k}\sigma_i^2$. Notebook đo trực tiếp sai số này trên ma trận TF-IDF và thấy trùng với lý thuyết đến sai số làm tròn, đồng thời nhỏ hơn hẳn sai số của không gian ngẫu nhiên và của phép chọn từ.
4. **Tâm hóa thay đổi ý nghĩa của thành phần đầu** (mục 2.6): không tâm hóa, $v_1$ gần song song với $\mu$ và mọi tài liệu có cùng dấu trên tọa độ thứ nhất; có tâm hóa, thành phần đầu mới đo sự khác biệt giữa các tài liệu.

### 5.2. Liên hệ với kết quả thực nghiệm

- **Phổ suy giảm chậm.** Ma trận TF-IDF cần hàng trăm thành phần mới đạt một nửa năng lượng (8). Đây không phải là lỗi của SVD mà là tính chất của dữ liệu thưa: mỗi tài liệu dùng một tập từ nhỏ, gần như trực giao với các tài liệu khác. Hệ quả là (i) randomized SVD phải power iteration nhiều hơn thường lệ (mục 4.5), và (ii) với $k$ nhỏ, một phần lớn của $X$ bị bỏ đi theo nghĩa Frobenius.
- **Nhưng thông tin phân loại tập trung hơn năng lượng.** Dù $k = 100$ chỉ giữ một phần năng lượng, bộ phân loại trên 100 tọa độ đã đạt F1 gần mức của mô hình dùng toàn bộ hơn 5000 đặc trưng. Thành phần chính thứ nhất tương quan mạnh với nhãn spam vì spam là nguồn phương sai lớn nhất của bộ dữ liệu; PCA tìm ra điều đó mà không nhìn vào nhãn.
- **Mô hình giảm chiều vẫn thua baseline một chút.** Đây là hệ quả trực tiếp của Định lý 5: phép chiếu giữ lại các hướng nhiều năng lượng, còn các từ hiếm nhưng mang tính phân biệt cao (xuất hiện trong vài tin spam) nằm ở phần đuôi của phổ và bị cắt. Lợi ích đổi lại là một biểu diễn đặc, số chiều nhỏ, có thể dùng cho các mô hình không xử lý được dữ liệu thưa, và cho phép trực quan hóa, diễn giải các trục.
- **LSA và PCA gần như tương đương khi $k$ lớn.** Theo (20), $X^\mathsf{T}X$ và $X_c^\mathsf{T}X_c$ chỉ khác nhau một ma trận hạng một; vì vậy hai phổ chỉ khác nhau đáng kể ở một vài thành phần đầu, và khi $k$ vượt vài chục, không gian con $\operatorname{span}(V_k)$ của hai phương pháp gần như trùng nhau. Đường F1 của hai phương pháp bám sát nhau trong notebook phản ánh đúng điều này.

### 5.3. Ý nghĩa hình học

SVD cho một cách nhìn thống nhất: mọi ánh xạ tuyến tính $x \mapsto Ax$ là *xoay* ($V^\mathsf{T}$), *co giãn theo các trục* ($\Sigma$), rồi *xoay* lần nữa ($U$). Phổ $\{\sigma_i\}$ cho biết dữ liệu "dày" theo những hướng nào; PCA chọn các hướng dày nhất quanh tâm, LSA chọn quanh gốc tọa độ, và Định lý Eckart–Young bảo đảm rằng không có không gian con nào cùng số chiều giữ được nhiều hơn. Toàn bộ phần thực nghiệm của đồ án là việc kiểm tra rằng những khẳng định đó đúng đến từng chữ số trên một ma trận dữ liệu thực.

---

## TÀI LIỆU THAM KHẢO

1. C. Eckart, G. Young. *The approximation of one matrix by another of lower rank.* Psychometrika 1(3), 1936, 211–218.
2. L. Mirsky. *Symmetric gauge functions and unitarily invariant norms.* Quarterly Journal of Mathematics 11, 1960, 50–59.
3. N. Halko, P. G. Martinsson, J. A. Tropp. *Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions.* SIAM Review 53(2), 2011, 217–288.
4. G. Strang. *Linear Algebra and Learning from Data.* Wellesley-Cambridge Press, 2019, chương I.8–I.9.
5. T. A. Almeida, J. M. Gómez Hidalgo, A. Yamakami. *Contributions to the study of SMS spam filtering: new collection and results.* ACM DocEng 2011. Bộ dữ liệu: UCI Machine Learning Repository, SMS Spam Collection.
6. Tài liệu scikit-learn: `sklearn.decomposition.TruncatedSVD`, `sklearn.decomposition.PCA`, `sklearn.feature_extraction.text.TfidfVectorizer`.
