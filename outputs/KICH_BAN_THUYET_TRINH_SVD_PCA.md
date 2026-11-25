# Kịch bản thuyết trình: Phát hiện SMS spam bằng SVD và PCA

**Người nói:** một sinh viên, xưng "em".
**Thời lượng:** khoảng 9 phút (khung 8–10 phút), chưa tính hỏi đáp.
**Cách dùng:** đọc phần **Lời nói**. Phần **Trình chiếu** là gợi ý để làm slide, không đọc thành tiếng. Mốc thời gian là ước lượng, cần bấm giờ khi tập.

Mọi số liệu lấy từ lần chạy notebook hiện tại ([KET_QUA_MO_HINH.md](../KET_QUA_MO_HINH.md), [test_metrics.csv](results/test_metrics.csv), [validation_metrics.csv](results/validation_metrics.csv)). Ngoại lệ duy nhất là câu hỏi cuối về bản trước; cách lấy số đó được ghi trong [README.md](../README.md), mục 4. Chứng minh đầy đủ của các công thức (để trả lời câu hỏi sâu) nằm trong [script.md](../script.md).

---

## 1. Mở đầu (0:00 – 0:40)

**Trình chiếu:** tên đề tài; hai tin ví dụ: "Can we meet at the library tomorrow?" và "Congratulations! You have won a free prize…".

**Lời nói**

> Em chào thầy cô và các bạn. Em xin trình bày đề tài "Phát hiện SMS spam bằng SVD và PCA".
>
> Mỗi tin nhắn chỉ dài vài dòng, nhưng khi chuyển thành số thì cần tới 1000 đặc trưng. Câu hỏi của em là: nếu dùng SVD hoặc PCA giảm xuống còn 10, 50 hay 100 chiều thì máy còn nhận ra spam tốt không?
>
> Toàn bộ thuật toán, gồm TF-IDF, SVD, PCA và Logistic Regression, em tự cài bằng NumPy. Phần em muốn nhấn mạnh nhất là mạch đi từ loss function đến score function.

## 2. Dữ liệu (0:40 – 1:30)

**Trình chiếu:** 5.574 → 5.159 tin; bảng chia 3.611 / 774 / 774; tỷ lệ spam 12,4–12,5%; hình [class_distribution.png](figures/class_distribution.png).

**Lời nói**

> Em dùng bộ SMS Spam Collection của UCI, gồm 5.574 tin nhắn tiếng Anh có nhãn ham hoặc spam. Sau khi bỏ 415 tin trùng nội dung, còn 5.159 tin.
>
> Em chia 70% train, 15% validation và 15% test, tức 3.611, 774 và 774 tin. Em chia theo từng lớp nên tập nào cũng có khoảng 12,5% spam.
>
> Train dùng để học, validation dùng để chọn số chiều k, còn test chỉ được dùng đúng một lần ở cuối.
>
> Vì spam ít như vậy, một mô hình luôn đoán ham đã đạt accuracy khoảng 87,5%. Nên em không chọn mô hình bằng accuracy mà bằng macro-F1.

## 3. TF-IDF (1:30 – 2:15)

**Trình chiếu:** công thức $\mathrm{tf} = 1 + \log f$, $\mathrm{idf} = \log\frac{1+N}{1+\mathrm{df}} + 1$; ma trận $X$: 3.611 × 1.000, 1,04% phần tử khác 0.

**Lời nói**

> Để máy xử lý được văn bản, em biến mỗi tin nhắn thành một hàng của ma trận X. Mỗi cột ứng với một từ, em giữ 1000 từ xuất hiện nhiều nhất trong train.
>
> Trọng số của một từ bằng tf nhân idf. tf tăng khi từ lặp lại trong tin. idf lớn khi từ hiếm, nên từ như "the" có trọng số thấp, còn từ như "operator" hay "rates" có trọng số cao. Cuối cùng em chuẩn hóa để mỗi hàng có độ dài bằng 1. Riêng vài tin không chứa từ nào trong danh sách thì hàng đó bằng 0.
>
> Ma trận train có kích thước 3.611 nhân 1.000, nhưng chỉ khoảng 1% phần tử khác 0, vì mỗi tin chỉ dùng chừng 10 từ.

## 4. SVD (2:15 – 3:25)

**Trình chiếu:** $X = U\Sigma V^T$, $Z = XV_k$; vòng lặp $Q \leftarrow \mathrm{QR}(X^TXQ)$; loss $= \sum_{j \gt k}\sigma_j^2$, score = năng lượng giữ lại; hình [retention.png](figures/retention.png).

**Lời nói**

> SVD viết X thành U nhân Sigma nhân V chuyển vị. Các cột của V là vector riêng của X chuyển vị nhân X, còn các giá trị kỳ dị sigma là căn bậc hai của trị riêng. Giữ k hướng đầu thì mỗi tin chỉ còn k tọa độ: Z bằng X nhân V_k. Mỗi tọa độ là tổ hợp của cả 1000 từ, không phải k từ được chọn ra.
>
> Để tìm V_k, em dùng orthogonal iteration: lấy một ma trận ngẫu nhiên, nhân với X chuyển vị X, rồi dùng QR để giữ các cột vuông góc. Lặp lại đến khi các trị riêng không đổi nữa.
>
> Phần này cũng có loss và score. Loss là sai số tái tạo, bằng tổng bình phương các sigma bị bỏ đi. Định lý Eckart–Young cho biết không có ma trận hạng k nào có loss nhỏ hơn. Em kiểm tra lại: loss của SVD là 2043, còn nếu chiếu lên 100 hướng ngẫu nhiên thì loss là 3232. Score là tỷ lệ năng lượng giữ lại.
>
> Với k bằng 100, tức 10% số cột, SVD chỉ giữ được 43% năng lượng. Năng lượng của tin nhắn ngắn trải ra trên rất nhiều hướng nhỏ, nên khó nén.

## 5. PCA (3:25 – 4:05)

**Trình chiếu:** $X_c = X - \mu$, $C = \frac{X_c^TX_c}{N-1} = V\frac{\Sigma^2}{N-1}V^T$; $\cos(v_1, \mu) = 0{,}989$.

**Lời nói**

> PCA tìm các hướng có phương sai lớn nhất quanh giá trị trung bình. Em trừ mean của train rồi làm SVD trên ma trận đã trừ mean. Như công thức trên màn hình, ma trận hiệp phương sai có cùng vector riêng với X_c chuyển vị X_c, nên PCA chính là SVD của dữ liệu đã trừ mean. Code PCA của em dùng lại đúng vòng lặp của SVD.
>
> Một quan sát thú vị: hướng đầu tiên của SVD gần trùng với vector mean, cosin bằng 0,989. Tức là SVD dùng hướng mạnh nhất để mô tả "tin nhắn trung bình", còn PCA đã trừ phần đó từ trước. Ở k bằng 100, PCA giữ 41,5% phương sai.

## 6. Từ loss function đến score function (4:05 – 7:05)

Đây là phần trọng tâm, dành khoảng 3 phút.

**Trình chiếu (một slide cho mỗi bước):**

1. Mô hình: $z = Xw + b$, $p = \sigma(z) = \frac{1}{1+e^{-z}}$
2. Loss: $J = -\frac1N\sum[y\log p + (1-y)\log(1-p)] + \frac{\lambda}{2N}\Vert w\Vert^2$, λ = 1
3. Gradient: $\frac{\partial J}{\partial w} = \frac1N X^T(p - y) + \frac{\lambda}{N}w$, $\frac{\partial J}{\partial b} = \frac1N\sum(p - y)$; gradient check: sai số $2 \cdot 10^{-9}$
4. Gradient descent: $w \leftarrow w - \eta\nabla_w J$; hình [learning_rate.png](figures/learning_rate.png) và [loss_curve.png](figures/loss_curve.png)
5. Score: bảng TP/FP/FN/TN và ví dụ 6 tin

**Lời nói**

> Bây giờ em đi qua Logistic Regression theo đúng năm bước: mô hình, loss, gradient, tối ưu, rồi dự đoán và score. Mỗi bước là một hoặc hai hàm trong code.
>
> **Bước 1, mô hình.** Em tính điểm z bằng X nhân w cộng b, rồi đưa qua hàm sigmoid để được xác suất p là spam, nằm giữa 0 và 1.
>
> **Bước 2, loss function.** Em dùng cross-entropy. Nếu tin là spam mà mô hình cho p nhỏ, số hạng trừ log p rất lớn, tức là đoán sai mà tự tin thì bị phạt nặng. Em cộng thêm phạt L2 với lambda bằng 1 để w không quá lớn. Khi w và b bằng 0, p bằng 0,5 với mọi tin, nên loss ban đầu bằng log 2, tức 0,693.
>
> **Bước 3, gradient.** Nhờ đạo hàm của sigmoid bằng p nhân 1 trừ p, gradient có dạng rất gọn: X chuyển vị nhân với sai số p trừ y, chia N, cộng thêm phần của L2. Để chắc công thức đúng, em làm gradient check: so với đạo hàm tính bằng sai phân hữu hạn trên 50 tin nhắn. Sai số lớn nhất chỉ khoảng 2 nhân 10 mũ trừ 9, nhỏ hơn nhiều so với ngưỡng 10 mũ trừ 6.
>
> **Bước 4, tối ưu.** Em dùng gradient descent: mỗi bước trừ đi eta nhân gradient. Chọn eta rất quan trọng. Gọi L là mức thay đổi nhanh nhất của gradient. Lý thuyết cho biết nếu eta nhỏ hơn 2 chia L thì loss chắc chắn giảm. Em tính L bằng chính hàm orthogonal iteration của phần SVD, chỉ lấy 1 hướng, được L bằng 0,26, tức ngưỡng an toàn là 7,8. Trên hình, eta bằng 50 làm loss dao động, eta bằng 0,5 thì quá chậm, nên em chọn eta bằng 2. Vòng lặp dừng khi loss thay đổi ít hơn 10 mũ trừ 10. Hình loss theo epoch cho thấy cả ba pipeline đều có loss giảm sau mọi epoch và hội tụ sau khoảng 7 đến 10 nghìn epoch.
>
> **Bước 5, score function.** Loss dùng để học, còn score dùng để đánh giá. Em dự đoán spam khi p từ 0,5 trở lên, rồi đếm bốn ô TP, FP, FN, TN.
>
> Em lấy ví dụ 6 tin: 2 spam và 4 ham. Giả sử mô hình bắt được 1 spam, bỏ sót 1 spam và báo nhầm 1 tin ham. Vậy TP bằng 1, FN bằng 1, FP bằng 1, TN bằng 3. Accuracy bằng 4 trên 6, khoảng 0,67. Precision spam bằng 1 trên 2, recall spam cũng bằng 1 trên 2, nên F1 spam bằng 0,5. Làm tương tự cho lớp ham được F1 bằng 0,75. Macro-F1 là trung bình hai số này, bằng 0,625. Code của em cho ra đúng các số này.
>
> Cuối cùng, phần SVD và PCA cũng đi theo đúng mạch đó: loss là sai số tái tạo, score là tỷ lệ năng lượng hoặc phương sai giữ lại. Còn với cả bài toán, score cuối cùng là macro-F1 trên dữ liệu mô hình chưa thấy.

## 7. Chọn k và kết quả trên test (7:05 – 8:35)

**Trình chiếu:** hình [validation_f1.png](figures/validation_f1.png); bảng test bên dưới; hình [test_confusion.png](figures/test_confusion.png).

| Pipeline | Số chiều | Accuracy | Recall spam | Macro-F1 test |
|---|---:|---:|---:|---:|
| Baseline (TF-IDF) | 1.000 | 97,80% | 84,5% | 0,9468 |
| SVD | 100 | 97,03% | 78,4% | 0,9259 |
| PCA | 100 | 97,03% | 78,4% | 0,9259 |

**Lời nói**

> Em thử k bằng 10, 50 và 100 trên validation. Macro-F1 tăng dần theo k và cao nhất ở k bằng 100: SVD đạt 0,907, PCA đạt 0,903. Cả hai vẫn thấp hơn baseline dùng đủ 1000 chiều, với 0,922. Vì vậy em khóa k bằng 100, fit lại trên train cộng validation, rồi đánh giá test đúng một lần.
>
> Trên test, baseline đạt macro-F1 0,947. SVD và PCA cùng đạt 0,926, tức thấp hơn khoảng 0,02.
>
> Nhìn confusion matrix thì rõ hơn. Cả ba mô hình chỉ báo nhầm 2 trong 677 tin ham. Khác biệt nằm ở spam bị bỏ sót: baseline bỏ sót 15 trên 97 tin, còn SVD và PCA bỏ sót 21 tin.

## 8. Kết luận (8:35 – 9:20)

**Trình chiếu:** ba ý chính và giới hạn.

**Lời nói**

> Em rút ra ba điều.
>
> Thứ nhất, mọi bước học trong đồ án đi cùng một mạch: định nghĩa mô hình, đặt loss, tối ưu loss, rồi đánh giá bằng score.
>
> Thứ hai, giảm từ 1000 xuống 100 chiều bớt được 90% số cột, nhưng macro-F1 giảm khoảng 0,02 và bỏ sót thêm 6 tin spam.
>
> Thứ ba, em cho rằng nguyên nhân là hai bước tối ưu hai mục tiêu khác nhau. SVD và PCA làm loss tái tạo nhỏ nhất mà không nhìn nhãn, còn phân loại cần những hướng phân biệt được spam. Giữ được nhiều năng lượng nhất chưa chắc là giữ được nhiều tín hiệu spam nhất. Đây là giả thuyết, em chưa có thí nghiệm riêng để kiểm chứng.
>
> Đồ án còn giới hạn: em mới chia dữ liệu một lần, thử ba giá trị k, chỉ dùng SMS tiếng Anh và chưa kiểm định thống kê các chênh lệch.
>
> Phần trình bày của em đến đây là hết. Em cảm ơn thầy cô và các bạn đã lắng nghe.

---

## Câu hỏi thường gặp

**SVD và PCA có phải là mô hình phân loại không?**
Không. Chúng chỉ tạo biểu diễn ít chiều hơn. Logistic Regression mới là bộ phân loại.

**SVD và PCA khác nhau ở đâu?**
PCA trừ mean trước rồi mới làm SVD. SVD làm trực tiếp trên TF-IDF. Trong bài, hướng đầu của SVD gần trùng mean (cos = 0,989), và có thể vì vậy mà hai cách cho kết quả gần như nhau.

**100 chiều có phải là giữ lại 100 từ không?**
Không. Mỗi chiều là một tổ hợp tuyến tính của cả 1000 từ.

**Tại sao không dùng accuracy để chọn k?**
Spam chỉ chiếm khoảng 12,5%. Luôn đoán ham đã được accuracy khoảng 87,5% mà không bắt được spam nào. Macro-F1 lấy trung bình F1 của hai lớp nên không bị che như vậy.

**Tại sao chọn η = 2?**
L là hằng số Lipschitz của gradient, đo gradient đổi nhanh nhất đến đâu. L đo được ≈ 0,26, nên η < 2/L ≈ 7,8 bảo đảm loss giảm. Hơn nữa, mỗi hàng TF-IDF có độ dài không quá 1, nên trị riêng lớn nhất của $[X\ 1]^T[X\ 1]$ không vượt tổng đường chéo $\sum_i(\Vert x_i \Vert^2 + 1) \le 2N$. Với đặc trưng Z của SVD/PCA cũng có $\Vert Z \Vert_F^2 \le N$. Vì vậy L không vượt quá khoảng 0,5 cho cả ba pipeline, với mọi tập dữ liệu. Vì vậy η = 2 vẫn an toàn khi dữ liệu thay đổi. Hình learning rate cho thấy η = 50 làm loss dao động.

**Làm sao biết code SVD tự viết đúng?**
Bộ test chạy vòng lặp với sai số dừng chặt (10⁻¹⁴) trên một ma trận nhỏ 40 × 8 và so với `np.linalg.svd`, `np.linalg.eigh`: giá trị kỳ dị khớp tới 10⁻⁸, không gian con khớp tới 10⁻⁶. Hai hàm NumPy đó chỉ dùng trong test, không dùng trong pipeline. Trên dữ liệu thật, notebook kiểm tra phần dư $\Vert X^TXv_j - \lambda_j v_j \Vert$ chỉ cỡ $10^{-5}\lambda_1$, và loss của SVD (2043) nhỏ hơn nhiều so với 100 hướng ngẫu nhiên (3232).

**Tại sao vòng lặp SVD với k = 100 cần khoảng 2.400 vòng?**
Tốc độ hội tụ phụ thuộc tỷ số giữa các trị riêng kề nhau. Ở k = 100, các trị riêng rất sát nhau nên hội tụ chậm.

**Kết quả có khớp bản trước không?**
Khi chạy code mới trên đúng cách chia dữ liệu của bản trước, baseline cho macro-F1 test 0,9310, trùng bản trước. Con số 0,9468 khác là do cách chia mới tạo ra một tập test khác.

## Ghi nhớ khi tập nói

- Mạch chính: dữ liệu → TF-IDF → SVD → PCA → **loss → gradient → GD → score** → kết quả.
- Phần 6 là trọng tâm: không cắt, và nên chỉ vào hình loss giảm khi nói bước 4.
- Đọc số làm tròn: 0,947 và 0,926; bỏ sót 15 và 21 tin spam.
- Phân biệt macro-F1 validation (dùng để chọn k) với macro-F1 test (chỉ xem một lần).
- Nếu bị thiếu giờ, rút phần 3 và phần 5 còn khoảng 30 giây mỗi phần.
