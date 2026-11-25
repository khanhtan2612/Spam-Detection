# Giải thích code: phát hiện SMS spam bằng SVD và PCA

Tài liệu này cho biết mỗi file làm gì, mỗi hàm ứng với công thức nào và nên đọc theo thứ tự nào. Chứng minh và giải thích toán của từng công thức nằm ở [script.md](script.md); code, output và hình đã chạy nằm ở [KET_QUA_MO_HINH.md](KET_QUA_MO_HINH.md).

Toàn bộ thuật toán được tự cài bằng **NumPy**. pandas chỉ dùng để đọc và ghi bảng, matplotlib chỉ dùng để vẽ. `np.linalg.svd` và `np.linalg.eigh` chỉ xuất hiện trong `tests/` để đối chiếu kết quả.

## 1. Đọc theo thứ tự nào?

```text
data.py → tfidf.py → decomposition.py → logistic.py → metrics.py
   (dữ liệu)  (văn bản→X)  (SVD, PCA)        (mô hình→loss→gradient→GD)  (score)
                                  ↘             ↓
                                    pipeline.py (ghép 3 bước, lưu/đọc model)
                                          ↓
                                    train.py (chọn k → refit → test) → plots.py, predict.py
```

Notebook [svd_pca_spam.ipynb](svd_pca_spam.ipynb) đi đúng thứ tự này và gọi lại các hàm trong `src/`. Muốn sửa thuật toán thì sửa trong `src/`, rồi chạy lại notebook.

## 2. Bản đồ 17 ô code của notebook

| Ô | Mục | Việc làm | Hàm chính |
|---:|---|---|---|
| 1 | Chuẩn bị | Tìm thư mục project, import, hàm hiển thị bảng | `show_table`, `show_figure` |
| 2 | 2. Dữ liệu | Tải (nếu thiếu), làm sạch, chia 70/15/15, vẽ số tin mỗi lớp | `download_data`, `prepare_data` |
| 3 | 3. TF-IDF | Học vocabulary và idf trên train, xem một hàng của X | `TfidfVectorizer` |
| 4 | 4. SVD | Orthogonal iteration với k = 100; kiểm tra phần dư `XᵀXv_j − λ_j v_j` và so loss với 100 hướng ngẫu nhiên; bảng loss/score | `TruncatedSVD` |
| 5 | 5. PCA | Như SVD nhưng trừ mean trước; so hướng v_1 của SVD với μ; vẽ score theo k | `PCA`, `plot_retention` |
| 6 | 6.1 Mô hình | Giá trị σ(z) | `sigmoid` |
| 7 | 6.2 Loss | J tại w = 0, b = 0 bằng log 2 | `loss` |
| 8 | 6.3 Gradient | Gradient check trên 50 tin | `gradient_check` |
| 9 | 6.4 Tối ưu | Tính L bằng `top_eigenvectors` với k = 1 (power iteration), thử η = 0,5 / 2 / 50 | `top_eigenvectors`, `plot_learning_rates` |
| 10 | 6.4 Tối ưu | Huấn luyện baseline bằng GD, kiểm tra J giảm đều | `fit` |
| 11 | 6.5 Score | Ví dụ 6 tin tính tay | `confusion_matrix`, `classification_scores` |
| 12 | 6.5 Score | Score của baseline trên validation | `score` |
| 13 | 7. Chọn k | Thử k ∈ {10, 50, 100} trên validation, chọn k | `run_validation_sweep`, `choose_k` |
| 14 | 8. Test | Refit trên train + validation, đánh giá test, vẽ loss theo epoch | `refit_and_test`, `plot_loss_curves` |
| 15 | 8. Test | Confusion matrix trên test | `plot_confusion_matrices` |
| 16 | 8. Test | Load model đã lưu, dự đoán 2 tin mới | `load_pipeline`, `predict_message` |
| 17 | 9. Kết luận | Bảng so sánh với baseline | pandas |

## 3. Dữ liệu: `src/data.py`

| Hàm | Làm gì |
|---|---|
| `normalize_text(text)` | Chữ thường, gộp khoảng trắng. Dùng chung khi làm sạch, huấn luyện và dự đoán. |
| `download_data(path)` | Tải file `SMSSpamCollection` từ UCI nếu máy chưa có. |
| `load_data(path)` | Đọc từng dòng `nhãn<TAB>tin nhắn`, rồi gọi `clean_data`. |
| `clean_data(raw)` | Bỏ tin rỗng, bỏ nội dung có hai nhãn mâu thuẫn, giữ một bản cho mỗi nội dung. |
| `stratified_split(df, (0.7, 0.15, 0.15), seed=42)` | Xáo riêng từng lớp bằng `np.random.default_rng(42)`, cắt `round(0.7·n)` tin cho train, `round(0.15·n)` cho validation, phần còn lại cho test. |

Loại trùng làm **trước** khi chia, nên một nội dung không thể nằm ở hai tập. Chia theo từng lớp giúp ba tập có cùng tỷ lệ spam.

## 4. TF-IDF: `src/tfidf.py`

- `tokenize(text)`: tách từ bằng regex `\b\w\w+\b`, tức chuỗi chữ hoặc số dài từ 2 ký tự.
- `TfidfVectorizer.fit(texts)`: giữ các từ có `df ≥ 2`, rồi lấy 1000 từ có tổng số lần xuất hiện lớn nhất. Từ hòa nhau được xếp theo thứ tự chữ cái. Sau đó tính `idf = log((1+N)/(1+df)) + 1`.
- `TfidfVectorizer.transform(texts)`: với mỗi tin, `tf = 1 + log f`, nhân với idf, rồi chia cho chuẩn L2 của hàng. Từ không có trong vocabulary bị bỏ qua. Tin không chứa từ nào trong vocabulary cho hàng toàn số 0 (20 tin trong train), nên mọi hàng có chuẩn ≤ 1.

`fit` chỉ được gọi trên train. Validation và test chỉ đi qua `transform`, nên từ chỉ có trong validation không bao giờ lọt vào vocabulary. Ma trận được lưu dạng dense: 3.611 × 1.000 số thực chỉ khoảng 29 MB.

## 5. SVD và PCA: `src/decomposition.py`

| Hàm / lớp | Công thức |
|---|---|
| `top_eigenvectors(A, k)` | Orthogonal iteration: `Q ← QR(A·Q)`, λ_j = q_jᵀ A q_j. Dừng khi mọi λ_j đổi ít hơn `1e-8·λ_1` giữa hai vòng. Nếu chạm 5000 vòng mà chưa đạt thì in cảnh báo. |
| `TruncatedSVD.fit(X)` | `A = XᵀX`, lấy k vector riêng lớn nhất làm `V_k`, σ_j = √λ_j. |
| `transform(X)` | `Z = X·V_k` |
| `inverse_transform(Z)` | `X_k = Z·V_kᵀ` |
| `loss()` | `‖X − X_k‖²_F = Σ_{j>k} σ_j² = ‖X‖²_F − Σ_{j≤k} σ_j²` |
| `score()` | `Σ_{j≤k} σ_j² / ‖X‖²_F` (tỷ lệ năng lượng giữ lại) |
| `PCA` | Kế thừa `TruncatedSVD`. `fit` học `μ` từ train rồi làm SVD trên `X − μ`. `explained_variance_ = σ_j²/(N−1)`. `score()` là tỷ lệ phương sai giữ lại. |

Tại sao orthogonal iteration hội tụ: mỗi lần nhân với `A = XᵀX`, thành phần theo hướng v_j được nhân với λ_j. Các hướng có λ lớn vì vậy lấn át dần, còn QR giữ các cột trực chuẩn để chúng không cùng đổ về v_1. Khi các λ_j gần bằng nhau (như ở k = 100), vòng lặp hội tụ chậm hơn. Số vòng thực tế được ghi trong cột `decomposition_iterations` của `outputs/results/*_metrics.csv`.

Trị riêng hội tụ nhanh hơn vector riêng: sai số vector cỡ căn bậc hai sai số trị riêng. Vì vậy test chạy với `tol = 1e-14` (chặt hơn mặc định `1e-8`) trên một ma trận nhỏ 40 × 8, rồi so σ với NumPy tới `1e-8` và projector của không gian con tới `1e-6`. Với dữ liệu thật và `tol = 1e-8`, notebook kiểm tra phần dư `‖XᵀXv_j − λ_j v_j‖/λ_1` (cỡ 1e-5) và so loss với 100 hướng ngẫu nhiên. Dự đoán của Logistic Regression (có phạt L2) chỉ phụ thuộc vào không gian con mà V_k sinh ra, không phụ thuộc cách chọn cơ sở trực chuẩn trong không gian đó.

## 6. Logistic Regression: `src/logistic.py`

Lớp `LogisticRegressionGD` đi đúng mạch 5 bước **mô hình → loss → gradient → tối ưu → score**. Bước cuối gồm hai hàm `predict` và `score`, nên trong code có 6 hàm, đánh số 1–6 theo đúng thứ tự. Docstring của mỗi hàm ghi công thức:

| Bước | Hàm | Công thức |
|---|---|---|
| 1. Mô hình | `sigmoid`, `predict_proba` | `z = Xw + b`, `p = σ(z) = 1/(1 + e^(−z))` |
| 2. Loss | `loss` | `J = −(1/N) Σ [y log p + (1−y) log(1−p)] + (λ/2N)‖w‖²`, p clip vào `[1e-12, 1−1e-12]` |
| 3. Gradient | `gradient` | `∂J/∂w = (1/N)Xᵀ(p − y) + (λ/N)w`, `∂J/∂b = (1/N)Σ(p − y)` |
| 4. Tối ưu | `fit` | `w ← w − η∂J/∂w`, `b ← b − η∂J/∂b`; lưu `loss_history_`; dừng khi đủ epoch hoặc `|ΔJ| < tol` |
| 5. Dự đoán và score | `predict`, `score` | spam nếu `p ≥ 0.5`; accuracy, precision/recall/F1 lớp spam, macro-F1 (gọi `metrics.py`) |

`gradient_check(X, y)` so gradient công thức với sai phân trung tâm `(J(θ+ε) − J(θ−ε))/2ε` tại một điểm ngẫu nhiên và trả sai số lớn nhất. Yêu cầu sai số < 1e-6.

**Siêu tham số đã chọn:** λ = 1, η = 2, tối đa 20.000 epoch, tol = 1e-10.

- λ = 1 cho cùng hàm mục tiêu với hệ số `C = 1` của bản trước.
- η = 2. GD bảo đảm J giảm khi `η < 2/L`, với L là hằng số Lipschitz của gradient: `L ≤ ¼·λ_max([X 1]ᵀ[X 1])/N + λ/N` vì σ' ≤ 1/4. Trị riêng lớn nhất không vượt tổng đường chéo, `λ_max ≤ Σ_i(‖x_i‖² + 1) ≤ 2N`, vì mỗi hàng có chuẩn ≤ 1. Với Z của SVD/PCA cũng có `‖Z‖²_F ≤ N`. Suy ra `L ≤ 0,5 + λ/N` cho cả ba pipeline. Notebook tính cận trên này trên dữ liệu thật (0,2575, tức J chắc chắn giảm khi η < 7,77) bằng `top_eigenvectors` với k = 1, và cho thấy η = 50 làm loss dao động.
- tol = 1e-10 và 20.000 epoch: mọi lần huấn luyện đều dừng vì `|ΔJ| < tol` trước khi chạm giới hạn epoch (cột `gd_epochs` trong kết quả).

## 7. Score function: `src/metrics.py`

| Hàm | Làm gì |
|---|---|
| `confusion_matrix(y_true, y_pred)` | Trả `[[TN, FP], [FN, TP]]`, với 1 = spam. Hàng là nhãn thật, cột là nhãn dự đoán. |
| `precision_recall_f1(tp, fp, fn)` | `P = TP/(TP+FP)`, `R = TP/(TP+FN)`, `F1 = 2PR/(P+R)`. Mẫu số bằng 0 thì trả 0. |
| `classification_scores(y_true, y_pred)` | accuracy, precision/recall/F1 lớp spam, F1 lớp ham (đổi vai: TP_ham = TN, FP_ham = FN, FN_ham = FP), macro-F1 = trung bình F1 hai lớp. |

Ví dụ 6 tin tính tay (TP = 1, FN = 1, FP = 1, TN = 3, macro-F1 = 0,625) có trong notebook mục 6.5 và trong `tests/test_metrics.py`.

## 8. Ghép pipeline và lưu model: `src/pipeline.py`

- `SpamPipeline(method, k)`: `method` là `baseline`, `svd` hoặc `pca`. `fit` lần lượt học TF-IDF → reducer (nếu có) → Logistic Regression trên train. `predict`, `predict_proba` và `score` chỉ dùng lại tham số đã học.
- `save_pipeline(model, path)` ghi `path.npz` (idf, V_k, σ, ‖X‖²_F, μ và phương sai nếu là PCA, w, b) bằng `np.savez`, và `path.json` (method, k, vocabulary).
- `load_pipeline(path)` đọc lại hai file đó, không fit lại.
- Hằng số: `SEED = 42`, `MAX_FEATURES = 1000`, `MIN_DF = 2`, `K_VALUES = (10, 50, 100)`.

## 9. Thí nghiệm: `src/train.py`

`run()` (hoặc `python -m src.train`) thực hiện 4 bước:

1. `prepare_data`: đọc, làm sạch, chia tập, lưu `data_summary.json`.
2. `run_validation_sweep`: với baseline và mỗi k của SVD/PCA, gọi `fit_and_score` (fit trên train, score trên validation). Kết quả lưu ở `validation_metrics.csv`. σ của k = 100 được lưu ở `spectrum.json` để vẽ score theo k.
3. `choose_k`: chọn k có macro-F1 validation cao nhất; bằng nhau thì lấy k nhỏ hơn. Kết quả lưu ở `selected_k.json`.
4. `refit_and_test`: fit lại trên train + validation với k đã khóa, đánh giá test một lần. Kết quả lưu ở `test_metrics.csv`, `confusion_matrices.json`, `loss_history.json`, `demo_predictions.json` và model trong `outputs/models/`.

Cuối cùng `save_run_config` ghi lại mọi tham số, và `make_plots` vẽ lại các biểu đồ.

Các cột chính trong CSV: `accuracy`, `spam_precision`, `spam_recall`, `spam_f1`, `ham_f1`, `macro_f1` (score của classifier); `reconstruction_loss`, `retained_ratio` (loss và score của SVD/PCA trên tập fit); `decomposition_iterations`, `gd_epochs` (số vòng lặp đến khi hội tụ); `final_train_loss` (J cuối của GD).

## 10. Biểu đồ, dự đoán và scripts

- [src/plots.py](src/plots.py): mỗi hình là một hàm, vẽ từ số đã lưu. Confusion matrix vẽ bằng `plt.imshow` và `plt.text` ghi số lên từng ô. Các hình lưu trong `outputs/figures/`: `class_distribution`, `retention`, `learning_rate`, `validation_f1`, `loss_curve`, `test_confusion`.
- [src/predict.py](src/predict.py): `predict_message(model, text)` trả nhãn và P(spam). Dòng lệnh: `python -m src.predict --model outputs/models/svd --text "..."`.
- [scripts/run_notebook.py](scripts/run_notebook.py): chạy notebook từ đầu rồi xuất `KET_QUA_MO_HINH.md`.
- [scripts/ipynb_to_markdown.py](scripts/ipynb_to_markdown.py): chỉ chuyển notebook đã chạy thành Markdown, không chạy lại.

## 11. Kiểm thử: `tests/`

Chạy: `python -m unittest discover -s tests -v`

| File | Kiểm tra |
|---|---|
| `test_data.py` | Làm sạch đúng; ba tập không trùng nhau, phủ đủ dữ liệu, giữ tỷ lệ lớp; cùng seed cho cùng kết quả. |
| `test_tfidf.py` | Công thức tf-idf khớp ví dụ tính tay; `min_df`/`max_features`; hàng có chuẩn 1, hoặc bằng 0 nếu tin không có từ nào trong vocabulary; từ chỉ có trong validation không vào vocabulary. |
| `test_decomposition.py` | SVD/PCA tự viết khớp `np.linalg.svd`/`np.linalg.eigh` (σ, phương sai, projector của không gian con); loss = Σ_{j>k} σ_j²; PCA dùng mean của train; cảnh báo khi chưa hội tụ. |
| `test_logistic.py` | Gradient check < 1e-6; J(0) = log 2; loss không tăng khi huấn luyện; ∇J ≈ 0 tại điểm dừng; λ lớn cho w nhỏ hơn. |
| `test_metrics.py` | Ví dụ 6 tin tính tay; trường hợp không dự đoán spam nào. |
| `test_pipeline.py` | Ba pipeline học được dữ liệu nhỏ; predict không đổi tham số đã học; lưu/đọc model cho cùng xác suất; quy tắc chọn k. |
| `test_notebook_export.py` | Xuất Markdown giữ đúng code, output, hình; từ chối notebook chưa chạy hoặc có lỗi. |
