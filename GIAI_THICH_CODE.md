# Giải thích source code: SMS spam với PCA và SVD

Tài liệu này giúp bạn biết **phần nào làm việc gì**, input/output ra sao và nên đọc theo thứ tự nào. Hướng dẫn chạy nằm ở [README.md](README.md); số liệu và hình đã chạy nằm ở [KET_QUA_MO_HINH.md](KET_QUA_MO_HINH.md).

## 1. Hai cách đọc cùng một project

- [svd_pca_spam.ipynb](svd_pca_spam.ipynb): source theo mạch học và thuyết trình, gồm phần giải thích, 16 ô code và output ngay bên dưới.
- [src/](src/): các hàm Python được notebook gọi lại. Đọc ở đây để hiểu chi tiết xử lý, huấn luyện và dự đoán.

Notebook gọi các hàm dùng chung để cách tính giữa notebook và terminal nhất quán. Khi sửa thuật toán, sửa hàm trong `src/` rồi chạy lại notebook để cập nhật kết quả. Các script trong [scripts/](scripts/) chỉ điều phối notebook hoặc xuất Markdown.

Luồng chính:

```text
data.py → models.py → train.py → outputs/
              ↑                    ↓
       math_demo.py              predict.py
                              plots.py đọc CSV/JSON
```

`math_demo.py` minh họa toán trên ma trận nhỏ, độc lập với huấn luyện SMS. Các ô notebook nối những phần này thành một bài trình bày liên tục.

## 2. Bản đồ 16 ô code trong notebook

Số dưới đây là **thứ tự ô code**, không tính các ô Markdown và không phụ thuộc số execution count hiển thị trong Jupyter.

| Ô | Công việc | Input → Output | Hàm hoặc module chính |
|---:|---|---|---|
| 1 | Xác định thư mục project, kiểm tra/cài thư viện, giới hạn CPU, tạo nơi lưu kết quả và hàm hiển thị bảng | Thư mục project → các biến dùng chung và logger | `ensure_dependencies`, `prepare_output_and_logger`, `show_table` |
| 2 | Đọc, làm sạch, chia SMS và xem mẫu | File UCI → `splits`, thống kê và bảng nhãn | `download_data`, `prepare_data_and_audit` |
| 3 | Vẽ phân bố ham/spam | Thống kê đã làm sạch → biểu đồ phân bố lớp | Matplotlib |
| 4 | Khảo sát TF-IDF, sparsity và dung lượng | SMS train → `X_train`, số chiều `D`, trọng số từ mẫu | `make_pipeline`, `TfidfVectorizer` |
| 5 | Minh họa SVD và giữ `k=2` | Ma trận 6×4 → `U`, singular values, `Vt`, `Z` và sai số | `run_demo`, `np.linalg.svd` |
| 6 | Minh họa PCA và quan hệ với SVD centered | Ma trận nhỏ → mean, covariance, eigenvalues và kiểm chứng | `np.linalg.eigh`, `np.linalg.svd` |
| 7 | In các bước của ba pipeline | Tên phương pháp → chuỗi transformer/classifier | `make_pipeline` |
| 8 | Fit baseline, PCA/SVD với các `k` hợp lệ trên train | Train/validation → bảng validation metrics và CSV | `fit_evaluate`, fingerprint TF-IDF |
| 9 | Chọn `k`, refit và đánh giá test | Metrics validation → `selected`, ba model, test metrics và hình | `choose_k`, `refit_and_evaluate_test`, `make_plots` |
| 10 | Xem tác động của `k` đến phân loại | Validation CSV → đồ thị macro-F1 | Ảnh `validation_f1.png` |
| 11 | Phân tích lỗi dự đoán trên test | Confusion matrix → số spam bị bỏ sót và ham báo nhầm | `confusion_matrices.json` |
| 12 | Xem thông tin giữ lại | Metrics train của từng cấu hình → PCA variance và SVD energy | Ảnh `retention.png` |
| 13 | So thời gian, số chiều và dung lượng | Metrics đã đo → bảng/biểu đồ chi phí | `fit_time.png`, `representation_size.png` |
| 14 | Nạp model và dự đoán hai SMS minh họa | File `.joblib` + SMS mới → nhãn và P(spam) | `joblib.load`, `predict_message` |
| 15 | Chạy kiểm thử tự động | Thư mục `tests/` → log, số test và trạng thái đạt/lỗi | `unittest` |
| 16 | Tính nhận xét từ số liệu thật | Test metrics → mức giảm chiều, giảm F1 và tỷ lệ dung lượng | Pandas |

Ô 1 tìm thư mục chứa `src/` và `requirements.txt` trước khi import, rồi dùng `scripts/notebook_setup.py` để cài thư viện còn thiếu vào đúng Python của kernel. Ô 4 chỉ khảo sát TF-IDF trên train; các pipeline ở ô 8 tự fit TF-IDF cùng cấu hình trên đúng tập train đó. Ô 9 khóa `k` trước khi đánh giá test. Ô 16 chỉ mô tả kết quả, không chọn lại tham số.

## 3. Dữ liệu: `src/data.py`

| Hàm | Nhận gì? | Trả hoặc tạo gì? |
|---|---|---|
| `normalize_text(text)` | Một chuỗi SMS | Chuỗi chữ thường, bỏ khoảng trắng thừa |
| `download_data(destination)` | Đường dẫn dataset | File UCI, README nguồn và thông tin gói tải |
| `load_data(path)` | File nhãn và nội dung ngăn bằng tab | DataFrame sạch và thống kê |
| `clean_data(raw)` | DataFrame `source_id`, `label`, `text` | Loại rỗng/trùng/mâu thuẫn; thêm hash nội dung |
| `split_data(frame, seed)` | Dữ liệu sạch | Dict chứa `train`, `validation`, `test` |

Mỗi hàng là một SMS. Mỗi nội dung chuẩn hóa giữ một bản. Nếu cùng nội dung có nhãn mâu thuẫn, loại cả nhóm. Làm sạch trước khi chia để tránh nội dung giống nhau xuất hiện ở nhiều tập.

Tỷ lệ gần **70/15/15** được thực hiện bằng hai lần `train_test_split`, có stratification để giữ tỷ lệ nhãn. Seed là 42. Các bước này không học vocabulary hoặc hướng chiếu.

## 4. Pipeline: `src/models.py`

Đây là nơi nên đọc đầu tiên khi muốn hiểu hoặc thay cấu hình. `MAX_FEATURES=1000`, `K_VALUES=(10, 50, 100)` và `DEFAULT_SEED=42` được đặt ở đầu file.

`make_pipeline(method, k, seed)` trả về **pipeline chưa fit**:

| `method` | Các bước | Biểu diễn trước classifier |
|---|---|---|
| `baseline` | TF-IDF → Logistic Regression | Sparse, tối đa 1.000 chiều |
| `pca` | TF-IDF → `ToDense` → PCA → Logistic Regression | Dense, `k` chiều |
| `svd` | TF-IDF → Truncated SVD → Logistic Regression | Dense, `k` chiều |

TF-IDF dùng unigram, `min_df=2`, sublinear TF và chuẩn hóa L2. Logistic Regression dùng `C=1`, `lbfgs`, `max_iter=2000`. Classifier giống nhau để khảo sát tác động của giảm chiều.

`ToDense.transform(X)` đổi sparse thành dense cho PCA. Hàm ước lượng dung lượng đầu vào và chặn nếu vượt 128 MiB; đây chưa phải peak RAM vì PCA còn dùng bộ nhớ trung gian. Truncated SVD nhận sparse trực tiếp.

## 5. Phần toán: `src/math_demo.py`

- `pca_eigh(X, k)` học mean và các hướng chính từ covariance; trả dict chứa mean, components, eigenvalues.
- `pca_transform(X, fit)` chiếu bằng mean và components đã học; trả ma trận giảm chiều `Z`.
- `pca_inverse(Z, fit)` tái tạo gần đúng dữ liệu, có cộng lại mean.
- `run_demo()` tạo ma trận 6×4 và chạy kiểm chứng; trả dict lưu được vào JSON.

SVD dùng `X = UΣVᵀ`, giữ `k` thành phần và chiếu `Z = XV_k`. PCA center `X_c = X − mean`, rồi tìm các hướng phương sai lớn. Demo kiểm chứng:

```text
eigenvalue_j = singular_value_j(X_c)² / (N − 1)
```

Code so projector của hai không gian con để không nhầm các vector chỉ khác dấu. Demo cũng kiểm tra trực giao, reconstruction, sai số giảm khi tăng `k` và đối chiếu NumPy với scikit-learn. Đây là ví dụ toán nhỏ, không thay cho thí nghiệm SMS.

## 6. Huấn luyện: `src/train.py`

`run(data_path, output_root)` điều phối sáu bước: tạo output → chuẩn bị dữ liệu → thử `k` → khóa lựa chọn → refit/test → lưu cấu hình và vẽ hình.

| Hàm | Vai trò và đầu ra |
|---|---|
| `prepare_output_and_logger` | Tạo `outputs/results`, `outputs/figures`, `outputs/models`; trả logger và đường dẫn kết quả |
| `prepare_data_and_audit` | Đọc/chia dữ liệu; lưu thống kê và ID từng split; trả `splits` |
| `fit_evaluate` | Nhận phương pháp, `k`, train và evaluation; trả pipeline đã fit, một dòng metrics và nhãn dự đoán |
| `score_predictions` | Nhận nhãn thật/dự đoán; trả accuracy, macro-F1 và precision/recall/F1 lớp spam |
| `run_validation_sweep` | Fit baseline và grid `k`; lưu `validation_metrics.csv` |
| `choose_k` | Chọn macro-F1 validation cao nhất; hòa thì lấy `k` nhỏ hơn |
| `refit_and_evaluate_test` | Fit train+validation với `k` đã khóa; lưu model, test metrics, confusion matrix và demo |
| `save_run_config` | Lưu seed, tham số, phiên bản và phạm vi đo thời gian |

`feature_fingerprint` băm vocabulary/IDF để kiểm tra các nhánh có cùng TF-IDF. `array_bytes` đo dữ liệu CSR hoặc dense. Thời gian fit gồm cả TF-IDF, reducer nếu có và classifier; inference lấy median của 5 batch sau warm-up.

Train dùng để học; validation dùng để chọn `k`; test dùng để đánh giá cuối. Khi refit trên train+validation, TF-IDF và reducer được học lại trên phần development này. Test vẫn chỉ được transform. Không dùng output test để sửa `k`.

## 7. Dự đoán, biểu đồ và công cụ notebook

[src/predict.py](src/predict.py) có `predict_message(model, text)`: nhận pipeline đã học và SMS, trả dict `text`, `prediction`, `spam_probability`. Hàm tìm đúng cột `spam` theo `model.classes_`. Lệnh terminal chỉ load `.joblib`, không fit lại.

[src/plots.py](src/plots.py) đọc CSV/JSON đã lưu để vẽ hình; không train model. [scripts/run_notebook.py](scripts/run_notebook.py) chạy 16 ô bằng Python hiện tại rồi xuất báo cáo. [scripts/ipynb_to_markdown.py](scripts/ipynb_to_markdown.py) chỉ chuyển notebook đã có output thành Markdown và ảnh.

Các file [tests/test_math.py](tests/test_math.py), [tests/test_pipeline.py](tests/test_pipeline.py), [tests/test_notebook_export.py](tests/test_notebook_export.py) và [tests/test_notebook_setup.py](tests/test_notebook_setup.py) kiểm tra toán, dữ liệu/pipeline, lưu/load, quá trình xuất output và cài thư viện bằng đúng Python của kernel. Log kiểm thử ở [outputs/results/test_output.txt](outputs/results/test_output.txt).

## 8. Đọc output và giải thích khi bảo vệ

- **Số liệu chính:** [validation_metrics.csv](outputs/results/validation_metrics.csv), [test_metrics.csv](outputs/results/test_metrics.csv).
- **Cấu hình và lựa chọn:** [run_config.json](outputs/results/run_config.json), [selected_k.json](outputs/results/selected_k.json).
- **Minh chứng toán/demo:** [math_demo.json](outputs/results/math_demo.json), [demo_predictions.json](outputs/results/demo_predictions.json).
- **Mô hình đã học:** [outputs/models/](outputs/models/); cần source và thư viện tương thích để nạp pipeline.

Macro-F1 trung bình F1 hai lớp nên hữu ích khi ham/spam mất cân bằng. Spam recall cho biết bắt được bao nhiêu spam; spam precision cho biết các dự đoán spam đúng bao nhiêu. Xác suất demo là ước lượng, chưa được hiệu chỉnh riêng.

PCA variance đo dữ liệu đã center; SVD energy đo dữ liệu không center. Không xem chúng là cùng tỷ lệ. Giảm `D` xuống `k` cũng chưa bảo đảm giảm dung lượng vì sparse có thể thành dense. Trong snapshot kèm project, baseline tốt hơn hai nhánh giảm chiều; hãy dùng số liệu thật để giải thích trade-off thay vì mặc định giảm chiều luôn tốt hơn.
