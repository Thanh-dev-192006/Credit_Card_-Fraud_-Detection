# Giải thích Notebook: Balanced Random Forest Classifier

**File notebook:** `BalancedRandomForestClassifier.ipynb`  
**Dataset:** Dữ liệu BankSim gốc, **không qua SMOTE** (từ thư mục `data/processed/`)  
**Mục tiêu:** Huấn luyện mô hình phát hiện giao dịch gian lận sử dụng chiến lược cân bằng dữ liệu nội sinh, không phụ thuộc vào bước tiền xử lý SMOTE bên ngoài.

---

## Tổng quan pipeline

```text
Processed Data (*.pkl) — dữ liệu GỐC, chưa SMOTE
  │
  ▼
Step 1 — Load Data
  │  Đọc X_train (gốc), y_train (gốc), X_test (gốc), y_test (gốc)
  ▼
Step 2 — Training
  │  Khởi tạo và huấn luyện BalancedRandomForestClassifier
  │  (Undersampling lớp đa số được thực hiện tự động trong mỗi cây)
  ▼
Step 3 — Dự đoán
  │  Sinh nhãn dự đoán (predict) và xác suất (predict_proba)
  ▼
Step 4 — Evaluation Metrics
  │  Tính toán Precision, Recall, F1, ROC-AUC, PR-AUC, Classification Report
  ▼
Step 5 — Confusion Matrix
  │  Trực quan hóa ma trận nhầm lẫn
  ▼
Step 6 — Precision-Recall Curve
  │  Vẽ đường cong PR với baseline
  ▼
Step 7 — Feature Importance
  │  Xếp hạng 20 đặc trưng quan trọng nhất theo Mean Decrease Impurity
  ▼
Step 8 — Lưu mô hình
     Serialize model ra file brf_model.pkl để dùng cho bước so sánh tổng hợp
```

---

## Step 1 — Load Data

Điểm khác biệt căn bản so với LightGBM/XGBoost: Notebook này nạp **tập dữ liệu gốc chưa qua SMOTE** (`X_train.pkl`, `y_train.pkl`).

| Tập dữ liệu | Số mẫu | Số features |
|---|---|---|
| Huấn luyện (gốc) | 475,714 | 28 |
| Kiểm tra (gốc) | 118,929 | 28 |

**Phân phối nhãn tập huấn luyện (thực tế):**

| Nhãn | Số mẫu | Tỷ lệ |
|---|---|---|
| 0 (Hợp lệ) | 469,794 | ~98.8% |
| 1 (Gian lận) | 5,920 | ~1.2% |

> **Tại sao không dùng SMOTE?** `BalancedRandomForestClassifier` có cơ chế tự xử lý mất cân bằng bên trong: ở mỗi cây, nó **random under-sample** lớp đa số xuống đúng bằng kích thước lớp thiểu số trước khi huấn luyện cây đó. Tức là mỗi cây học trên một bootstrap sample cân bằng 50/50, nhưng toàn bộ thông tin gốc vẫn được khai thác xuyên suốt 200 cây.

---

## Step 2 — Huấn luyện Balanced Random Forest

Mô hình sử dụng `BalancedRandomForestClassifier` từ thư viện `imbalanced-learn`.

**Siêu tham số (Hyperparameters) được cấu hình:**

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `n_estimators` | `200` | Số lượng cây trong rừng. 200 cây là đủ cho BRF vì mỗi cây đã thấy tập cân bằng. |
| `max_depth` | `None` | Không giới hạn độ sâu — để mỗi cây phát triển đầy đủ đến khi thuần nhãn. |
| `max_features` | `'sqrt'` | Mỗi lần tách nút chỉ xét `√28 ≈ 5` features, đảm bảo tính ngẫu nhiên giữa các cây. |
| `sampling_strategy` | `'auto'` | Tự động undersample lớp đa số xuống bằng lớp thiểu số (5,920 mẫu) cho mỗi cây. |
| `replacement` | `False` | Sampling không hoàn lại — mỗi mẫu chỉ xuất hiện một lần trong một bootstrap. |
| `bootstrap` | `True` | Dùng bootstrap sampling khi xây cây, đảm bảo tính đa dạng giữa các cây. |
| `random_state` | `42` | Cố định seed để tái lập kết quả. |
| `n_jobs` | `-1` | Sử dụng toàn bộ CPU để song song hóa việc xây cây. |

> **Lưu ý quan trọng:** Không đặt `class_weight='balanced'` cùng với BRF. Tham số này sẽ tạo ra **double-balancing** — vừa undersample nội sinh (qua `sampling_strategy`), vừa nhân thêm trọng số cho lớp thiểu số — khiến model bị lệch thái quá sang lớp gian lận, dẫn đến Precision rất thấp và FP tăng vọt.

---

## Step 3 — Dự đoán

```python
y_pred = brf.predict(X_test)          # Nhãn dự đoán (threshold mặc định 0.5)
y_prob = brf.predict_proba(X_test)[:, 1]  # Xác suất thuộc lớp gian lận
```

| | Số lượng |
|---|---|
| Mẫu dự đoán là lớp 1 (positive) | 2,855 |
| Mẫu thực tế là lớp 1 (positive) | 1,280 |

---

## Step 4 — Evaluation Metrics

Giống LightGBM/XGBoost, `Accuracy` bị loại bỏ. Các chỉ số đánh giá thực chiến:

**Kết quả trên tập Test:**

| Chỉ số | Giá trị |
|---|---|
| **Precision** | 0.4371 |
| **Recall** | 0.9750 |
| **F1-Score** | 0.6036 |
| **ROC-AUC** | 0.9981 |
| **PR-AUC** | 0.8715 |

**Classification Report:**

```
              precision    recall  f1-score   support

           0       1.00      0.99      0.99    117,649
           1       0.44      0.97      0.60      1,280

    accuracy                           0.99    118,929
   macro avg       0.72      0.98      0.80    118,929
weighted avg       0.99      0.99      0.99    118,929
```

---

## Step 5 — Confusion Matrix

|  | Dự đoán: 0 | Dự đoán: 1 |
|---|---|---|
| **Thực tế: 0** | TN = 116,042 | FP = 1,607 |
| **Thực tế: 1** | FN = 32 | TP = 1,248 |

**Giải thích:**
- **TP = 1,248 / 1,280:** Model bắt được **97.5%** giao dịch gian lận thực sự — tỷ lệ Recall vượt trội.
- **FN = 32:** Chỉ bỏ sót 32 ca gian lận — rất tốt cho mục tiêu giảm thiệt hại tài chính.
- **FP = 1,607:** Gắn cờ nhầm 1,607 giao dịch hợp lệ — là hệ quả trực tiếp của chiến lược undersampling aggressive: model học để "không bỏ sót gian lận" nên có xu hướng cảnh báo thận trọng hơn.

---

## Step 6 — Precision-Recall Curve

Đường PR Curve đi kèm đường baseline (prevalence = 1,280 / 118,929 ≈ **1.08%**).

**Ý nghĩa:** PR-AUC = **0.8715** nghĩa là model đạt diện tích 87.15% dưới đường cong Precision-Recall — cao hơn rất nhiều so với baseline ngẫu nhiên (1.08%), xác nhận khả năng phân biệt gian lận tốt ngay cả trên dữ liệu cực kỳ mất cân bằng.

---

## Step 7 — Feature Importance

Top 20 features theo Mean Decrease Impurity (MDI):

| Hạng | Feature | Importance |
|---|---|---|
| 1 | `merchant_risk` | 0.4023 |
| 2 | `category_risk` | 0.2961 |
| 3 | `category` | 0.0480 |
| 4 | `cust_std` | 0.0415 |
| 5 | `cust_mean` | 0.0348 |
| 6 | `cust_max` | 0.0294 |
| 7 | `merchant` | 0.0210 |
| 8 | `amount_zscore` | 0.0176 |
| 9 | `amount_log` | 0.0154 |
| 10 | `is_new_merchant` | 0.0152 |

**Phân tích:** Hai feature `merchant_risk` (40.2%) và `category_risk` (29.6%) chiếm gần **70% tổng importance** — cho thấy các đặc trưng kỹ thuật rủi ro được tạo trong bước Feature Engineering là tín hiệu quyết định nhất để nhận diện gian lận. Điều này khẳng định bước Feature Engineering đã tạo ra thông tin cực kỳ discriminative cho cả ba model.

---

## Tổng kết & So sánh 3 mô hình

### Bảng so sánh đầy đủ

| Mô hình | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **LightGBM** | 0.8915 | 0.7828 | 0.8336 | — | 0.9335 |
| **XGBoost** | 0.8926 | 0.7789 | 0.8319 | — | 0.9323 |
| **BRF** | 0.4371 | **0.9750** | 0.6036 | **0.9981** | 0.8715 |

### Chiến lược xử lý mất cân bằng

| Model | Phương pháp | Thời điểm xử lý |
|---|---|---|
| LightGBM | SMOTE (oversampling) | Trước khi train |
| XGBoost | SMOTE (oversampling) | Trước khi train |
| BRF | Random Undersampling | Trong quá trình train (nội sinh) |

### Đọc kết quả theo góc nhìn nghiệp vụ

**LightGBM / XGBoost — "Cân bằng giữa bắt gian lận và tránh phiền khách"**
- Recall ~78%: Bỏ sót khoảng 22% giao dịch gian lận.
- Precision ~89%: Cứ 10 cảnh báo thì có ~9 ca thực sự gian lận.
- F1 ~0.83: Điểm cân bằng tốt nhất trong ba model.
- Phù hợp khi hệ thống cần **vừa bắt gian lận vừa tránh khóa oan thẻ khách hàng**.

**Balanced Random Forest — "Không bỏ sót bằng mọi giá"**
- Recall 97.5%: Chỉ bỏ sót 32 / 1,280 ca gian lận.
- Precision 43.7%: Cứ 10 cảnh báo thì có ~6 ca nhầm (FP cao).
- F1 ~0.60: Thấp hơn vì Precision thấp kéo xuống.
- Phù hợp khi **chi phí bỏ sót gian lận >> chi phí gắn cờ nhầm** (ví dụ: giao dịch giá trị lớn, thanh toán quốc tế).

> **Kết luận:** Không có model nào tuyệt đối tốt hơn — lựa chọn phụ thuộc vào **ngưỡng chấp nhận rủi ro nghiệp vụ**. Nếu tối ưu theo F1 và PR-AUC, LightGBM/XGBoost vượt trội. Nếu tối ưu theo Recall (zero-miss policy), BRF là lựa chọn phù hợp hơn. Trong thực tế, có thể điều chỉnh **decision threshold** của LightGBM/XGBoost để đẩy Recall lên cao hơn mà không cần đổi model.