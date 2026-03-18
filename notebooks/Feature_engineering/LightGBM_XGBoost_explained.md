# Giải thích Notebook: LightGBM & XGBoost Training

**File notebook:** `LightGBM_XGBoost.ipynb`
**Dataset:** Dữ liệu BankSim đã qua tiền xử lý và áp dụng SMOTE (từ thư mục `data/processed/`)
**Mục tiêu:** Huấn luyện và so sánh hai mô hình phân loại gian lận mạnh mẽ nhất trên dữ liệu dạng bảng.

---

## Tổng quan pipeline

```text
Processed Data (*.pkl)
  │
  ▼
Step 1 — Load Data
  │  Đọc X_train_res (đã SMOTE), y_train_res, X_test (gốc), y_test (gốc)
  ▼
Step 2 — LightGBM Training
  │  Khởi tạo và huấn luyện mô hình LGBMClassifier
  ▼
Step 3 — XGBoost Training
  │  Khởi tạo và huấn luyện mô hình XGBClassifier (hist method)
  ▼
Step 4 — Evaluation Metrics
  │  Dự đoán test set, tính toán F1, Recall, Precision, AUC-PR
  ▼
Step 5 — Visualization
     Vẽ biểu đồ Precision-Recall Curve để so sánh trực quan hiệu năng
```

---

## Step 1 — Load Data

Notebook sử dụng hàm `load_data()` để đọc trực tiếp các file `.pkl` được xuất ra từ quá trình Feature Engineering & Resampling. 

* **Tập huấn luyện (đã SMOTE):** Chứa **516,773** mẫu và 28 features. Đây là tập dữ liệu đã được cân bằng, giúp mô hình học được đặc trưng của nhóm giao dịch gian lận.
* **Tập kiểm tra (gốc):** Chứa **118,929** mẫu và 28 features.

> **Nguyên tắc chống Data Leakage:** Tập `X_test` tuyệt đối là tập gốc, giữ nguyên tỷ lệ mất cân bằng của thực tế (~1.2%). Đánh giá (evaluate) trên tập test có áp dụng SMOTE sẽ dẫn đến ảo tưởng về độ chính xác.

---

## Step 2 — LightGBM Training

Huấn luyện mô hình sử dụng thuật toán Gradient Boosting của Microsoft (`lightgbm.LGBMClassifier`). LightGBM tối ưu hóa tốc độ và bộ nhớ nhờ chiến lược phát triển cây theo lá (leaf-wise).

**Siêu tham số (Hyperparameters) được cấu hình:**

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `n_estimators` | `1000` | Số lượng vòng lặp boosting (số cây). Số lượng cây lớn giúp mô hình hội tụ tốt hơn. |
| `learning_rate` | `0.05` | Tốc độ học. Mức 0.05 kết hợp với 1000 cây là công thức tiêu chuẩn để đạt hiệu năng cao mà không bị overfit nhanh. |
| `num_leaves` | `31` | Số lượng lá tối đa trên mỗi cây (Mặc định). Tham số cốt lõi kiểm soát độ phức tạp của thuật toán leaf-wise. |
| `n_jobs` | `-1` | Sử dụng toàn bộ số luồng (cores) của CPU để huấn luyện song song. |

---

## Step 3 — XGBoost Training

Huấn luyện mô hình Gradient Boosting kinh điển `xgboost.XGBClassifier` với chiến lược phát triển cây theo độ sâu (level-wise).

**Siêu tham số (Hyperparameters) được cấu hình:**

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `n_estimators` | `1000` | Cùng số lượng cây với LightGBM để dễ dàng so sánh công bằng. |
| `learning_rate` | `0.05` | Tốc độ học tương đương LightGBM. |
| `max_depth` | `6` | Giới hạn độ sâu tối đa của cây để bắt các tương tác phi tuyến tính giữa các tính năng. |
| `tree_method` | `'hist'` | Tối ưu hóa cốt lõi: Sử dụng thuật toán xây dựng cây dựa trên Histogram thay vì thuật toán tham lam truyền thống, giúp **tăng tốc độ huấn luyện trên dữ liệu lớn** tương đương với LightGBM. |

---

## Step 4 — Evaluation Metrics

Trong bài toán Fraud Detection, chỉ số `Accuracy` bị loại bỏ hoàn toàn do tính chất mất cân bằng cực đoan. Notebook tập trung tính toán 4 chỉ số thực chiến:

1. **Recall (Độ phủ):** Bắt được bao nhiêu % ca gian lận? (Tối quan trọng để giảm thiệt hại tài chính).
2. **Precision (Độ chuẩn):** Trong các ca bị gắn cờ gian lận, đúng bao nhiêu %? (Tránh khóa thẻ oan của khách).
3. **F1-Score:** Trung bình điều hòa giữa Precision và Recall.
4. **AUC-PR (Average Precision):** Diện tích dưới đường cong Precision-Recall, là chỉ số tốt nhất cho Imbalanced Data.

**Kết quả từ tập dữ liệu Test:**

| Mô hình | F1-Score | Recall | Precision | AUC-PR |
|----------|----------|--------|-----------|--------|
| **LightGBM** | 0.833611 | 0.782813 | 0.891459 | 0.933470 |
| **XGBoost** | 0.831873 | 0.778906 | 0.892569 | 0.932269 |

> **Phân tích kết quả:** Hai mô hình cho hiệu năng gần như tương đương nhau, bám đuổi rất sát sao. Cả hai đều đạt mức Precision rất ấn tượng (~89%) và duy trì Recall ở mức tốt (~78%), mang lại điểm F1 trên 0.83. Điều này cho thấy bước Feature Engineering và SMOTE trước đó đã hoạt động cực kỳ hiệu quả.

---

## Step 5 — Visualization

Bước cuối cùng là vẽ biểu đồ **Precision-Recall Curve** (PR Curve).

**Tại sao PR Curve vượt trội hơn ROC Curve trong phát hiện gian lận?**

* **ROC Curve** sử dụng tỷ lệ False Positive (`FPR = FP / (FP + TN)`). Vì tập BankSim có lượng giao dịch bình thường (`TN`) quá khổng lồ, `FPR` luôn rất nhỏ, làm đường ROC bị đẩy vút lên trên một cách "ảo giác".
* **PR Curve** hoàn toàn **không quan tâm đến True Negatives (`TN`)**. Nó chỉ đánh giá dựa trên các ca được dự đoán là gian lận (Precision) và các ca thực sự là gian lận (Recall), mang lại góc nhìn trung thực 100% về hiệu năng trên class thiểu số.

Biểu đồ so sánh AUC-PR xác nhận trực quan những gì thể hiện trên bảng số liệu: Cả hai đường cong của LightGBM và XGBoost gần như chồng lên nhau, cùng đạt diện tích AUC-PR ấn tượng (lần lượt là ~0.9335 và ~0.9323), chứng tỏ khả năng nhận diện gian lận xuất sắc của cả hai mô hình trên tập dữ liệu này.