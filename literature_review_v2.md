# Literature Review: Phát hiện Gian lận Giao dịch Tài chính

---

## 1. Mô tả Dataset BankSim

BankSim là một dataset tổng hợp (synthetic dataset) được sinh ra từ dữ liệu giao dịch thực tế tổng hợp của một ngân hàng Tây Ban Nha, sử dụng kỹ thuật multi-agent simulation. Dataset này được thiết kế đặc biệt cho nghiên cứu phát hiện gian lận, với đặc điểm mô phỏng hành vi thanh toán thực tế trong khoảng 6 tháng (180 bước mô phỏng).

### 1.1 Ý nghĩa các cột

Dataset chính (`bs140513_032310.csv`) gồm **594.643 bản ghi** và **10 cột** như sau:

| Tên cột | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `step` | int64 | Bước thời gian mô phỏng, từ 0 đến 179 (tương đương ~6 tháng, mỗi step ≈ 1 ngày) |
| `customer` | object | Mã định danh khách hàng duy nhất (ví dụ: `'C1093826151'`) |
| `age` | object | Nhóm tuổi khách hàng: `'0'`–`'6'` (trẻ đến già), `'U'` = unknown |
| `gender` | object | Giới tính: `'M'` (nam), `'F'` (nữ), `'E'` (doanh nghiệp), `'U'` (không xác định) |
| `zipcodeOri` | object | Mã bưu chính nơi ở của khách hàng (ví dụ: `'28007'`) |
| `merchant` | object | Mã định danh người bán/đơn vị chấp nhận thanh toán (ví dụ: `'M348934600'`) |
| `zipMerchant` | object | Mã bưu chính của người bán |
| `category` | object | Danh mục giao dịch (15 loại, ví dụ: `es_transportation`, `es_food`, `es_health`) |
| `amount` | float64 | Giá trị giao dịch (đơn vị: EUR), khoảng từ 0 đến 8.329,96 |
| `fraud` | int64 | Nhãn mục tiêu: `0` = giao dịch bình thường, `1` = giao dịch gian lận |

**15 danh mục giao dịch** bao gồm: `es_transportation`, `es_health`, `es_otherservices`, `es_food`, `es_hotelservices`, `es_barsandrestaurants`, `es_tech`, `es_sportsandtoys`, `es_wellnessandbeauty`, `es_hyper`, `es_fashion`, `es_home`, `es_contents`, `es_travel`, `es_leisure`.

Ngoài ra, file `bsNET140513_032310.csv` cung cấp biểu diễn đồ thị (graph representation) của cùng dữ liệu với 5 cột: `Source` (khách hàng), `Target` (người bán), `Weight` (giá trị giao dịch), `typeTrans` (loại giao dịch), `fraud` — phục vụ cho các phương pháp phân tích mạng lưới (Graph-based).

### 1.2 Thống kê cơ bản và quan sát từ notebook

**Phân phối nhãn (Class Distribution):**

| Nhãn | Số lượng | Tỷ lệ |
|---|---|---|
| 0 (bình thường) | 587.443 | ~98,79% |
| 1 (gian lận) | 7.200 | ~1,21% |

Đây là mức mất cân bằng nghiêm trọng với tỷ lệ gian lận chỉ khoảng 1 trên 82 giao dịch.

**Thống kê cột `amount`:**

| Chỉ số | Bình thường | Gian lận |
|---|---|---|
| Mean | 31,85 EUR | 530,93 EUR |
| Median | 26,61 EUR | 319,18 EUR |
| Max | 2.144,86 EUR | 8.329,96 EUR |
| Std | 31,47 | 835,59 |

**Giá trị trung bình của giao dịch gian lận cao hơn ~16 lần** so với giao dịch bình thường. Điều này gợi ý `amount` là một feature phân biệt quan trọng, phản ánh hành vi thực tế của kẻ gian lận khi cố ý thực hiện các giao dịch lớn.

**Không có giá trị thiếu (missing values):** Tất cả 10 cột đều đầy đủ trên toàn bộ 594.643 bản ghi — cho phép tập trung vào xử lý class imbalance thay vì imputation.

---

## 2. Bài toán là gì?

Phát hiện gian lận giao dịch tài chính là bài toán nhận dạng tự động các giao dịch có dấu hiệu gian lận trong dòng chảy liên tục của các giao dịch thẻ tín dụng, thẻ ghi nợ, hoặc thanh toán điện tử. Về bản chất, đây là bài toán phân loại nhị phân (binary classification) với nhãn `0` (hợp lệ) và `1` (gian lận), song cũng có thể được tiếp cận theo hướng phát hiện bất thường (anomaly detection) khi nhãn gian lận khan hiếm hoặc không đáng tin cậy.

Theo báo cáo của Nilson Report (2023), tổng thiệt hại toàn cầu do gian lận thẻ thanh toán vượt **33 tỷ USD** mỗi năm. Bài toán yêu cầu tối ưu hóa cân bằng giữa precision và recall, thường được đo bằng F1-score, AUPRC, hoặc cost-sensitive metrics — vì cả false positive (chặn nhầm giao dịch hợp lệ) lẫn false negative (bỏ sót gian lận) đều gây thiệt hại đáng kể.

---

## 3. Input và Output

**Input (Features đầu vào):** Thông tin giao dịch (thời gian, giá trị, danh mục), thông tin khách hàng (mã định danh, tuổi, giới tính, địa chỉ), thông tin người bán, và các features kỹ thuật được tính toán (velocity features, độ lệch so với lịch sử khách hàng, v.v.).

**Output (Nhãn đầu ra):** Nhãn nhị phân `fraud ∈ {0, 1}` trong dạng supervised, hoặc điểm bất thường (anomaly score) trong dạng unsupervised. Một số hệ thống còn trả về xác suất gian lận kèm giải thích (explanation) phục vụ fraud analyst.

**Dạng bài toán:** Chủ yếu là **binary classification** với class imbalance nghiêm trọng, ngoài ra còn có các biến thể temporal modeling và graph-based classification.

---

## 4. Các kỹ thuật và Phân loại

### 4.1 Supervised Learning

Đây là nhóm phương pháp phổ biến nhất khi dữ liệu có nhãn đầy đủ. **Logistic Regression** là baseline dễ diễn giải; **các phương pháp Tree-based** (Random Forest, XGBoost, LightGBM) là tiêu chuẩn vàng trên dữ liệu tabular với khả năng xử lý class imbalance tích hợp; **Neural Networks** (MLP, LSTM) được dùng khi dữ liệu đủ lớn hoặc cần nắm bắt temporal patterns.

### 4.2 Unsupervised / Anomaly Detection

Khi nhãn gian lận khan hiếm, các phương pháp này học phân phối dữ liệu bình thường và đánh dấu các điểm lệch là bất thường. **Isolation Forest** phù hợp với dữ liệu lớn chiều cao. **Autoencoder** được huấn luyện chỉ trên giao dịch bình thường — giao dịch gian lận sẽ có reconstruction error cao. **One-Class SVM** học siêu phẳng phân tách dữ liệu bình thường, nhưng kém hiệu quả hơn với dữ liệu quy mô lớn.

Ngoài hai nhóm chính trên, **Graph-based methods** khai thác mối quan hệ giữa khách hàng, người bán và các thực thể khác trong mạng lưới giao dịch để phát hiện các pattern gian lận mang tính tập thể. BankSim cung cấp sẵn file đồ thị `bsNET140513_032310.csv` cho phép áp dụng hướng tiếp cận này.

| Phương pháp | Ưu điểm | Nhược điểm |
|---|---|---|
| Logistic Regression | Nhanh, diễn giải được | Hiệu suất thấp với quan hệ phi tuyến |
| Random Forest | Ổn định, ít overfit | Chậm khi inference, khó tuning |
| XGBoost / LightGBM | Hiệu suất cao, hỗ trợ imbalance | Cần hyperparameter tuning kỹ |
| MLP | Linh hoạt, scalable | Cần nhiều dữ liệu, khó diễn giải |
| LSTM / Transformer | Nắm bắt temporal patterns | Phức tạp, cần dữ liệu lịch sử |
| Isolation Forest | Không cần nhãn, nhanh | Khó kiểm soát ngưỡng phân loại |
| Autoencoder | Học phân phối bình thường | Cần dữ liệu sạch để huấn luyện |

---

## 5. Thách thức chính

**Class Imbalance (Mất cân bằng lớp)** là thách thức nổi bật nhất trong fraud detection. Tỷ lệ gian lận thực tế thường dưới 1%, khiến các mô hình có xu hướng học thiên về lớp đa số và bỏ sót gian lận. Các kỹ thuật giải quyết gồm: oversampling (SMOTE, ADASYN), undersampling, cost-sensitive learning (điều chỉnh trọng số loss function), và thay đổi metric đánh giá (dùng AUPRC, F1, G-mean thay vì Accuracy).

**Concept Drift (Thay đổi mẫu theo thời gian)** xảy ra khi hành vi gian lận tiến hóa theo thời gian — kẻ gian lận học cách né tránh hệ thống phát hiện. Một mô hình được huấn luyện trên dữ liệu 6 tháng trước có thể nhanh chóng lỗi thời. Giải pháp bao gồm: online learning, sliding window retraining, và drift detection algorithms.

**Cost Asymmetry (Bất cân xứng chi phí)** phản ánh thực tế rằng chi phí bỏ sót một giao dịch gian lận (false negative) thường cao hơn nhiều so với chi phí chặn nhầm một giao dịch hợp lệ (false positive). Điều này đòi hỏi phải tích hợp cost matrix vào quá trình huấn luyện và đánh giá mô hình.

**Latency (Độ trễ thời gian thực):** Hệ thống sản xuất phải đưa ra quyết định trong vài mili giây, điều này giới hạn độ phức tạp của mô hình có thể triển khai thực tế.

**Thiếu dữ liệu nhãn (Lack of Labeled Data):** Chi phí điều tra thủ công cao khiến chỉ một phần nhỏ giao dịch nghi ngờ được xác nhận, dẫn đến thiếu nhãn đáng tin cậy để huấn luyện.

---

## 6. Các nhánh nghiên cứu chính

**Graph Neural Networks (GNN) cho Fraud Detection** là hướng đang được nghiên cứu nhiều nhất. Các mô hình GNN học embedding cho từng node (khách hàng, giao dịch) bằng cách tổng hợp thông tin từ các node lân cận, giúp phát hiện các fraud pattern mang tính tập thể như money mule networks hay collusion fraud.

**Transformer-based Sequence Modeling** áp dụng kiến trúc Transformer để mô hình hóa lịch sử giao dịch của từng khách hàng như một chuỗi token. Hướng này vượt qua giới hạn của LSTM trong việc nắm bắt long-range dependencies và đang cho kết quả tốt trên nhiều benchmark.

**Explainable AI (XAI) trong Fraud Detection** giải quyết nhu cầu diễn giải quyết định của mô hình — quan trọng cả về mặt quy định (ngân hàng phải giải thích lý do từ chối giao dịch) lẫn vận hành. Các kỹ thuật như SHAP và LIME được áp dụng rộng rãi để giải thích dự đoán của các mô hình phức tạp.

**Synthetic Data Generation** sử dụng các mô hình sinh (GAN, VAE) để tạo dữ liệu gian lận tổng hợp, giải quyết vấn đề thiếu dữ liệu nhãn. BankSim chính là một ví dụ điển hình của hướng này — được sinh từ dữ liệu thực tế thông qua multi-agent simulation.

---

## 7. Kết luận và Định hướng cho BankSim

| Bước | Kỹ thuật | Lý do |
|---|---|---|
| Baseline | Logistic Regression + SMOTE | Nhanh, diễn giải được, kiểm tra feature importance |
| Nâng cao | XGBoost/LightGBM + class weight | Hiệu suất cao với tabular, xử lý imbalance tích hợp |
| Temporal | Feature engineering (velocity) + LSTM | Khai thác thông tin chuỗi thời gian từ cột `step` |
| Graph | GNN trên bsNET | Khai thác cấu trúc mạng lưới khách hàng-người bán |
| Explainability | SHAP values | Giải thích dự đoán, phân tích feature importance |

---

*Tài liệu được biên soạn phục vụ nghiên cứu và giảng dạy. Dataset BankSim được công bố bởi Lopez-Rojas, E. A., Axelsson, S. (2014), Chalmers University of Technology.*
