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

Đây là một quan sát rất đáng chú ý: **giá trị trung bình của giao dịch gian lận cao hơn ~16 lần** so với giao dịch bình thường. Điều này gợi ý `amount` là một feature phân biệt quan trọng, nhưng cũng phản ánh hành vi thực tế của kẻ gian lận — cố ý thực hiện các giao dịch lớn khi đã chiếm được thẻ.

**Phân phối nhóm tuổi (`age`):** Tập trung ở nhóm `'2'` (187.310 giao dịch) và `'3'` (147.131 giao dịch), tương ứng độ tuổi trung niên — phản ánh nhóm dân số có mức chi tiêu cao nhất.

**Phân phối giới tính (`gender`):** Nữ (`'F'`) chiếm 54,6%, Nam (`'M'`) chiếm 45,1%, một tỷ lệ nhỏ là doanh nghiệp (`'E'`) và không xác định (`'U'`).

**Không có giá trị thiếu (missing values):** Tất cả 10 cột đều đầy đủ trên toàn bộ 594.643 bản ghi — đây là một thuận lợi so với nhiều dataset thực tế, và cho phép tập trung vào xử lý class imbalance thay vì imputation.

**Phạm vi thời gian (`step`):** Từ 0 đến 179, mô phỏng khoảng 6 tháng hoạt động. Theo thiết kế dataset, mỗi bước có khoảng 3 thẻ bị đánh cắp và ~2 giao dịch gian lận được thực hiện mỗi ngày.

---

## 2. Bài toán là gì?

Phát hiện gian lận giao dịch tài chính (Credit Card / Financial Transaction Fraud Detection) là bài toán nhận dạng tự động các giao dịch có dấu hiệu gian lận trong dòng chảy liên tục của các giao dịch thẻ tín dụng, thẻ ghi nợ, hoặc thanh toán điện tử. Về bản chất, đây là bài toán phân loại nhị phân (binary classification) với nhãn `0` (hợp lệ) và `1` (gian lận), song cũng có thể được tiếp cận theo hướng phát hiện bất thường (anomaly detection) khi nhãn gian lận khan hiếm hoặc không đáng tin cậy.

Bối cảnh thực tế của bài toán này ngày càng cấp bách hơn trong kỷ nguyên số. Theo báo cáo của Nilson Report (2023), tổng thiệt hại toàn cầu do gian lận thẻ thanh toán vượt **33 tỷ USD** mỗi năm. Tại Việt Nam, Ngân hàng Nhà nước ghi nhận hàng nghìn vụ gian lận thanh toán điện tử mỗi năm, gây thiệt hại hàng trăm tỷ đồng. Với sự bùng nổ của thanh toán không tiền mặt, e-commerce và ví điện tử, quy mô và độ phức tạp của gian lận cũng gia tăng tương ứng.

Tầm quan trọng của bài toán không chỉ nằm ở thiệt hại tài chính trực tiếp. Việc phát hiện sai (false positive — chặn nhầm giao dịch hợp lệ) gây ra trải nghiệm khách hàng kém, mất doanh thu và tốn chi phí xử lý khiếu nại. Ngược lại, bỏ sót gian lận (false negative) gây thiệt hại tài chính trực tiếp và mất niềm tin người dùng. Do đó, bài toán yêu cầu tối ưu hóa không chỉ độ chính xác mà còn phải cân bằng giữa precision và recall, thường được đo bằng F1-score, AUPRC (Area Under Precision-Recall Curve), hoặc cost-sensitive metrics.

---

## 3. Input và Output

**Input (Features đầu vào)** trong bài toán fraud detection thường bao gồm nhiều loại thông tin khác nhau:

- **Thông tin giao dịch**: thời gian, giá trị giao dịch (`amount`), loại giao dịch (`category`), phương thức thanh toán.
- **Thông tin khách hàng**: mã định danh (`customer`), nhóm tuổi (`age`), giới tính (`gender`), địa chỉ (`zipcodeOri`).
- **Thông tin người bán**: mã định danh (`merchant`), địa điểm (`zipMerchant`), loại hình kinh doanh.
- **Features kỹ thuật (engineered)**: tần suất giao dịch trong cửa sổ thời gian (velocity features), độ lệch so với lịch sử khách hàng, khoảng cách địa lý, số lần giao dịch thất bại liên tiếp, v.v.

**Output (Nhãn đầu ra):** Trong dạng supervised, đầu ra là nhãn nhị phân `fraud ∈ {0, 1}`. Trong dạng anomaly detection, đầu ra là điểm bất thường (anomaly score), sau đó áp dụng ngưỡng để phân loại. Một số hệ thống hiện đại còn trả về xác suất gian lận kèm giải thích (explanation), phục vụ cho nhân viên kiểm soát gian lận (fraud analyst).

**Dạng bài toán:** Chủ yếu là **binary classification** với class imbalance nghiêm trọng. Ngoài ra còn có các biến thể: **sequential/temporal modeling** (xem lịch sử giao dịch như một chuỗi), **graph-based classification** (xem mạng lưới khách hàng-người bán như đồ thị), và **anomaly/novelty detection** (khi dữ liệu gian lận không đủ hoặc mẫu gian lận thay đổi liên tục).

---

## 4. Các kỹ thuật và Phân loại

### 4.1 Supervised Learning

Đây là nhóm phương pháp phổ biến nhất khi dữ liệu có nhãn đầy đủ. Ý tưởng cốt lõi là học một hàm ánh xạ từ features sang nhãn gian lận/bình thường từ dữ liệu lịch sử đã được gán nhãn.

**Logistic Regression** là nền tảng cơ bản, dễ diễn giải và tính toán nhanh. Dù không nắm bắt được các mối quan hệ phi tuyến phức tạp, nó vẫn là baseline quan trọng và thường được kết hợp với feature engineering tốt để đạt hiệu quả ổn định. Trong môi trường sản xuất, sự diễn giải được của Logistic Regression còn giúp đáp ứng yêu cầu tuân thủ quy định (regulatory compliance).

**Các phương pháp dựa trên cây quyết định (Tree-based)** — bao gồm Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost) — hiện đang là tiêu chuẩn vàng trong fraud detection dạng tabular. Random Forest giảm overfitting qua bagging và chọn features ngẫu nhiên. XGBoost và LightGBM cho phép xử lý class imbalance thông qua tham số `scale_pos_weight`, huấn luyện nhanh và đạt AUROC cao trên nhiều benchmark. CatBoost đặc biệt mạnh với features dạng categorical như `merchant`, `category`.

**Mạng nơ-ron nhân tạo (Neural Networks)** được áp dụng khi dữ liệu đủ lớn. Các kiến trúc phổ biến gồm MLP (Multi-Layer Perceptron) cho tabular data, LSTM/GRU cho dữ liệu chuỗi thời gian lịch sử giao dịch, và Transformer-based models cho sequence modeling phức tạp hơn.

| Phương pháp | Ưu điểm | Nhược điểm |
|---|---|---|
| Logistic Regression | Nhanh, diễn giải được | Hiệu suất thấp với quan hệ phi tuyến |
| Random Forest | Ổn định, ít overfit | Chậm khi inference, khó tuning |
| XGBoost / LightGBM | Hiệu suất cao, hỗ trợ imbalance | Cần hyperparameter tuning kỹ |
| MLP | Linh hoạt, scalable | Cần nhiều dữ liệu, khó diễn giải |
| LSTM / Transformer | Nắm bắt temporal patterns | Phức tạp, cần dữ liệu lịch sử |

### 4.2 Unsupervised / Anomaly Detection

Khi nhãn gian lận khan hiếm hoặc không đáng tin cậy, các phương pháp không giám sát học phân phối của dữ liệu bình thường và đánh dấu các điểm lệch khỏi phân phối đó là bất thường.

**Isolation Forest** xây dựng các cây ngẫu nhiên và đo độ "dễ bị cô lập" của một điểm dữ liệu — điểm gian lận thường bị cô lập nhanh hơn vì chúng nằm ở vùng thưa thớt của không gian feature. Thuật toán này hiệu quả với dữ liệu lớn chiều cao và không cần giả định về phân phối dữ liệu.

**Autoencoder** là mạng nơ-ron được huấn luyện để tái tạo (reconstruct) dữ liệu đầu vào. Khi chỉ huấn luyện trên giao dịch bình thường, mô hình học cách nén và giải nén mẫu bình thường. Giao dịch gian lận sẽ có reconstruction error cao vì chúng không khớp với phân phối đã học. Biến thể Variational Autoencoder (VAE) còn cho phép mô hình hóa xác suất không gian ẩn.

**One-Class SVM** học một siêu phẳng phân tách dữ liệu bình thường khỏi không gian còn lại. Phù hợp khi chỉ có dữ liệu bình thường để huấn luyện, nhưng kém hiệu quả hơn với dữ liệu quy mô lớn do độ phức tạp tính toán O(n²).

### 4.3 Semi-supervised

Semi-supervised learning tận dụng cả dữ liệu có nhãn (ít) và dữ liệu không nhãn (nhiều) — đây là tình huống phổ biến trong thực tế khi việc gán nhãn tốn kém và chỉ một phần nhỏ giao dịch được điều tra thủ công.

**Self-training** bắt đầu với một classifier đơn giản trên dữ liệu có nhãn, sau đó dần dần gán nhãn giả (pseudo-labels) cho dữ liệu không nhãn có độ tin cậy cao và đưa vào huấn luyện tiếp. **Label Propagation** lan truyền nhãn qua đồ thị similarity giữa các giao dịch. **Co-training** sử dụng hai mô hình độc lập trên hai tập features khác nhau, mỗi mô hình gán nhãn cho dữ liệu không nhãn để huấn luyện mô hình kia. Gần đây, các phương pháp dựa trên **contrastive learning** như SimCLF cũng được áp dụng để học representation tốt từ dữ liệu không nhãn trước khi fine-tune trên dữ liệu có nhãn.

### 4.4 Graph-based / Network Analysis

Fraud detection theo hướng đồ thị khai thác mối quan hệ giữa các thực thể: khách hàng, thẻ, người bán, địa chỉ IP, thiết bị. Gian lận thường để lại dấu vết đồ thị đặc trưng: một thẻ bị đánh cắp kết nối với nhiều người bán bất thường, hoặc một người bán độc hại nhận giao dịch từ nhiều thẻ khác nhau trong thời gian ngắn.

**Graph Neural Networks (GNN)** — đặc biệt là GraphSAGE, GAT (Graph Attention Network), và Graph Convolutional Network (GCN) — học embedding cho từng node bằng cách tổng hợp thông tin từ các node lân cận. Trong fraud detection, node có thể là khách hàng hoặc giao dịch, cạnh biểu thị mối quan hệ, và bài toán trở thành node classification hoặc edge classification trên đồ thị.

Dataset BankSim cung cấp sẵn file `bsNET140513_032310.csv` với cấu trúc `(Source, Target, Weight, typeTrans, fraud)` — đây là biểu diễn đồ thị hai phía (bipartite graph) giữa khách hàng và người bán, cho phép áp dụng trực tiếp GNN.

---

## 5. Thách thức chính

**Class Imbalance (Mất cân bằng lớp)** là thách thức nổi bật nhất trong fraud detection. Tỷ lệ gian lận thực tế thường dưới 1%, khiến các mô hình có xu hướng học thiên về lớp đa số và bỏ sót gian lận. Các kỹ thuật giải quyết gồm: oversampling (SMOTE, ADASYN), undersampling, cost-sensitive learning (điều chỉnh trọng số loss function), và thay đổi metric đánh giá (dùng AUPRC, F1, G-mean thay vì Accuracy).

**Concept Drift (Thay đổi mẫu theo thời gian)** xảy ra khi hành vi gian lận tiến hóa theo thời gian — kẻ gian lận học cách né tránh hệ thống phát hiện. Một mô hình được huấn luyện trên dữ liệu 6 tháng trước có thể nhanh chóng lỗi thời. Giải pháp bao gồm: online learning, sliding window retraining, và drift detection algorithms.

**Cost Asymmetry (Bất cân xứng chi phí)** phản ánh thực tế rằng chi phí bỏ sót một giao dịch gian lận (false negative) thường cao hơn nhiều so với chi phí chặn nhầm một giao dịch hợp lệ (false positive). Điều này đòi hỏi phải tích hợp cost matrix vào quá trình huấn luyện và đánh giá mô hình.

**Data Privacy và Quy định bảo mật** tạo ra rào cản lớn trong việc chia sẻ và sử dụng dữ liệu giao dịch thực. GDPR (châu Âu), PCI DSS, và các quy định địa phương hạn chế khả năng thu thập và lưu trữ dữ liệu chi tiết, đặc biệt khi cần đào tạo mô hình tập trung với dữ liệu của nhiều tổ chức tài chính khác nhau.

**Latency (Độ trễ thời gian thực)** là yêu cầu bắt buộc trong môi trường sản xuất: hệ thống phải đưa ra quyết định phê duyệt/từ chối giao dịch trong vài mili giây. Điều này giới hạn độ phức tạp của mô hình có thể triển khai thực tế, và đòi hỏi thiết kế đặc biệt cho inference pipeline.

**Thiếu dữ liệu nhãn (Lack of Labeled Data)** do chi phí điều tra thủ công cao và nhiều vụ gian lận không bao giờ được phát hiện. Trong thực tế, chỉ một phần nhỏ giao dịch nghi ngờ được điều tra, và ngay cả khi điều tra cũng có thể không xác định được chắc chắn.

---

## 6. Các nhánh nghiên cứu chính

**Graph Neural Networks (GNN) cho Fraud Detection** là hướng nghiên cứu đang nóng nhất hiện nay. Các công trình như GAS (Graph-based Anti-Spam), RGTAN, và Pick-and-Choose GNN cho thấy hiệu quả vượt trội so với phương pháp tabular truyền thống khi khai thác cấu trúc mạng lưới. Hướng nghiên cứu này đặc biệt phù hợp với fraud patterns có tính tập thể (collusion fraud, money mule networks).

**Transformer-based Sequence Modeling** áp dụng kiến trúc Transformer để mô hình hóa lịch sử giao dịch của từng khách hàng như một chuỗi tokens. Các mô hình như FraudBERT, TabTransformer, hay FT-Transformer đã chứng minh khả năng nắm bắt temporal patterns phức tạp, vượt qua giới hạn của LSTM trong việc xử lý long-range dependencies.

**Federated Learning (Học liên kết)** là giải pháp cho bài toán data privacy: nhiều ngân hàng cùng huấn luyện một mô hình chung mà không cần chia sẻ dữ liệu thô. Mỗi tổ chức huấn luyện mô hình cục bộ và chỉ chia sẻ gradient/model weights. Đây là hướng có tiềm năng lớn trong bối cảnh quy định bảo mật dữ liệu ngày càng chặt chẽ.

**Explainable AI (XAI) trong Fraud Detection** giải quyết nhu cầu diễn giải quyết định của mô hình — quan trọng cả về mặt quy định (ngân hàng phải giải thích tại sao từ chối giao dịch) lẫn vận hành (giúp fraud analyst ưu tiên điều tra). SHAP (SHapley Additive exPlanations) và LIME (Local Interpretable Model-agnostic Explanations) được áp dụng rộng rãi để giải thích dự đoán của các mô hình phức tạp như XGBoost hay Neural Networks.

**Synthetic Data Generation** sử dụng GAN (Generative Adversarial Networks) và VAE để sinh dữ liệu gian lận tổng hợp, giải quyết vấn đề thiếu dữ liệu nhãn. CTGAN và TVAE được thiết kế riêng cho tabular data. Chính BankSim cũng là một ví dụ điển hình của hướng này — dataset được sinh từ dữ liệu thực tế của ngân hàng Tây Ban Nha thông qua multi-agent simulation.

---

## 7. Kết luận và Định hướng cho BankSim

### Liên kết lý thuyết với dataset BankSim

**Thách thức class imbalance** biểu hiện rõ ràng trong BankSim: chỉ 7.200 giao dịch gian lận trên tổng số 594.643 (1,21%). Với tỷ lệ này, một mô hình ngây thơ luôn dự đoán "bình thường" đạt accuracy 98,79% nhưng hoàn toàn vô dụng. Kỹ thuật phù hợp gồm SMOTE để oversampling lớp thiểu số, `scale_pos_weight` trong XGBoost/LightGBM, và đánh giá bằng AUPRC hoặc F1-score thay vì Accuracy.

**Thách thức cost asymmetry** cũng đặc biệt rõ trong BankSim khi giao dịch gian lận có giá trị trung bình ~531 EUR so với ~32 EUR của giao dịch bình thường — bỏ sót một giao dịch gian lận thiệt hại gấp ~16 lần so với chặn nhầm một giao dịch hợp lệ. Điều này gợi ý cần thiết kế cost matrix bất đối xứng trong quá trình tối ưu mô hình.

**Cấu trúc đồ thị có sẵn** trong BankSim (file `bsNET140513_032310.csv`) tạo điều kiện thuận lợi để áp dụng **Graph-based methods**. Bipartite graph giữa `customer` và `merchant` cho phép phát hiện các pattern như: một merchant bất thường nhận giao dịch từ nhiều khách hàng trong thời gian ngắn, hoặc một khách hàng đột ngột giao dịch với nhiều merchant khác lạ. GNN như GraphSAGE có thể học các embedding phản ánh ngữ cảnh mạng lưới này.

**Temporal patterns** trong dữ liệu BankSim (cột `step` từ 0–179) cho phép xây dựng velocity features (số giao dịch trong N bước gần nhất, tổng số tiền trong cửa sổ thời gian) và áp dụng LSTM/Transformer để mô hình hóa chuỗi lịch sử giao dịch của từng khách hàng — một hướng có thể cải thiện đáng kể so với các mô hình tabular thông thường không tận dụng thông tin thời gian.

**Định hướng đề xuất cho BankSim:**

| Bước | Kỹ thuật | Lý do |
|---|---|---|
| Baseline | Logistic Regression + SMOTE | Nhanh, diễn giải được, kiểm tra feature importance |
| Nâng cao | XGBoost/LightGBM + class weight | Hiệu suất cao với tabular, xử lý imbalance tích hợp |
| Temporal | Feature engineering (velocity) + LSTM | Khai thác thông tin chuỗi thời gian từ cột `step` |
| Graph | GraphSAGE trên bsNET | Khai thác cấu trúc mạng lưới khách hàng-người bán |
| Explainability | SHAP values | Giải thích dự đoán, phân tích feature importance |

BankSim là một benchmark lý tưởng để nghiên cứu fraud detection vì nó kết hợp đủ các thách thức thực tế (imbalance, temporal evolution, network structure) trong một môi trường an toàn về bảo mật dữ liệu. Kết quả nghiên cứu trên BankSim có thể cung cấp insights có giá trị cho việc triển khai hệ thống phát hiện gian lận thực tế tại các tổ chức tài chính.

---

*Tài liệu được biên soạn phục vụ nghiên cứu và giảng dạy. Dataset BankSim được công bố bởi Lopez-Rojas, E. A., Axelsson, S. (2014), Chalmers University of Technology.*
