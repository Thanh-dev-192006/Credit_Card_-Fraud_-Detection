# Giải thích Notebook: Data Preprocessing & Feature Engineering

**File notebook:** `Data_Preprocessing_FE.ipynb`
**Dataset:** BankSim — ~594,643 giao dịch ngân hàng tổng hợp
**Mục tiêu:** Phát hiện gian lận (binary classification), tỉ lệ fraud ~1.2%
**Mô hình downstream:** LightGBM, XGBoost, BalancedRandomForestClassifier

---

## Tổng quan pipeline

```
Raw CSV
  │
  ▼
Step 1 — Data Cleaning
  │  làm sạch chuỗi, xử lý age='U', loại cột zero-variance
  ▼
Step 2 — Temporal Train/Test Split
  │  sort theo step, 80% train / 20% test, không shuffle
  ▼
Step 3 — Categorical Encoding
  │  LabelEncoder fit trên train only, unseen → -1
  ▼
Step 4 — Feature Engineering
  │  add_features(df, ref_df=train_df): tạo 20 features mới
  ▼
Step 5 — SMOTE Variant Comparison
  │  so sánh 4 cấu hình resampling bằng LightGBM proxy (3-fold CV)
  ▼
Step 6 — Export
     lưu X_train, X_test, y_train, y_test, X_train_res, y_train_res,
     encoders, feature_cols vào data/processed/*.pkl
```

---

## Step 1 — Data Cleaning

Dữ liệu raw trong file CSV có tất cả các giá trị string được bao bởi dấu nháy đơn (ví dụ: `'C1093826151'`, `'4'`, `'M'`). Step này chuẩn hoá và làm sạch trước khi xử lý.

| Thao tác | Mô tả |
|----------|-------|
| Strip whitespace & quotes | `.str.strip().str.strip("'")` trên các cột: `customer`, `age`, `gender`, `zipcodeOri`, `merchant`, `zipMerchant`, `category` |
| `age_unknown` flag | Binary flag = 1 nếu giá trị `age` gốc là `'U'` (không xác định tuổi), = 0 nếu không — tạo trước khi map |
| Map `'U'` → `-1` | Chuyển `age = 'U'` thành `-1` để có thể cast toàn bộ cột sang `int` |
| Drop zero-variance | Loại `zipcodeOri` và `zipMerchant` vì chỉ có 1 giá trị duy nhất (`'28007'`) — không cung cấp thông tin phân biệt |

**Lý do tạo `age_unknown` riêng:** Nếu chỉ map `'U'` → `-1`, mô hình không phân biệt được "tuổi -1 vì dữ liệu thiếu" với "tuổi -1 có ý nghĩa thực". Flag nhị phân giúp mô hình học pattern riêng cho nhóm khách hàng không rõ tuổi.

---

## Step 2 — Temporal Train/Test Split

```python
df = df.sort_values('step').reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]   # ~475,714 records
test_df  = df.iloc[split_idx:]   # ~118,929 records
```

**Tại sao không dùng random split?**
BankSim mô phỏng giao dịch theo thời gian (cột `step` = đơn vị thời gian). Nếu split ngẫu nhiên, test set sẽ chứa thông tin từ quá khứ lẫn tương lai của train, gây **data leakage thời gian**. Temporal split đảm bảo mô hình chỉ học từ quá khứ và dự đoán tương lai — phản ánh đúng điều kiện thực tế.

---

## Step 3 — Categorical Encoding

Các cột categorical (`gender`, `category`, `merchant`, `customer`) được chuyển sang số nguyên bằng `sklearn.preprocessing.LabelEncoder`.

**Nguyên tắc quan trọng:**
- `LabelEncoder` được `.fit()` **chỉ trên `train_df`**
- `test_df` được transform bằng encoder đã fit — không re-fit
- Nhãn trong test set **không xuất hiện trong train** → map thành `-1` (thay vì raise error)

```python
classes_map = {v: i for i, v in enumerate(le.classes_)}
test_df[col] = test_df[col].map(classes_map).fillna(-1).astype(int)
```

Tất cả encoder được lưu vào dict `encoders` để export và tái sử dụng lúc inference.

---

## Step 4 — Feature Engineering

Hàm `add_features(df, ref_df)` nhận thêm tham số `ref_df` — luôn là `train_df`. Mọi thống kê (mean, std, fraud rate...) đều được tính từ `ref_df`, sau đó map/merge vào `df`. Cách này đảm bảo test set **không bị "nhìn thấy" thông tin của chính nó** khi tạo features.

```python
train_df_fe = add_features(train_df, ref_df=train_df)
test_df_fe  = add_features(test_df,  ref_df=train_df)  # ref_df vẫn là train!
```

---

### 4.1 — Nhóm Amount (Đặc trưng về số tiền giao dịch)

| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| `amount_log` | `log1p(amount)` | Log-transform để giảm skewness của phân phối `amount` — giao dịch fraud thường có amount rất lớn, log giúp mô hình xử lý tốt hơn |
| `amount_zscore` | `(amount − μ_train) / (σ_train + ε)` | Z-score chuẩn hoá so với **toàn bộ tập train** — giá trị càng lớn/nhỏ so với trung bình thì càng bất thường |

> `μ_train`, `σ_train` được tính từ `ref_df['amount']` — không dùng thống kê của test.

---

### 4.2 — Nhóm Customer Behavioral Aggregation (Hành vi khách hàng)

Tính từ `ref_df.groupby('customer')['amount']`, sau đó **merge vào `df`** theo `customer`. Khách hàng không xuất hiện trong ref_df sẽ được fill bằng giá trị global.

| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| `cust_mean` | `mean(amount)` theo customer | Mức chi tiêu trung bình của khách hàng đó trong lịch sử (train) |
| `cust_std` | `std(amount)` theo customer | Độ biến động chi tiêu — khách hàng chi tiêu đều hay thất thường |
| `cust_max` | `max(amount)` theo customer | Giao dịch lớn nhất từng thực hiện — ngưỡng tham chiếu |
| `cust_tx_count` | `count(amount)` theo customer | Tổng số giao dịch — phân biệt khách hàng mới ít lịch sử vs. khách quen |
| `amount_deviation` | `(amount − cust_mean) / (cust_std + ε)` | Giao dịch hiện tại lệch bao nhiêu sigma so với thói quen của chính khách hàng đó — phát hiện giao dịch bất thường so với cá nhân |
| `amount_to_max_ratio` | `amount / (cust_max + ε)` | Tỉ lệ so với giao dịch lớn nhất lịch sử — giá trị > 1 nghĩa là phá vỡ kỷ lục chi tiêu cá nhân |

> **Tại sao chia theo customer thay vì global?** Một giao dịch $500 có thể bình thường với người thường xuyên mua đồ xa xỉ, nhưng là dấu hiệu đáng ngờ với người chỉ quen mua $20. Behavioral aggregation cá nhân hoá mức độ "bất thường".

---

### 4.3 — Nhóm Risk Encoding (Mã hoá rủi ro theo danh mục)

Tính **fraud rate** (tỉ lệ gian lận thực tế) của từng nhóm từ `ref_df`. Đây là dạng **target encoding** nhưng chỉ dùng train để tránh leakage.

| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| `category_risk` | `mean(fraud)` groupby `category` (từ ref_df) | Xác suất fraud lịch sử của loại hàng hoá — ví dụ `es_leisure` có fraud rate ~95%, `es_food` ~0% |
| `merchant_risk` | `mean(fraud)` groupby `merchant` (từ ref_df) | Xác suất fraud lịch sử của từng merchant cụ thể |

Nhãn không có trong ref_df (unseen category/merchant trong test) → fill bằng **global fraud rate** của train.

> **Tại sao không dùng category/merchant code trực tiếp?** Sau LabelEncoder, các mã số chỉ là nhãn tuỳ ý không có thứ tự. Risk encoding chuyển chúng thành xác suất có ý nghĩa kinh tế thực — mô hình tree-based có thể split trực tiếp tại ngưỡng rủi ro.

---

### 4.4 — Nhóm Behavioral Flags (Cờ hành vi bất thường)

Trước khi tính nhóm này, `df` được **sort theo `step` tăng dần** để đảm bảo thứ tự thời gian đúng khi dùng `duplicated()`.

| Feature | Điều kiện | Ý nghĩa |
|---------|-----------|---------|
| `is_new_merchant` | `1` nếu đây là lần **đầu tiên** khách hàng giao dịch với merchant này | Khách hàng bị đánh cắp thẻ thường bị dùng ở những nơi họ chưa từng mua — tín hiệu mạnh về fraud |
| `is_new_category` | `1` nếu đây là lần **đầu tiên** khách hàng mua ở category này | Tương tự — pattern chi tiêu đột ngột khác thói quen |
| `is_round_amount` | `1` nếu `amount % 10 == 0` | Giao dịch thử nghiệm (card testing) thường dùng số tròn như $10, $50, $100 |
| `is_micro_tx` | `1` nếu `amount < 1` | Giao dịch kiểm tra thẻ với giá trị cực nhỏ (< $1) — kỹ thuật phổ biến trước khi thực hiện fraud lớn |

**Cách tính `is_new_merchant` và `is_new_category`:**

```python
df = df.sort_values('step').reset_index(drop=True)
df['is_new_merchant'] = (
    ~df.groupby('customer')['merchant'].transform(lambda x: x.duplicated())
).astype(int)
```

`x.duplicated()` đánh dấu `True` cho lần xuất hiện thứ 2 trở đi trong group, `False` cho lần đầu tiên. Phủ định (`~`) cho ra `True` (= 1) tại lần đầu tiên. Vì đã sort theo `step`, "lần đầu tiên" theo thứ tự thời gian là đúng nghĩa.

---

### 4.5 — Nhóm Time Features (Đặc trưng thời gian)

Cột `step` trong BankSim đại diện cho thời gian tổng hợp (1 step = 1 giờ theo thiết kế dataset). Từ đó có thể suy ra giờ trong ngày và ngày trong tuần.

| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| `hour` | `step % 24` | Giờ trong ngày (0–23) — khung giờ giao dịch |
| `is_night` | `1` nếu `hour >= 23` hoặc `hour <= 5` | Giao dịch trong đêm khuya (23:00–05:00) — thường ít được giám sát, tỉ lệ fraud cao hơn |
| `is_weekend` | `1` nếu `(step // 24) % 7 >= 5` | Giao dịch vào cuối tuần (thứ 7, CN) — thời điểm hệ thống giám sát có thể giảm |

---

### 4.6 — Nhóm Demographic Deviation (Lệch chuẩn nhân khẩu học)

Tương tự behavioral aggregation nhưng ở cấp độ **nhóm tuổi** thay vì cá nhân.

| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| `age_group_mean` | `mean(amount)` groupby `age` (từ ref_df) | Mức chi tiêu trung bình của nhóm tuổi — đặc trưng nhân khẩu học |
| `age_group_std` | `std(amount)` groupby `age` (từ ref_df) | Độ biến động chi tiêu trong nhóm tuổi |
| `age_group_deviation` | `(amount − age_group_mean) / (age_group_std + ε)` | Giao dịch hiện tại lệch bao nhiêu so với mức bình thường của người cùng nhóm tuổi — phát hiện chi tiêu bất thường theo demographic |

> `age_group_mean` và `age_group_std` được giữ lại như features riêng (không chỉ dùng để tính deviation) vì chúng phản ánh đặc trưng chi tiêu của nhóm — có ích cho mô hình.

---

### Tổng hợp tất cả features sau Step 4

| # | Feature | Nhóm |
|---|---------|------|
| 1 | `step` | Raw |
| 2 | `customer` | Raw (encoded) |
| 3 | `age` | Raw (cleaned) |
| 4 | `gender` | Raw (encoded) |
| 5 | `merchant` | Raw (encoded) |
| 6 | `category` | Raw (encoded) |
| 7 | `amount` | Raw |
| 8 | `age_unknown` | Cleaning flag |
| 9 | `amount_log` | Amount |
| 10 | `amount_zscore` | Amount |
| 11 | `cust_mean` | Customer Behavioral |
| 12 | `cust_std` | Customer Behavioral |
| 13 | `cust_max` | Customer Behavioral |
| 14 | `cust_tx_count` | Customer Behavioral |
| 15 | `amount_deviation` | Customer Behavioral |
| 16 | `amount_to_max_ratio` | Customer Behavioral |
| 17 | `category_risk` | Risk Encoding |
| 18 | `merchant_risk` | Risk Encoding |
| 19 | `is_new_merchant` | Behavioral Flag |
| 20 | `is_new_category` | Behavioral Flag |
| 21 | `is_round_amount` | Behavioral Flag |
| 22 | `is_micro_tx` | Behavioral Flag |
| 23 | `hour` | Time |
| 24 | `is_night` | Time |
| 25 | `is_weekend` | Time |
| 26 | `age_group_mean` | Demographic |
| 27 | `age_group_std` | Demographic |
| 28 | `age_group_deviation` | Demographic |

**Tổng: 28 features** (không tính cột target `fraud`).

---

## Step 5 — SMOTE Variant Comparison

### Vấn đề class imbalance

Dataset có ~1.2% giao dịch fraud — mô hình train trực tiếp trên dữ liệu gốc thường bị bias về class 0 (non-fraud), bỏ sót nhiều fraud thực sự. Oversampling tăng synthetic samples cho class thiểu số.

### Phương pháp so sánh

Dùng **LightGBM** làm proxy model với **3-fold Stratified K-Fold** (không shuffle để giữ thứ tự thời gian). Metric: **Average Precision (AUC-PR)** — phù hợp hơn AUC-ROC cho imbalanced data vì tập trung vào precision/recall của class thiểu số.

| Variant | Mô tả |
|---------|-------|
| **No resampling** | Train trực tiếp trên dữ liệu gốc, để mô hình tự xử lý imbalance |
| **SMOTE(0.1)** | Tạo synthetic samples cho class fraud bằng nội suy tuyến tính giữa các điểm fraud lân cận (k-NN). `sampling_strategy=0.1` → tỉ lệ fraud:non-fraud = 1:10 sau resampling |
| **BorderlineSMOTE(0.1)** | Chỉ oversample các điểm fraud nằm gần **ranh giới quyết định** (dễ bị phân loại sai) — tập trung vào vùng khó hơn SMOTE thông thường |
| **SMOTEENN(0.1)** | Kết hợp SMOTE (oversample fraud) với **Edited Nearest Neighbours** (xoá các điểm non-fraud nằm trong vùng của fraud) — vừa tăng fraud vừa làm sạch ranh giới |

**Tham số cố định:** `sampling_strategy=0.1`, `random_state=42` cho cả 4 variant.

### Chọn variant tốt nhất

```python
best_variant = results_df.loc[results_df['Mean AUC-PR'].idxmax(), 'Variant']
```

Variant có **Mean AUC-PR cao nhất** trên 3 fold được chọn, sau đó **re-fit trên toàn bộ train set** để tạo `X_train_res`, `y_train_res`.

> **Lưu ý:** SMOTE chỉ áp dụng cho **train set**, không bao giờ áp dụng cho test set — test set phải giữ nguyên phân phối thực tế để đánh giá chính xác.

---

## Step 6 — Export

Tất cả artifacts được lưu bằng `joblib` vào `data/processed/`:

| File | Nội dung | Dùng cho |
|------|----------|---------|
| `X_train.pkl` | Feature matrix train (trước SMOTE) | Baseline training / evaluation |
| `X_test.pkl` | Feature matrix test | Final evaluation |
| `y_train.pkl` | Labels train (trước SMOTE) | Baseline training |
| `y_test.pkl` | Labels test | Final evaluation |
| `X_train_res.pkl` | Feature matrix train **sau SMOTE** tốt nhất | Training chính thức |
| `y_train_res.pkl` | Labels train **sau SMOTE** tốt nhất | Training chính thức |
| `encoders.pkl` | Dict `{col: LabelEncoder}` cho 4 categorical cols | Inference / transform data mới |
| `feature_cols.pkl` | List tên 28 features theo đúng thứ tự | Đảm bảo input shape nhất quán |

---

## Nguyên tắc chống Data Leakage

Toàn bộ pipeline tuân thủ nguyên tắc: **test set không được tham gia vào bất kỳ bước fitting nào**.

| Bước | Fit trên | Transform |
|------|----------|-----------|
| LabelEncoder | `train_df` only | train + test |
| amount z-score (μ, σ) | `train_df` only | train + test |
| Customer behavioral stats | `ref_df = train_df` | train + test |
| Category/merchant risk | `ref_df = train_df` | train + test |
| Age group stats | `ref_df = train_df` | train + test |
| SMOTE | `train_df` only | train only |
