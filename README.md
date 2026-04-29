# Credit Card Fraud Detection — BankSim Dataset

A multi-tier machine learning system for financial fraud detection built on the BankSim synthetic transaction dataset. The project explores three complementary paradigms: tabular gradient boosting (Tier 0), graph-enhanced feature engineering (Tier 1), and a Graph Neural Network (Tier 2), comparing their effectiveness on a highly imbalanced classification task.

---

In the 'src' folder, create 'data' folder and add the 'raw' and 'processed' subfolders. In the 'raw' folder, add the 'bs140513_032310.csv' file. In the 'processed' folder, add the 'X_train.pkl', 'y_train.pkl', 'X_val.pkl', 'y_val.pkl', 'X_test.pkl', 'y_test.pkl', 'X_train_res.pkl', 'y_train_res.pkl', 'X_train_graph.pkl', 'X_val_graph.pkl', 'X_test_graph.pkl', 'feature_cols.pkl', 'feature_cols_graph.pkl', and 'encoders.pkl' files.

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Project Structure](#2-project-structure)
3. [Methodology Overview](#3-methodology-overview)
4. [Data Preprocessing & Feature Engineering](#4-data-preprocessing--feature-engineering)
5. [Tier 0 — Tabular ML Models](#5-tier-0--tabular-ml-models)
6. [Tier 0 — Ensemble Strategies](#6-tier-0--ensemble-strategies)
7. [Tier 1 — Graph Feature Engineering](#7-tier-1--graph-feature-engineering)
8. [Tier 2 — GraphSAGE (GNN)](#8-tier-2--graphsage-gnn)
9. [Cross-Tier Comparison](#9-cross-tier-comparison)
10. [Error Analysis](#10-error-analysis)
11. [How to Run](#11-how-to-run)
12. [Requirements](#12-requirements)

---

## 1. Dataset Overview

**Source:** BankSim — synthetic bank transaction dataset  
**Files:**
- `bs140513_032310.csv` — tabular transaction records (used for all ML models)
- `bsNET140513_032310.csv` — network/graph format (used for GraphSAGE only)

### Basic Statistics

| Property | Value |
|---|---|
| Total transactions | 594,643 |
| Features (raw) | 10 columns |
| Fraudulent transactions | 7,200 |
| Legitimate transactions | 587,443 |
| Fraud rate | **1.21%** |
| Missing values | 0 |
| Memory (tabular CSV) | 45.4+ MB |

### Columns

| Column | Type | Description |
|---|---|---|
| `step` | int | Time step (0–179, proxy for hour-of-week) |
| `customer` | string | Customer ID |
| `age` | string | Age group (0–6, U=unknown) |
| `gender` | string | M / F / E / U |
| `zipcodeOri` | string | Customer zip code (zero-variance — dropped) |
| `merchant` | string | Merchant ID (50 unique merchants) |
| `zipMerchant` | string | Merchant zip code (zero-variance — dropped) |
| `category` | string | Transaction category (15 unique) |
| `amount` | float | Transaction amount (0.0 – 8,329.96) |
| `fraud` | int | Target label: 1 = fraud, 0 = legitimate |

### Amount Statistics

| Statistic | Normal | Fraud |
|---|---|---|
| Mean | 31.85 | 530.93 |
| Median | 26.61 | 319.18 |
| Max | 2,144.86 | 8,329.96 |
| Min | 0.00 | 0.03 |
| Std | 31.47 | 835.59 |

### Fraud Rate by Category (Top 5)

| Category | Fraud Rate |
|---|---|
| es_leisure | 94.99% |
| es_travel | 79.40% |
| es_sportsandtoys | 49.53% |
| es_otherservices | 25.00% |
| es_hotelservices | 31.42% |

### Cardinality

| Column | Unique Values |
|---|---|
| step | 180 |
| customer | 4,112 |
| age | 8 (0–6 + U) |
| gender | 4 (M, F, E, U) |
| merchant | 50 |
| category | 15 |

### Train / Val / Test Split

**Method:** Temporal split — sorted by `step` ascending, no shuffle (prevents future data leakage).

| Split | Rows | Fraud Count | Fraud Rate |
|---|---|---|---|
| Train (70%) | 416,250 | ~5,040 | ~1.21% |
| Validation (15%) | 89,196 | ~972 | ~1.09% |
| Test (15%) | 89,197 | 962 | ~1.08% |

After SMOTE(0.1) resampling on train:

| Split | Rows | Fraud Rate |
|---|---|---|
| Train (after SMOTE) | 452,082 | 9.09% |

---

## 2. Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── raw/
│   │   │   └── bs140513_032310.csv          # Original BankSim dataset
│   │   └── processed/                       # Serialized .pkl feature splits
│   │       ├── X_train.pkl / y_train.pkl
│   │       ├── X_val.pkl   / y_val.pkl
│   │       ├── X_test.pkl  / y_test.pkl
│   │       ├── X_train_res.pkl / y_train_res.pkl   # SMOTE-resampled
│   │       ├── X_train_graph.pkl / X_val_graph.pkl / X_test_graph.pkl
│   │       ├── feature_cols.pkl / feature_cols_graph.pkl
│   │       └── encoders.pkl
│   ├── features/
│   │   └── feature_engineering.py           # Pipeline: time / amount / frequency / target-encoding
│   ├── models/
│   │   ├── trainer.py                       # MLflow-backed training orchestrator
│   │   └── predictor.py                     # Inference wrapper for FastAPI
│   └── evaluation/
│       └── evaluator.py                     # Metrics, threshold tuning, cost analysis
│
├── notebooks/
│   ├── exploration/
│   │   ├── Data_exploration.ipynb           # Dataset overview, amount stats
│   │   ├── EDA.ipynb                        # Distribution plots, fraud by category/age
│   │   └── Data_exploration_NET.ipynb       # Network graph EDA
│   ├── data_preprocessing/
│   │   └── preprocessing.ipynb              # Raw → cleaned parquet splits
│   ├── Feature_engineering/
│   │   └── Data_Preprocessing_FE.ipynb      # FE pipeline + SMOTE comparison + export
│   ├── Model_Training/
│   │   ├── LightGBM_XGBoost.ipynb           # Optuna tuning, loss curves, SHAP
│   │   ├── BalancedRandomForestClassifier.ipynb
│   │   └── Catboost.ipynb
│   ├── Model_ensemble/
│   │   └── Ensemble_Strategies.ipynb        # Hybrid Funnel + Weighted Stacking
│   └── Graph_Base/
│       ├── Graph_Features.ipynb             # Bipartite graph construction + 6 graph features
│       ├── Retrain_Graph_ErrorAnalysis.ipynb # Retrain with 34 features + error analysis
│       └── GraphSAGE.ipynb                  # Edge-level GNN training + evaluation
│
├── tests/
│   ├── unit/
│   │   ├── test_feature_engineering.py
│   │   └── test_preprocessing.py
│   └── integration/
│
├── api/
│   └── main.py                              # FastAPI scoring endpoint (POST /predict)
│
├── scripts/
│   └── run_pipeline.py                      # CLI: end-to-end pipeline runner
│
├── requirements.txt
└── README.md
```

---

## 3. Methodology Overview

The project is structured as three sequential tiers of increasing modeling complexity:

```
Raw BankSim Data (594,643 transactions)
         │
         ▼
┌─────────────────────────────────┐
│  Tier 0 — Tabular ML            │  ← 28 engineered features
│  LightGBM / XGBoost / CatBoost  │    + SMOTE(0.1) resampling
│  BalancedRandomForest            │    + Optuna hyperparameter tuning
│  + 3 Ensemble Strategies        │
└─────────────┬───────────────────┘
              │ if performance insufficient
              ▼
┌─────────────────────────────────┐
│  Tier 1 — Graph Feature Eng.    │  ← 28 + 6 = 34 features
│  Bipartite customer↔merchant    │    PageRank / degree / fraud rates
│  graph from train set only      │    Same 4 models retrained
└─────────────┬───────────────────┘
              │ exploring GNN
              ▼
┌─────────────────────────────────┐
│  Tier 2 — GraphSAGE (GNN)       │  ← Edge-level classification
│  2-layer SAGEConv, edge MLP     │    on bsNET graph
│  BCEWithLogitsLoss + pos_weight  │
└─────────────────────────────────┘
```

**Primary metric:** PR-AUC (Precision-Recall Area Under Curve) — chosen because the dataset is highly imbalanced (1.21% fraud), making accuracy and ROC-AUC misleading.

**Secondary metric:** F2-score — weights recall twice as heavily as precision, reflecting the business cost of missed fraud (FN) being greater than false alarms (FP).

---

## 4. Data Preprocessing & Feature Engineering

### 4.1 Data Cleaning

- Strip quotes from string columns (`customer`, `age`, `gender`, `merchant`, `category`)
- Map `age='U'` (unknown) to `-1`; create binary `age_unknown` flag
- Drop `zipcodeOri` and `zipMerchant` (zero-variance: single value `28007`)
- Label-encode categorical columns (`gender`, `category`, `merchant`, `customer`) — fitted on train only; unseen labels → `-1`

### 4.2 SMOTE Variant Comparison

3-fold `StratifiedKFold` (no shuffle), LightGBM proxy model, metric = PR-AUC:

| Resampling Variant | Mean PR-AUC | Std | Avg Train Size |
|---|---|---|---|
| No resampling | 0.7898 | 0.1533 | 277,500 |
| **SMOTE(0.1)** | **0.9168** | **0.0165** | **301,387** |
| BorderlineSMOTE(0.1) | 0.9146 | 0.0197 | 301,387 |
| SMOTEENN(0.1) | 0.8983 | 0.0168 | 297,079 |

**Winner: SMOTE(0.1)** — applied only to the training set; val/test remain untouched.  
Post-SMOTE train size: **452,082** rows at fraud rate **9.09%**.

### 4.3 Engineered Features (28 total)

All aggregations and risk encodings are computed **from the training set only** and then mapped to val/test — no data leakage.

| # | Feature | Group | Computation |
|---|---|---|---|
| 1 | `step` | Raw | Original time step |
| 2 | `customer` | Raw | Label-encoded |
| 3 | `age` | Raw | Cleaned integer (−1 = unknown) |
| 4 | `gender` | Raw | Label-encoded |
| 5 | `merchant` | Raw | Label-encoded |
| 6 | `category` | Raw | Label-encoded |
| 7 | `amount` | Raw | Original float |
| 8 | `age_unknown` | Raw | Binary flag for age='U' |
| 9 | `amount_log` | Amount Transform | `log1p(amount)` |
| 10 | `amount_zscore` | Amount Transform | `(amount − train_mean) / train_std` |
| 11 | `cust_mean` | Customer Behavioral | Mean amount per customer (train agg) |
| 12 | `cust_std` | Customer Behavioral | Std amount per customer (train agg) |
| 13 | `cust_max` | Customer Behavioral | Max amount per customer (train agg) |
| 14 | `cust_tx_count` | Customer Behavioral | Transaction count per customer (train agg) |
| 15 | `amount_deviation` | Customer Behavioral | `(amount − cust_mean) / (cust_std + ε)` |
| 16 | `amount_to_max_ratio` | Customer Behavioral | `amount / (cust_max + ε)` |
| 17 | `category_risk` | Risk Encoding | Fraud rate per category (train-only mean) |
| 18 | `merchant_risk` | Risk Encoding | Fraud rate per merchant (train-only mean) |
| 19 | `is_new_merchant` | Behavioral Flags | 1 if first transaction with this merchant |
| 20 | `is_new_category` | Behavioral Flags | 1 if first transaction in this category |
| 21 | `is_round_amount` | Behavioral Flags | `amount % 10 == 0` |
| 22 | `is_micro_tx` | Behavioral Flags | `amount < 1` |
| 23 | `hour` | Time | `step % 24` |
| 24 | `is_night` | Time | 1 if hour ≥ 23 or hour ≤ 5 |
| 25 | `is_weekend` | Time | 1 if `(step // 24) % 7 >= 5` |
| 26 | `age_group_mean` | Demographic Deviation | Mean amount per age group (train agg) |
| 27 | `age_group_std` | Demographic Deviation | Std amount per age group (train agg) |
| 28 | `age_group_deviation` | Demographic Deviation | `(amount − age_group_mean) / (age_group_std + ε)` |

### 4.4 Anti-Leakage Protocol

- All `groupby` aggregations (`cust_mean`, `category_risk`, `merchant_risk`, etc.) use `ref_df=train_df`
- Val/test receive these values via left-join mapping (no recomputation)
- Label encoders: `le.fit(train)` → `val/test` mapped via class dict, unseen → `-1`
- **Merchant risk leakage audit:** PR-AUC with `merchant_risk` = 0.9247; without = 0.9095; delta = **+0.0152** — feature is informative but not leaking (independent train-only encoding confirmed)

---

## 5. Tier 0 — Tabular ML Models

All models use Optuna hyperparameter optimization (objective: maximize PR-AUC on validation set, TPE sampler).

### 5.1 LightGBM

**Training data:** SMOTE-resampled (`X_train_res`, 452,082 rows)  
**Optuna:** 45 trials, TPE sampler, objective = val PR-AUC

| Parameter | Baseline | Optuna Best |
|---|---|---|
| n_estimators | 1,000 | **849** |
| max_depth | 6 | **9** |
| learning_rate | 0.05 | **0.00902** |
| num_leaves | 31 | **236** |
| min_child_samples | — | **69** |
| subsample | — | **0.9091** |
| colsample_bytree | — | **0.5455** |

| Metric | Baseline Val | Optuna Val | Optuna Test |
|---|---|---|---|
| PR-AUC | 0.9354 | 0.9440 | **0.9317** |
| F1 | 0.8471 | 0.8590 | 0.8262 |
| Recall | 0.7778 | 0.8117 | 0.7536 |
| Precision | 0.9299 | 0.9121 | 0.9142 |
| ROC-AUC | 0.9989 | 0.9991 | 0.9989 |

**Optuna gain:** Val PR-AUC +0.0087 | Val/Test gap: 0.0123 (no overfitting concern)  
**Confusion matrix (test, threshold=0.5):** TN≈88,167 | FP≈68 | FN≈237 | TP≈725

---

### 5.2 XGBoost

**Training data:** SMOTE-resampled  
**Optuna:** 100 trials, TPE sampler, objective = val PR-AUC

| Parameter | Optuna Best |
|---|---|
| n_estimators | 771 |
| max_depth | 6 |
| learning_rate | 0.02379 |
| subsample | 0.8997 |
| colsample_bytree | 0.5739 |
| min_child_weight | 13 |
| scale_pos_weight | 1.055 |
| tree_method | hist |
| eval_metric | aucpr |

| Metric | Optuna Val | Optuna Test |
|---|---|---|
| PR-AUC | 0.9434 | **0.9293** |
| F1 | 0.8548 | 0.8312 |
| Recall | 0.8056 | 0.7703 |
| Precision | 0.9105 | 0.9026 |
| ROC-AUC | 0.9991 | 0.9989 |

**Optuna gain:** Val PR-AUC +0.0083 | Val/Test gap: 0.0141  
**Confusion matrix (test, threshold=0.5):** TN≈88,155 | FP≈80 | FN≈221 | TP≈741

---

### 5.3 CatBoost

**Training data:** SMOTE-resampled  
**Optuna:** 45 trials, TPE sampler, objective = val PR-AUC

| Parameter | Optuna Best |
|---|---|
| iterations | 1,983 |
| depth | 8 |
| learning_rate | 0.006463 |
| l2_leaf_reg | 1.028 |
| subsample | 0.6831 |
| colsample_bylevel | 0.8616 |
| min_data_in_leaf | 13 |
| eval_metric | AUC |

| Metric | Baseline Val | Optuna Val | Optuna Test |
|---|---|---|---|
| PR-AUC | 0.9372 | 0.9430 | **0.9312** |
| F1 | 0.8523 | 0.8533 | 0.8363 |
| Recall | 0.7953 | 0.8169 | 0.7859 |
| Precision | 0.9181 | 0.8931 | 0.8936 |
| ROC-AUC | 0.9990 | 0.9991 | 0.9989 |

**Optuna gain:** Val PR-AUC +0.0058  
**Confusion matrix (test, threshold=0.5):** TN≈88,145 | FP≈90 | FN≈206 | TP≈756

---

### 5.4 BalancedRandomForestClassifier

**Training data:** Original imbalanced train set (`X_train`, 416,250 rows) — BRF handles class imbalance internally via balanced bootstrap. SMOTE not applied.  
**Optuna:** 45 trials, TPE sampler, objective = val PR-AUC

| Parameter | Optuna Best |
|---|---|
| n_estimators | 1,162 |
| max_depth | 11 |
| min_samples_leaf | 5 |
| min_samples_split | 10 |
| max_features | sqrt |
| sampling_strategy | auto |
| replacement | False |
| bootstrap | True |

| Metric | Baseline Val | Optuna Val | Optuna Test |
|---|---|---|---|
| PR-AUC | 0.8730 | 0.8996 | **0.8833** |
| F1 | 0.5647 | 0.5843 | 0.5974 |
| Recall | 0.9928 | 0.9897 | 0.9802 |
| Precision | 0.3945 | 0.4145 | 0.4296 |
| ROC-AUC | 0.9981 | 0.9985 | 0.9983 |

**Optuna gain:** Val PR-AUC +0.0266  
**Confusion matrix (test, threshold=0.5):** TN≈86,983 | FP≈1,252 | FN≈19 | TP≈943  
**Note:** BRF trades precision for recall — only 19 missed fraud at the cost of ~1,252 false alarms.

---

### 5.5 Individual Model Summary (Test Set)

| Model | PR-AUC | F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|---|
| **LightGBM** | **0.9317** | 0.8262 | 0.7536 | 0.9142 | 0.9989 |
| CatBoost | 0.9312 | 0.8363 | 0.7859 | 0.8936 | 0.9989 |
| XGBoost | 0.9293 | 0.8312 | 0.7703 | 0.9026 | 0.9989 |
| BalancedRF | 0.8833 | 0.5974 | 0.9802 | 0.4296 | 0.9983 |

---

## 6. Tier 0 — Ensemble Strategies

Three ensemble strategies are evaluated. Base models are retrained with best-known parameters.

**Base model params in ensemble context:**
- BRF: n_estimators=500, max_depth=None, min_samples_leaf=2, sampling_strategy='auto'
- LightGBM: n_estimators=1588, max_depth=8, lr=0.0111, num_leaves=262
- XGBoost: n_estimators=1373, max_depth=8, lr=0.0070, scale_pos_weight=5.08
- CatBoost: iterations=1000, depth=6, lr=0.05, l2_leaf_reg=3.0

### 6.1 Hybrid Funnel (Two-Stage Filtering)

Exploits BRF's near-perfect recall as a coarse net and boosting models' precision as a confirmation filter.

```
All Transactions
     │
 [Stage 1: BRF threshold = 0.4]
  │               │
Flagged       Not Flagged ──► label = 0 (Legitimate)
  │
 [Stage 2: avg(XGB, CAT) threshold = 0.5]
  │                   │
Confirmed         Not Confirmed
  │                   │
label=2           label=1
(High-Risk)       (Review Required)
```

**Three-tier distribution on test set:**

| Label | Count | % |
|---|---|---|
| 0 — Legitimate | 86,672 | 97.17% |
| 1 — Review Required | 1,647 | 1.85% |
| 2 — High-Risk Fraud | 878 | 0.98% |
| Actual fraud | 962 | 1.08% |

**Conservative mode** (label=2 only = fraud):

| Metric | Value |
|---|---|
| PR-AUC | 0.9283 |
| F1 | 0.8337 |
| Recall | 0.7973 |
| Precision | 0.8736 |
| FN | ~195 |
| FP | ~111 |

**Aggressive mode** (label≥1 = fraud):

| Metric | Value |
|---|---|
| PR-AUC | 0.9283 |
| F1 | 0.5489 |
| Recall | **0.9948** |
| Precision | 0.3790 |
| FN | **~5** |
| FP | ~1,566 |

> Aggressive mode misses only **5 fraudulent transactions** out of 962 — the best recall of any strategy evaluated.

---

### 6.2 Weighted Stacking (Meta-Learner)

**Protocol:**
1. Train 4 base models on `X_train_res`
2. Generate fraud probabilities on `X_val` → 4-column meta-feature matrix (89,196 × 4)
3. Train `LogisticRegression(C=1.0, class_weight='balanced')` on val meta-features
4. Apply to test meta-features for final predictions

**Meta-learner coefficients (Logistic Regression):**

| Base Model | Coefficient | Interpretation |
|---|---|---|
| P_BRF | **+1.3883** | Highest trust — BRF's recall signal is most informative |
| P_CAT | +0.1804 | Second most trusted |
| P_LGB | +0.1725 | Third |
| P_XGB | **−0.0139** | Negative — multicollinearity with LGB/CAT, redundant signal |

| Metric | Value |
|---|---|
| PR-AUC | 0.9260 |
| F1 | 0.6062 |
| Recall | 0.9834 |
| Precision | 0.4382 |
| FN | ~16 |
| FP | ~1,214 |

---

### 6.3 Full Ensemble Comparison (Test Set)

| Strategy | PR-AUC | F1 | Recall | Precision | FN | FP |
|---|---|---|---|---|---|---|
| LightGBM (base) | 0.9323 | 0.8221 | 0.7349 | 0.9327 | ~255 | ~51 |
| XGBoost (base) | 0.9304 | 0.8357 | 0.8669 | 0.8066 | ~128 | ~200 |
| CatBoost (base) | 0.9203 | 0.8192 | 0.7464 | 0.9077 | ~244 | ~73 |
| BRF (base) | 0.8871 | 0.5851 | 0.9886 | 0.4155 | ~11 | ~1,342 |
| Funnel – Conservative | 0.9283 | 0.8337 | 0.7973 | 0.8736 | ~195 | ~111 |
| **Funnel – Aggressive** | 0.9283 | 0.5489 | **0.9948** | 0.3790 | **~5** | ~1,566 |
| Weighted Stacking | 0.9260 | 0.6062 | 0.9834 | 0.4382 | ~16 | ~1,214 |

**Key observation:** No ensemble strategy improves PR-AUC over the best individual model (LightGBM, 0.9317/0.9323). This is consistent with high error correlation among the boosting models — they tend to miss the same hard cases.

---

## 7. Tier 1 — Graph Feature Engineering

### 7.1 Graph Construction

A **bipartite undirected graph** is built from the training set only (no val/test data leakage):

- **Customer nodes:** 4,111 unique customers (bipartite side 0)
- **Merchant nodes:** 50 unique merchants (bipartite side 1)
- **Edges:** 38,924 unique customer-merchant pairs (one edge per unique pair)
- **Tool:** NetworkX 3.5

### 7.2 Graph Features (6 additional)

| Feature | Definition | Computation |
|---|---|---|
| `customer_degree` | Number of unique merchants a customer transacted with | `G.degree(customer_node)` — train graph |
| `merchant_degree` | Number of unique customers a merchant served | `G.degree(merchant_node)` — train graph |
| `merchant_fraud_rate` | Fraction of merchant's transactions that are fraud | `groupby(merchant)[fraud].mean()` — train only |
| `customer_fraud_rate` | Fraction of customer's transactions that are fraud | `groupby(customer)[fraud].mean()` — train only |
| `customer_pagerank` | PageRank score of customer node | `nx.pagerank(G, alpha=0.85)` — train graph |
| `merchant_pagerank` | PageRank score of merchant node | Same PageRank run |

**Global mean fallback** applied for val/test nodes unseen in the train graph.

**Global mean values (train graph):**

| Feature | Global Mean |
|---|---|
| customer_degree | 9.47 |
| merchant_degree | 778.48 |
| merchant_fraud_rate | 0.2233 |
| customer_fraud_rate | 0.0283 |
| customer_pagerank | 0.000131 |
| merchant_pagerank | 0.009190 |

Total features after adding graph: **34** (28 tabular + 6 graph).

### 7.3 Sanity Check

Quick LightGBM validation (300 trees, same params, no Optuna):

| | PR-AUC (Val) |
|---|---|
| Without graph features | 0.9365 |
| With graph features | 0.9296 |
| Delta | **−0.0068** |

This negative delta on the sanity check foreshadows the full retrain results.

### 7.4 Full Retrain — Baseline vs Graph-Enhanced (Test Set)

All 4 models retrained with **same Optuna best params** (no re-tuning):

| Model | Baseline PR-AUC | Graph PR-AUC | Δ PR-AUC | Δ F2 |
|---|---|---|---|---|
| LightGBM | 0.9317 | 0.9221 | **−0.0096** | — |
| XGBoost | 0.9293 | 0.9086 | **−0.0207** | — |
| CatBoost | 0.9312 | 0.9234 | **−0.0078** | — |
| BalancedRF | 0.8833 | 0.8404 | **−0.0429** | — |

**All four models degrade when graph features are added.**

### 7.5 Root Cause Analysis

1. **Overlap with tabular features:** `merchant_fraud_rate` (graph) and `merchant_risk` (tabular FE) are both computed from `train.groupby('merchant')['fraud'].mean()` — they carry nearly identical signal but from slightly different column names, causing confusion to the model rather than adding information.

2. **PageRank is uninformative for edge-level fraud:** With only 50 merchants, PageRank scores are tightly clustered (merchant PageRank range: 0.000061–0.044710). All high-degree merchants get similar PageRank, providing no additional discriminative power.

3. **Degree features are noisy:** All customers transact with a very limited set of merchants (median customer_degree = 9). Degree doesn't correlate well with fraud risk in BankSim.

4. **Conclusion:** Tier 1 is discarded. **Tier 0 (28 tabular features) is the best performing configuration.** Graph features hurt all models — graph topology in BankSim is not informative enough to compensate for the noise introduced.

---

## 8. Tier 2 — GraphSAGE (GNN)

### 8.1 Data Source

Uses `bsNET140513_032310.csv` — the network representation of BankSim with columns: `Source`, `Target`, `Weight`, `typeTrans`, `fraud`.

**Split:** Random 80/10/10 (edge-level), not temporal.

### 8.2 Architecture

```python
class EdgeGraphSAGE(torch.nn.Module):
    conv1 = SAGEConv(in_channels=2,  out_channels=64, aggr='mean')
    conv2 = SAGEConv(in_channels=64, out_channels=64, aggr='mean')
    fc    = Sequential(
        Linear(64*2 + 2, 64),   # concat src_emb + tgt_emb + edge_attr(2)
        ReLU(),
        Dropout(0.3),
        Linear(64, 1)
    )
```

**Node features (2-dimensional):**
- Feature 0: total degree (in-degree + out-degree, all edges)
- Feature 1: fraud rate per node (fraud occurrences / degree, train edges only)

**Edge features (2-dimensional):**
- Feature 0: `Weight` (transaction amount, StandardScaler fitted on train)
- Feature 1: `typeTrans` (LabelEncoded, 0–14)

**Edge representation:** `concat(src_emb[64], tgt_emb[64], edge_attr[2])` → 130-dim → MLP → scalar logit

### 8.3 Training Configuration

| Hyperparameter | Value |
|---|---|
| Loss function | BCEWithLogitsLoss |
| pos_weight | dynamic: `num_neg / num_pos` (from train edges) |
| Optimizer | Adam(lr=1e-4, weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau(mode='max', factor=0.5, patience=5) |
| Gradient clipping | max_norm=1.0 |
| Dropout | 0.3 (in both SAGEConv and MLP) |
| Max epochs | 300 |
| Early stopping | patience=20, monitor=val PR-AUC |
| Threshold | Optimized via F2-score on val PR curve |

### 8.4 Training Progress

| Epoch | Train Loss | Val Loss | Val PR-AUC |
|---|---|---|---|
| 10 | 2.9375 | 2.4423 | 0.0096 |
| 20 | 2.2808 | 1.7680 | 0.0487 |
| 30 | 1.6217 | 1.2343 | 0.7337 |
| 40 | 1.2218 | 0.9018 | 0.7886 |
| 50 | 1.0810 | 0.7055 | 0.7937 |
| 60 | 0.6526 | 0.5789 | 0.7962 |
| 70 | 0.5096 | 0.4800 | 0.7962 |
| 80 | 0.4735 | 0.4434 | 0.7957 |
| **64** (best) | — | — | **0.7964** |
| 84 (early stop) | — | — | — |

### 8.5 Test Set Results

| Metric | Value |
|---|---|
| **PR-AUC** | **0.7985** |
| F2-Score | 0.7845 |
| Precision | 0.6234 |
| Recall | 0.8387 |
| Optimal Threshold | 0.6082 |
| Best Epoch | 64 |

---

## 9. Cross-Tier Comparison

| Tier | Best Model | PR-AUC | Recall | Precision | Features | Complexity |
|---|---|---|---|---|---|---|
| **Tier 0 Tabular** | **LightGBM** | **0.9317** | 0.7536 | 0.9142 | 28 | Low |
| Tier 1 Graph FE | CatBoost | 0.9234 | — | — | 34 | Low+ |
| Tier 2 GNN | GraphSAGE | 0.7985 | 0.8387 | 0.6234 | 2 (raw) | High |

**Winner: Tier 0 Tabular ML**

### Why Tabular ML Dominates

1. **BankSim graph topology is too simple:** With only 50 merchants, the customer-merchant bipartite graph has very low structural diversity. Every customer connects to a small overlapping subset of the same 50 merchants, making GNN neighborhood aggregation uninformative — neighboring nodes carry nearly the same signal.

2. **Rich domain features outperform structural signals:** `merchant_risk`, `amount_deviation`, and `category_risk` encode the fraud signal directly from transactions. GNNs need to rediscover these patterns through message passing, starting from just 2 node features, which is an information disadvantage.

3. **Temporal structure ignored in GNN:** The transductive GraphSAGE setup uses random 80/10/10 splits, discarding the temporal ordering. The tabular models use temporal splits, which is more realistic and avoids future data contamination.

4. **GraphSAGE PR-AUC gap: −0.1332 vs LightGBM.** The 13-point gap cannot be closed purely by architecture changes given the structural limitations of BankSim's graph.

---

## 10. Error Analysis

**Model:** CatBoost graph-enhanced (best PR-AUC in Tier 1: 0.9234), evaluated at threshold=0.5.

**Error counts:**
- FN = 202 (21.0% of actual fraud missed)
- TP = 760 (79.0% caught)

### 10.1 Amount Distribution

| Group | Median Amount |
|---|---|
| FN (missed fraud) | lower |
| TP (caught fraud) | higher |

`amount_log` is the top differentiating feature (normalized diff = 0.991). The model consistently misses **low-amount fraud** — transactions that fall closer to the distribution of legitimate transactions.

### 10.2 Merchant Risk Profile

| Group | Mean merchant_fraud_rate |
|---|---|
| FN | lower |
| TP | higher |

Missed fraud disproportionately occurs at merchants with **low historical fraud rates** — "clean-looking" merchants where the risk signal is weak.

### 10.3 Predicted Probability

FN transactions cluster near the decision boundary (predicted probability median well below threshold=0.5), while TP transactions have high-confidence predictions. This indicates the missed frauds are **genuinely hard borderline cases**, not systematic model failures.

### 10.4 Top 5 Differentiating Features (FN vs TP)

| Rank | Feature | FN Signal | TP Signal | Direction |
|---|---|---|---|---|
| 1 | `amount_log` | Lower | Higher | FN = low-amount fraud |
| 2 | `merchant_fraud_rate` | Lower | Higher | FN = clean merchants |
| 3 | `customer_fraud_rate` | Lower | Higher | FN = low-risk customers |
| 4 | `amount_deviation` | Lower | Higher | FN = normal-looking amounts |
| 5 | `merchant_degree` | varies | varies | Graph topology signal |

### 10.5 Root Cause Conclusion

- Missed frauds are **borderline cases** with low predicted probability (near threshold)
- They occur at **clean merchants** (low `merchant_risk`) and involve **smaller amounts** close to the legitimate distribution
- These are **structurally hard to detect** — the fraud patterns overlap with legitimate transaction patterns
- Cannot be fixed by threshold adjustment alone without significantly increasing false positives
- **Possible improvements:** velocity features (transactions per hour per customer), merchant-time interaction features, or temporal graph features

---

## 11. How to Run

### Setup

```bash
pip install -r requirements.txt
```

### Step 1 — Preprocessing & Feature Engineering

```bash
# Run the FE notebook to generate all .pkl splits in src/data/processed/
jupyter notebook notebooks/Feature_engineering/Data_Preprocessing_FE.ipynb
```

Or via the CLI pipeline:

```bash
python scripts/run_pipeline.py
```

### Step 2 — Train Individual Models

```bash
# LightGBM + XGBoost (Optuna, 45/100 trials each)
jupyter notebook notebooks/Model_Training/LightGBM_XGBoost.ipynb

# BalancedRandomForest (Optuna, 45 trials)
jupyter notebook notebooks/Model_Training/BalancedRandomForestClassifier.ipynb

# CatBoost (Optuna, 45 trials)
jupyter notebook notebooks/Model_Training/Catboost.ipynb
```

### Step 3 — Ensemble Strategies

```bash
jupyter notebook notebooks/Model_ensemble/Ensemble_Strategies.ipynb
```

### Step 4 — Graph Feature Engineering

```bash
# Build bipartite graph and extract 6 graph features
jupyter notebook notebooks/Graph_Base/Graph_Features.ipynb

# Retrain with 34 features + error analysis
jupyter notebook notebooks/Graph_Base/Retrain_Graph_ErrorAnalysis.ipynb
```

### Step 5 — GraphSAGE

```bash
jupyter notebook notebooks/Graph_Base/GraphSAGE.ipynb
```

### API Server

```bash
uvicorn api.main:app --reload
# POST /predict with JSON body: {"features": [...]}
```

### Tests

```bash
pytest tests/
```

---

## 12. Requirements

Key dependencies (see `requirements.txt` for pinned versions):

| Package | Purpose |
|---|---|
| `lightgbm` | LightGBM gradient boosting |
| `xgboost` | XGBoost gradient boosting |
| `catboost` | CatBoost gradient boosting |
| `imbalanced-learn` | BalancedRandomForestClassifier + SMOTE |
| `optuna` | Hyperparameter optimization (TPE) |
| `scikit-learn` | Metrics, encoders, LogisticRegression meta-learner |
| `networkx` | Bipartite graph construction + PageRank |
| `torch` | PyTorch — GraphSAGE training |
| `torch-geometric` | SAGEConv layers |
| `shap` | Feature importance (SHAP values) |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `joblib` | Model serialization |
| `fastapi` / `uvicorn` | Scoring API |
| `mlflow` | Experiment tracking |

---

## Key Results Summary

| Metric | Best Value | Achieved By |
|---|---|---|
| Highest PR-AUC | **0.9317** | LightGBM (Tier 0, Optuna) |
| Highest Recall | **0.9948** | Hybrid Funnel Aggressive |
| Fewest Missed Fraud | **5 FN** | Hybrid Funnel Aggressive |
| Best F1 | **0.8363** | CatBoost (Tier 0, Optuna) |
| GNN PR-AUC | 0.7985 | GraphSAGE (Tier 2) |
| Graph FE impact | **−0.0078 to −0.0429** | All 4 models degraded |
