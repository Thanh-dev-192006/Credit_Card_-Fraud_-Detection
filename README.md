# Credit Card Fraud Detection — BankSim Dataset

Fraud detection system trained on the BankSim synthetic dataset (~594K transactions, 1.21% fraud rate).

## Project Structure

```
credit-card-fraud-detection/
├── data/
│   ├── raw/                        # Original BankSim CSV (never modified)
│   ├── processed/                  # Cleaned parquet splits (train/val/test)
│   └── features/                   # Engineered features + saved encoders
│
├── notebooks/
│   ├── exploration/                # EDA: distributions, correlations, fraud patterns
│   │   ├── 01_eda_overview.ipynb
│   │   ├── 02_fraud_patterns.ipynb
│   │   └── 03_feature_analysis.ipynb
│   └── reporting/                  # Final results, business cost analysis
│       ├── 04_model_comparison.ipynb
│       └── 05_threshold_tuning.ipynb
│
├── src/
│   ├── data/
│   │   └── preprocessing.py        # Ingest, validate schema, stratified split
│   ├── features/
│   │   └── feature_engineering.py  # Time/amount/frequency/category features  (*)
│   ├── models/
│   │   ├── trainer.py              # Train all models + MLflow logging           (*)
│   │   └── predictor.py            # Inference wrapper (used by API)
│   └── evaluation/
│       └── evaluator.py            # Metrics, threshold tuning, cost analysis
│
├── configs/
│   ├── model_config.yaml           # Hyperparameters + imbalance strategy        (*)
│   └── pipeline_config.yaml        # Stage toggles and logging config
│
├── tests/
│   ├── unit/
│   │   ├── test_feature_engineering.py
│   │   └── test_preprocessing.py
│   └── integration/                # End-to-end pipeline smoke tests
│
├── artifacts/
│   ├── models/                     # Serialized .pkl model files
│   ├── plots/                      # PR curves, ROC curves, confusion matrices
│   └── reports/                    # test_metrics.csv, pipeline.log
│
├── api/
│   └── main.py                     # FastAPI scoring endpoint (POST /predict)
│
├── mlruns/                         # MLflow auto-generated tracking data
├── scripts/
│   └── run_pipeline.py             # CLI: python scripts/run_pipeline.py
├── requirements.txt
├── Makefile
└── README.md
```

Files marked (*) are the three most critical — start there.


