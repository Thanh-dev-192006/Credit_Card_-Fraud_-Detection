"""
Model training orchestrator with MLflow experiment tracking.

Trains all models enabled in configs/model_config.yaml, logs:
  - Parameters, metrics, feature importances
  - Serialized model artifacts
  - PR / ROC curve plots
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_training(config_path: str = "configs/model_config.yaml") -> dict[str, Any]:
    """
    Train all enabled models, log to MLflow, return {model_name: run_id}.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])

    X_train, y_train = _load_split(cfg, "train")
    X_val, y_val = _load_split(cfg, "val")

    X_train, y_train = _apply_imbalance_strategy(X_train, y_train, cfg)

    results = {}
    for model_name, model_cfg in cfg["models"].items():
        if not model_cfg.get("enabled", True):
            logger.info("Skipping disabled model: %s", model_name)
            continue

        logger.info("Training %s ...", model_name)
        with mlflow.start_run(run_name=model_name) as run:
            model = _build_model(model_name, model_cfg)
            model = _fit(model, model_name, model_cfg, X_train, y_train, X_val, y_val)

            val_metrics = _evaluate(model, X_val, y_val, model_name)
            mlflow.log_params({f"{model_name}/{k}": v for k, v in model_cfg.items() if k != "enabled"})
            mlflow.log_metrics(val_metrics)
            mlflow.sklearn.log_model(model, artifact_path=model_name)

            _save_artifact(model, cfg, model_name)
            results[model_name] = run.info.run_id
            logger.info("%s val PR-AUC=%.4f  ROC-AUC=%.4f", model_name,
                        val_metrics["pr_auc"], val_metrics["roc_auc"])

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_split(cfg: dict, split: str) -> tuple[pd.DataFrame, pd.Series]:
    features_dir = Path(cfg["data"]["features_dir"])
    df = pd.read_parquet(features_dir / f"{split}_features.parquet")
    target = cfg["data"]["target_col"]
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def _apply_imbalance_strategy(
    X: pd.DataFrame, y: pd.Series, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    strategy = cfg["imbalance"]["strategy"]

    if strategy in ("smote", "both"):
        smote = SMOTE(
            k_neighbors=cfg["imbalance"]["smote_k_neighbors"],
            sampling_strategy=cfg["imbalance"]["sampling_strategy"],
            random_state=cfg["data"]["random_state"],
        )
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(
            "SMOTE resampling: %d → %d samples (fraud: %d → %d)",
            len(y), len(y_res), y.sum(), y_res.sum(),
        )
        return X_res, y_res

    return X.values, y.values


def _build_model(name: str, model_cfg: dict) -> Any:
    """Instantiate the sklearn-compatible model from config."""
    params = {k: v for k, v in model_cfg.items() if k not in ("enabled",)}

    factories = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier,
    }

    if name not in factories:
        raise ValueError(f"Unknown model: {name}. Add it to trainer._build_model().")

    # XGBoost/LightGBM have early_stopping_rounds handled at fit-time
    init_params = {
        k: v for k, v in params.items()
        if k not in ("early_stopping_rounds", "eval_metric")
    }
    return factories[name](**init_params)


def _fit(model: Any, name: str, cfg: dict, X_train, y_train, X_val, y_val) -> Any:
    """Fit with early stopping for boosting models, plain fit otherwise."""
    early_stop = cfg.get("early_stopping_rounds")

    if name == "xgboost" and early_stop:
        model.set_params(early_stopping_rounds=early_stop, eval_metric=cfg.get("eval_metric", "aucpr"))
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "lightgbm" and early_stop:
        from lightgbm import early_stopping, log_evaluation
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(early_stop), log_evaluation(period=-1)],
        )
    else:
        model.fit(X_train, y_train)

    return model


def _evaluate(model: Any, X_val, y_val, model_name: str) -> dict[str, float]:
    y_prob = model.predict_proba(X_val)[:, 1]
    return {
        "pr_auc": average_precision_score(y_val, y_prob),
        "roc_auc": roc_auc_score(y_val, y_prob),
    }


def _save_artifact(model: Any, cfg: dict, model_name: str) -> None:
    out_dir = Path(cfg["experiment"]["artifact_location"]) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model artifact: %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_ids = run_training()
    print("\nCompleted runs:")
    for model, run_id in run_ids.items():
        print(f"  {model}: {run_id}")
