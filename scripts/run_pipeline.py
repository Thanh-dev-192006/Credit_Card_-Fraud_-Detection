"""
CLI entry point — runs the full training pipeline end-to-end.

Usage:
    python scripts/run_pipeline.py [--config configs/model_config.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

# Make src importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import run_preprocessing
from src.features.feature_engineering import build_features
from src.models.trainer import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fraud Detection Training Pipeline")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["preprocess", "features", "train"],
        default=["preprocess", "features", "train"],
        help="Pipeline stages to run (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if "preprocess" in args.stages:
        logger.info("=== Stage: preprocess ===")
        run_preprocessing(args.config)

    if "features" in args.stages:
        logger.info("=== Stage: features ===")
        import pickle
        import pandas as pd
        import yaml
        from pathlib import Path

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        processed_dir = Path(cfg["data"]["processed_dir"])
        features_dir = Path(cfg["data"]["features_dir"])
        features_dir.mkdir(parents=True, exist_ok=True)

        for split in ("train", "val", "test"):
            raw = pd.read_parquet(processed_dir / f"{split}.parquet")
            is_train = split == "train"

            if is_train:
                feats, fitted_encoders = build_features(raw, fit_encoders=True)
                with open(features_dir / "encoders.pkl", "wb") as f:
                    pickle.dump(fitted_encoders, f)
            else:
                with open(features_dir / "encoders.pkl", "rb") as f:
                    enc = pickle.load(f)
                feats, _ = build_features(raw, fit_encoders=False, encoders=enc)

            feats.to_parquet(features_dir / f"{split}_features.parquet", index=False)

    if "train" in args.stages:
        logger.info("=== Stage: train ===")
        run_ids = run_training(args.config)
        logger.info("MLflow run IDs: %s", run_ids)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
