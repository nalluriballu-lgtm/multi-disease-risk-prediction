"""Dataset configuration. Verify each CSV schema before training."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets" / "raw"
MODEL_DIR = ROOT / "saved_models"

# Edit filename and target to match the exact dataset CSV you download.
DATASETS = {
    "diabetes": {"filename": "diabetes.csv", "target": "Outcome", "positive_label": 1},
    "heart": {"filename": "heart.csv", "target": "target", "positive_label": 1},
    "kidney": {"filename": "kidney.csv", "target": "classification", "positive_label": "ckd"},
    "breast_cancer": {"filename": "breast_cancer.csv", "target": "diagnosis", "positive_label": "M"},
}
