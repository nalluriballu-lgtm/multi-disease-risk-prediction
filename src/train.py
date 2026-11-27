"""Train one disease/model pipeline from a local CSV."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier
from src.config import DATASETS, RAW_DIR, MODEL_DIR
from src.data_processing import make_preprocessor

def build_model(name):
    if name == "svm":
        return SVC(probability=True, class_weight="balanced", random_state=42)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    if name == "xgboost":
        return XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                             eval_metric="logloss", random_state=42)
    raise ValueError("Model must be svm, random_forest, or xgboost")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", required=True, choices=DATASETS)
    parser.add_argument("--model", default="random_forest",
                        choices=["svm", "random_forest", "xgboost"])
    args = parser.parse_args()
    cfg = DATASETS[args.disease]
    path = RAW_DIR / cfg["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Place the dataset at {path}")
    df = pd.read_csv(path)
    if cfg["target"] not in df:
        raise ValueError(f"Target '{cfg['target']}' not found. Available: {list(df.columns)}")
    df = df.dropna(subset=[cfg["target"]])
    X, y = df.drop(columns=[cfg["target"]]), df[cfg["target"]]
    if y.nunique() != 2:
        raise ValueError("This starter training script currently expects a binary target.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = Pipeline([("preprocess", make_preprocessor(X_train)),
                     ("model", build_model(args.model))])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(classification_report(y_test, pred, zero_division=0))
    try:
        print("ROC-AUC:", round(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]), 4))
    except (ValueError, AttributeError):
        print("ROC-AUC unavailable for this split/model.")
    out = MODEL_DIR / args.disease
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": list(X.columns)}, out / f"{args.model}.joblib")
    print("Saved model to", out / f"{args.model}.joblib")

if __name__ == "__main__":
    main()
