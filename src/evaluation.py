"""Evaluation helpers for binary classification."""
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

def evaluate_binary(y_true, y_pred, y_score=None):
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }
    if y_score is not None:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            result["roc_auc"] = None
    return result
