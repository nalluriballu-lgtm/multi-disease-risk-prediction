"""Reusable baseline classifiers."""
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_classifiers():
    return {
        "svm": SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced",
                   random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                                eval_metric="logloss", random_state=42),
    }

def get_stacking_model():
    models = get_classifiers()
    return StackingClassifier(
        estimators=[("rf", models["random_forest"]), ("svm", models["svm"])],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba", cv=5
    )
