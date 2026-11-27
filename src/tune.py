"""Small GridSearchCV helper. Tune only within training folds."""
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def tune_pipeline(pipeline, parameter_grid, X_train, y_train, scoring="f1"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, parameter_grid, scoring=scoring,
                          cv=cv, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    return search
