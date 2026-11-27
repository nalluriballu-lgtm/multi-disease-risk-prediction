# Dataset setup

The repository does not redistribute patient-level data. Download each dataset from an official or otherwise trusted source and check its terms of use.

| Dataset | Suggested CSV filename | Starter target in config |
|---|---|---|
| Pima Indians Diabetes | diabetes.csv | Outcome |
| UCI Heart Disease | heart.csv | target |
| UCI Chronic Kidney Disease | kidney.csv | classification |
| Wisconsin Diagnostic Breast Cancer | breast_cancer.csv | diagnosis |

**Important:** these are suggested defaults only. Different copies use different column names, encodings and labels. Inspect each CSV and edit `src/config.py` to match it exactly. The CKD dataset may require cleaning inconsistent labels; breast-cancer dataset may contain ID or unnamed columns that should be removed.

Never put identifiable patient records in this public repository. Keep raw data local; the `.gitignore` excludes files under `datasets/raw/`.
