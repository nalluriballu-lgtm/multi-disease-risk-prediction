# Multi-Disease Risk Prediction System

An educational machine-learning project that explores classification models for four public tabular datasets: diabetes, heart disease, chronic kidney disease (CKD), and breast cancer (Wisconsin Diagnostic Breast Cancer).

## Important safety notice
This is an academic demonstration, **not a medical device or diagnostic system**. Do not use its outputs to diagnose, screen, or make treatment decisions. Model scores are not clinical risk estimates. Use only permitted, de-identified datasets and follow each dataset's license and terms.

## Algorithms
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- Optional experiments: Artificial Neural Network, TabNet, and stacking
- Explainability experiments: SHAP and LIME

## Project layout
```
datasets/raw/                 # Place permitted CSV datasets here (not included)
src/
  config.py                   # Dataset target-column configuration
  data_processing.py          # Cleaning, splitting and preprocessing
  train.py                    # Train one disease/model pipeline
  predict.py                  # Load a saved model and predict a row
  evaluation.py               # Classification metrics and plots
  explainability.py           # Optional SHAP explanation
backend/                      # FastAPI API
frontend/                     # React/Vite interface starter
tests/                         # Basic tests
saved_models/                  # Generated model artifacts
reports/                       # Generated evaluation reports
```

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy your CSV files into `datasets/raw/`. Edit `src/config.py` to set the correct filename and target column for each dataset. Dataset schemas vary; confirm column names and target labels before training.

## Train
```powershell
python -m src.train --disease diabetes --model random_forest
python -m src.train --disease heart --model svm
```

## Start the API
```powershell
uvicorn backend.app:app --reload
```
Open http://127.0.0.1:8000/docs.

## Frontend
```powershell
cd frontend
npm install
npm run dev
```

## Data
This repository intentionally does not include patient-level datasets. Obtain the relevant public datasets from their official sources and check their permitted use:
- UCI Machine Learning Repository (Heart Disease, Chronic Kidney Disease)
- Pima Indians Diabetes dataset (verify source and usage terms)
- Wisconsin Diagnostic Breast Cancer dataset (UCI)

The breast-cancer dataset is a diagnostic classification dataset, not a symptom-based risk calculator.

## Notes
- Preprocessing is fitted on training data only to avoid leakage.
- Compare models using precision, recall, F1, ROC-AUC and confusion matrices; accuracy alone is insufficient.
- For small or imbalanced datasets, use stratified splits and cross-validation.
- SHAP/LIME explain model behavior, not biological causation.
- Authentication, deployment hardening, privacy review, and clinical validation are out of scope for this student starter project.

## License
No license has been added. Add a license only after deciding how you want others to reuse the project.
