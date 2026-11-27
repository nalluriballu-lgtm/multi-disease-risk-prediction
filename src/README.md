# Training notes

1. Download a permitted public dataset and place its CSV in `datasets/raw/`.
2. Inspect its columns and edit `src/config.py` so filename and target match.
3. Ensure the target has exactly two classes for this starter script.
4. Run, for example: `python -m src.train --disease diabetes --model random_forest`.

This is a teaching scaffold. Dataset-specific validation, robust nested cross-validation,
calibration, fairness analysis, model cards and external validation should be added
before drawing research conclusions. It is not for clinical use.
