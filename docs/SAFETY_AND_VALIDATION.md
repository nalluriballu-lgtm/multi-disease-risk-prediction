# Safety and validation checklist

- Educational research prototype only; not for diagnosis, screening or treatment.
- No clinical risk claim should be made from a classifier's raw probability.
- Keep patient-identifiable information out of the repository and logs.
- Confirm data provenance, license, consent basis and permitted use.
- Split data before fitting preprocessing to avoid data leakage.
- Use stratified cross-validation and report confidence intervals where feasible.
- Evaluate recall, precision, F1, ROC-AUC, PR-AUC and calibration, not accuracy alone.
- Check subgroup performance and possible dataset bias.
- Validate on an independent external dataset before making generalisation claims.
- SHAP/LIME explanations describe model behavior, not medical causation.
- Obtain appropriate expert and ethics review before any real-world health research.
