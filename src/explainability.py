"""Optional SHAP explanation helper; explanations describe model behavior only."""
def explain_with_shap(pipeline, X_background, X_to_explain):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install shap with: pip install shap") from exc
    # KernelExplainer is model-agnostic and can be slow; use a small background sample.
    background = X_background.sample(min(50, len(X_background)), random_state=42)
    sample = X_to_explain.sample(min(20, len(X_to_explain)), random_state=42)
    predict_fn = lambda values: pipeline.predict_proba(values)[:, 1]
    explainer = shap.KernelExplainer(predict_fn, background)
    values = explainer.shap_values(sample, nsamples=100)
    return {"sample": sample, "shap_values": values, "explainer": explainer}
