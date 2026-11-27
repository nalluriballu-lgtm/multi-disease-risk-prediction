import { useState } from "react";

const diseases = [
  ["diabetes", "Diabetes"],
  ["heart", "Heart Disease"],
  ["kidney", "Chronic Kidney Disease"],
  ["breast_cancer", "Breast Cancer (WDBC)"],
];

export default function App() {
  const [disease, setDisease] = useState("diabetes");
  const [model, setModel] = useState("random_forest");
  const [json, setJson] = useState("{}");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function submit(event) {
    event.preventDefault();
    setError(""); setResult(null);
    let features;
    try { features = JSON.parse(json); }
    catch { setError("Enter valid JSON for the selected dataset's feature names."); return; }
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ disease, model, features })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(JSON.stringify(data.detail || data));
      setResult(data);
    } catch (err) { setError(err.message || "Could not contact the API."); }
  }

  return <main className="container">
    <header><p className="eyebrow">ACADEMIC MACHINE LEARNING PROJECT</p>
      <h1>Multi-Disease Prediction Demo</h1>
      <p>Compare machine-learning models using public tabular datasets.</p>
    </header>
    <section className="notice"><strong>Research demo only.</strong> This application is not a medical device. Its output must not be used for diagnosis or treatment.</section>
    <form onSubmit={submit} className="panel">
      <label>Disease dataset<select value={disease} onChange={e => setDisease(e.target.value)}>{diseases.map(([v,l]) => <option value={v} key={v}>{l}</option>)}</select></label>
      <label>Model<select value={model} onChange={e => setModel(e.target.value)}><option value="random_forest">Random Forest</option><option value="svm">SVM</option><option value="xgboost">XGBoost</option></select></label>
      <label>Feature values (JSON)<textarea rows="8" value={json} onChange={e => setJson(e.target.value)} placeholder='{"feature_name": 1}' /></label>
      <button type="submit">Run model</button>
    </form>
    {error && <section className="panel error">{error}</section>}
    {result && <section className="panel"><h2>Model output</h2><pre>{JSON.stringify(result, null, 2)}</pre></section>}
    <footer>Use only de-identified public datasets with appropriate permissions.</footer>
  </main>;
}
