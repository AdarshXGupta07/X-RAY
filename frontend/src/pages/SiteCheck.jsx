import { useEffect, useState } from "react";
import { api } from "../api.js";
import VerdictCard from "../components/VerdictCard.jsx";

function ModelCard({ health }) {
  if (!health) return <section className="card muted">Connecting to the server…</section>;
  const ref = health.reference;
  if (!ref) {
    return (
      <section className="card warn">
        No model reference is loaded. {health.notes.join(" ")}
      </section>
    );
  }
  return (
    <section className="card model">
      <div>
        <div className="eyebrow">Model under test</div>
        <div className="model-name">{ref.model}</div>
        <div className="muted small">
          Trained and certified at <strong>{ref.source_site}</strong> on {ref.n_images} images ({ref.n_tb} TB).
        </div>
      </div>
      <div className="model-facts">
        <div><span className="fact">{Math.round(ref.target_sensitivity * 100)}%</span> sensitivity promised</div>
        <div><span className="fact">{Math.round(ref.confidence * 100)}%</span> confidence</div>
        <div><span className="fact">{ref.threshold_probability.toFixed(3)}</span> rule-out threshold</div>
        <div><span className="fact">{ref.tb_labels_needed}</span> TB labels to re-certify</div>
      </div>
    </section>
  );
}

function ScenarioPicker({ scenarios, onRun, busy }) {
  return (
    <div className="scenarios">
      {scenarios.map((s) => (
        <button key={s.name} className="scenario" disabled={busy} onClick={() => onRun(s.name)}>
          <span className="scenario-title">{s.title}</span>
          <span className="scenario-desc">{s.description}</span>
          <span className="scenario-meta">
            {s.n_images} images{s.n_labelled ? `, ${s.n_labelled} labelled` : ", no labels"}
          </span>
        </button>
      ))}
    </div>
  );
}

function Uploader({ onRun, busy, modelLoaded }) {
  const [files, setFiles] = useState([]);
  const [labels, setLabels] = useState(null);
  return (
    <form className="uploader" onSubmit={(e) => { e.preventDefault(); onRun(files, labels); }}>
      <p className="muted small">
        Images are scored in memory on this server and never stored. Use only images you are
        allowed to process, never identifiable patient data.
      </p>
      <label className="field">
        <span>Chest X-rays from one site (PNG or JPEG, 20 or more)</span>
        <input type="file" accept="image/png,image/jpeg" multiple
               onChange={(e) => setFiles([...e.target.files])} />
      </label>
      <label className="field">
        <span>Optional labels CSV (columns <code>filename,label</code>; 1 = TB, 0 = normal)</span>
        <input type="file" accept=".csv,text/csv" onChange={(e) => setLabels(e.target.files[0] || null)} />
      </label>
      <button className="primary" type="submit" disabled={busy || !files.length || !modelLoaded}>
        {busy ? "Scoring…" : `Check ${files.length || ""} image${files.length === 1 ? "" : "s"}`}
      </button>
      {!modelLoaded && <p className="error">The server has no model loaded, so uploads are disabled.</p>}
    </form>
  );
}

export default function SiteCheck() {
  const [health, setHealth] = useState(null);
  const [scenarios, setScenarios] = useState([]);
  const [mode, setMode] = useState("simulated");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [seed, setSeed] = useState(0);

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(e.message));
    api.scenarios().then(setScenarios).catch(() => {});
  }, []);

  async function run(promise) {
    setBusy(true);
    setError(null);
    try {
      setResult(await promise);
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <section className="intro">
        <h1>Site check</h1>
        <p>
          A TB rule-out threshold was certified at one hospital to catch at least 90% of TB cases.
          Before a new hospital relies on it, this check compares the new site's scores with the
          source, says whether the promise can be trusted, and re-certifies it when enough TB cases
          are labelled.
        </p>
      </section>

      <ModelCard health={health} />

      <section className="card">
        <div className="segmented" role="tablist">
          <button role="tab" aria-selected={mode === "simulated"}
                  className={mode === "simulated" ? "seg active" : "seg"}
                  onClick={() => setMode("simulated")}>Simulated hospital</button>
          <button role="tab" aria-selected={mode === "upload"}
                  className={mode === "upload" ? "seg active" : "seg"}
                  onClick={() => setMode("upload")}>Upload images</button>
        </div>
        {mode === "simulated" ? (
          <>
            <p className="muted small">
              Simulated batches built from the source model's scores, so the true answer is known.
              No patient images are used or shown.{" "}
              <label className="inline">
                Random draw{" "}
                <input type="number" min="0" max="999" value={seed}
                       onChange={(e) => setSeed(Number(e.target.value) || 0)} />
              </label>
            </p>
            <ScenarioPicker scenarios={scenarios} busy={busy}
                            onRun={(name) => run(api.runScenario(name, seed))} />
          </>
        ) : (
          <Uploader busy={busy} modelLoaded={!!health?.model_loaded}
                    onRun={(files, labels) => run(api.checkImages(files, labels))} />
        )}
        {error && <p className="error" role="alert">{error}</p>}
      </section>

      {busy && <section className="card muted">Running the check…</section>}
      {result && !busy && <VerdictCard result={result} sourceSite={health?.reference?.source_site} />}
    </>
  );
}
