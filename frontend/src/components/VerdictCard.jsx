import Histogram from "./Histogram.jsx";

const LABELS = {
  plausible: "Promise plausible",
  recalibrate: "Re-certify before use",
  recertified: "Re-certified at this site",
  defer: "Not enough data",
};

const pct = (v) => (v == null ? "n/a" : `${Math.round(v * 100)}%`);

function Stat({ label, value, hint }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

function GroundTruth({ gt, recertified }) {
  const broken = gt.source_threshold_sensitivity < gt.target_sensitivity;
  return (
    <div className={broken ? "truth truth-bad" : "truth truth-ok"}>
      <div className="truth-title">What the check could not see (known only because this site is simulated)</div>
      <p>
        At this site the source threshold actually catches{" "}
        <strong>{pct(gt.source_threshold_sensitivity)}</strong> of TB cases (promise:{" "}
        {pct(gt.target_sensitivity)}) and clears {pct(gt.source_threshold_specificity)} of healthy people.
        {broken ? " Below the target in this batch." : " At or above the target in this batch."}
      </p>
      <p className="small muted">
        A batch with a few dozen TB cases can land a little either side of 90% by chance; a gap of
        10 points or more is unlikely to be chance.
      </p>
      {recertified && (
        <p>
          With the re-certified threshold: <strong>{pct(gt.in_force_sensitivity)}</strong> of TB caught,{" "}
          {pct(gt.in_force_specificity)} of healthy people cleared.
        </p>
      )}
    </div>
  );
}

export default function VerdictCard({ result, sourceSite }) {
  const r = result;
  return (
    <section className={`card verdict verdict-${r.verdict}`} aria-live="polite">
      <div className="verdict-tag">{LABELS[r.verdict]}</div>
      <h2 className="verdict-headline">{r.headline}</h2>

      <div className="stats">
        <Stat label="images checked" value={r.n_images} />
        <Stat label="shift test p-value" value={r.ks_p_value == null ? "n/a" : r.ks_p_value.toPrecision(2)}
              hint={r.shift_detected == null ? "batch too small" : r.shift_detected ? "shift detected" : "no shift detected"} />
        <Stat label="ruled out here (source threshold)" value={pct(r.ruled_out_fraction_site)}
              hint={`${pct(r.ruled_out_fraction_source)} at the source`} />
        <Stat label="labelled TB cases" value={r.n_labelled_tb}
              hint={r.tb_labels_needed > 0 ? `${r.tb_labels_needed} more needed to re-certify` : "enough to re-certify"} />
      </div>

      <ul className="reasons">
        {r.reasons.map((t) => <li key={t}>{t}</li>)}
      </ul>

      {r.ground_truth && <GroundTruth gt={r.ground_truth} recertified={r.verdict === "recertified"} />}

      <Histogram data={r.histogram} sourceLabel={sourceSite} />

      <details className="images">
        <summary>Per-image scores ({r.images.length})</summary>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Image</th><th>TB probability</th><th>Decision</th><th>Label given</th></tr>
            </thead>
            <tbody>
              {r.images.map((im) => (
                <tr key={im.name}>
                  <td>{im.name}</td>
                  <td>{im.probability.toFixed(3)}</td>
                  <td>{im.ruled_out ? "ruled out" : "refer for testing"}</td>
                  <td>{im.label == null ? "" : im.label === 1 ? "TB" : "normal"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}
