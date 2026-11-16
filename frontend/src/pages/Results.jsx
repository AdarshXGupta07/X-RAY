import { useEffect, useState } from "react";
import { api } from "../api.js";

const pct = (v) => `${(v * 100).toFixed(1)}%`;

export default function Results() {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.headline().then(setRows).catch((e) => setError(e.message));
  }, []);

  return (
    <>
      <section className="intro">
        <h1>Study results</h1>
        <p>
          Five training runs of the same model (DenseNet-121, trained on Shenzhen), identical data
          splits, different training randomness. Each threshold is certified on Shenzhen to give at
          least 90% sensitivity, then tested at Montgomery, a hospital the model never saw. A
          violation is a calibration draw (of 1,000) where measured sensitivity fell below 90%.
        </p>
      </section>

      <section className="card">
        {error && <p className="error">{error}</p>}
        {!rows && !error && <p className="muted">Loading…</p>}
        {rows && (
          <div className="table-wrap">
            <table className="results">
              <thead>
                <tr>
                  <th>Seed</th>
                  <th>Violations at source (control)</th>
                  <th>AUC at new site</th>
                  <th>Violations at new site</th>
                  <th>Sensitivity at new site</th>
                  <th>Specificity at new site</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.seed} className={r.target_violation > 0.5 ? "row-bad" : ""}>
                    <td>{r.seed}</td>
                    <td>{pct(r.control_violation)}</td>
                    <td>{r.target_auc.toFixed(3)}</td>
                    <td><strong>{pct(r.target_violation)}</strong></td>
                    <td>{r.target_sensitivity.toFixed(2)}</td>
                    <td>{r.target_specificity.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="card findings">
        <h2>What this shows</h2>
        <ol>
          <li>
            <strong>The promise can fail silently.</strong> One of five equally good runs missed the
            90% target at the new hospital in every draw, while its source control and its AUC looked
            like the others'.
          </li>
          <li>
            <strong>The other runs fail the other way.</strong> They keep sensitivity but clear only
            5–34% of healthy people at the new site.
          </li>
          <li>
            <strong>The unlabelled shift test cannot tell these apart.</strong> With 50 images it
            fires for every run.
          </li>
          <li>
            <strong>About 29 labelled TB cases fix it.</strong> Re-certifying from new-site TB cases
            kept violations at 11–13% for all ten models (five seeds, with and without corner masks),
            close to the 10% allowed. Moving the threshold without labels violated in 27–80% of draws.
          </li>
          <li>
            <strong>Masking the image corners did not help.</strong> It lowered new-site AUC in 9 of
            10 paired comparisons.
          </li>
        </ol>
      </section>
    </>
  );
}
