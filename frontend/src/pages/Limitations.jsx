export default function Limitations() {
  return (
    <>
      <section className="intro">
        <h1>Limitations</h1>
        <p>Read these before drawing any conclusion from this prototype.</p>
      </section>
      <section className="card findings">
        <ul>
          <li><strong>Not a medical device.</strong> Research prototype for a student project. It must not be used to make or support any clinical decision.</li>
          <li><strong>Two hospitals.</strong> Shenzhen (China) and Montgomery County (USA), public NLM datasets. Results may not carry over to other sites, scanners or populations.</li>
          <li><strong>Small evaluation sets.</strong> Montgomery has 58 TB cases, so each per-run violation rate has its own uncertainty.</li>
          <li><strong>Five training runs, one architecture.</strong> How often the promise fails could differ for other models or more runs.</li>
          <li><strong>The shift test only sees scores.</strong> It can miss shifts that leave the score distribution unchanged, and in this study it missed a strong contrast change. "No shift detected" is not proof that the promise holds.</li>
          <li><strong>Re-certification assumes the labelled TB cases are representative</strong> of the TB cases the site will see. Labels chosen by convenience (e.g. only severe cases) would make the new threshold too optimistic.</li>
          <li><strong>Simulated scenarios are illustrations.</strong> They resample the source model's scores and shift them; real shifts can look different.</li>
          <li><strong>Shortcut signals exist.</strong> Montgomery's burned-in corner markers alone predict TB (AUC 0.90). Masking them did not improve transfer, but other shortcuts may remain.</li>
          <li><strong>No patient images are shown.</strong> The NLM data terms do not allow sharing the images; the demo works on simulated scores or on images you supply yourself.</li>
        </ul>
      </section>
    </>
  );
}
