// Source vs site score distribution as two overlaid step outlines, with threshold lines.
// Plain SVG: no chart library needed for 20 bins.

const W = 640;
const H = 220;
const PAD = { left: 40, right: 12, top: 12, bottom: 34 };

function stepPath(shares, x, y) {
  let d = `M ${x(0)} ${y(0)}`;
  shares.forEach((s, i) => {
    d += ` L ${x(i)} ${y(s)} L ${x(i + 1)} ${y(s)}`;
  });
  return d + ` L ${x(shares.length)} ${y(0)}`;
}

export default function Histogram({ data, sourceLabel }) {
  const bins = data.source.length;
  const top = Math.max(...data.source, ...data.site, 0.05) * 1.1;
  const x = (i) => PAD.left + (i / bins) * (W - PAD.left - PAD.right);
  const xp = (p) => PAD.left + p * (W - PAD.left - PAD.right);
  const y = (s) => H - PAD.bottom - (s / top) * (H - PAD.top - PAD.bottom);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const yTicks = [0, top / 2, top].map((t) => Math.round(t * 100) / 100);

  return (
    <figure className="histogram">
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Distribution of TB probabilities at the source and at this site">
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} className="grid" />
            <text x={PAD.left - 6} y={y(t) + 4} className="axis" textAnchor="end">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        {ticks.map((p) => (
          <text key={p} x={xp(p)} y={H - PAD.bottom + 16} className="axis" textAnchor="middle">
            {p.toFixed(2)}
          </text>
        ))}
        <text x={(PAD.left + W - PAD.right) / 2} y={H - 4} className="axis" textAnchor="middle">
          TB probability (temperature-scaled)
        </text>
        <path d={stepPath(data.source, x, y)} className="series-source" />
        <path d={stepPath(data.site, x, y)} className="series-site" />
        {data.source_threshold != null && (
          <line x1={xp(data.source_threshold)} x2={xp(data.source_threshold)}
                y1={PAD.top} y2={H - PAD.bottom} className="thr-source" />
        )}
        {data.new_threshold != null && (
          <line x1={xp(data.new_threshold)} x2={xp(data.new_threshold)}
                y1={PAD.top} y2={H - PAD.bottom} className="thr-new" />
        )}
      </svg>
      <figcaption className="legend">
        <span><i className="key key-source" /> Source ({sourceLabel})</span>
        <span><i className="key key-site" /> This site</span>
        <span><i className="key key-thr-source" /> Source threshold</span>
        {data.new_threshold != null && <span><i className="key key-thr-new" /> Re-certified threshold</span>}
      </figcaption>
    </figure>
  );
}
