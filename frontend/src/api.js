// Small wrapper around fetch: JSON in, JSON out, FastAPI's error "detail" as the message.
async function request(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body?.detail;
    const msg = Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail;
    throw new Error(msg || `Request failed (${res.status})`);
  }
  return body;
}

export const api = {
  health: () => request("/api/health"),
  scenarios: () => request("/api/scenarios"),
  runScenario: (name, seed = 0) =>
    request(`/api/site-check/scenario/${encodeURIComponent(name)}?seed=${seed}`, { method: "POST" }),
  checkImages: (files, labelsFile) => {
    const form = new FormData();
    for (const f of files) form.append("images", f);
    if (labelsFile) form.append("labels_csv", labelsFile);
    return request("/api/site-check/images", { method: "POST", body: form });
  },
  headline: () => request("/api/results/headline"),
};
