import { useEffect, useState } from "react";
import SiteCheck from "./pages/SiteCheck.jsx";
import Results from "./pages/Results.jsx";
import Limitations from "./pages/Limitations.jsx";

const PAGES = {
  check: { label: "Site check", component: SiteCheck },
  results: { label: "Study results", component: Results },
  limits: { label: "Limitations", component: Limitations },
};

function pageFromHash() {
  const key = window.location.hash.replace("#", "");
  return key in PAGES ? key : "check";
}

export default function App() {
  const [page, setPage] = useState(pageFromHash);

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const Page = PAGES[page].component;
  return (
    <>
      <div className="banner" role="note">
        Research prototype. Not for clinical use. Do not upload real patient images.
      </div>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <div className="brand-name">TB-Recertify</div>
            <div className="brand-sub">Does a 90% TB rule-out promise survive a new hospital?</div>
          </div>
        </div>
        <nav className="tabs">
          {Object.entries(PAGES).map(([key, p]) => (
            <a key={key} href={`#${key}`} className={key === page ? "tab active" : "tab"}>
              {p.label}
            </a>
          ))}
        </nav>
      </header>
      <main className="container">
        <Page />
      </main>
    </>
  );
}
