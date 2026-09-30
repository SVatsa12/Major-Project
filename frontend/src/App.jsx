import React, { useState, useEffect, useRef } from "react";
import {
  ShieldCheck, AlertTriangle, Gavel, Activity,
  Cpu, ChevronRight, Zap, BarChart2, BookOpen, Search
} from "lucide-react";
import "./index.css";

/* ─────────────────────────────────────────
   DATA
───────────────────────────────────────── */
const FALLBACK_MATRIX = [
  { platform: "WhatsApp",  dpdpAct2023: 31.2, dpdpRules2025: 50.0, itAct2000:  7.1, final: 29.4 },
  { platform: "Telegram",  dpdpAct2023: 18.8, dpdpRules2025: 30.0, itAct2000: 14.3, final: 21.0 },
  { platform: "Snapchat",  dpdpAct2023: 15.6, dpdpRules2025: 20.0, itAct2000:  0.0, final: 11.9 },
  { platform: "Google",    dpdpAct2023: 12.5, dpdpRules2025: 13.3, itAct2000:  7.1, final: 11.0 },
  { platform: "YouTube",   dpdpAct2023:  3.1, dpdpRules2025: 13.3, itAct2000:  7.1, final:  7.8 },
  { platform: "Meta",      dpdpAct2023:  3.1, dpdpRules2025:  3.3, itAct2000: 14.3, final:  6.9 },
];

const PREDICT_SAMPLES = {
  compliant: "We have implemented a comprehensive data deletion policy. Upon user request or account deletion, all personal data is securely erased within 30 days, with no backups retained.",
  vague:     "We may retain certain information for operational purposes as needed for service provision.",
  irrelevant:"Our mobile app is available for iOS 14.5 and later. Download from the App Store today.",
};

const REGULATORY_GAPS = [
  { label: "Breach notification",               detail: "addressed by 1 of 6 platforms",        severity: "high"   },
  { label: "India grievance redressal officer",  detail: "missing on 3 of 6 platforms",          severity: "high"   },
  { label: "15 of 38 taxonomy rules",            detail: "never matched in corpus",              severity: "medium" },
  { label: "Verbosity bias",                     detail: "policy length vs. coverage r = 0.84",  severity: "low"    },
];

const PLATFORM_COLORS = {
  WhatsApp: "#25D366", Telegram: "#26A5E4", Snapchat: "#FFFC00",
  Google:   "#4285F4", YouTube:  "#FF0000", Meta:     "#0082FB",
};

const PLATFORM_INITIALS = (name) => name.slice(0, 2).toUpperCase();

const mockPredict = (text) => {
  const lower = text.toLowerCase();
  if (lower.includes("delet") || lower.includes("erase") || lower.includes("remove"))
    return { verdict: "Compliant",         confidence: 0.92, rule: "DPDP_ACT-0047", category: "Data Retention & Erasure" };
  if (lower.includes("may") || lower.includes("certain") || lower.includes("as needed"))
    return { verdict: "Partially Compliant", confidence: 0.68, rule: "DPDP_RUL-0227", category: "Transparency" };
  return { verdict: "Not Addressed", confidence: 0.78, rule: null, category: null };
};

/* ─────────────────────────────────────────
   TINY HELPERS
───────────────────────────────────────── */
const scoreColor = (v) =>
  v >= 25 ? "var(--jade)" : v >= 10 ? "var(--saffron)" : "var(--rose)";
const scoreBg = (v) =>
  v >= 25 ? "rgba(16,201,143,0.1)" : v >= 10 ? "rgba(245,166,35,0.1)" : "rgba(245,101,101,0.1)";

/* Radial progress ring (SVG) */
function Ring({ pct, color, size = 80, stroke = 7, label, sublabel }) {
  const r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (pct / 100) * circ;
  return (
    <div style={{ textAlign: "center" }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={stroke} />
        <circle
          cx={size/2} cy={size/2} r={r} fill="none"
          stroke={color} strokeWidth={stroke}
          strokeDasharray={circ} strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(0.22,1,0.36,1)" }}
        />
      </svg>
      <div style={{ marginTop: 6, fontSize: 13, fontWeight: 700, color: "var(--text-1)", fontVariantNumeric: "tabular-nums" }}>{label}</div>
      {sublabel && <div style={{ fontSize: 10, color: "var(--text-3)", marginTop: 2, letterSpacing: "0.04em" }}>{sublabel}</div>}
    </div>
  );
}

/* Horizontal bar */
function HBar({ val, max = 55, color }) {
  const pct = Math.min((val / max) * 100, 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, height: 5, background: "rgba(255,255,255,0.06)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 3, transition: "width 1s cubic-bezier(0.22,1,0.36,1)" }} />
      </div>
      <span style={{ fontSize: 11, color: "var(--text-2)", minWidth: 30, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
        {val.toFixed(1)}
      </span>
    </div>
  );
}

/* ─────────────────────────────────────────
   MAIN APP
───────────────────────────────────────── */
export default function App() {
  const [activeTab, setActiveTab] = useState("rankings");
  const [matrix, setMatrix] = useState(FALLBACK_MATRIX);
  const [metrics, setMetrics] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [clauseInput, setClauseInput] = useState(PREDICT_SAMPLES.compliant);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("http://localhost:8000/api/statutory-matrix")
      .then(r => r.json())
      .then(data => {
        if (data && data.length > 0) {
          setMatrix(data.map(row => ({
            platform:      row.Platform || row.platform,
            dpdpAct2023:   parseFloat(row["DPDP Act 2023"]) || 0,
            dpdpRules2025: parseFloat(row["DPDP Rules 2025"]) || 0,
            itAct2000:     parseFloat(row["IT Act 2000"]) || 0,
            final:         parseFloat(row["Combined Final %"]) || 0,
          })));
        }
      }).catch(() => {});

    fetch("http://localhost:8000/api/metrics")
      .then(r => r.json())
      .then(setMetrics)
      .catch(() => {});
  }, []);

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clause_text: clauseInput }),
      });
      setPrediction(await res.json());
    } catch {
      setPrediction(mockPredict(clauseInput));
    }
    setLoading(false);
  };

  const doPredictSample = (key) => { setClauseInput(PREDICT_SAMPLES[key]); setPrediction(null); };

  const meanAct    = matrix.reduce((s, r) => s + r.dpdpAct2023,   0) / matrix.length;
  const meanRules  = matrix.reduce((s, r) => s + r.dpdpRules2025, 0) / matrix.length;
  const meanIt     = matrix.reduce((s, r) => s + r.itAct2000,     0) / matrix.length;
  const totalMean  = matrix.reduce((s, r) => s + r.final,         0) / matrix.length;

  const TABS = [
    { id: "rankings", label: "Rankings",  icon: <BarChart2 size={13} /> },
    { id: "matrix",   label: "Matrix",    icon: <BookOpen  size={13} /> },
    { id: "predict",  label: "Predict",   icon: <Search    size={13} /> },
    { id: "models",   label: "Models",    icon: <Cpu       size={13} /> },
  ];

  const KPI_CARDS = [
    {
      label: "Audited Platforms", value: "6",
      sub: "WhatsApp · Telegram · Snapchat · YouTube · Google · Meta",
      icon: <ShieldCheck size={18} />, accent: "var(--indigo)", accentDim: "var(--indigo-dim)",
    },
    {
      label: "Active In-Scope Rules", value: "35",
      sub: "of 38 canonical · 3 governance-only excluded",
      icon: <Gavel size={18} />, accent: "var(--saffron)", accentDim: "var(--saffron-dim)",
    },
    {
      label: "Production Macro-F1", value: metrics ? metrics.baseline.macro_f1.toFixed(4) : "0.4895",
      sub: "TF-IDF + LogReg · target ≥ 0.50",
      icon: <Activity size={18} />, accent: "var(--jade)", accentDim: "var(--jade-dim)",
    },
    {
      label: "Baseline Test Accuracy", value: metrics ? `${(metrics.baseline.accuracy * 100).toFixed(1)}%` : "88.8%",
      sub: "116 test clauses",
      icon: <Zap size={18} />, accent: "var(--jade)", accentDim: "var(--jade-dim)",
    },
  ];

  return (
    <div style={{ position: "relative", minHeight: "100vh", backgroundColor: "var(--bg-void)" }}>
      <div className="mesh-bg" />

      {/* ── HEADER ─────────────────────────────── */}
      <header style={{
        position: "sticky", top: 0, zIndex: 50,
        backdropFilter: "blur(18px)",
        backgroundColor: "rgba(5,7,13,0.85)",
        borderBottom: "1px solid rgba(99,120,180,0.12)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 28px", height: 60,
      }}>
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 10,
            background: "linear-gradient(135deg,#5B6CF9,#7B8CF9)",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "0 0 16px rgba(91,108,249,0.4)",
          }}>
            <ShieldCheck size={17} color="#fff" />
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "var(--font-display)", letterSpacing: "-0.01em" }}>
              DPDP Compliance
            </div>
            <div style={{ fontSize: 10, color: "var(--text-3)", letterSpacing: "0.04em" }}>AUDIT INTELLIGENCE PLATFORM</div>
          </div>

          {/* live badge */}
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 8,
            background: "rgba(16,201,143,0.08)", border: "1px solid rgba(16,201,143,0.2)",
            borderRadius: 999, padding: "3px 10px" }}>
            <div className="pulse-dot" />
            <span style={{ fontSize: 10, fontWeight: 700, color: "var(--jade)", letterSpacing: "0.07em" }}>LIVE</span>
          </div>
        </div>

        {/* Nav */}
        <nav style={{ display: "flex", gap: 4, background: "rgba(255,255,255,0.03)", borderRadius: 999, padding: 4 }}>
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`nav-pill ${activeTab === t.id ? "active" : "inactive"}`}
              style={{ display: "flex", alignItems: "center", gap: 6 }}
            >
              {t.icon}{t.label}
            </button>
          ))}
        </nav>

        <span style={{ fontSize: 11, color: "var(--text-3)", letterSpacing: "0.04em" }}>Updated 28 Sep 2026</span>
      </header>

      {/* ── MAIN ──────────────────────────────── */}
      <main style={{ padding: "28px 28px", position: "relative", zIndex: 1 }}>

        {/* KPI STRIP */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 28 }}>
          {KPI_CARDS.map((card, i) => (
            <div key={i} className="kpi-card" style={{ "--accent-color": card.accent }}>
              <div className="accent-line" />
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-3)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
                  {card.label}
                </div>
                <div style={{ width: 30, height: 30, borderRadius: 8, background: card.accentDim,
                  display: "flex", alignItems: "center", justifyContent: "center", color: card.accent, flexShrink: 0 }}>
                  {card.icon}
                </div>
              </div>
              <div className="kpi-value" style={{ fontSize: 30, fontWeight: 800, color: card.accent,
                fontFamily: "var(--font-display)", letterSpacing: "-0.02em", marginBottom: 6 }}>
                {card.value}
              </div>
              <div style={{ fontSize: 11, color: "var(--text-3)", lineHeight: 1.5 }}>{card.sub}</div>
            </div>
          ))}
        </div>

        {/* ── RANKINGS TAB ──────────────────────── */}
        {activeTab === "rankings" && (
          <div className="tab-content" style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 16 }}>

            {/* Platform bar chart */}
            <div className="card" style={{ padding: 24 }}>
              <div className="section-title">Platform Compliance Rankings</div>
              {/* Column header row */}
              <div style={{

                display: "grid",
                gridTemplateColumns: "60px 1fr 1fr 1fr 1fr 80px",
                gap: 0,
                paddingBottom: 8,
                borderBottom: "1px solid rgba(99,120,180,0.08)",
                marginBottom: 4,
              }}>
                <div />
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-3)", letterSpacing: "0.08em", textTransform: "uppercase" }}>Platform</div>
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--indigo)", letterSpacing: "0.08em", textTransform: "uppercase", textAlign: "center" }}>DPDP Act</div>
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--saffron)", letterSpacing: "0.08em", textTransform: "uppercase", textAlign: "center" }}>DPDP Rules</div>
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--jade)", letterSpacing: "0.08em", textTransform: "uppercase", textAlign: "center" }}>IT Act</div>
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-3)", letterSpacing: "0.08em", textTransform: "uppercase", textAlign: "right" }}>Final</div>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {[...matrix].sort((a, b) => b.final - a.final).map((row, i) => (
                  <div key={i} style={{
                    display: "grid",
                    gridTemplateColumns: "60px 1fr 1fr 1fr 1fr 80px",
                    gap: 0,
                    alignItems: "center",
                    padding: "10px 0",
                    borderBottom: "1px solid rgba(99,120,180,0.05)",
                    borderRadius: 8,
                    transition: "background 0.15s",
                  }}
                    onMouseEnter={e => e.currentTarget.style.background = "rgba(91,108,249,0.04)"}
                    onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                  >
                    {/* Rank + Avatar */}
                    <div style={{ display: "flex", alignItems: "center", gap: 8, paddingLeft: 8 }}>
                      <span style={{
                        fontSize: 11, fontWeight: 700, width: 16, textAlign: "center", flexShrink: 0,
                        color: i === 0 ? "var(--saffron)" : i === 1 ? "var(--text-2)" : "var(--text-3)",
                      }}>#{i + 1}</span>
                      <div style={{
                        width: 28, height: 28, borderRadius: 7, flexShrink: 0,
                        background: `${PLATFORM_COLORS[row.platform] || "var(--indigo)"}22`,
                        border: `1px solid ${PLATFORM_COLORS[row.platform] || "var(--indigo)"}44`,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 9, fontWeight: 700, color: PLATFORM_COLORS[row.platform] || "var(--indigo)",
                      }}>
                        {PLATFORM_INITIALS(row.platform)}
                      </div>
                    </div>

                    {/* Platform name + overall bar */}
                    <div style={{ paddingRight: 16 }}>
                      <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-1)", marginBottom: 5 }}>{row.platform}</div>
                      <div style={{ height: 4, background: "rgba(255,255,255,0.05)", borderRadius: 2, overflow: "hidden" }}>
                        <div style={{
                          width: `${Math.min((row.final / 55) * 100, 100)}%`,
                          height: "100%", borderRadius: 2,
                          background: `linear-gradient(90deg, ${scoreColor(row.final)}88, ${scoreColor(row.final)})`,
                          transition: "width 1s cubic-bezier(0.22,1,0.36,1)",
                        }} />
                      </div>
                    </div>

                    {/* DPDP Act */}
                    <div style={{ padding: "0 12px", textAlign: "center" }}>
                      <div style={{ fontSize: 14, fontWeight: 700, color: "var(--indigo)", fontVariantNumeric: "tabular-nums", marginBottom: 4 }}>
                        {row.dpdpAct2023.toFixed(1)}
                      </div>
                      <div style={{ height: 4, background: "rgba(91,108,249,0.1)", borderRadius: 2, overflow: "hidden" }}>
                        <div style={{ width: `${Math.min((row.dpdpAct2023 / 55) * 100, 100)}%`, height: "100%", background: "var(--indigo)", borderRadius: 2, transition: "width 1s" }} />
                      </div>
                    </div>

                    {/* DPDP Rules */}
                    <div style={{ padding: "0 12px", textAlign: "center" }}>
                      <div style={{ fontSize: 14, fontWeight: 700, color: "var(--saffron)", fontVariantNumeric: "tabular-nums", marginBottom: 4 }}>
                        {row.dpdpRules2025.toFixed(1)}
                      </div>
                      <div style={{ height: 4, background: "rgba(245,166,35,0.1)", borderRadius: 2, overflow: "hidden" }}>
                        <div style={{ width: `${Math.min((row.dpdpRules2025 / 55) * 100, 100)}%`, height: "100%", background: "var(--saffron)", borderRadius: 2, transition: "width 1s" }} />
                      </div>
                    </div>

                    {/* IT Act */}
                    <div style={{ padding: "0 12px", textAlign: "center" }}>
                      <div style={{ fontSize: 14, fontWeight: 700, color: "var(--jade)", fontVariantNumeric: "tabular-nums", marginBottom: 4 }}>
                        {row.itAct2000.toFixed(1)}
                      </div>
                      <div style={{ height: 4, background: "rgba(16,201,143,0.1)", borderRadius: 2, overflow: "hidden" }}>
                        <div style={{ width: `${Math.min((row.itAct2000 / 55) * 100, 100)}%`, height: "100%", background: "var(--jade)", borderRadius: 2, transition: "width 1s" }} />
                      </div>
                    </div>

                    {/* Final score */}
                    <div style={{ textAlign: "right", paddingRight: 8 }}>
                      <span style={{
                        display: "inline-block", padding: "4px 10px", borderRadius: 999,
                        background: `${scoreColor(row.final)}18`,
                        border: `1px solid ${scoreColor(row.final)}44`,
                        fontSize: 13, fontWeight: 800, color: scoreColor(row.final),
                        fontVariantNumeric: "tabular-nums",
                      }}>
                        {row.final.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>


            {/* Right column */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

              {/* Statutory coverage rings */}
              <div className="card" style={{ padding: 24 }}>
                <div className="section-title">Statutory Coverage</div>
                <div style={{ display: "flex", justifyContent: "space-around", marginBottom: 20 }}>
                  <Ring pct={Math.min(meanAct, 100)}   color="var(--indigo)"  label={`${meanAct.toFixed(1)}%`}   sublabel="DPDP Act" />
                  <Ring pct={Math.min(meanRules, 100)} color="var(--saffron)" label={`${meanRules.toFixed(1)}%`} sublabel="DPDP Rules" />
                  <Ring pct={Math.min(meanIt, 100)}    color="var(--jade)"    label={`${meanIt.toFixed(1)}%`}    sublabel="IT Act" />
                </div>
                <div style={{ borderTop: "1px solid var(--border)", paddingTop: 14, textAlign: "center" }}>
                  <div style={{ fontSize: 10, color: "var(--text-3)", letterSpacing: "0.06em", marginBottom: 4 }}>OVERALL MEAN COMPLIANCE</div>
                  <div style={{ fontSize: 26, fontWeight: 800, color: scoreColor(totalMean), fontFamily: "var(--font-display)" }}>
                    {totalMean.toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Regulatory gaps */}
              <div className="card" style={{ padding: 24, flex: 1 }}>
                <div className="section-title">Regulatory Gaps</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {REGULATORY_GAPS.map((g, i) => (
                    <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start",
                      background: "rgba(255,255,255,0.02)", borderRadius: 8, padding: "10px 12px",
                      border: "1px solid rgba(255,255,255,0.04)" }}>
                      <div style={{ width: 6, height: 6, borderRadius: "50%", flexShrink: 0, marginTop: 5,
                        background: g.severity === "high" ? "var(--rose)" : g.severity === "medium" ? "var(--saffron)" : "var(--jade)" }} />
                      <div>
                        <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-1)", marginBottom: 2 }}>{g.label}</div>
                        <div style={{ fontSize: 11, color: "var(--text-3)" }}>{g.detail}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── MATRIX TAB ────────────────────────── */}
        {activeTab === "matrix" && (
          <div className="tab-content">
            <div className="card" style={{ padding: 24 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
                <div className="section-title" style={{ marginBottom: 0 }}>Multi-Act Compliance Heatmap</div>
                <div style={{ display: "flex", gap: 12 }}>
                  {[
                    { label: "≥ 25% High",   color: "var(--jade)"    },
                    { label: "10–25% Mid",    color: "var(--saffron)" },
                    { label: "< 10% Low",     color: "var(--rose)"    },
                  ].map(l => (
                    <div key={l.label} className="heat-legend">
                      <div className="heat-swatch" style={{ background: l.color, opacity: 0.7 }} />
                      {l.label}
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ overflowX: "auto" }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Platform</th>
                      <th>DPDP Act 2023</th>
                      <th>DPDP Rules 2025</th>
                      <th>IT Act 2000</th>
                      <th>Combined Final %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...matrix].sort((a, b) => b.final - a.final).map((row, i) => (
                      <tr key={i}>
                        <td>
                          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                            <div style={{ width: 28, height: 28, borderRadius: 7, flexShrink: 0,
                              background: `${PLATFORM_COLORS[row.platform] || "#5B6CF9"}22`,
                              display: "flex", alignItems: "center", justifyContent: "center",
                              fontSize: 10, fontWeight: 700, color: PLATFORM_COLORS[row.platform] || "#5B6CF9" }}>
                              {PLATFORM_INITIALS(row.platform)}
                            </div>
                            <span style={{ fontWeight: 600, color: "var(--text-1)" }}>{row.platform}</span>
                          </div>
                        </td>
                        {[row.dpdpAct2023, row.dpdpRules2025, row.itAct2000].map((v, ci) => (
                          <td key={ci} style={{ textAlign: "right" }}>
                            <div style={{ display: "inline-flex", alignItems: "center", gap: 8, justifyContent: "flex-end" }}>
                              <div style={{ width: 48, height: 5, background: "rgba(255,255,255,0.05)", borderRadius: 2, overflow: "hidden" }}>
                                <div style={{ width: `${Math.min((v / 55) * 100, 100)}%`, height: "100%",
                                  background: scoreColor(v), borderRadius: 2, transition: "width 1s" }} />
                              </div>
                              <span style={{ color: scoreColor(v), fontWeight: 600, fontVariantNumeric: "tabular-nums", minWidth: 36 }}>
                                {v.toFixed(1)}
                              </span>
                            </div>
                          </td>
                        ))}
                        <td style={{ textAlign: "right" }}>
                          <span style={{
                            display: "inline-block", padding: "3px 10px", borderRadius: 999,
                            background: scoreBg(row.final), color: scoreColor(row.final),
                            fontWeight: 700, fontVariantNumeric: "tabular-nums", fontSize: 12,
                          }}>
                            {row.final.toFixed(1)}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ── PREDICT TAB ───────────────────────── */}
        {activeTab === "predict" && (
          <div className="tab-content" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

            {/* Input */}
            <div className="card" style={{ padding: 24 }}>
              <div className="section-title">Analyze a Privacy Clause</div>
              <textarea
                className="clause-input"
                value={clauseInput}
                onChange={e => setClauseInput(e.target.value)}
                placeholder="Paste a privacy policy clause here..."
              />
              <button className="btn-primary" onClick={handleAnalyze} disabled={loading} style={{ marginTop: 14 }}>
                {loading ? "Analyzing…" : "Analyze Clause →"}
              </button>
              <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <button className="btn-ghost" onClick={() => doPredictSample("compliant")}>Load: Compliant</button>
                <button className="btn-ghost" onClick={() => doPredictSample("vague")}>Load: Vague</button>
                <button className="btn-ghost" onClick={() => doPredictSample("irrelevant")} style={{ gridColumn: "1/-1" }}>
                  Load: Irrelevant
                </button>
              </div>
            </div>

            {/* Result */}
            <div className="card" style={{ padding: 24 }}>
              <div className="section-title">Classification Result</div>
              {prediction ? (() => {
                const isCompliant  = prediction.verdict === "Compliant";
                const isPartial    = prediction.verdict === "Partially Compliant";
                const accent = isCompliant ? "var(--jade)" : isPartial ? "var(--saffron)" : "var(--rose)";
                const accentDim = isCompliant ? "var(--jade-dim)" : isPartial ? "var(--saffron-dim)" : "var(--rose-dim)";
                const confPct = ((prediction.confidence || 0) * 100).toFixed(0);
                return (
                  <div>
                    {/* Verdict banner */}
                    <div style={{
                      textAlign: "center", padding: "20px 16px", borderRadius: 12, marginBottom: 20,
                      background: accentDim, border: `1px solid ${accent}44`,
                    }}>
                      <div style={{ fontSize: 10, letterSpacing: "0.12em", color: "var(--text-3)", marginBottom: 8, fontWeight: 700 }}>VERDICT</div>
                      <div style={{ fontSize: 24, fontWeight: 800, color: accent, fontFamily: "var(--font-display)" }}>
                        {prediction.verdict}
                      </div>
                    </div>

                    {/* Confidence */}
                    <div style={{ marginBottom: 16 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ fontSize: 11, color: "var(--text-3)", fontWeight: 600, letterSpacing: "0.04em" }}>CONFIDENCE</span>
                        <span style={{ fontSize: 13, fontWeight: 700, color: accent, fontVariantNumeric: "tabular-nums" }}>{confPct}%</span>
                      </div>
                      <div style={{ height: 8, background: "rgba(255,255,255,0.06)", borderRadius: 4, overflow: "hidden" }}>
                        <div style={{ width: `${confPct}%`, height: "100%",
                          background: `linear-gradient(90deg,${accent}88,${accent})`, borderRadius: 4,
                          transition: "width 1s cubic-bezier(0.22,1,0.36,1)" }} />
                      </div>
                    </div>

                    {/* Rule match */}
                    {prediction.rule && (
                      <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: 8, padding: "12px 14px",
                        border: "1px solid rgba(255,255,255,0.06)" }}>
                        <div style={{ fontSize: 10, color: "var(--text-3)", letterSpacing: "0.06em", fontWeight: 700, marginBottom: 6 }}>
                          MATCHED RULE
                        </div>
                        <div style={{ fontSize: 14, fontWeight: 700, color: "var(--indigo)", marginBottom: 4 }}>{prediction.rule}</div>
                        <div style={{ fontSize: 12, color: "var(--text-2)" }}>{prediction.category}</div>
                      </div>
                    )}
                  </div>
                );
              })() : (
                <div style={{ height: "100%", display: "flex", flexDirection: "column",
                  alignItems: "center", justifyContent: "center", color: "var(--text-3)", gap: 12, paddingTop: 40 }}>
                  <Search size={32} style={{ opacity: 0.3 }} />
                  <div style={{ fontSize: 13 }}>Enter a clause and click Analyze</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── MODELS TAB ────────────────────────── */}
        {activeTab === "models" && (
          <div className="tab-content" style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Production model card — full width */}
            <div className="card" style={{ padding: 28, borderColor: "rgba(16,201,143,0.2)" }}>
              <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2,
                background: "linear-gradient(90deg, transparent, var(--jade), transparent)" }} />
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 24 }}>
                <div>
                  <div className="section-title" style={{ marginBottom: 4 }}>TF-IDF + Logistic Regression</div>
                  <div style={{ fontSize: 12, color: "var(--text-3)" }}>Production baseline model · Trained on 771 labelled privacy clauses</div>
                </div>
                <span className="badge badge-success">● PRODUCTION</span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
                {[
                  { label: "Test Accuracy",    value: "88.8%",  color: "var(--jade)",    sub: "116 test clauses"       },
                  { label: "Macro-F1",          value: "0.4895", color: "var(--indigo)",  sub: "target ≥ 0.50"         },
                  { label: "F1 Not Addressed",  value: "0.94",   color: "var(--jade)",    sub: "highest class F1"       },
                  { label: "F1 Partially Compliant", value: "0.53", color: "var(--saffron)", sub: "mid-range class"    },
                ].map((m, i) => (
                  <div key={i} style={{ background: "rgba(255,255,255,0.03)", borderRadius: 10,
                    padding: "16px 18px", border: "1px solid rgba(255,255,255,0.06)" }}>
                    <div style={{ fontSize: 10, color: "var(--text-3)", letterSpacing: "0.06em", fontWeight: 700, marginBottom: 8, textTransform: "uppercase" }}>{m.label}</div>
                    <div style={{ fontSize: 28, fontWeight: 800, color: m.color, fontFamily: "var(--font-display)", marginBottom: 4 }}>{m.value}</div>
                    <div style={{ fontSize: 11, color: "var(--text-3)" }}>{m.sub}</div>
                  </div>
                ))}
              </div>
            </div>


          </div>
        )}
      </main>

      {/* ── FOOTER ────────────────────────────── */}
      <footer style={{
        borderTop: "1px solid rgba(99,120,180,0.1)",
        padding: "14px 28px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        fontSize: 11,
        color: "var(--text-3)",
        position: "relative", zIndex: 1,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--jade)" }} />
          <span>Production · TF-IDF + LogReg · Macro-F1 <strong style={{ color: "var(--text-2)" }}>0.4895</strong></span>
        </div>
        <span>Taxonomy v38 · 35 in-scope · Corpus 771 clauses</span>
      </footer>
    </div>
  );
}
