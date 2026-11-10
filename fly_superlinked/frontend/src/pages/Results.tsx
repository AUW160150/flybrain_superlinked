import { useEffect, useState } from "react";
import { api, fmt } from "../lib/sim";

const ARM_COLOR: Record<string, string> = { real: "var(--c-real)", rewired: "var(--c-rewired)", rewired_class: "var(--c-class)", reactive: "var(--c-reactive)" };
const ARM_LABEL: Record<string, string> = { real: "real connectome", rewired: "rewired, whole graph", rewired_class: "rewired, within classes", reactive: "reactive baseline" };
const MEM_COLOR: Record<string, string> = { shared: "var(--c-real)", private: "var(--c-rewired)", none: "var(--c-reactive)" };
const MEM_LABEL: Record<string, string> = { shared: "shared swarm memory", private: "private memory only", none: "no memory" };
const pct = (x: number | null | undefined) => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`);

type Tip = { x: number; y: number; text: string } | null;

/** Horizontal bars, values in [0, max]; direct labels; per-bar hover. */
function HBars({ rows, max = 1, format = pct, onTip }: { rows: { key: string; label: string; v: number | null; color: string; tip: string }[]; max?: number; format?: (v: number) => string; onTip: (t: Tip) => void }) {
  const W = 520, rowH = 30, left = 170, H = rows.length * rowH + 22;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: 640 }} role="img">
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <g key={f}><line x1={left + f * (W - left - 50)} x2={left + f * (W - left - 50)} y1={0} y2={H - 18} stroke="var(--line)" strokeWidth={1} />
          <text x={left + f * (W - left - 50)} y={H - 4} fontSize={10} textAnchor="middle" fill="var(--ink-3)" fontFamily="IBM Plex Mono, monospace">{format(f * max)}</text></g>))}
      {rows.map((r, i) => {
        const w = r.v === null ? 0 : Math.max(2, (r.v / max) * (W - left - 50));
        const y = i * rowH + 6;
        return (
          <g key={r.key} onMouseMove={(e) => onTip({ x: e.clientX, y: e.clientY, text: r.tip })} onMouseLeave={() => onTip(null)}>
            <rect x={0} y={y - 4} width={W} height={rowH - 2} fill="transparent" />
            <text x={0} y={y + 13} fontSize={12} fill="var(--ink-2)" fontFamily="IBM Plex Sans, sans-serif">{r.label}</text>
            <rect x={left} y={y} width={w} height={18} rx={4} fill={r.color} />
            <text x={left + w + 6} y={y + 13} fontSize={12} fill="var(--ink)" fontFamily="IBM Plex Mono, monospace">{r.v === null ? "–" : format(r.v)}</text>
          </g>);
      })}
    </svg>
  );
}

function Controllers({ d, onTip }: { d: any; onTip: (t: Tip) => void }) {
  if (!d || d.status === "not run") return <div className="empty">Controller experiment not run yet.</div>;
  const arms = ["real", "rewired", "rewired_class", "reactive"].filter((a) => d.arms[a]);
  const s = (a: string) => d.arms[a].summary;
  const cmp = d.comparisons ?? {};
  const best = arms.reduce((b, a) => (s(a).success_rate > s(b).success_rate ? a : b), arms[0]);
  const rr = cmp.real_minus_rewired?.success, rc = cmp.real_minus_rewired_class?.success, rx = cmp.real_minus_reactive?.success;
  return (<>
    <HBars onTip={onTip} rows={arms.map((a) => ({ key: a, label: ARM_LABEL[a], v: s(a).success_rate, color: ARM_COLOR[a],
      tip: `${ARM_LABEL[a]}: ${pct(s(a).success_rate)} of ${s(a).n} missions completed; delivered ${pct(s(a).delivery_rate)}` }))} />
    <div className="tablewrap"><table>
      <thead><tr><th>arm</th><th className="num">success</th><th className="num">delivered</th><th className="num">time to goal</th><th className="num">collisions</th><th className="num">gust entries</th><th className="num">pond entries</th><th className="num">escapes</th></tr></thead>
      <tbody>{arms.map((a) => (<tr key={a}><td>{ARM_LABEL[a]}</td><td className="num">{pct(s(a).success_rate)}</td><td className="num">{pct(s(a).delivery_rate)}</td>
        <td className="num">{s(a).mean_time_to_goal_s == null ? "–" : `${fmt(s(a).mean_time_to_goal_s, 1)} s`}</td><td className="num">{fmt(s(a).mean_collisions, 2)}</td><td className="num">{fmt(s(a).mean_gust_entries, 1)}</td>
        <td className="num">{fmt(s(a).mean_pond_entries, 1)}</td><td className="num">{fmt(s(a).mean_escapes, 1)}</td></tr>))}</tbody>
    </table></div>
    <p className="takeaway">
      {best === "reactive" ? "The hand-written reactive controller is the most reliable. " : `${ARM_LABEL[best]} completes the most missions. `}
      {rr && `The real wiring completes ${Math.round(rr.diff * 100)} points more missions than the whole-graph shuffle (95% CI ${Math.round(rr.ci95[0] * 100)} to ${Math.round(rr.ci95[1] * 100)})`}
      {rc && `, and ${Math.round(rc.diff * 100)} points more than the within-class shuffle (${Math.round(rc.ci95[0] * 100)} to ${Math.round(rc.ci95[1] * 100)})`}
      {rx && `; it completes ${Math.abs(Math.round(rx.diff * 100))} points ${rx.diff < 0 ? "fewer" : "more"} than the reactive baseline (${Math.round(rx.ci95[0] * 100)} to ${Math.round(rx.ci95[1] * 100)}).`}
      {" "}Same {d.seeds?.length} seeds for every arm; decoder gains were tuned on separate seeds with the real circuit.
    </p>
  </>);
}

function Memory({ d, onTip }: { d: any; onTip: (t: Tip) => void }) {
  if (!d || d.status === "not run" || !d.arms) return <div className="empty">Memory ablation not run yet.</div>;
  const arms = ["none", "private", "shared"].filter((a) => d.arms[a]);
  const s = (a: string) => d.arms[a].summary;
  const runs = (a: string) => d.arms[a].runs as any[];
  const W = 520, cell = 20, left = 150;
  return (<>
    <div className="grid g-2">
      <div><div className="eyebrow">missions that entered the gust zone</div>
        <HBars onTip={onTip} rows={arms.map((a) => ({ key: a, label: MEM_LABEL[a], v: s(a).gust_entry_rate, color: MEM_COLOR[a], tip: `${MEM_LABEL[a]}: ${pct(s(a).gust_entry_rate)} of missions entered the gust zone; first five missions ${pct(s(a).first5_gust_entry_rate)}` }))} /></div>
      <div><div className="eyebrow">missions completed</div>
        <HBars onTip={onTip} rows={arms.map((a) => ({ key: a, label: MEM_LABEL[a], v: s(a).success_rate, color: MEM_COLOR[a], tip: `${MEM_LABEL[a]}: ${pct(s(a).success_rate)} completed, mean ${fmt(s(a).mean_mission_time_s, 1)} s` }))} /></div>
    </div>
    <div className="eyebrow" style={{ marginTop: 12 }}>gust entries per mission, in the order flown (seed 0 → 19)</div>
    <svg viewBox={`0 0 ${W} ${arms.length * 26 + 8}`} style={{ width: "100%", maxWidth: 640 }} role="img" aria-label="Gust entries per mission by memory mode">
      {arms.map((a, i) => (<g key={a}>
        <text x={0} y={i * 26 + 17} fontSize={12} fill="var(--ink-2)">{MEM_LABEL[a]}</text>
        {runs(a).map((r, j) => (<rect key={j} x={left + j * (cell + 2)} y={i * 26 + 4} width={cell} height={18} rx={3}
          fill={r.gust_entries > 0 ? MEM_COLOR[a] : "var(--paper-2)"} stroke="var(--line)" opacity={r.gust_entries > 0 ? Math.min(1, 0.45 + 0.2 * r.gust_entries) : 1}
          onMouseMove={(e) => onTip({ x: e.clientX, y: e.clientY, text: `${MEM_LABEL[a]}, seed ${r.seed}: ${r.gust_entries} gust entries, ${r.success ? "completed" : "not completed"}, ${r.hazards_used} remembered hazards used` })} onMouseLeave={() => onTip(null)} />))}
      </g>))}
    </svg>
    <div className="tablewrap"><table>
      <thead><tr><th>memory</th><th className="num">completed</th><th className="num">mean time</th><th className="num">gust entries / mission</th><th className="num">missions in gust</th><th className="num">first 5 in gust</th><th className="num">collisions</th></tr></thead>
      <tbody>{arms.map((a) => (<tr key={a}><td>{MEM_LABEL[a]}</td><td className="num">{pct(s(a).success_rate)}</td><td className="num">{s(a).mean_mission_time_s == null ? "–" : `${fmt(s(a).mean_mission_time_s, 1)} s`}</td>
        <td className="num">{fmt(s(a).mean_gust_entries, 2)}</td><td className="num">{pct(s(a).gust_entry_rate)}</td><td className="num">{pct(s(a).first5_gust_entry_rate)}</td><td className="num">{fmt(s(a).mean_collisions, 2)}</td></tr>))}</tbody>
    </table></div>
    {arms.length === 3 && <p className="takeaway">
      With shared memory the courier entered the gust zone on {pct(s("shared").gust_entry_rate)} of missions, against {pct(s("none").gust_entry_rate)} with no memory and {pct(s("private").gust_entry_rate)} with only its own notes
      {` (first five missions: ${pct(s("shared").first5_gust_entry_rate)} vs ${pct(s("none").first5_gust_entry_rate)} vs ${pct(s("private").first5_gust_entry_rate)})`}. Brain: {d.brain}; memory backend: {d.arms.shared?.memory_backend}.
    </p>}
  </>);
}

function Sie({ d }: { d: any }) {
  if (!d || d.status === "not run") return <div className="empty">SIE benchmark not run yet.</div>;
  const P = d.providers ?? {};
  const jobs: [string, string][] = [["mission_planner", "Mission planner (30 missions)"], ["perception", "Perception to facts (40 keyframes)"]];
  const fr = P.frontier && P.frontier.mission_planner ? P.frontier : null;
  const take = jobs.map(([j, name]) => {
    const a = P.sie?.[j], b = fr?.[j];
    if (!a) return null;
    if (!b) return `${name}: SIE ${pct(a.exact_match)} exact at $${fmt(a.cost_per_1000_calls_usd, 2)} per 1,000 calls; frontier not run.`;
    const ratio = b.cost_per_1000_calls_usd / a.cost_per_1000_calls_usd;
    return `${name}: SIE reaches ${pct(a.field_accuracy)} field accuracy vs ${pct(b.field_accuracy)} for ${b.model} (exact ${pct(a.exact_match)} vs ${pct(b.exact_match)}) at 1/${fmt(ratio, 0)} of the cost.`;
  }).filter(Boolean);
  return (<>
    <div className="tablewrap"><table>
      <thead><tr><th>job</th><th>provider · model</th><th className="num">valid JSON</th><th className="num">exact match</th><th className="num">field accuracy</th><th className="num">median latency</th><th className="num">$ per 1,000 calls</th></tr></thead>
      <tbody>{jobs.flatMap(([j, name]) => ["sie", "frontier"].map((p) => {
        const r = P[p]?.[j];
        return (<tr key={j + p}><td>{p === "sie" ? name : ""}</td><td>{p === "sie" ? "SIE" : "frontier"} · <span className="mono small">{r?.model ?? "not run"}</span></td>
          <td className="num">{r ? pct(r.valid_json_rate) : "–"}</td><td className="num">{r ? pct(r.exact_match) : "not run"}</td><td className="num">{r ? pct(r.field_accuracy) : "–"}</td>
          <td className="num">{r ? `${fmt(r.median_latency_s, 2)} s` : "–"}</td><td className="num">{r ? `$${fmt(r.cost_per_1000_calls_usd, 3)}` : "–"}</td></tr>);
      }))}
      {d.place_recognition?.model && <tr><td>Place recognition (image embedding)</td><td>SIE · <span className="mono small">{d.place_recognition.model}</span></td><td className="num">–</td><td className="num">–</td><td className="num">–</td><td className="num">{fmt(d.place_recognition.batch_latency_s, 2)} s / {d.place_recognition.batch_size}</td><td className="num">${fmt(d.place_recognition.cost_per_1000_images_usd, 4)}</td></tr>}
      </tbody>
    </table></div>
    {take.map((t, i) => <p key={i} className="takeaway">{t}</p>)}
    <p className="cap">Cost = measured tokens from each API's usage field × published list price (sources in backend/config/prices.yaml, read {String(d.generated).slice(0, 10)}). Keyframe labels come from simulator geometry. Precomputed: {String(d.generated)}.</p>
  </>);
}

export default function Results() {
  const [c, setC] = useState<any>(null), [m, setM] = useState<any>(null), [s, setS] = useState<any>(null);
  const [tip, setTip] = useState<Tip>(null);
  useEffect(() => { api("/api/bench/controllers").then(setC).catch(() => setC(null)); api("/api/bench/memory").then(setM).catch(() => setM(null)); api("/api/bench/sie").then(setS).catch(() => setS(null)); }, []);
  return (
    <div className="page wrap">
      <div className="page-head"><div>
        <div className="section-n">04 · Results</div>
        <h2>What the numbers <em>say</em></h2>
        <p className="lead" style={{ marginTop: 10 }}>All three panels read the bench JSON files shipped with the app (precomputed, dated). Nothing on this page is typed by hand.</p>
      </div></div>
      <div className="grid" style={{ gap: 20 }}>
        <section className="card"><div className="section-n">4.1</div><h3>Controller comparison: real vs rewired vs reactive</h3>
          <p className="cap" style={{ marginBottom: 10 }}>{c?.task}</p><Controllers d={c} onTip={setTip} /></section>
        <section className="card"><div className="section-n">4.2 · cognee shared memory</div><h3>Memory ablation: does sharing what scouts learned help the courier?</h3>
          <p className="cap" style={{ marginBottom: 10 }}>{m?.protocol}</p><Memory d={m} onTip={setTip} /></section>
        <section className="card"><div className="section-n">4.3 · Superlinked SIE</div><h3>Everyday agent jobs on SIE open models vs a frontier model</h3><Sie d={s} /></section>
      </div>
      {tip && <div className="tip" style={{ left: tip.x + 12, top: tip.y + 12 }}>{tip.text}</div>}
    </div>
  );
}
