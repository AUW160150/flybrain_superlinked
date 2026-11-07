import { useState } from "react";
import Arena3D, { ArenaInfo } from "../components/Arena3D";
import { api, DRONE_COLORS, fmt, Frame } from "../lib/sim";

const EXAMPLES = [
  "Scouts, map the north side and come home.",
  "Courier, take the package to the red tower and come home, avoid the pond.",
  "Scouts, map the north side. Courier, take the package to the red tower and come home, avoid the pond.",
  "Courier, deliver to the blue arch and return home, keeping away from the gust zone.",
];

export default function MissionControl({ frame, arena, selected, setSelected, log, dark, onReplay, replaying }: {
  frame: Frame | null; arena: ArenaInfo | null; selected: string; setSelected: (s: string) => void;
  log: { t: number; msg: string }[]; dark: boolean; onReplay: (name: string | null) => void; replaying: boolean;
}) {
  const [text, setText] = useState(EXAMPLES[2]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [brain, setBrain] = useState("real");
  const [mem, setMem] = useState("shared");
  const [seed, setSeed] = useState(0);
  const [replays, setReplays] = useState<{ name: string }[] | null>(null);

  const send = async () => {
    setBusy(true); setErr("");
    try { await api("/api/missions", { text }); } catch (e) { setErr(String(e).slice(0, 300)); }
    setBusy(false);
  };
  const reset = async () => {
    setBusy(true); setErr("");
    try { await api("/api/sim/reset", { seed, brain, memory_mode: mem }); } catch (e) { setErr(String(e)); }
    setBusy(false);
  };
  const loadReplays = async () => setReplays(await api("/api/replays"));

  const drones = frame?.drones ?? [];
  return (
    <div className="page wrap">
      <div className="page-head">
        <div>
          <div className="section-n">01 · Mission control</div>
          <h1 className="hero-title">A fly brain <em>flies</em>, small open models see and plan, the swarm <em>remembers</em> together.</h1>
        </div>
      </div>
      <div className="grid g-main">
        <div className="card flush">
          <div className="stage">
            <Arena3D arena={arena} drones={drones} selected={selected} showGust={!!frame?.gust_known} dark={dark} />
            <div className="overlay">
              <span className="badge">t = {fmt(frame?.t, 1)} s</span>
              <span className="badge">brain: {frame?.brain ?? "–"}</span>
              <span className="badge">memory: {frame?.memory_mode ?? "–"}</span>
              <span className="badge">seed {frame?.seed ?? "–"}</span>
            </div>
            <div className="legend">
              {Object.entries(DRONE_COLORS).map(([n, c]) => <div key={n}><span className="sw" style={{ background: c }} />{n}</div>)}
              <div><span className="sw" style={{ background: "#29507f" }} />pond (no-fly)</div>
              <div><span className="sw" style={{ background: "#cf9a2f", opacity: 0.5 }} />gust zone, once felt</div>
            </div>
          </div>
        </div>
        <div className="grid" style={{ alignContent: "start" }}>
          <div className="card">
            <div className="card-head"><div className="eyebrow">Operator mission</div><span className="small muted">planned by {frame?.sie ? "SIE open model" : "keyword fallback"}</span></div>
            <div className="chips">{EXAMPLES.map((e) => <button key={e} className="chip" onClick={() => setText(e)}>{e}</button>)}</div>
            <div className="field"><label htmlFor="mission">Mission sentence</label>
              <textarea id="mission" value={text} onChange={(e) => setText(e.target.value)} maxLength={500} /></div>
            <button className="btn primary" style={{ width: "100%" }} onClick={send} disabled={busy || replaying}>{busy ? "Planning…" : "Send mission"}</button>
            {err && <p className="err">{err}</p>}
            {frame?.plan_source && <p className="cap">Last plan from {frame.plan_source}.</p>}
          </div>
          <div className="card">
            <div className="eyebrow">Experiment settings (applied on reset)</div>
            <div className="row">
              <div className="field"><label htmlFor="brain">Brain</label>
                <select id="brain" value={brain} onChange={(e) => setBrain(e.target.value)}>
                  <option value="real">real connectome</option><option value="rewired">rewired (whole graph)</option>
                  <option value="rewired_class">rewired (within classes)</option><option value="reactive">reactive baseline</option>
                </select></div>
              <div className="field"><label htmlFor="mem">Courier memory</label>
                <select id="mem" value={mem} onChange={(e) => setMem(e.target.value)}>
                  <option value="shared">shared swarm memory</option><option value="private">private only</option><option value="none">none</option>
                </select></div>
              <div className="field" style={{ maxWidth: 80 }}><label htmlFor="seed">Seed</label>
                <input id="seed" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></div>
            </div>
            <div className="row">
              <button className="btn" onClick={reset} disabled={busy || replaying}>Reset simulation</button>
              <button className="btn" onClick={() => (replaying ? onReplay(null) : loadReplays())}>{replaying ? "Back to live" : "Replays…"}</button>
            </div>
            {replays && !replaying && (
              <div className="chips" style={{ marginTop: 10 }}>
                {replays.length === 0 && <span className="small muted">No recorded replays yet.</span>}
                {replays.map((r) => <button key={r.name} className="chip" onClick={() => onReplay(r.name)}>▶ {r.name}</button>)}
              </div>
            )}
          </div>
          <div className="card">
            <div className="eyebrow">Drones</div>
            <div className="drones">
              {drones.map((d) => (
                <button key={d.name} className="drone" aria-pressed={d.name === selected} onClick={() => setSelected(d.name)}>
                  <span className="swatch" style={{ background: DRONE_COLORS[d.name] }} />
                  <span><b>{d.name}</b><br /><span className="meta">{d.done ? (d.delivered ? "done · delivered" : "done") : d.active ? `→ ${d.target?.label?.replace(/_/g, " ") ?? "…"}` : "idle"}</span></span>
                  <span className="meta num">{fmt(d.speed, 1)} m/s<br />hits {d.collisions} · gust {d.gust_entries}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <div className="eyebrow">Mission log</div>
        <div className="log">
          {log.length === 0 && <span className="muted">Send a mission to start. Scouts first, then the courier, shows shared memory best.</span>}
          {[...log].reverse().map((l, i) => <div key={i}><span className="t">{fmt(l.t, 1)}s</span>{l.msg}</div>)}
        </div>
      </div>
    </div>
  );
}
