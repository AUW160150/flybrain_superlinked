import { useEffect, useRef } from "react";
import { DRONE_COLORS, fmt, Frame } from "../lib/sim";

type Layout = { az: number[]; el: number[]; left: number[]; source: string };
type RasterNeuron = { type: string; side: string; group?: string };

function decodeMosaic(b64: string) {
  const bin = atob(b64); const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Each ommatidium drawn as a hexagon at its measured azimuth/elevation (equirectangular projection, left eye on the left). */
function Mosaic({ layout, mosaic, stripe }: { layout: Layout | null; mosaic?: string; stripe?: number[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c || !layout) return;
    const W = c.width, H = c.height, ctx = c.getContext("2d")!;
    const css = getComputedStyle(document.documentElement);
    ctx.fillStyle = css.getPropertyValue("--paper-2"); ctx.fillRect(0, 0, W, H);
    const cols = mosaic ? decodeMosaic(mosaic) : null;
    const st = new Set(stripe ?? []);
    const r = 4.6;
    for (let i = 0; i < layout.az.length; i++) {
      const x = W / 2 - (layout.az[i] / 180) * (W / 2 - 8);
      const y = H / 2 - (layout.el[i] / 90) * (H / 2 - 8);
      ctx.beginPath();
      for (let k = 0; k < 6; k++) { const a = Math.PI / 6 + (k * Math.PI) / 3; ctx[k ? "lineTo" : "moveTo"](x + r * Math.cos(a), y + r * Math.sin(a)); }
      ctx.closePath();
      ctx.fillStyle = cols ? `rgb(${cols[i * 3]},${cols[i * 3 + 1]},${cols[i * 3 + 2]})` : "#999";
      ctx.fill();
      if (st.has(i)) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.4; ctx.stroke(); }
    }
    ctx.strokeStyle = css.getPropertyValue("--ink-3"); ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = css.getPropertyValue("--ink-2"); ctx.font = "11px IBM Plex Mono, monospace";
    ctx.fillText("left eye", 10, 14); ctx.fillText("right eye", W - 70, 14); ctx.fillText("ahead", W / 2 + 4, H - 6);
  }, [layout, mosaic, stripe]);
  return <canvas ref={ref} width={720} height={340} style={{ width: "100%", borderRadius: 4 }} aria-label="Compound eye mosaic" />;
}

function Raster({ neurons, spikes, t }: { neurons: RasterNeuron[]; spikes: { t: number; n: number }[]; t: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const W = c.width, H = c.height, ctx = c.getContext("2d")!;
    const css = getComputedStyle(document.documentElement);
    ctx.fillStyle = css.getPropertyValue("--paper"); ctx.fillRect(0, 0, W, H);
    const left = 86, n = Math.max(1, neurons.length), rowH = (H - 20) / n, now = t * 1000;
    ctx.font = "11px IBM Plex Mono, monospace";
    let prev = "";
    neurons.forEach((nr, i) => {
      const y = 10 + i * rowH;
      const label = `${nr.group ?? nr.type} ${nr.side}`;
      if (label !== prev) { ctx.fillStyle = css.getPropertyValue("--ink-2"); ctx.fillText(label, 4, y + 10); prev = label;
        ctx.fillStyle = css.getPropertyValue("--line"); ctx.fillRect(left, y, W - left, 1); }
      if (i % 2 === 0) { ctx.fillStyle = css.getPropertyValue("--paper-2"); ctx.fillRect(left, y, W - left, rowH); }
    });
    const col = (type: string) => type.startsWith("DN") || type === "MDN" ? css.getPropertyValue("--accent") : type.startsWith("AOTU") ? css.getPropertyValue("--c-rewired") : css.getPropertyValue("--ink");
    for (const s of spikes) {
      const x = left + ((s.t - (now - 1000)) / 1000) * (W - left);
      if (x < left) continue;
      const nr = neurons[s.n]; if (!nr) continue;
      ctx.fillStyle = col(nr.type);
      ctx.fillRect(x, 10 + s.n * rowH + 1, 2, Math.max(1.5, rowH - 2));
    }
    ctx.fillStyle = css.getPropertyValue("--ink-3"); ctx.fillText("−1 s", left, H - 2); ctx.fillText("now", W - 26, H - 2);
  }, [neurons, spikes, t]);
  return <canvas ref={ref} width={720} height={560} style={{ width: "100%" }} aria-label="Spike raster of 64 displayed neurons" />;
}

function Bar({ label, v, max, color }: { label: string; v: number; max: number; color: string }) {
  return (<>
    <span>{label}</span>
    <div className="bar"><span style={{ width: `${Math.min(100, (100 * v) / max)}%`, background: color }} /></div>
    <span className="num">{fmt(v, 0)} Hz</span>
  </>);
}

export default function InsideFly({ frame, selected, setSelected, layout, neurons, spikes }: {
  frame: Frame | null; selected: string; setSelected: (s: string) => void; layout: Layout | null; neurons: RasterNeuron[]; spikes: { t: number; n: number }[];
}) {
  const d = frame?.detail;
  const cmd = d?.cmd ?? {};
  const rates = d?.rates ?? {};
  const isFly = frame?.brain && frame.brain !== "reactive";
  return (
    <div className="page wrap">
      <div className="page-head">
        <div>
          <div className="section-n">02 · Inside the fly</div>
          <h2>What <em>{selected}</em> sees, and what its connectome does with it</h2>
          <p className="lead" style={{ marginTop: 10 }}>The eye samples the arena along {layout?.az.length ?? "…"} measured ommatidial directions. The circled facets are the target stripe the planner shows the fly. Spikes come from the {frame?.brain === "rewired" ? "rewired" : frame?.brain === "rewired_class" ? "class-rewired" : "real"} MaleCNS sub-circuit, simulated in 1 ms steps.</p>
        </div>
        <div className="row" style={{ flex: "none" }}>
          {Object.keys(DRONE_COLORS).map((n) => (
            <button key={n} className="btn" aria-pressed={n === selected} style={n === selected ? { background: "var(--ink)", color: "var(--paper)" } : {}} onClick={() => setSelected(n)}>{n}</button>))}
        </div>
      </div>
      {!isFly && frame && <p className="empty">This run uses the reactive baseline, so no neurons are simulated. Reset with brain = real to watch the circuit.</p>}
      <div className="grid g-2">
        <div className="card">
          <div className="card-head"><div className="eyebrow">Compound eye</div><span className="small muted">{layout?.source}</span></div>
          <Mosaic layout={layout} mosaic={d?.mosaic} stripe={d?.stripe} />
          <p className="cap">Projection: azimuth across (left eye left), elevation up. White outlines: target stripe. Motion (temporal contrast) L/R: {fmt(d?.eye?.motion_l, 2)} / {fmt(d?.eye?.motion_r, 2)} · near-object cover L/R: {fmt(d?.eye?.cover_l, 3)} / {fmt(d?.eye?.cover_r, 3)}</p>
        </div>
        <div className="card">
          <div className="card-head"><div className="eyebrow">Spike raster · last second</div><span className="small muted">64 neurons, labelled by cell type and side</span></div>
          <Raster neurons={neurons} spikes={spikes} t={frame?.t ?? 0} />
          <p className="cap">Black: visual inputs (LC10, LC4, LPLC2, HS). Blue: AOTU relay. Red: DN (descending neuron) and motor outputs.</p>
        </div>
      </div>
      <div className="grid g-3" style={{ marginTop: 16 }}>
        <div className="card">
          <div className="eyebrow">Descending drive</div>
          <div className="bars">
            <Bar label="left" v={Number(cmd.dn_left ?? 0)} max={200} color="var(--c-rewired)" />
            <Bar label="right" v={Number(cmd.dn_right ?? 0)} max={200} color="var(--accent)" />
            <Bar label="DNp09" v={Number(cmd.dnp09 ?? 0)} max={200} color="var(--ink-2)" />
          </div>
          <p className="cap">Left/right = DNa02 + ½ DNa01 per side. DNp09 (walk command neuron) sets forward speed.</p>
        </div>
        <div className="card">
          <div className="eyebrow">Body command</div>
          <dl className="kv">
            <dt>yaw rate</dt><dd className="num">{fmt((Number(cmd.yaw_rate ?? 0) * 180) / Math.PI, 0)} °/s {Number(cmd.yaw_rate ?? 0) > 0.05 ? "(left)" : Number(cmd.yaw_rate ?? 0) < -0.05 ? "(right)" : ""}</dd>
            <dt>forward</dt><dd className="num">{fmt(Number(cmd.v ?? 0), 2)} m/s</dd>
            <dt>L−R drive</dt><dd className="num">{fmt(-Number(cmd.delta ?? 0), 0)} Hz (after bias)</dd>
            <dt>LC10 L / R</dt><dd className="num">{fmt(rates.LC10_L, 0)} / {fmt(rates.LC10_R, 0)} Hz</dd>
            <dt>AOTU L / R</dt><dd className="num">{fmt(rates.AOTU_L, 0)} / {fmt(rates.AOTU_R, 0)} Hz</dd>
            <dt>altitude</dt><dd>plain PID, not the fly</dd>
          </dl>
        </div>
        <div className="card">
          <div className="eyebrow">Loom and escape</div>
          <div className="loom" style={{ marginBottom: 10 }}><span className={`lamp ${cmd.escape ? "on" : ""}`} />{cmd.escape ? "giant fiber fired: evasive climb and sidestep" : "no escape"}</div>
          <dl className="kv">
            <dt>expansion L/R</dt><dd className="num">{fmt(d?.eye?.expand_l, 2)} / {fmt(d?.eye?.expand_r, 2)}</dd>
            <dt>LC4 L / R</dt><dd className="num">{fmt(rates.LC4_L, 0)} / {fmt(rates.LC4_R, 0)} Hz</dd>
            <dt>DNp01 L / R</dt><dd className="num">{fmt(rates.DNp01_L, 0)} / {fmt(rates.DNp01_R, 0)} Hz</dd>
            <dt>GF spikes</dt><dd className="num">{String(cmd.giant_fiber_spikes ?? 0)} in last 40 ms</dd>
          </dl>
          <p className="cap">DNp01 is the giant fiber. Looming input reaches it through LC4/LPLC2 as in FlyWorld's calibration.</p>
        </div>
      </div>
    </div>
  );
}
