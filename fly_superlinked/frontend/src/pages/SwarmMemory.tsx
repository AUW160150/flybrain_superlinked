import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, SimulationNodeDatum } from "d3-force";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, DRONE_COLORS, Frame, MemEvent } from "../lib/sim";

type GNode = SimulationNodeDatum & { id: string; label: string; dataset: string; by: string; type?: string };
type GEdge = { source: string | GNode; target: string | GNode; label?: string };

const BY_COLOR: Record<string, string> = { ...DRONE_COLORS, operator: "#4b4a40", shared: "#87857a", entity: "#87857a", error: "#b3261e" };

function kindPill(e: MemEvent) {
  if (e.kind === "denied") return <span className="pill denied">denied</span>;
  if (e.kind === "write") return <span className="pill allowed">write ✓</span>;
  if (e.kind === "read") return <span className="pill read">read</span>;
  return <span className="pill other">{e.kind}</span>;
}

function Graph({ data }: { data: { nodes: GNode[]; edges: GEdge[] } | null }) {
  const ref = useRef<SVGSVGElement>(null);
  const [nodes, setNodes] = useState<GNode[]>([]);
  const [edges, setEdges] = useState<GEdge[]>([]);
  const [hover, setHover] = useState<{ n: GNode; x: number; y: number } | null>(null);
  const W = 760, H = 520;
  useEffect(() => {
    if (!data) return;
    const ns = data.nodes.slice(0, 260).map((n) => ({ ...n }));
    const ids = new Set(ns.map((n) => n.id));
    const es = data.edges.filter((e) => ids.has(String(e.source)) && ids.has(String(e.target))).map((e) => ({ ...e }));
    const sim = forceSimulation<GNode>(ns)
      .force("link", forceLink<GNode, any>(es).id((d: any) => d.id).distance(38).strength(0.5))
      .force("charge", forceManyBody().strength(-60))
      .force("center", forceCenter(W / 2, H / 2))
      .force("collide", forceCollide(9))
      .stop();
    for (let i = 0; i < 260; i++) sim.tick();
    ns.forEach((n) => { n.x = Math.max(12, Math.min(W - 12, n.x ?? 0)); n.y = Math.max(12, Math.min(H - 12, n.y ?? 0)); });
    setNodes(ns); setEdges(es);
  }, [data]);
  if (!data) return <div className="empty">Loading graph…</div>;
  if (data.nodes.length === 0) return <div className="empty">The graph is empty. Fly the scouts to fill swarm_map.</div>;
  return (
    <div style={{ position: "relative" }}>
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", background: "var(--paper-2)", borderRadius: 4 }} role="img" aria-label="Knowledge graph of swarm_map and outcomes">
        {edges.map((e, i) => {
          const s = e.source as GNode, t = e.target as GNode;
          return <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y} stroke="var(--line)" strokeWidth={1} />;
        })}
        {nodes.map((n) => (
          <circle key={n.id} cx={n.x} cy={n.y} r={n.type === "DocumentChunk" || n.dataset === "outcomes" && n.by === "operator" ? 6 : 5}
            fill={BY_COLOR[n.by] ?? "#87857a"} stroke="var(--paper)" strokeWidth={2}
            onMouseMove={(ev) => setHover({ n, x: ev.clientX, y: ev.clientY })} onMouseLeave={() => setHover(null)} />
        ))}
        {nodes.filter((n) => n.type && !["DocumentChunk", "TextDocument", "TextSummary"].includes(n.type)).slice(0, 40).map((n) => (
          <text key={"t" + n.id} x={(n.x ?? 0) + 7} y={(n.y ?? 0) + 3} fontSize={9.5} fontFamily="IBM Plex Mono, monospace" fill="var(--ink-2)">{n.label.slice(0, 22)}</text>
        ))}
      </svg>
      {hover && <div className="tip" style={{ left: hover.x + 12, top: hover.y + 12 }}>{hover.n.label}<br />{hover.n.type ? `${hover.n.type} · ` : ""}{hover.n.dataset} · from {hover.n.by}</div>}
    </div>
  );
}

export default function SwarmMemory({ frame, mem }: { frame: Frame | null; mem: MemEvent[] }) {
  const [graph, setGraph] = useState<any>(null);
  const [filter, setFilter] = useState("all");
  const load = async () => setGraph(await api("/api/memory/graph").catch(() => ({ nodes: [], edges: [] })));
  useEffect(() => { load(); }, []);
  const counts = useMemo(() => ({ write: mem.filter((e) => e.kind === "write").length, denied: mem.filter((e) => e.kind === "denied").length, read: mem.filter((e) => e.kind === "read").length }), [mem]);
  const shown = [...mem].reverse().filter((e) => filter === "all" || e.kind === filter);
  return (
    <div className="page wrap">
      <div className="page-head">
        <div>
          <div className="section-n">03 · Swarm memory</div>
          <h2>Shared memory with <em>real</em> permissions</h2>
          <p className="lead" style={{ marginTop: 10 }}>Every drone is its own cognee user. Scouts may write to <code>swarm_map</code>; everyone may read it. Only the operator writes <code>outcomes</code>. Each drone's <code>private_*</code> dataset is its own. A denied write below was rejected by cognee itself.</p>
        </div>
        <span className="badge">backend: {frame?.memory_backend ?? "…"}</span>
      </div>
      <div className="grid g-2">
        <div className="card">
          <div className="card-head"><div className="eyebrow">Memory events</div>
            <div className="row" style={{ flex: "none" }}>
              {["all", "write", "denied", "read", "feedback", "improve"].map((f) => (
                <button key={f} className="chip" aria-pressed={f === filter} style={f === filter ? { borderColor: "var(--ink)" } : {}} onClick={() => setFilter(f)}>{f}</button>))}
            </div>
          </div>
          <p className="small muted" style={{ marginTop: 0 }}>{counts.write} writes · {counts.denied} denied · {counts.read} reads this session</p>
          <div className="feed">
            {shown.length === 0 && <div className="empty">No memory events yet. Send a mission on Mission control.</div>}
            {shown.map((e, i) => (
              <div key={i} className={`ev ${e.kind === "denied" ? "denied" : ""}`}>
                <div>{kindPill(e)}</div>
                <div><div className="who" style={{ color: BY_COLOR[e.user] }}>{e.user}</div><div className="ds">{e.dataset}</div></div>
                <div>{e.text.replace(/\[fact [^\]]+\]/g, "").slice(0, 260)}
                  {e.kind === "denied" && <div className="small" style={{ color: "var(--bad)", marginTop: 4 }}>{String(e.meta?.reason ?? "").slice(0, 160)}</div>}
                  {e.kind === "read" && <div className="small muted">{String(e.meta?.hits ?? 0)} hits</div>}
                </div>
              </div>))}
          </div>
        </div>
        <div className="card">
          <div className="card-head"><div className="eyebrow">Knowledge graph · swarm_map + outcomes</div><button className="btn" onClick={load}>Refresh</button></div>
          <Graph data={graph} />
          <p className="cap">{graph?.source ?? ""}. Colour = which drone contributed the text a node came from: {Object.entries(DRONE_COLORS).map(([n, c]) => <span key={n} style={{ marginRight: 10 }}><span className="sw" style={{ background: c }} />{n}</span>)}<span className="sw" style={{ background: "#4b4a40" }} />operator. Grey: shared or not traceable to one drone.</p>
        </div>
      </div>
    </div>
  );
}
