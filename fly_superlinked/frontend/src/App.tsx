import { useEffect, useState } from "react";
import { ArenaInfo } from "./components/Arena3D";
import { api, useSim } from "./lib/sim";
import InsideFly from "./pages/InsideFly";
import Method from "./pages/Method";
import MissionControl from "./pages/MissionControl";
import Results from "./pages/Results";
import SwarmMemory from "./pages/SwarmMemory";

const PAGES = [
  ["mission", "Mission control"], ["fly", "Inside the fly"], ["memory", "Swarm memory"], ["results", "Results"], ["method", "Method and limitations"],
] as const;

function useHash() {
  const [h, setH] = useState(location.hash.slice(1) || "mission");
  useEffect(() => { const f = () => setH(location.hash.slice(1) || "mission"); addEventListener("hashchange", f); return () => removeEventListener("hashchange", f); }, []);
  return h;
}

function useTheme() {
  const [theme, setTheme] = useState<string>(() => { try { return localStorage.getItem("theme") || ""; } catch { return ""; } });
  const sysDark = matchMedia("(prefers-color-scheme: dark)").matches;
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme; else delete document.documentElement.dataset.theme;
    try { localStorage.setItem("theme", theme); } catch { /* storage unavailable */ }
  }, [theme]);
  const dark = theme ? theme === "dark" : sysDark;
  return { dark, toggle: () => setTheme(dark ? "light" : "dark") };
}

export default function App() {
  const page = useHash();
  const { dark, toggle } = useTheme();
  const [selected, setSelected] = useState("courier");
  const sim = useSim(selected);
  const [arena, setArena] = useState<ArenaInfo | null>(null);
  const [layout, setLayout] = useState<any>(null);
  const [neurons, setNeurons] = useState<{ type: string; side: string }[]>([]);
  useEffect(() => {
    api("/api/arena").then(setArena).catch(() => {});
    api("/api/eye/layout").then(setLayout).catch(() => {});
    api("/api/status").then((s) => setNeurons(s.raster_neurons)).catch(() => {});
  }, []);
  const onReplay = async (name: string | null) => {
    if (!name) { sim.setReplay(null); return; }
    const r = await api(`/api/replays/${name}`);
    sim.setReplay({ name, frames: r.frames });
  };
  const mode = sim.replay ? "replay" : "live";
  return (
    <>
      <nav className="nav"><div className="wrap">
        <a className="wordmark" href="#mission"><i />Flybrain Swarm</a>
        <div className="tabs" role="tablist">
          {PAGES.map(([id, label], i) => <a key={id} href={`#${id}`} aria-current={page === id ? "page" : undefined}><span className="n">0{i + 1}</span>{label}</a>)}
        </div>
        <div className="status">
          <span className={`badge ${mode === "live" && sim.connected ? "live" : "replay"}`} title={mode === "replay" ? `Replay: ${sim.replay?.name}` : "Live simulation stream"}>
            <span className="dot" />{mode === "replay" ? `Replay · ${sim.replay?.name}` : sim.connected ? "Live" : "Connecting"}</span>
          <button className="themebtn" onClick={toggle} aria-label="Toggle light or dark mode">{dark ? "light" : "dark"}</button>
        </div>
      </div></nav>
      <main>
        {page === "mission" && <MissionControl frame={sim.frame} arena={arena} selected={selected} setSelected={setSelected} log={sim.log} dark={dark} onReplay={onReplay} replaying={!!sim.replay} />}
        {page === "fly" && <InsideFly frame={sim.frame} selected={selected} setSelected={setSelected} layout={layout} neurons={neurons} spikes={sim.spikes} />}
        {page === "memory" && <SwarmMemory frame={sim.frame} mem={sim.mem} />}
        {page === "results" && <Results />}
        {page === "method" && <Method />}
      </main>
      <footer><div className="wrap">
        Flybrain Swarm · MaleCNS v1.0 (CC BY 4.0) · built on <a href="https://flybrain-q2h5.onrender.com/">Flybrain</a> · open models on Superlinked SIE · memory on cognee.
      </div></footer>
    </>
  );
}
