import { useEffect, useRef, useState } from "react";

export type Drone = {
  name: string; role: string; pos: number[]; yaw: number; speed: number; active: boolean; done: boolean; delivered: boolean;
  target: { x: number; y: number; label: string; kind: string } | null; waypoints: [number, number, string, string][]; wp_i: number;
  collisions: number; gust_entries: number; pond_entries: number; in_gust: boolean; trail: number[][];
};
export type MemEvent = { kind: string; user: string; dataset: string; text: string; ok: boolean; t: number; meta: Record<string, unknown> };
export type Detail = {
  name: string; brain: string; mosaic?: string; stripe?: number[];
  eye?: Record<string, number>; cmd?: Record<string, number | boolean>; rates?: Record<string, number>; spikes?: [number, number][];
};
export type Frame = {
  t: number; brain: string; seed: number; memory_mode: string; mode: "live" | "replay"; memory_backend: string; sie: boolean; frontier: string;
  drones: Drone[]; plan: unknown; plan_source?: string; detail: Detail | null; place_matches: Record<string, unknown>[]; gust_known: boolean;
  memory_events: MemEvent[]; log: { t: number; msg: string }[]; calls: { job: string; ok: boolean; model: string; latency_s: number; cost_usd: number; t: number }[];
};

export async function api<T = any>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, body === undefined ? undefined : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}

/** One live WebSocket (or a replay player) feeding the whole app. Memory events, log lines and spikes accumulate. */
export function useSim(selected: string) {
  const [frame, setFrame] = useState<Frame | null>(null);
  const [connected, setConnected] = useState(false);
  const [replay, setReplay] = useState<{ name: string; frames: Frame[] } | null>(null);
  const memRef = useRef<MemEvent[]>([]);
  const logRef = useRef<{ t: number; msg: string }[]>([]);
  const callsRef = useRef<Frame["calls"]>([]);
  const spikeRef = useRef<{ t: number; n: number }[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const [, bump] = useState(0);

  const ingest = (f: Frame) => {
    if (f.memory_events?.length) memRef.current = [...memRef.current, ...f.memory_events].slice(-300);
    if (f.log?.length) logRef.current = [...logRef.current, ...f.log].slice(-200);
    if (f.calls?.length) callsRef.current = [...callsRef.current, ...f.calls].slice(-200);
    if (f.detail?.spikes) {
      const base = Math.round(f.t * 1000) - 40;
      const add = f.detail.spikes.map(([n, ms]) => ({ t: base + ms, n }));
      const cutoff = Math.round(f.t * 1000) - 1000;
      spikeRef.current = [...spikeRef.current.filter((s) => s.t >= cutoff && s.t <= f.t * 1000 + 50), ...add];
    }
    setFrame(f);
  };

  useEffect(() => {
    if (replay) return;
    let alive = true;
    let retry: number | undefined;
    const open = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws/sim`);
      wsRef.current = ws;
      ws.onopen = () => { setConnected(true); ws.send(JSON.stringify({ select: selected })); };
      ws.onclose = () => { setConnected(false); if (alive) retry = window.setTimeout(open, 1500); };
      ws.onmessage = (m) => ingest(JSON.parse(m.data));
    };
    open();
    return () => { alive = false; clearTimeout(retry); wsRef.current?.close(); };
  }, [replay]);

  useEffect(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ select: selected }));
  }, [selected]);

  useEffect(() => {
    if (!replay) return;
    let i = 0;
    memRef.current = []; logRef.current = []; callsRef.current = []; spikeRef.current = [];
    const id = window.setInterval(() => {
      if (i >= replay.frames.length) { i = 0; memRef.current = []; logRef.current = []; }
      ingest({ ...replay.frames[i], mode: "replay" });
      i += 1;
    }, 100);
    return () => clearInterval(id);
  }, [replay]);

  const clearFeeds = () => { memRef.current = []; logRef.current = []; callsRef.current = []; bump((x) => x + 1); };

  return { frame, connected, mem: memRef.current, log: logRef.current, calls: callsRef.current, spikes: spikeRef.current, replay, setReplay, clearFeeds };
}

export const DRONE_COLORS: Record<string, string> = { "scout-1": "#2f6db0", "scout-2": "#4e6b3b", courier: "#c2412a" };
export const fmt = (x: number | null | undefined, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? "–" : x.toFixed(d));
