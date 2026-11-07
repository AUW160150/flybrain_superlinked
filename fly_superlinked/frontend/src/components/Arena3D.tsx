import { Canvas } from "@react-three/fiber";
import { Line, OrbitControls, Text } from "@react-three/drei";
import { useMemo } from "react";
import type { Drone } from "../lib/sim";
import { DRONE_COLORS } from "../lib/sim";

// Simulator frame: x east, y north, z up. three.js: x east, y up, z south  ->  (x, z, -y)
const P = (x: number, y: number, z = 0): [number, number, number] => [x, z, -y];

export type ArenaInfo = {
  half: number; ceiling: number; home: [number, number]; home_radius: number;
  pond: { center: [number, number]; radius: number };
  gust: { min: [number, number]; max: [number, number] };
  landmarks: { name: string; label: string; color: string; goal: [number, number];
    prims: { kind: string; center: number[]; size: number[]; color: number[] }[] }[];
};

function rgb(c: number[]) { return `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`; }

function DroneMesh({ d, selected }: { d: Drone; selected: boolean }) {
  const col = DRONE_COLORS[d.name] ?? "#888";
  const [x, y, z] = d.pos;
  return (
    <group position={P(x, y, z)} rotation={[0, d.yaw, 0]}>
      <mesh><boxGeometry args={[0.9, 0.18, 0.9]} /><meshStandardMaterial color={col} /></mesh>
      {[[0.55, 0.55], [-0.55, 0.55], [0.55, -0.55], [-0.55, -0.55]].map(([a, b], i) => (
        <mesh key={i} position={[a, 0.12, b]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.18, 0.3, 20]} /><meshBasicMaterial color={col} transparent opacity={0.8} />
        </mesh>))}
      <mesh position={[0.65, 0, 0]}><coneGeometry args={[0.16, 0.4, 10]} /><meshStandardMaterial color="#1a1a12" /></mesh>
      {selected && <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.12, 0]}><ringGeometry args={[1.0, 1.15, 40]} /><meshBasicMaterial color={col} /></mesh>}
      <Text position={[0, 1.0, 0]} fontSize={0.7} color={col} anchorX="center" outlineWidth={0.04} outlineColor="#f4f1ea">{d.name}</Text>
    </group>
  );
}

export default function Arena3D({ arena, drones, selected, showGust, dark }: { arena: ArenaInfo | null; drones: Drone[]; selected: string; showGust: boolean; dark: boolean }) {
  const ground = dark ? "#2b3323" : "#c9d3b2";
  const grid = useMemo(() => {
    const lines: [number, number, number][][] = [];
    if (!arena) return lines;
    for (let v = -arena.half; v <= arena.half; v += 5) {
      lines.push([P(v, -arena.half, 0.01), P(v, arena.half, 0.01)]);
      lines.push([P(-arena.half, v, 0.01), P(arena.half, v, 0.01)]);
    }
    return lines;
  }, [arena]);
  if (!arena) return <div className="empty">Loading arena…</div>;
  const g = arena.gust;
  return (
    <Canvas camera={{ position: [26, 30, 34], fov: 42 }} dpr={[1, 2]}>
      <color attach="background" args={[dark ? "#1e1e16" : "#ebe7dc"]} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[20, 40, 10]} intensity={1.1} />
      <mesh rotation={[-Math.PI / 2, 0, 0]}><planeGeometry args={[arena.half * 2, arena.half * 2]} /><meshStandardMaterial color={ground} /></mesh>
      {grid.map((l, i) => <Line key={i} points={l} color={dark ? "#3a4430" : "#b3bf98"} lineWidth={1} />)}
      {/* pond: no-fly zone */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={P(arena.pond.center[0], arena.pond.center[1], 0.03)}>
        <circleGeometry args={[arena.pond.radius, 48]} /><meshStandardMaterial color="#29507f" />
      </mesh>
      <Text position={P(arena.pond.center[0], arena.pond.center[1], 0.2)} rotation={[-Math.PI / 2, 0, 0]} fontSize={0.9} color="#f4f1ea">pond · no-fly</Text>
      {/* home pad */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={P(arena.home[0], arena.home[1], 0.04)}>
        <ringGeometry args={[arena.home_radius - 0.3, arena.home_radius, 40]} /><meshBasicMaterial color={dark ? "#ede9df" : "#1a1a12"} />
      </mesh>
      <Text position={P(arena.home[0], arena.home[1] - 2.6, 0.1)} rotation={[-Math.PI / 2, 0, 0]} fontSize={0.9} color={dark ? "#ede9df" : "#1a1a12"}>home</Text>
      {/* gust zone: drawn only once a drone has felt it */}
      {showGust && (
        <group>
          <mesh position={P((g.min[0] + g.max[0]) / 2, (g.min[1] + g.max[1]) / 2, 4)}>
            <boxGeometry args={[g.max[0] - g.min[0], 8, g.max[1] - g.min[1]]} />
            <meshStandardMaterial color="#cf9a2f" transparent opacity={0.16} depthWrite={false} />
          </mesh>
          <Text position={P((g.min[0] + g.max[0]) / 2, (g.min[1] + g.max[1]) / 2, 8.6)} fontSize={0.8} color="#b5832a">gust zone (discovered)</Text>
        </group>
      )}
      {arena.landmarks.map((lm) => lm.prims.map((p, i) =>
        p.kind === "cyl" ? (
          <mesh key={lm.name + i} position={P(p.center[0], p.center[1], p.center[2] + p.size[1] / 2)}>
            <cylinderGeometry args={[p.size[0], p.size[0], p.size[1], 24]} /><meshStandardMaterial color={rgb(p.color)} />
          </mesh>
        ) : (
          <mesh key={lm.name + i} position={P(p.center[0], p.center[1], p.center[2])}>
            <boxGeometry args={[p.size[0] * 2, p.size[2] * 2, p.size[1] * 2]} /><meshStandardMaterial color={rgb(p.color)} />
          </mesh>
        )))}
      {arena.landmarks.filter((l) => !l.name.startsWith("green_tree")).map((lm) => (
        <Text key={lm.name} position={P(lm.goal[0], lm.goal[1], 12)} fontSize={0.9} color={dark ? "#ede9df" : "#1a1a12"}>{lm.label}</Text>
      ))}
      {drones.map((d) => (
        <group key={d.name}>
          {d.trail.length > 1 && <Line points={d.trail.map((p) => P(p[0], p[1], p[2]))} color={DRONE_COLORS[d.name]} lineWidth={2} transparent opacity={0.85} />}
          {d.active && !d.done && d.waypoints.length > d.wp_i && (
            <Line points={[P(d.pos[0], d.pos[1], d.pos[2]), ...d.waypoints.slice(d.wp_i).map((w) => P(w[0], w[1], 4))]}
              color={DRONE_COLORS[d.name]} lineWidth={1} dashed dashSize={0.6} gapSize={0.5} transparent opacity={0.6} />
          )}
          <DroneMesh d={d} selected={d.name === selected} />
        </group>
      ))}
      <OrbitControls target={[0, 0, 0]} maxPolarAngle={Math.PI / 2.1} minDistance={12} maxDistance={90} />
    </Canvas>
  );
}
