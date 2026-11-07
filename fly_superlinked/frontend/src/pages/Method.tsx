import { useEffect, useState } from "react";
import { api, fmt } from "../lib/sim";

export default function Method() {
  const [st, setSt] = useState<any>(null);
  const [probe, setProbe] = useState<any>(null);
  useEffect(() => { api("/api/status").then(setSt).catch(() => {}); api("/api/bench/probe").then(setProbe).catch(() => {}); }, []);
  const real = probe?.brains?.real, rew = probe?.brains?.rewired;
  const row = (k: string) => real?.[k];
  return (
    <div className="page wrap prose">
      <div className="section-n">05 · Method and limitations</div>
      <h2>What the connectome <em>does</em> and does not control</h2>
      <div className="grid g-2" style={{ marginTop: 20 }}>
        <div className="card">
          <h3>Controlled by the fly circuit</h3>
          <ul>
            <li><b>Turning.</b> Yaw rate comes from the left minus right firing of DNa02 and DNa01 (DN = descending neuron), through FlyWorld's decoder.</li>
            <li><b>Forward speed.</b> Read from the firing rate of DNp09, a walking command neuron.</li>
            <li><b>Escape.</b> When looming input fires DNp01 (the giant fiber), the drone climbs and sidesteps away from the threat side.</li>
            <li><b>Goal following.</b> The target landmark appears as a bright vertical stripe in the eye. The stripe drives LC10 visual projection neurons on that eye's side, which reach DNa02 through AOTU (anterior optic tubercle) neurons.</li>
          </ul>
          <h3>Engineered, not the fly</h3>
          <ul>
            <li><b>Altitude hold</b> is a plain PID controller.</li>
            <li><b>The 'go' current</b> on DNp09 while a mission is active (as FlyWorld's walk button).</li>
            <li><b>Which target is shown</b>: chosen by the planner and route planner, around the pond, landmarks and remembered gusts.</li>
            <li><b>Decoder gains</b> (dead zone 4 Hz, tanh scale 20 Hz, LC10 current 0.3) chosen on tuning seeds 100–102 with the real circuit, then frozen.</li>
            <li><b>Gust sensing</b>: a drone notices a gust from the gap between commanded and measured velocity, not from the brain.</li>
          </ul>
        </div>
        <div className="card">
          <h3>Data and code sources</h3>
          <ul>
            <li><b>MaleCNS v1.0</b> (Berg et al., 2026; FlyEM / Janelia, Cambridge, MRC LMB, Google; CC BY 4.0). The controller uses FlyWorld's 10,203-neuron sensory-to-descending sub-circuit (401,680 edges with at least 5 synapses), unchanged.</li>
            <li><b>LIF dynamics</b> from FlyWorld: 1 ms steps, τ = 20 ms, threshold 1, 2 ms refractory, weight = signed synapse count × 0.005.</li>
            <li><b>Eye</b>: {st ? "1,709" : "1,709"} microCT ommatidial directions from reiserlab/eyemap_T4 (as in FlyVision), hash-checked. The eye is from a female specimen and is not registered to MaleCNS photoreceptors; the eye-to-LC10 mapping is a retinotopic stand-in by side only.</li>
            <li><b>Models</b>: {st ? Object.entries(st.models).map(([k, v]) => `${k} = ${v}`).join("; ") : "…"}. Memory backend: {st?.memory_backend ?? "…"}.</li>
          </ul>
          <h3>Controls</h3>
          <ul>
            <li><b>Rewired, whole graph</b>: every edge keeps its source and signed weight; targets are shuffled across the graph so each neuron's in- and out-degree is unchanged (fixed seed).</li>
            <li><b>Rewired, within classes</b>: targets shuffled only among edges joining the same superclass × side blocks, the style of the owner's Project 08 control.</li>
            <li><b>Reactive baseline</b>: hand-written steer-to-target with obstacle repulsion from the same eye.</li>
            <li>The walk-coupled turn bias is re-measured for each wiring with the same procedure. Same seeds for every arm.</li>
          </ul>
        </div>
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <h3>Measured before the controller was built</h3>
        <p className="cap">Probe: 'go' current on DNp09, then one side of a visual group stimulated for 300 ms (bench/probe_circuit.py). Rates in Hz.</p>
        {real ? (
          <div className="tablewrap"><table>
            <thead><tr><th>condition</th><th className="num">DNa02 L</th><th className="num">DNa02 R</th><th className="num">DNa01 L</th><th className="num">DNa01 R</th><th className="num">DNp01 L</th><th className="num">DNp01 R</th><th className="num">rewired DNa02 L / R</th></tr></thead>
            <tbody>{["go", "go+LC4_L", "go+LC4_R", "go+LPLC2_L", "go+HS_L", "go+LC11_L"].filter((k) => row(k)).map((k) => (
              <tr key={k}><td className="mono small">{k}</td>{["DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp01_L", "DNp01_R"].map((n) => <td key={n} className="num">{fmt(row(k)[n], 0)}</td>)}
                <td className="num">{rew?.[k] ? `${fmt(rew[k].DNa02_L, 0)} / ${fmt(rew[k].DNa02_R, 0)}` : "–"}</td></tr>))}</tbody>
          </table></div>) : <div className="empty">Probe results not found.</div>}
        {st?.calibration && <p className="cap">Turn-bias calibration for the current wiring: go alone gives DNp09 {fmt(st.calibration.go_dnp09_hz, 0)} Hz, left drive {fmt(st.calibration.go_left_hz, 0)} Hz, right {fmt(st.calibration.go_right_hz, 0)} Hz (FlyWorld's published bias: {fmt(st.calibration.flyworld_published_bias, 3)} per Hz).</p>}
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <h3>Known limits</h3>
        <ul>
          <li><b>Thin left input.</b> The sub-circuit holds 3 left-side LC10 cells and 27 right-side ones (an artifact of how FlyWorld extracted it), so left turns are weak. The real circuit reaches targets with right-handed loops.</li>
          <li><b>Not the owner's steering circuit.</b> Project 08's 443-neuron LC10a → AOTU → DN graph is not in the flybrain repo; this uses the same pathway as it exists inside the FlyWorld subset.</li>
          <li><b>No training.</b> The owner's prior result (rewired ≈ real) came from circuits trained by imitation, which can compensate for wiring. Here nothing is trained, so the built-in pathways carry the behaviour; that is a different question, and the large real-vs-rewired gap here should be read in that light. The rewired graphs are also not activity-matched.</li>
          <li><b>Kinematic drones.</b> No rotor dynamics, no pitch or roll; collisions push the drone back but are not fatal.</li>
          <li><b>Labels from geometry.</b> Perception keyframes are labelled from the simulator's geometry, not by people; distance classes are thresholds on depth.</li>
          <li><b>Precomputed.</b> Results and replays are precomputed and dated; live runs on the free Render plan can be slow.</li>
        </ul>
      </div>
    </div>
  );
}
