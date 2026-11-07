# Prompt 1: Flybrain Swarm (fly connectome flies a drone swarm with shared memory)

Paste everything below this line into Claude Code (or your coding agent) from an empty working folder.

---

You are building a hackathon project for LOCALHOST NYC (Saturday, November 7, 2026). Time is short, so work in milestones and keep the app demoable after every milestone. Commit after each milestone and append a short entry to `LOG.md`.

## What we are building

**Flybrain Swarm**: a small swarm of simulated drones, each steered by a real fruit fly connectome circuit (from the MaleCNS v1.0 connectome) driven by a simulated compound eye. The drones share what they learn through a cognee knowledge graph with per-drone permissions, and every routine "agent job" (turning a mission sentence into a plan, turning what the drone sees into text facts, recognizing places it has seen before) runs on small open models through Superlinked SIE Serverless instead of a frontier model.

The pitch in one line: *a fly brain flies, small open models see and plan, and the swarm remembers together.*

## Prize targets (design every choice around these)

1. **cognee sponsor prize: "best project where agents share what they learn between each other."** Several agents with shared memory and clear read/write permissions, plus a loop where the system learns from outcomes, feedback, and fixes.
2. **Superlinked sponsor prize: "best project that moves an agent's everyday jobs off a frontier model and onto SIE."** Judges want a measured comparison of quality against cost.

Both need visible proof in the UI, not only in the README.

## Owner and existing work

The owner (Rodela) already maintains a fly connectome research site at https://flybrain-q2h5.onrender.com/ with repo `github.com/prove-ai/flybrain` (deployed on Render). It contains, among other things:
- **FlyVision** (pygame, numpy): a first-person world sampled through 1,709 eye directions from a microCT reference fly.
- **Steering circuit** (PyTorch): a 443-neuron MaleCNS circuit used as a robot target-tracking controller. Its loops act as a short memory. Rewired copies performed about as well as the real wiring.
- **FlyWorld** (Python, panda3d, torch): a 10,203-neuron escape-and-feeding sub-circuit with leaky integrate-and-fire neurons in 1 ms steps, where a loom stimulus fires the giant fiber (escape) and DNp09 stimulation drives walking.

**Step 0 before writing code:** clone `github.com/prove-ai/flybrain` (ask me for access if it fails), read its README and the code for FlyVision, the steering circuit, and FlyWorld, and reuse their connectome loading, eye sampling, and LIF simulation code instead of rewriting them. Write a short `NOTES.md` listing which files and functions you will reuse and the neuron types available as visual inputs and descending (motor) outputs in each subset. Do not invent neuron IDs or cell types; use only what the existing subsets define.

## Scientific honesty rules (non-negotiable)

- The owner's prior result is that rewired controls perform about as well as the real circuit. Every experiment here must include a **degree-preserving rewired connectome** control and a **simple reactive baseline controller**, run on the same seeds.
- Report whatever the numbers say, including null or negative results. Never hardcode or cherry-pick results. The UI has a "Method and limitations" panel that states plainly what the fly circuit does and does not control (for example, altitude hold is a normal PID, not the fly brain).
- Label anything precomputed as precomputed.

## Docs to read before using each library

Superlinked SIE:
- SIE Serverless overview and pricing: https://superlinked.com/serverless
- Python SDK reference: https://superlinked.com/docs/reference/sdk (package `sie-sdk`, `from sie_sdk import SIEClient`, `from sie_sdk.types import Item`; methods `encode`, `score`, `extract`, `generate`)
- HTTP API reference: https://superlinked.com/docs/reference/api (also `POST /v1/chat/completions`, OpenAI style, supports `response_format` for structured output)
- Model catalog for exact model IDs: https://superlinked.com/models
- Cookbook for idioms (`rules.yaml` one line per model, `SIE_API_KEY`, `SIE_BASE_URL`): https://github.com/superlinked/cookbook

cognee:
- Core API (remember/recall/improve/forget, and add/cognify/search/memify): https://docs.cognee.ai/python-api.md and https://docs.cognee.ai/core-concepts/main-operations/recall.md
- Permissions and multi-user mode: https://docs.cognee.ai/setup-configuration/permissions.md, https://docs.cognee.ai/guides/permission-snippets.md, https://docs.cognee.ai/examples/multi-tenant-access-control.md
- Datasets and node sets: https://docs.cognee.ai/core-concepts/further-concepts/datasets.md, https://docs.cognee.ai/core-concepts/further-concepts/node-sets.md
- Learning loops: https://docs.cognee.ai/guides/feedback-system.md, https://docs.cognee.ai/guides/self-improving-memory-quickstart.md, https://docs.cognee.ai/core-concepts/main-operations/improve.md
- LLM and embedding providers: https://docs.cognee.ai/setup-configuration/llm-providers.md, https://docs.cognee.ai/setup-configuration/embedding-providers.md
- Deployment: https://docs.cognee.ai/guides/deploy-rest-api-server.md

If the docs disagree with anything in this prompt, follow the docs and note the difference in `LOG.md`.

## Architecture

```
backend/   Python 3.11, FastAPI, numpy, torch (CPU), sie-sdk, cognee
  sim/        arena, quadrotor model, compound eye ray casting
  brain/      connectome loader, LIF simulator, eye->input and DN->command mappings, rewired control
  agents/     drone agent loop, mission planner (SIE), perception-to-facts (SIE), place recognition (SIE)
  memory/     cognee setup, users, datasets, permissions, write/recall helpers, feedback loop
  bench/      SIE vs frontier benchmark, controller experiments, ablations
  api/        REST + WebSocket
frontend/  React + Vite + TypeScript + react-three-fiber
render.yaml, Dockerfile, README.md, LOG.md, NOTES.md
```

### 1. Arena and drones (`sim/`)
- A 40 m x 40 m x 12 m outdoor arena with distinct, brightly colored landmarks (a red tower, a blue arch, a yellow cube cluster, green trees as cylinders), a pond marked as a no-fly zone, a wind gust zone (unknown to drones at the start), and a home pad.
- Quadrotor as a kinematic model with velocity and yaw-rate commands, simple drag, gust disturbance, 50 Hz physics. Collisions with landmarks and the ground are counted, not fatal.
- Three drones: `scout-1`, `scout-2`, `courier`.

### 2. Compound eye (`sim/eye.py`)
- Reuse FlyVision's eye direction set (1,709 directions) if available; otherwise sample about 700 directions on a sphere with a Fibonacci lattice. Ray cast against the arena primitives to get per-ommatidium color and luminance at 25 Hz.
- Compute per-ommatidium motion (temporal contrast) and a looming signal (rapid expansion of a dark object).
- Render the eye output two ways: a hexagonal mosaic for the UI, and an equirectangular image (for example 128 x 64 PNG) for SIE vision models.

### 3. Fly brain controller (`brain/`)
- Load the existing steering circuit subset (and, if cheap enough, the escape sub-circuit). Run LIF dynamics in 1 ms steps on CPU.
- Map the eye onto the subset's visual input neurons retinotopically (left eye drives left-side inputs, azimuth bins onto columns), using whatever visual input types the subset actually contains.
- Read out descending neurons: left-right activity difference becomes yaw rate, total drive becomes forward speed. If the escape sub-circuit is loaded and the giant fiber fires on a loom, trigger an evasive climb and sidestep. Altitude hold and goal-height changes are a plain PID, and the UI says so.
- Goal following: the target landmark's direction is presented as a bright visual stripe in the eye input (the fly steers toward it, like the owner's pursuit work). The planner chooses which landmark or waypoint is the current target.
- Controls: `brain=real`, `brain=rewired` (degree-preserving shuffle, fixed seed), `brain=reactive` (hand-written steer-toward-target with obstacle repulsion). Same seeds for all.

### 4. SIE jobs (moving everyday agent work off a frontier model) (`agents/`)
Use `SIEClient(base_url=os.environ.get("SIE_BASE_URL", "https://api.superlinked.com"), api_key=os.environ["SIE_API_KEY"])`. Put every model choice in `backend/models.yaml`, one line per job, so models can be swapped in one line.
1. **Mission planner:** turn an operator sentence ("Scouts, map the north side. Courier, take the package to the red tower and come home, avoid the pond") into a JSON plan matching a strict schema (per drone: ordered goals with landmark names, constraints, return-home flag). Use SIE chat completions or `generate` with a JSON schema on an open model from the catalog (for example the Qwen chat model SIE lists). Validate with pydantic; on invalid JSON retry once, then fail loudly.
2. **Perception to facts:** at keyframes (every 2 s or on a new landmark), send the eye's equirectangular image plus pose to an SIE image-question or vision model and get a short structured observation (`{landmark, bearing, approx_distance, hazard}`). These observations are what drones write to shared memory.
3. **Place recognition:** embed keyframe images with an SIE image embedding model (for example SigLIP, via `encode`) and store vectors per drone; on revisit, nearest neighbor match plus optional rerank tells the drone "I have seen this place before" and corrects drift in its pose estimate. Show matches in the UI.
4. **Benchmark (the evidence for the Superlinked prize):** write 30 operator missions with hand-written gold JSON plans in `bench/missions.jsonl`, and 40 labeled keyframes for perception. Run each job on (a) the SIE open model and (b) a frontier model if `FRONTIER_API_KEY` is set (OpenAI-compatible or Anthropic; make it pluggable). Record exact-match and field-level accuracy, median latency, and cost per 1,000 calls (use each provider's published per-token or per-call price, put the prices in a config file with a source URL, and note the date). Save to `bench/results.json`. If the frontier key is missing, still report SIE numbers and show the frontier column as "not run". Never fabricate the comparison.

### 5. Shared swarm memory with cognee (`memory/`)
- Turn on cognee multi-user mode with access control as the permissions docs describe. Create one cognee user per drone plus an `operator` user.
- Datasets and permissions:
  - `private_<drone>`: only that drone reads and writes (its raw notes and failed attempts).
  - `swarm_map`: scouts have write; everyone has read. Holds landmark observations with poses and hazards.
  - `outcomes`: only the operator (the orchestrator) writes; everyone reads. Holds mission results: success, time, collisions, which route was taken, and why it failed if it did.
- Write path: perception facts become short natural-language memories with structured metadata (drone, time, pose, landmark, hazard type) and go to `swarm_map` or `private_<drone>` based on the drone's permissions. Attempting to write where the drone has no permission must fail and show up in the UI memory feed as "denied" (this demonstrates the permissions are real).
- Read path: before planning each leg, the courier recalls from `swarm_map` and `outcomes` ("where is the red tower, what hazards lie between home and it, which routes failed before") and the planner turns the answer into waypoints that avoid known hazards.
- Learning loop: after every mission, the orchestrator writes the outcome and uses cognee's feedback and improve mechanisms so routes and facts tied to failures are down-weighted and confirmed facts are reinforced. If a scout's fact turns out wrong (the courier gets there and the landmark is not where stated), write a correction.
- Configure cognee's LLM to use SIE's OpenAI-compatible chat endpoint with an open model (check llm-providers docs for the custom/OpenAI-compatible provider settings). For embeddings, use SIE if cognee's embedding provider config supports it; otherwise use a local embedding option from the docs. Record the final config in `NOTES.md`. This keeps the whole stack on open models.
- **Ablation (the evidence for the cognee prize):** run the courier mission on 20 seeds (a) with no memory, (b) with only its private memory, (c) with shared swarm memory after the scouts fly. Report success rate, time to goal, collisions, and gust-zone entries. Save to `bench/memory_ablation.json`.

### 6. API (`api/`)
- `WS /ws/sim`: 20 Hz stream of drone poses, current goals, eye mosaic (compressed), a spike raster window for about 64 displayed neurons (labeled by type), descending neuron readouts, and memory events (write, read, denied, feedback).
- `POST /api/missions` (operator text), `POST /api/sim/reset` (seed, brain type, memory mode), `GET /api/bench/sie`, `GET /api/bench/controllers`, `GET /api/bench/memory`, `GET /api/memory/graph` (nodes and edges for visualization, scoped to what the operator can read), `GET /healthz`.
- **Replay mode:** record full runs to JSON and serve them so the demo works even if the live sim is slow on Render. The UI clearly labels "Live" or "Replay".

### 7. Frontend (`frontend/`)
Match the visual language of the owner's Flybrain site: sober, scientific, light and dark modes, large serif headline with italic emphasis, generous whitespace, numbered sections, honest captions. Pages:
- **Mission Control (home):** a 3D arena (react-three-fiber) with drones, trails, landmarks, pond, and the gust zone revealed once discovered. A mission text box with example missions. Brain selector (real / rewired / reactive) and memory mode selector. Live or Replay badge.
- **Inside the fly:** for the selected drone, the hexagonal compound-eye mosaic, a live spike raster with neuron type labels, left and right descending drive bars, yaw and speed readouts, and a loom/escape indicator.
- **Swarm memory:** a live feed of memory events with drone, dataset, permission result (allowed or denied), and text; plus an interactive knowledge graph view (force-directed) of `swarm_map` and `outcomes`, with nodes colored by which drone contributed them.
- **Results:** three panels from the bench JSON files: controller comparison (real vs rewired vs reactive), memory ablation, and SIE vs frontier quality and cost table with a short plain-English takeaway computed from the data.
- **Method and limitations:** what the connectome controls, what is engineered, data sources (MaleCNS v1.0), and known limits.
Use plain words in all labels; define acronyms on first use (for example "DN (descending neuron)").

### 8. Deploy on Render
- One Docker web service: multi-stage Dockerfile builds the frontend with Node, then serves it from FastAPI as static files along with the API and WebSocket. Bind to `$PORT`.
- `render.yaml` Blueprint with env vars `SIE_API_KEY`, `SIE_BASE_URL`, `FRONTIER_API_KEY` (optional), cognee settings, and a persistent disk mounted at `/var/data` for cognee's local stores and recorded replays. Health check path `/healthz`. Note in the README that the free plan sleeps and has limited memory, so use at least the Starter plan for the demo.
- Keep torch CPU-only to fit the image size. Ship precomputed replays and bench results in the repo so a cold deploy still has a full demo.

## Milestones (keep the app runnable after each)

1. Repo scaffold, `NOTES.md` from reading the flybrain repo, arena and kinematic drones streaming to a basic three.js page. Deploy to Render early with a hello-world so deployment is not a last-minute surprise.
2. Compound eye plus real connectome controller steering one drone toward a target, with rewired and reactive controls.
3. SIE mission planner and perception-to-facts working end to end.
4. cognee multi-user memory with the three datasets, permission denials visible, courier using recalled facts.
5. Feedback and improve loop, ablation run, SIE benchmark run, Results page.
6. Frontend polish, replay recordings, README, demo script.
If time runs out, cut place recognition first, then the escape sub-circuit. Never cut the controls or the honesty panel.

## Deliverables
- Live Render URL.
- `README.md`: one-paragraph pitch, architecture diagram (Mermaid), how to run locally, env vars, results tables pulled from the bench JSON, limitations, and which prize each part targets.
- `DEMO.md`: a 2-minute demo script: (1) type a mission, (2) open "Inside the fly" while the courier flies, (3) show a scout writing to shared memory and a denied write, (4) show the courier rerouting around the gust zone it never saw, (5) show the Results page with the memory ablation and the SIE cost table.
- Tests for the JSON plan schema, the permission rules (a drone cannot write to a dataset it lacks write on), and the rewired shuffle preserving degree.

Work autonomously. Ask me only for credentials or repo access.

Render link: https://dashboard.render.com/ 

Claude API: stored in `.env` as `FRONTIER_API_KEY` (not committed) 

superlinked: stored in `.env` as `SIE_API_KEY` (not committed)

import os
from sie_sdk import SIEClient

client = SIEClient(
    api_key=os.environ["SIE_API_KEY"],
    base_url=os.environ.get("SIE_BASE_URL", "https://api.superlinked.com"),
)