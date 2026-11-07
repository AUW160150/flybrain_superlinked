# Gates: Flybrain Swarm hackathon build

OWNS: fly_superlinked/**, render.yaml

Scope: the full prompt in prompt.md. That covers sim, compound eye, connectome controller with controls, SIE jobs and benchmark, cognee swarm memory with permissions and a learning loop, the ablation, API with replays, five-page frontend, Render deploy files, README/DEMO/NOTES/LOG, and tests.

- [ ] G1: unit tests for plan schema, permission rules and degree-preserving rewiring pass
  CHECK: .venv/bin/python -m pytest -q && echo UNIT_TESTS_OK
  EXPECT: UNIT_TESTS_OK
  CWD: fly_superlinked
  EVIDENCE: pending

- [ ] G2: bench results are complete and consistent (4 controller arms x 20 seeds; SIE and frontier on 30 missions and 40 keyframes with costs; 3 memory arms x 20 seeds on cognee with a recorded denial)
  CHECK: .venv/bin/python -m backend.scripts.verify bench
  EXPECT: BENCH_FILES_OK
  CWD: fly_superlinked
  EVIDENCE: pending

- [ ] G3: the frontend type-checks and builds
  CHECK: npm run build && echo FRONTEND_BUILD_OK
  EXPECT: FRONTEND_BUILD_OK
  CWD: fly_superlinked/frontend
  EVIDENCE: pending

- [ ] G4: a fresh server serves health, arena, eye layout and bench endpoints, plans a mission, and streams frames where the courier moves
  CHECK: .venv/bin/python -m backend.scripts.verify api
  EXPECT: API_OK
  CWD: fly_superlinked
  EVIDENCE: pending

- [ ] G5: README has the Mermaid diagram and generated results tables for all three experiments; DEMO, NOTES, LOG, Dockerfile, requirements and render.yaml exist with the required settings
  CHECK: .venv/bin/python -m backend.scripts.verify docs
  EXPECT: DOCS_OK
  CWD: fly_superlinked
  EVIDENCE: pending

- [ ] G6: at least one recorded replay with moving drones ships in the repo
  CHECK: .venv/bin/python -m backend.scripts.verify replay
  EXPECT: REPLAY_OK
  CWD: fly_superlinked
  EVIDENCE: pending

- [ ] G7: everything is committed and pushed to github.com/AUW160150/superlinked
  CHECK: .venv/bin/python -m backend.scripts.verify pushed
  EXPECT: PUSHED_OK
  CWD: fly_superlinked
  EVIDENCE: pending

- [ ] G8: the UI was inspected in light and dark mode and at phone width: arena with drones, eye mosaic and raster, memory feed with a denied write and the graph, results charts, method panel
  EVIDENCE: pending

- [ ] G9: a live Render URL answers /healthz
  EVIDENCE: pending
