# Gates: Flybrain Swarm hackathon build

OWNS: fly_superlinked/**, render.yaml

Scope: the full prompt in prompt.md. That covers sim, compound eye, connectome controller with controls, SIE jobs and benchmark, cognee swarm memory with permissions and a learning loop, the ablation, API with replays, five-page frontend, Render deploy files, README/DEMO/NOTES/LOG, and tests.

- [x] G1: unit tests for plan schema, permission rules and degree-preserving rewiring pass
  CHECK: .venv/bin/python -m pytest -q && echo UNIT_TESTS_OK
  EXPECT: UNIT_TESTS_OK
  CWD: .
  EVIDENCE: automatic-evidence=v1; definition-sha256=f5142b659981120dd0d1040dd1fb24e70c95f96aef1f5d89baa07e2c53bc14fc; exit=0; EXPECT=matched; output-sha256=167e0f78554be96408b6f75d458984aba6a12163e3c9992c5db2b881e46174c8; output-bytes=113; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/fly_superlinked; path=850bd16ad3bb/21 entries

- [x] G2: bench results are complete and consistent (4 controller arms x 20 seeds; SIE and frontier on 30 missions and 40 keyframes with costs; 3 memory arms x 20 seeds on cognee with a recorded denial)
  CHECK: .venv/bin/python -m backend.scripts.verify bench
  EXPECT: BENCH_FILES_OK
  CWD: .
  EVIDENCE: automatic-evidence=v1; definition-sha256=5446c2d588bc4ea6c9abead62d66f756593a93f69705b36509edd7ec9687ed59; exit=0; EXPECT=matched; output-sha256=27c1d6868ce96fad99995d4f2f16478139d4f26c09b5a9ff15a9414bd342e6e3; output-bytes=15; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/fly_superlinked; path=850bd16ad3bb/21 entries

- [x] G3: the frontend type-checks and builds
  CHECK: npm run build && echo FRONTEND_BUILD_OK
  EXPECT: FRONTEND_BUILD_OK
  CWD: frontend
  EVIDENCE: automatic-evidence=v1; definition-sha256=41f45e3caab634eedeb5da56e60a7b06d292e398592d2ec64eedcd281bf92f73; exit=0; EXPECT=matched; output-sha256=857b7b5f0692104a4116514fe48ba0381e957bff4de144a9153d5d2b1160feba; output-bytes=744; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/fly_superlinked/frontend; path=850bd16ad3bb/21 entries

- [x] G4: a fresh server serves health, arena, eye layout and bench endpoints, plans a mission, and streams frames where the courier moves
  CHECK: .venv/bin/python -m backend.scripts.verify api
  EXPECT: API_OK
  CWD: .
  EVIDENCE: automatic-evidence=v1; definition-sha256=0101d90158256246d7daab2d24afed863b334d8f275d036c158d11bf700878ed; exit=0; EXPECT=matched; output-sha256=dc776c5c26016877254dd9b96645d5f7c37f59a9100cec8b78d3238dcbf85aae; output-bytes=7; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/fly_superlinked; path=850bd16ad3bb/21 entries

- [x] G5: README has the Mermaid diagram and generated results tables for all three experiments; DEMO, NOTES, LOG, Dockerfile, requirements and render.yaml exist with the required settings
  CHECK: .venv/bin/python -m backend.scripts.verify docs
  EXPECT: DOCS_OK
  CWD: .
  EVIDENCE: automatic-evidence=v1; definition-sha256=c7132d00a098800c7307e4e3a9294c0c73b57759d7957f4c1b1f712c3f4f0c5e; exit=0; EXPECT=matched; output-sha256=6d36de704b81554dfb84505a4da24630d2ffb6dc40ba29dd836d0c858704f7bc; output-bytes=8; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/fly_superlinked; path=850bd16ad3bb/21 entries

- [x] G6: at least one recorded replay with moving drones ships in the repo
  CHECK: .venv/bin/python -m backend.scripts.verify replay
  EXPECT: REPLAY_OK
  CWD: .
  EVIDENCE: automatic-evidence=v1; definition-sha256=a65676689b4e1cb30f8c4b6e6925f488d0d1674417f748b80cedc0310d5aa01e; exit=0; EXPECT=matched; output-sha256=d446d98a07381156efe35c1729f9852784bbfa72a55174a6b86d85869750c558; output-bytes=10; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/fly_superlinked; path=850bd16ad3bb/21 entries

- [x] G7: everything is committed and pushed to github.com/AUW160150/flybrain_superlinked
  CHECK: .venv/bin/python -m backend.scripts.verify pushed
  EXPECT: PUSHED_OK
  CWD: .
  EVIDENCE: automatic-evidence=v1; definition-sha256=5cdcb193af3f20e13ec526b3384e7b5da79869243d0798af906c123f680dfda5; exit=0; EXPECT=matched; output-sha256=e495d43a6efbc67041a8c877da1066334801d1f74ebf999085b82e6be0d3bd52; output-bytes=10; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/fly_superlinked; path=850bd16ad3bb/21 entries

- [x] G8: the UI was inspected in light and dark mode and at phone width: arena with drones, eye mosaic and raster, memory feed with a denied write and the graph, results charts, method panel
  EVIDENCE: 2026-11-07 Playwright screenshots of all five pages at 1440 px (light and dark) and 390 px (light), no console errors. Seen: live arena with three drones, trails, dashed routes and the revealed gust zone; eye mosaic with stripe outlines and a raster showing LC10 L -> AOTU L -> DNa02 L; memory feed with cognee 403 denials and the cognee graph; results bars and tables; method panel. Fixed on the way: oversized chart text, empty space under the arena, missing phone gutter, phone overflow from filter chips (all pages now exactly 390 px wide).

- [x] G9: the live Render URL answers /healthz and serves the bench results
  CHECK: curl -sf --max-time 60 https://flybrain-swarm.onrender.com/healthz | grep -q '"ok":true' && curl -sf --max-time 60 https://flybrain-swarm.onrender.com/api/bench/controllers | grep -q '"real"' && echo LIVE_URL_OK
  EXPECT: LIVE_URL_OK
  CWD: .
  EVIDENCE: automatic-evidence=v1; definition-sha256=6890ae9c67114bf4dbba85551901e93ee5a99d959828179717ffb0c223778edd; exit=0; EXPECT=matched; output-sha256=db926b37f7f5f9619f2bfad44d009f130907c051a3b1afea9c0a989f3a72465b; output-bytes=12; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/fly_superlinked; path=850bd16ad3bb/21 entries
