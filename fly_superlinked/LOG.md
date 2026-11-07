# LOG

## 2026-11-07 Milestone 1 (partial): scaffold, NOTES, arena, eye, connectome controller

- Read flybrain; NOTES.md lists reused files. The 443-neuron steering circuit code is not in the repo, so the controller uses the LC10 -> AOTU -> DNa02 pathway inside FlyWorld's 10,203-neuron sub-circuit.
- Eye: 1,709 microCT directions fetched from the pinned reiserlab/eyemap_T4 commit and hash-checked. Ray cast ~1 ms per frame.
- Probe (bench/probe_circuit.py): real wiring steers ipsilaterally via AOTU -> DNa02; the subset has only 3 left LC10 cells vs 27 right, so left turns are weak.
- Decoder gains tuned on seeds 100-102 (real circuit), frozen: dead zone 4 Hz, tanh scale 20 Hz, LC10 0.3.
- Unit tests caught a bug in the rewired control (fancy-index swaps duplicated targets, changing in-degrees). Fixed; degree preservation is now exact and tested.
- Docs differences: cognee 1.6.7 has no SearchType.FEEDBACK or save_interaction; feedback goes through sessions (cognee.session.add_feedback) and improve(). Using that instead of the prompt's "feedback and improve mechanisms" wording.
- SIE and cognee need SIE_API_KEY; until it is set, the app runs with a keyword fallback planner and a local permissioned memory, both labelled in the UI.

## 2026-11-07 Milestones 2-5: controller, SIE jobs, cognee memory, benchmarks, frontend

- SIE works end to end: planner and perception on `Qwen/Qwen3.5-4B` (strict JSON schema), place recognition on `google/siglip2-base-patch16-224`. Frontier comparison: `claude-opus-5-5` through the Anthropic SDK with structured output.
- cognee 1.6.7 runs fully on SIE (LLM `Qwen/Qwen3.8-27B-FP8` as a `custom` provider, embeddings `BAAI/bge-m3` via `openai_compatible`). Real 403 `PermissionDeniedError` on the courier's writes to `swarm_map`. Docs difference: feedback needs a session-scoped completion search (CHUNKS stores no QA), so recall runs CHUNKS (facts) plus GRAPH_COMPLETION in the session (answer that feedback scores).
- Perception benchmark v1 (128x64 image, no definitions) scored poorly for both models; v2 upscales the same eye image 4x (nearest) and defines bearing/distance/hazard, for both providers. v1 kept in `sie_benchmark_v1.json`.
- Added a class-preserving rewired control (superclass x side blocks), closer to Project 08's control. 123 of 401,680 edges stay parallel in blocks too small to avoid it.
- Bugs found by the experiments and fixed before the final runs (all arms rerun on the same code): tree goals at trunk centres were unreachable (scouts orbited forever, so the first shared-memory arm had no scout facts); the reactive baseline slowed to 40% for any landmark within 6 m and could not beat the wind; gust reports were separate small circles (now clustered into area hazards); the gust (up to 4.4 m/s) could exceed drone speed (now 2.1-3.1 m/s); the gust detector fired on sharp turns and arena walls (now measured against the drone's own no-wind model); routine observations crowded hazards out of recall (hazards now recalled through their cognee node set); the router dropped a hazard circle that contained the goal (now shrinks it).
- Live server bug: sim steps run in a worker thread and started cognee calls on a new event loop; now handed back to the main loop.
