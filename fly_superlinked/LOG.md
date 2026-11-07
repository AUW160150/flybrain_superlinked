# LOG

## 2026-11-07 Milestone 1 (partial): scaffold, NOTES, arena, eye, connectome controller

- Read flybrain; NOTES.md lists reused files. The 443-neuron steering circuit code is not in the repo, so the controller uses the LC10 -> AOTU -> DNa02 pathway inside FlyWorld's 10,203-neuron sub-circuit.
- Eye: 1,709 microCT directions fetched from the pinned reiserlab/eyemap_T4 commit and hash-checked. Ray cast ~1 ms per frame.
- Probe (bench/probe_circuit.py): real wiring steers ipsilaterally via AOTU -> DNa02; the subset has only 3 left LC10 cells vs 27 right, so left turns are weak.
- Decoder gains tuned on seeds 100-102 (real circuit), frozen: dead zone 4 Hz, tanh scale 20 Hz, LC10 0.3.
- Unit tests caught a bug in the rewired control (fancy-index swaps duplicated targets, changing in-degrees). Fixed; degree preservation is now exact and tested.
- Docs differences: cognee 1.6.7 has no SearchType.FEEDBACK or save_interaction; feedback goes through sessions (cognee.session.add_feedback) and improve(). Using that instead of the prompt's "feedback and improve mechanisms" wording.
- SIE and cognee need SIE_API_KEY; until it is set, the app runs with a keyword fallback planner and a local permissioned memory, both labelled in the UI.
