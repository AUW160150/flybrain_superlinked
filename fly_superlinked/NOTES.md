# NOTES: what is reused from the Flybrain repo, and what the circuit offers

Source repo: `github.com/prove-ai/flybrain` (read 2026-11-07, shallow clone of `main`).

## Reused

| Piece | From flybrain | Here | Change |
|---|---|---|---|
| Sensory-to-descending sub-circuit (10,203 neurons, 401,680 edges with >= 5 synapses, MaleCNS v1.0) | `projects/flyworld-game/processed/circuit_npc.npz` | `backend/brain/data/circuit_npc.npz` | copied unchanged; loaded with `allow_pickle=False` |
| Named neuron groups (with `_L`/`_R` subgroups) | `projects/flyworld-game/processed/circuit_npc_groups.json` (built by `flyworld/groups.py`) | `backend/brain/data/` | copied unchanged |
| LIF calibration (published turn bias) | `projects/flyworld-game/processed/calibration.json` | `backend/brain/data/` | kept for reference; the bias is re-measured per wiring with the same procedure (see below) |
| Batched event-driven LIF simulator | `flyworld/brain.py` `BatchedLIF` | `backend/brain/lif.py` | same parameters (1 ms, tau 20 ms, threshold 1, 2 ms refractory, gain 0.005, governor); CPU only; adds a per-neuron raster for the UI |
| Motor decoder rules | `flyworld/decoder.py` `MotorDecoder` | `backend/brain/controller.py` `FlyBrains.step` | same readout: forward from DNp09 rate (10 to 100 Hz), turn from (DNa02 + 0.5 DNa01) right minus left, minus the walk-coupled bias, 100 ms EMA; escape from DNp01 spikes |
| Looming encoder formulas | `flyworld/encoders.py` `SensoryEncoder` | `FlyBrains.encode` | expansion rate -> LC4, LC6; angular size while expanding -> LPLC2, LC16; driven from the ray-cast eye instead of geometric proxies |
| Eye directions (1,709 microCT ommatidial axes, 857 L / 852 R) | `projects/flyvision/config/eye_reference.json` pin and `flyvision/eye_reference.py` checks | `backend/scripts/prepare_eye.py` -> `backend/sim/data/eye_directions.npz` | same pinned file (reiserlab/eyemap_T4 @ 99d2a43), git-blob hash, unit-norm and head-basis checks |

Not reused, and why:
- **Project 08 steering circuit (443 neurons, LC10a -> AOTU -> DN).** Only its reports are in the repo (`site/src/results/sources/fly_track/`); the code and graph are not. Instead the controller uses the LC10 -> AOTU -> DNa02 pathway that exists inside the FlyWorld sub-circuit (types listed below). Ask the owner for `fly_track` if the 443-neuron graph should replace it.
- **FlyWorld panda3d game and FlyVision pygame viewer.** Replaced by a numpy ray caster (`backend/sim/eye.py`) and a react-three-fiber frontend.

## Neuron types in the subset (from the `type` column; nothing invented)

Visual inputs used (retinotopic by eye side):
- `LC10a, LC10b, LC10c-1, LC10c-2, LC10d` (group `LC10`): 3 left cells, 27 right cells in this subset. Target stripe.
- `LC4` (looming, expansion velocity), `LC6`, `LPLC2` (looming, angular size), `LC16`.
- `HSE, HSN, HSS` (group `HS`, horizontal optic flow).
Also present but not driven: `LC9`, `LC11`, `LC12`, `LC15`, `LC22`, `LPLC1`, `LPLC4`, `LT*` (several), `H2`.

Central relay read for the UI: `AOTU*` (62 cells in the subset).

Descending / motor outputs read out: `DNa02` (L 1, R 1), `DNa01` (L 1, R 1), `DNp09` (L 1, R 1), `DNp01` giant fiber (L 1, R 1), `MDN` (4), plus `DNg02`, `DN_all` for display.

## Measured before wiring the controller (backend/bench/probe_circuit.py, results/probe_circuit.json)

With the FlyWorld 'go' current on DNp09 (0.15), stimulating one side of a visual group for 300 ms:
- `LC10` right -> DNa02 right 77 to 130 Hz; `LC10` left -> no change (only 3 cells). `AOTU` left -> DNa02 left 163 Hz, right -> DNa02 right 97 to 167 Hz. So the real wiring has an ipsilateral pursuit pathway, but the subset's left LC10 input is thin.
- `LC4` left -> DNa01 right 93 Hz and both giant fibers (escape away from the threat).
- `LC11`, `HS` -> almost no effect on steering DNs.
- The 'go' current alone produces DNa02 left 50 Hz (the walk-coupled asymmetry FlyWorld calibrates out).
- In the degree-preserving rewired graph, the same stimuli either do nothing or recruit every DN at once (no lateralisation).

## Choices that affect the controller comparison

- Decoder gains (dead zone 4 Hz, tanh scale 20 Hz, LC10 current 0.3) were chosen on tuning seeds 100 to 102 with the **real** circuit, then frozen. Evaluation uses seeds 0 to 19. This favours the real circuit, if anything. Log: `backend/bench/results/decoder_tuning_seeds100-102.txt`.
- The walk-coupled turn bias is re-measured for each wiring with FlyWorld's procedure (go current alone, (left - right) / DNp09 rate), so the rewired circuit is not penalised by the real circuit's calibration.
- The rewired control shuffles targets across the whole graph (degrees and each neuron's outgoing signed weights kept). The owner's Project 08 rewiring was class-preserving (role x side blocks); this one is stronger.

## Final cognee and SIE configuration

Filled in once keys are available; see LOG.md.
