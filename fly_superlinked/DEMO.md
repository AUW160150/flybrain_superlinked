# DEMO: two minutes

Recorded walkthrough: https://www.loom.com/share/4cfdcb64c2374509bf463aa0c1fe609c

Before the demo: open the app, check the badge top right says **Live** (or pick a replay under *Replays…* if the live sim is slow; the badge then says **Replay**). Reset with brain = real connectome, memory = shared swarm memory, seed 0.

**0:00 – Mission (Mission control).** Click the example chip *"Scouts, map the north side. Courier, take the package to the red tower and come home, avoid the pond."* and **Send mission**. Point at the caption under the button: the plan came from **SIE `Qwen/Qwen3.5-4B`**, a small open model, as strict JSON. The dashed lines are each drone's route; the pond is a no-fly zone on every map; the gust zone is invisible because nobody has felt it yet.

**0:20 – Inside the fly.** Switch to *Inside the fly* with **courier** selected. Left: the compound eye, 1,709 real ommatidial directions; the white-outlined facets are the target stripe. Right: one second of spikes from 64 labelled neurons of the MaleCNS circuit. Watch LC10 and AOTU fire on the side of the stripe and DNa02 follow, then the yaw readout turn the drone. If something looms close, the giant fiber (DNp01) lamp lights and the drone climbs and sidesteps. Say plainly: altitude is a PID, not the fly.

**0:45 – Shared memory and a denied write (Swarm memory).** A scout flies through the gust and writes *"scout-2 felt a strong gust near (…)"* to `swarm_map`: green **write ✓**. When the courier feels it too and tries to share, cognee rejects it: red **denied** with cognee's own `PermissionDeniedError (403)`, and the note goes to `private_courier`. The graph on the right is cognee's knowledge graph of `swarm_map` and `outcomes`, coloured by contributing drone.

**1:10 – The courier reroutes around a gust it never saw (Mission control).** Send *"Courier, take the package to the red tower and come home, avoid the pond."* The log shows the courier recalling from `swarm_map`, `outcomes` and its private notes, the memory's answer, and "routes around 1 remembered gust hazard": its new dashed route bends west of the gust zone the scouts found. After landing, the operator writes the outcome, feedback and `improve()` (visible in the memory feed).

**1:35 – Results.** Open *Results*: (4.1) real vs both rewired controls vs reactive, same seeds, with the honest takeaway (the reactive baseline is best; the real wiring steers, both shuffles do not). (4.2) memory ablation: how often the courier entered the gust zone with no memory, private memory and shared memory, mission by mission. (4.3) SIE vs frontier: planner quality within a few points of `claude-opus-5-5` at about 1/23 of the cost; perception is weak for both models and is reported as such.

**1:55 – Method and limitations.** One sentence: what the connectome controls, what is engineered, and that nothing is trained here, unlike the owner's earlier rewired ≈ real result.
