"""Regenerate the results tables in README.md from the bench JSON files (between RESULTS markers).

Usage: python -m backend.scripts.readme_tables   (prints 'README tables updated')
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "backend/bench/results"
README = ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
pct = lambda x: "–" if x is None else f"{round(100 * x)}%"
num = lambda x, d=1: "–" if x is None else f"{x:.{d}f}"


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def controllers(d):
    if not d:
        return "_Controller experiment not run._\n"
    names = {"real": "real connectome", "rewired": "rewired, whole graph", "rewired_class": "rewired, within classes", "reactive": "reactive baseline"}
    out = [f"Task: {d['task']}. Seeds {d['seeds'][0]}–{d['seeds'][-1]} for every arm (tuning seeds {d['tuning_seeds_excluded']} excluded). Generated {d['generated']}.\n",
           "| arm | success | delivered | time to goal (s) | collisions | gust entries | pond entries | escapes |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in ("real", "rewired", "rewired_class", "reactive"):
        if a in d["arms"]:
            s = d["arms"][a]["summary"]
            out.append(f"| {names[a]} | {pct(s['success_rate'])} | {pct(s['delivery_rate'])} | {num(s['mean_time_to_goal_s'])} | {num(s['mean_collisions'], 2)} | "
                       f"{num(s['mean_gust_entries'])} | {num(s['mean_pond_entries'])} | {num(s['mean_escapes'])} |")
    for k, v in d.get("comparisons", {}).items():
        s = v["success"]
        out.append(f"\n{k.replace('_', ' ')}: success difference {s['diff']:+.2f} (95% paired bootstrap CI {s['ci95'][0]:+.2f} to {s['ci95'][1]:+.2f}).")
    return "\n".join(out) + "\n"


def memory(d):
    if not d or not d.get("arms"):
        return "_Memory ablation not run._\n"
    names = {"none": "no memory", "private": "private memory only", "shared": "shared swarm memory"}
    out = [f"Brain: {d['brain']}. Memory backend: {next(iter(d['arms'].values())).get('memory_backend')}. Seeds 0–19 flown in order. Generated {d['generated']}.\n",
           "| courier memory | completed | mean time (s) | gust entries / mission | missions entering gust | first 5 missions entering gust | collisions |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for a in ("none", "private", "shared"):
        if a in d["arms"]:
            s = d["arms"][a]["summary"]
            out.append(f"| {names[a]} | {pct(s['success_rate'])} | {num(s['mean_mission_time_s'])} | {num(s['mean_gust_entries'], 2)} | {pct(s['gust_entry_rate'])} | "
                       f"{pct(s['first5_gust_entry_rate'])} | {num(s['mean_collisions'], 2)} |")
    ev = {a: v.get("memory_event_counts", {}) for a, v in d["arms"].items()}
    out.append("\nMemory events per arm (writes / reads / denied / feedback / improve / errors): "
               + "; ".join(f"{a}: {e.get('write', 0)}/{e.get('read', 0)}/{e.get('denied', 0)}/{e.get('feedback', 0)}/{e.get('improve', 0)}/{e.get('error', 0)}" for a, e in ev.items()) + ".")
    return "\n".join(out) + "\n"


def sie(d):
    if not d:
        return "_SIE benchmark not run._\n"
    out = [f"{d['missions']} missions with hand-written gold plans; {d['keyframes']} keyframes labelled from simulator geometry. Generated {d['generated']}.\n",
           "| job | provider | model | valid JSON | exact match | field accuracy | median latency (s) | $ per 1,000 calls |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for job, name in (("mission_planner", "mission planner"), ("perception", "perception to facts")):
        for p in ("sie", "frontier"):
            r = d["providers"].get(p, {}).get(job)
            if r:
                out.append(f"| {name} | {p} | `{r['model']}` | {pct(r['valid_json_rate'])} | {pct(r['exact_match'])} | {pct(r['field_accuracy'])} | {num(r['median_latency_s'], 2)} | ${num(r['cost_per_1000_calls_usd'], 3)} |")
            else:
                out.append(f"| {name} | {p} | not run | | | | | |")
    pr = d.get("place_recognition", {})
    if pr.get("model"):
        out.append(f"| place recognition | sie | `{pr['model']}` | – | – | – | {num(pr['batch_latency_s'], 2)} per {pr['batch_size']} images | ${num(pr['cost_per_1000_images_usd'], 4)} per 1,000 images |")
    return "\n".join(out) + "\n"


def main():
    body = ("\n### Controllers (real vs rewired vs reactive)\n\n" + controllers(load("controllers.json"))
            + "\n### Memory ablation (cognee)\n\n" + memory(load("memory_ablation.json"))
            + "\n### SIE vs frontier (Superlinked)\n\n" + sie(load("sie_benchmark.json")))
    text = README.read_text()
    a, b = text.index(START) + len(START), text.index(END)
    README.write_text(text[:a] + "\n" + body + "\n" + text[b:])
    print("README tables updated")


if __name__ == "__main__":
    main()
