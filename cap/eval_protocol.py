"""AlphaNIST-style evaluation of a cap policy: 5 seeds x 32 trials = 160 held-out episodes.

Within a seed the 32 start states cycle through the five reset families
(loose, pickup, inhand[=transport], partial, near), so every family gets 6-7 starts per seed
and 32 per task. Sampler seeds 1..5 are disjoint from the fix loop's debug seed 0.

  PYTHONPATH=.:cap python cap/eval_protocol.py --run-dir cap/runs/fixloop_R16_near     # best version by log.jsonl
  PYTHONPATH=.:cap python cap/eval_protocol.py --policy cap/runs/fixloop_R16_near/v02.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import api

FAMILIES = ["loose", "pickup", "inhand", "partial", "near"]


def load_policy(path: Path):
    spec = importlib.util.spec_from_file_location(f"cap_policy_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.policy


def best_version(run_dir: Path) -> Path:
    best = None
    for line in (run_dir / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "turn" in r and "episodes" in r and not r.get("eval"):
            if best is None or r["n_ok"] > best[0]:
                best = (r["n_ok"], r["turn"])
    return run_dir / f"v{best[1]:02d}.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="R16")
    ap.add_argument("--policy", default=None, help="path to a vNN.py; default = best in --run-dir")
    ap.add_argument("--run-dir", default="cap/runs/fixloop_R16_near")
    ap.add_argument("--level", default="grasp")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    path = Path(a.policy) if a.policy else best_version(Path(a.run_dir))
    policy = load_policy(path)
    out = Path(a.out or (path.parent / f"eval_{path.stem}.json"))
    env = api.make_env(a.task)
    print(f"policy {path}  task {a.task}  {a.seeds} seeds x {a.trials} trials", flush=True)

    rows, by_fam, by_seed = [], defaultdict(list), defaultdict(list)
    t0 = time.time()
    for seed in range(1, a.seeds + 1):
        # per-family pools for this seed, consumed round-robin so the 32 trials cycle the families
        pools = {f: api.sample_starts(a.task, f, -(-a.trials // len(FAMILIES)), seed=seed) for f in FAMILIES}
        used = defaultdict(int)
        for k in range(a.trials):
            fam = FAMILIES[k % len(FAMILIES)]
            s = pools[fam][used[fam]]; used[fam] += 1
            try:
                r = api.run_episode(env, s, policy, level=a.level, seed=seed * 1000 + k)
            except api.LevelError as e:
                r = dict(success=False, safety=False, timeout=False, steps=0, error=str(e), d_asm=float("nan"))
            ok = bool(r["success"])
            rows.append(dict(seed=seed, trial=k, family=fam, success=ok, safety=bool(r.get("safety", False)),
                             timeout=bool(r.get("timeout", False)), steps=int(r.get("steps", 0)),
                             error=r.get("error"), d_asm_mm=round(float(r.get("d_asm", float("nan"))) * 1000, 2)))
            by_fam[fam].append(ok); by_seed[seed].append(ok)
        sr = 100 * np.mean(by_seed[seed])
        print(f"seed {seed}: {sum(by_seed[seed])}/{a.trials} = {sr:.1f}%   ({time.time() - t0:.0f}s)", flush=True)

    total = sum(r["success"] for r in rows)
    seed_rates = [100 * np.mean(by_seed[s]) for s in sorted(by_seed)]
    summary = dict(policy=str(path), task=a.task, seeds=a.seeds, trials=a.trials,
                   success_rate=round(100 * total / len(rows), 1),
                   seed_mean=round(float(np.mean(seed_rates)), 1), seed_std=round(float(np.std(seed_rates)), 1),
                   by_family={f: dict(n_ok=sum(v), n=len(v), rate=round(100 * np.mean(v), 1)) for f, v in by_fam.items()},
                   safety=sum(r["safety"] for r in rows), errors=sum(bool(r["error"]) for r in rows))
    out.write_text(json.dumps(dict(summary=summary, episodes=rows), indent=1))
    print("\nby family:")
    for f in FAMILIES:
        b = summary["by_family"][f]
        print(f"  {f:8s} {b['n_ok']:3d}/{b['n']:<3d} {b['rate']:5.1f}%")
    print(f"==> {path.name} on {a.task}: {total}/{len(rows)} = {summary['success_rate']}%  "
          f"(seed mean {summary['seed_mean']} +- {summary['seed_std']}, safety {summary['safety']}, "
          f"errors {summary['errors']})   wrote {out}", flush=True)


if __name__ == "__main__":
    main()
