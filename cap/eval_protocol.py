"""Official evaluation of a cap policy: the AlphaNIST protocol as implemented in assembly_policy.evaluate.

5 seeds x 32 trials drawn (rng 10_000 + seed, with replacement) from the task's frozen evaluation bank
<data>/banks/<TASK>_eval.npz (128 states per reset family, disjoint from the training bank). The cap
policy is deterministic, so a seed only decides which 32 of the 640 bank states are drawn.

  PYTHONPATH=.:cap python cap/eval_protocol.py --run-dir cap/runs/fixloop/R16_bank     # best version by log.jsonl
  PYTHONPATH=.:cap python cap/eval_protocol.py --policy cap/runs/fixloop/R16_bank/v08.py
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
from assembly_policy.sim.resets import FAMILIES
from assembly_policy.sim.tasks import BANK_DIR

SEEDS = (0, 1, 2, 3, 4)
TRIALS = 32


def load_policy(path: Path):
    spec = importlib.util.spec_from_file_location(f"cap_policy_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.policy


def best_version(run_dir: Path) -> Path:
    """Highest debug success rate; ties go to the version scored on the larger debug set, then the later one."""
    best = None
    for line in (run_dir / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("eval") or "turn" not in r:
            continue
        key = (r["n_ok"] / max(r["n"], 1), r["n"], r["turn"])
        if best is None or key > best[0]:
            best = (key, r["turn"])
    return run_dir / f"v{best[1]:02d}.py"


def official_eval(env, policy, task, level, bank_dir=BANK_DIR, seeds=SEEDS, trials=TRIALS, verbose=True):
    """Returns (summary, episodes). Mirrors assembly_policy.evaluate.evaluate for one task."""
    bank = np.load(Path(bank_dir) / f"{task}_eval.npz")
    states, fam_id = bank["states"], bank["family"]
    rows, by_fam, by_seed = [], defaultdict(list), defaultdict(list)
    t0 = time.time()
    for s in seeds:
        rng = np.random.default_rng(10_000 + s)
        for k, i in enumerate(rng.integers(0, len(states), trials)):
            fam = FAMILIES[int(fam_id[i])]
            try:
                r = api.run_episode(env, states[i], policy, level=level, seed=s * 1000 + k)
            except api.LevelError as e:
                r = dict(success=False, safety=False, timeout=False, steps=0, error=str(e), d_asm=float("nan"))
            ok = bool(r["success"])
            rows.append(dict(seed=s, trial=k, bank_idx=int(i), family=fam, success=ok,
                             safety=bool(r.get("safety", False)), timeout=bool(r.get("timeout", False)),
                             steps=int(r.get("steps", 0)), error=r.get("error"),
                             d_asm_mm=round(float(r.get("d_asm", float("nan"))) * 1000, 2)))
            by_fam[fam].append(ok); by_seed[s].append(ok)
        if verbose:
            print(f"[eval] seed {s}: {sum(by_seed[s])}/{trials} = {100 * np.mean(by_seed[s]):.1f}%   "
                  f"({time.time() - t0:.0f}s)", flush=True)
    seed_rates = [100 * np.mean(by_seed[s]) for s in seeds]
    total = sum(r["success"] for r in rows)
    summary = dict(task=task, protocol="AlphaNIST: 5 seeds x 32 trials from the frozen eval bank",
                   seeds=list(seeds), trials=trials, n=len(rows), n_ok=total,
                   success_rate=round(100 * total / len(rows), 1),
                   seed_mean=round(float(np.mean(seed_rates)), 1), seed_std=round(float(np.std(seed_rates)), 1),
                   by_family={f: dict(n_ok=sum(v), n=len(v), rate=round(100 * np.mean(v), 1))
                              for f, v in by_fam.items()},
                   safety=sum(r["safety"] for r in rows), errors=sum(bool(r["error"]) for r in rows))
    return summary, rows


def print_summary(summary, name):
    print("by family:")
    for f in FAMILIES:
        b = summary["by_family"].get(f)
        if b:
            print(f"  {f:8s} {b['n_ok']:3d}/{b['n']:<3d} {b['rate']:5.1f}%")
    print(f"==> {name} on {summary['task']}: {summary['n_ok']}/{summary['n']} = {summary['success_rate']}%  "
          f"(seed mean {summary['seed_mean']} +- {summary['seed_std']}, safety {summary['safety']}, "
          f"errors {summary['errors']})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="R16")
    ap.add_argument("--policy", default=None, help="path to a vNN.py; default = best in --run-dir")
    ap.add_argument("--run-dir", default="cap/runs/fixloop/R16_bank")
    ap.add_argument("--level", default="grasp")
    ap.add_argument("--bank-dir", default=str(BANK_DIR))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    path = Path(a.policy) if a.policy else best_version(Path(a.run_dir))
    policy = load_policy(path)
    out = Path(a.out or (path.parent / f"eval_{path.stem}.json"))
    env = api.make_env(a.task)
    print(f"policy {path}  task {a.task}  official protocol, bank {a.bank_dir}", flush=True)
    summary, rows = official_eval(env, policy, a.task, a.level, a.bank_dir)
    summary["policy"] = str(path)
    out.write_text(json.dumps(dict(summary=summary, episodes=rows), indent=1))
    print_summary(summary, path.name)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
