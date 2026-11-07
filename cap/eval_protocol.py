"""Held-out evaluation of a cap policy: the NIST task-board benchmark in simulation.

Every episode is a full assembly from scratch: the part lies loose on the board at a random pose, the robot
is at its home configuration with the gripper open (reset family `board`). Starts are the first `n` (default 100)
`board` states of the task's frozen evaluation bank (<data>/banks_board/<TASK>_eval.npz, disjoint from the
training bank the loop debugs on).
Reported: success rate and completion time of the successful episodes.

  PYTHONPATH=.:cap python cap/eval_protocol.py --run-dir cap/runs/fixloop/R16_agent     # best round
  PYTHONPATH=.:cap python cap/eval_protocol.py --policy <path>/policy.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

import api
from assembly_policy.sim.resets import FAMILIES
from assembly_policy.sim.tasks import BANK_DIR

CAP_BANK_DIR = BANK_DIR.parent / "banks_board"   # benchmark-protocol starts (family `board`)
START_FAMILY = "board"
DT = 0.04


def load_policy(path: Path):
    spec = importlib.util.spec_from_file_location(f"cap_policy_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.policy


def best_round(run_dir: Path):
    """Highest debug success rate; ties go to the larger set, then the later round. Returns (round, policy path)."""
    best = None
    for line in (run_dir / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "round" not in r:
            continue
        key = (r["n_ok"] / max(r["n"], 1), r["n"], r["round"])
        if best is None or key > best[0]:
            best = (key, r["round"])
    return best[1], run_dir / "rounds" / f"r{best[1]:02d}" / "policy.py"


N_EVAL = 100


def evaluate(env, policy, task, level, bank_dir=CAP_BANK_DIR, n=N_EVAL, verbose=True, record=False, on_episode=None):
    """on_episode(k, bank_idx, result) is called after every episode (with frames when record=True) so callers can
    save videos/traces; the protocol itself only returns the summary rows."""
    bank = np.load(Path(bank_dir) / f"{task}_eval.npz")
    idx = np.nonzero(bank["family"] == FAMILIES.index(START_FAMILY))[0][:n]
    rows, t0 = [], time.time()
    for k, i in enumerate(idx):
        try:
            r = api.run_episode(env, bank["states"][i], policy, level=level, seed=20_000 + k, record=record)
            if on_episode is not None:
                on_episode(k, int(i), r)
        except api.LevelError as e:
            r = dict(success=False, safety=False, timeout=False, steps=0, error=str(e), d_asm=float("nan"))
        rows.append(dict(trial=k, bank_idx=int(i), success=bool(r["success"]), safety=bool(r.get("safety", False)),
                         timeout=bool(r.get("timeout", False)), steps=int(r.get("steps", 0)),
                         time_s=round(int(r.get("steps", 0)) * DT, 2), error=r.get("error"),
                         d_asm_mm=round(float(r.get("d_asm", float("nan"))) * 1000, 2)))
        if verbose and (k + 1) % 32 == 0:
            print(f"[eval] {k + 1}/{len(idx)}: {sum(x['success'] for x in rows)} ok  ({time.time() - t0:.0f}s)", flush=True)
    ok = [x for x in rows if x["success"]]
    summary = dict(task=task, horizon_s=env.ecfg.horizon_s,
                   protocol=f"NIST task board: full assembly from the first {len(idx)} {START_FAMILY} starts (part loose on the board, robot at home, gripper open) of the frozen eval bank",
                   n=len(rows), n_ok=len(ok), success_rate=round(100 * len(ok) / max(len(rows), 1), 1),
                   time_mean_s=round(float(np.mean([x["time_s"] for x in ok])), 2) if ok else None,
                   time_median_s=round(float(np.median([x["time_s"] for x in ok])), 2) if ok else None,
                   safety=sum(x["safety"] for x in rows), errors=sum(bool(x["error"]) for x in rows),
                   timeouts=sum(x["timeout"] for x in rows))
    return summary, rows


def print_summary(s, name):
    print(f"==> {name} on {s['task']}: {s['n_ok']}/{s['n']} = {s['success_rate']}%  "
          f"time mean {s['time_mean_s']} s / median {s['time_median_s']} s  "
          f"(timeouts {s['timeouts']}, safety {s['safety']}, errors {s['errors']})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="R16")
    ap.add_argument("--policy", default=None, help="a policy.py; default = best round in --run-dir")
    ap.add_argument("--run-dir", default="cap/runs/fixloop/R16_agent")
    ap.add_argument("--level", default="grasp")
    ap.add_argument("--bank-dir", default=str(CAP_BANK_DIR))
    ap.add_argument("--n", type=int, default=N_EVAL, help="held-out starts (first n board states of the eval bank)")
    ap.add_argument("--horizon", type=float, default=40.0, help="episode time limit in seconds")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.policy:
        path, name = Path(a.policy), Path(a.policy).parent.name
    else:
        rnd, path = best_round(Path(a.run_dir)); name = f"r{rnd:02d}"
    policy = load_policy(path)
    out = Path(a.out or Path(a.run_dir) / f"eval_{name}.json")
    env = api.make_env(a.task, horizon_s=a.horizon)
    print(f"policy {path}  task {a.task}  horizon {a.horizon:g} s", flush=True)
    summary, rows = evaluate(env, policy, a.task, a.level, a.bank_dir, n=a.n)
    summary["policy"] = str(path)
    out.write_text(json.dumps(dict(summary=summary, episodes=rows), indent=1))
    print_summary(summary, name)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
