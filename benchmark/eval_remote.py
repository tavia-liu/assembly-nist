"""Score a sandbox's policy.py through the cell server (benchmark/server.py): the policy sees only robot.py.

  SERVER_TOKEN=... PYTHONPATH=.:benchmark python benchmark/eval_remote.py --sandbox <dir> --task R8 --n 100 --out <dir>
  SERVER_TOKEN=... PYTHONPATH=.:benchmark python benchmark/eval_remote.py --sandbox <dir> --board --seeds 10 --out <dir>

policy.py defines policy(api); for the board also, optionally, recover(api) (run before every part after the first)
and finish(api) (run after a success, unscored). Held-out starts are seeds RESERVED.. (reserved on the server).
Videos are rendered by the server into its --out under <name>/.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RESERVED = 1000


def load_policy(sandbox: Path):
    sys.path.insert(0, str(sandbox))
    spec = importlib.util.spec_from_file_location("submitted_policy", sandbox / "policy.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def run_to_end(a, fn, robot):
    """fn(a) then let the clock run out; returns the outcome."""
    err = None
    try:
        fn(a)
    except robot.EpisodeOver:
        pass
    except Exception as e:                       # a buggy policy is a result
        err = f"{type(e).__name__}: {e}"
    try:
        while not a.outcome().get("done"):
            a.wait(0.2)
    except robot.EpisodeOver:
        pass
    o = a.outcome(); o["error"] = err
    return o


def single(a, robot, pol, token, name):
    rows, t0 = [], time.time()
    for k in range(a.n):
        seed = RESERVED + k
        with robot.start(a.task, seed=seed, horizon=a.horizon, token=token) as ep:
            o = run_to_end(ep, pol.policy, robot)
            ep._conn.rpc(cmd="save_video", ep=ep._ep, name=f"{name}/e{k:03d}")
        rows.append(dict(trial=k, seed=seed, success=o["success"], safety=o["safety"], timeout=o["timeout"],
                         time_s=round(o["steps"] * 0.04, 2), error=o["error"]))
        if (k + 1) % 10 == 0:
            print(f"[eval] {k + 1}/{a.n}: {sum(r['success'] for r in rows)} ok ({time.time() - t0:.0f}s)", flush=True)
    ok = [r for r in rows if r["success"]]
    return dict(task=a.task, horizon_s=a.horizon, n=len(rows), n_ok=len(ok), success_rate=round(100 * len(ok) / max(1, len(rows)), 1),
                time_mean_s=round(float(np.mean([r["time_s"] for r in ok])), 2) if ok else None,
                timeouts=sum(r["timeout"] for r in rows), safety=sum(r["safety"] for r in rows),
                errors=sum(bool(r["error"]) for r in rows)), rows


def board(a, robot, pol, token, name):
    res = []
    for s in range(a.seeds):
        seed = RESERVED + s
        rng = np.random.default_rng(seed)
        with robot.board(seed=seed, tasks=a.tasks, per_part_s=a.horizon, token=token) as b:
            order = [str(p) for p in rng.permutation(b.parts)]
            stages, err = [], None
            for i, part in enumerate(order):
                ap = b.select(part)
                try:
                    if i > 0 and hasattr(pol, "recover"):
                        pol.recover(ap)
                    pol.policy(ap)
                except robot.EpisodeOver:
                    pass
                except Exception as e:
                    err = f"{part}: {type(e).__name__}: {e}"
                o = ap.outcome()
                stages.append(dict(part=part, success=bool(o.get("success")), steps=o.get("steps")))
                if o.get("timeout"):
                    break
                if o.get("success") and hasattr(pol, "finish"):
                    fa = b.free()
                    try:
                        pol.finish(fa)
                    except robot.EpisodeOver:
                        pass
                    except Exception as e:
                        err = f"finish {part}: {type(e).__name__}: {e}"
            o = b.outcome()
            b._conn.rpc(cmd="save_video", ep=b._ep, name=f"{name}/seed{s:02d}")
        res.append(dict(seed=seed, order=order, stages=stages, seated=o["seated"], n_seated=o["n_seated"], n=o["n"],
                        time_s=o["time_s"], timeout=o["timeout"], error=err))
        print(f"seed {seed}: {o['n_seated']}/{o['n']} seated, {o['time_s']} s{'  ' + err if err else ''}", flush=True)
    return dict(tasks=a.tasks, per_part_s=a.horizon, n_seeds=len(res),
                seated_mean=round(float(np.mean([r["n_seated"] for r in res])), 2),
                all_seated_rate=round(100 * float(np.mean([r["n_seated"] == r["n"] for r in res])), 1),
                per_part={p: round(100 * float(np.mean([r["seated"][p] for r in res])), 1) for p in res[0]["seated"]}), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", required=True)
    ap.add_argument("--task", default="R8"); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--board", action="store_true"); ap.add_argument("--tasks", default="board20"); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--horizon", type=float, default=240.0)
    ap.add_argument("--out", required=True, help="where eval.json goes (videos: server --out/<name>/)")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    token = os.environ.get("SERVER_TOKEN") or sys.exit("SERVER_TOKEN not set")
    sandbox = Path(a.sandbox).resolve()
    pol = load_policy(sandbox)
    import robot                                  # the sandbox's client (sys.path set by load_policy)
    name = a.name or (f"board_{a.tasks}" if a.board else f"single_{a.task}") + time.strftime("_%m%d_%H%M")
    summary, rows = (board if a.board else single)(a, robot, pol, token, name)
    summary.update(policy=str(sandbox / "policy.py"), videos=name)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(dict(summary=summary, episodes=rows), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
