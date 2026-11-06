"""Run a cap policy on n start states; print outcome and primitive trace per episode.

  PYTHONPATH=.:cap python cap/eval_policy.py --task R16 --family near --n 6 --policy oracle_r16:policy
"""
from __future__ import annotations

import argparse
import importlib
import json

import api


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="R16")
    ap.add_argument("--family", default="near")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--policy", default="oracle_r16:policy", help="module:attr")
    ap.add_argument("--level", default="grasp")
    ap.add_argument("--trace", action="store_true", help="print the primitive trace of every episode")
    a = ap.parse_args()

    mod, attr = a.policy.split(":")
    policy = getattr(importlib.import_module(mod), attr)
    env = api.make_env(a.task)
    ok = 0
    for i, s in enumerate(api.sample_starts(a.task, a.family, a.n, seed=0)):
        r = api.run_episode(env, s, policy, level=a.level, seed=i)
        ok += bool(r["success"])
        print(f"ep{i}: success={r['success']} d_asm={r['d_asm'] * 1000:.1f}mm steps={r['steps']} "
              f"safety={r['safety']} error={r['error']}", flush=True)
        if a.trace:
            for t in r["trace"]:
                print("   ", json.dumps(t), flush=True)
    print(f"==> {ok}/{a.n} success  ({a.policy}, {a.task} {a.family})")


if __name__ == "__main__":
    main()
