"""Several parts in a row with a cap policy (default: the code+RL hybrid), NIST board protocol per part.

All parts of --tasks lie at random settled board poses (non-overlapping, from data/banks_board), the robot starts at
home with the gripper open, a random order is drawn, and the policy assembles one part at a time; the robot state
carries over between parts (no scripted transitions: the policy itself releases, lifts and re-approaches). Each part
has --per-part-s seconds. As in assembly_policy.viz.demo, each stage simulates the active task's scene: the other
parts are drawn but not physical.

Output per seed: <out>/seedNN.json (per-part success and time, all-assembled flag) and <out>/seedNN.mp4
(main view + wrist close-up, result banner per part) for the first --video-seeds seeds; <out>/summary.json.

  PYTHONPATH=.:cap MUJOCO_GL=egl python cap/multi_part.py --tasks R8+P12+Gm+USB+N8 --n-seeds 10 --video-seeds 2
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np

import api
from assembly_policy.sim.resets import FAMILIES
from assembly_policy.sim.tasks import resolve_tasks
from assembly_policy.viz.demo import _separated, fixture_footprints, make_order, part_footprint, render
from eval_protocol import CAP_BANK_DIR, START_FAMILY

ROOT = Path(__file__).resolve().parents[1]


def board_starts(task, bank_dir, rng):
    z = np.load(Path(bank_dir) / f"{task}_eval.npz")
    idx = np.nonzero(z["family"] == FAMILIES.index(START_FAMILY))[0]
    return z["states"][rng.permutation(idx)]


def choose_layout(tasks, bank_dir, sp, rng, margin=0.006):
    """One board start per task such that no part touches another part or any fixture (demo.choose_layout, board)."""
    fixtures = fixture_footprints()
    chosen, polys = {}, []
    for t in rng.permutation(tasks):
        pool = board_starts(t, bank_dir, rng)
        for m in (margin, margin / 2, 0.0):
            for S in pool:
                poly = part_footprint(t, S[sp[:3]], S[sp[3:7]])
                if all(_separated(poly, f, m) for f in fixtures) and all(_separated(poly, q, m) for q in polys):
                    chosen[t] = S.copy(); polys.append(poly)
                    break
            if t in chosen:
                break
        if t not in chosen:
            chosen[t] = pool[0].copy(); polys.append(part_footprint(t, pool[0][sp[:3]], pool[0][sp[3:7]]))
            print(f"[layout] {t}: no overlap-free start found", flush=True)
    return chosen


class Run:
    """What demo.render needs: tasks, env (state layout) and frames = [(stage, state, phase)]."""

    def __init__(self, tasks, env):
        self.tasks, self.env, self.frames = tasks, env, []


def run_seed(seed, tasks, policy_mod, a, out):
    rng = np.random.default_rng(seed)
    order = make_order(tasks, "", rng)
    envs = {t: api.make_env(t, horizon_s=a.per_part_s) for t in tasks}
    e0 = envs[order[0]]
    m0 = e0.models[0].model
    robot_q = np.setdiff1d(np.arange(1, 1 + m0.nq), e0.sp)
    robot_v = np.setdiff1d(np.arange(1 + m0.nq, 1 + m0.nq + m0.nv), e0.spd)
    layout = choose_layout(tasks, a.bank_dir, e0.sp, rng)
    run = Run(tasks, e0)
    final_pose = {t: layout[t][e0.sp].copy() for t in tasks}
    results, cur, t_wall = [], None, time.time()
    dt = e0.ecfg.control_dt
    for i, t in enumerate(order):
        env = envs[t]
        S = layout[t].copy()                              # board start: robot at home, gripper open
        if cur is not None:                               # later parts: the robot where the previous part left it
            S[robot_q] = cur[robot_q]; S[robot_v] = 0.0
        pol = policy_mod.policy if i == 0 else (lambda api_, _p=policy_mod: (_p.recover(api_), _p.policy(api_)))
        r = api.run_episode(env, S, pol, level="grasp", seed=seed * 100 + i, record=True)
        run.frames += [(i, f, "policy") for f in r["frames"]]
        results.append(dict(task=t, success=bool(r["success"]), safety=bool(r.get("safety", False)),
                            seconds=round(r["steps"] * dt, 2), error=r.get("error")))
        print(f"seed {seed} part {i + 1}/{len(order)} {t}: {'assembled' if r['success'] else 'NOT assembled'} "
              f"in {r['steps'] * dt:.1f} s{'  error: ' + r['error'] if r.get('error') else ''}", flush=True)
        final_pose[t] = env.state[0, env.sp].copy()
        cur = env.state[0].copy()
    res = dict(seed=seed, order=order, parts=results, n_ok=sum(r["success"] for r in results), n=len(order),
               all_assembled=all(r["success"] for r in results),
               total_s=round(sum(r["seconds"] for r in results), 2), wall_s=round(time.time() - t_wall, 1))
    (out / f"seed{seed:02d}.json").write_text(json.dumps(res, indent=1))
    if seed - a.seed < a.video_seeds:
        n = render(run, layout, order, results, final_pose, str(out / f"seed{seed:02d}.mp4"), a.width, a.height,
                   None, slow=1, fast=1, banner_s=3.0)
        print(f"wrote {out / f'seed{seed:02d}.mp4'} ({n} frames)", flush=True)
    for e in envs.values():
        e.close()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="R8+P12+Gm+USB+N8")
    ap.add_argument("--policy", default="hybrid_rl", help="module in cap/ with policy(api) and recover(api)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--video-seeds", type=int, default=1, help="render the first k seeds")
    ap.add_argument("--per-part-s", type=float, default=40.0)
    ap.add_argument("--bank-dir", default=str(CAP_BANK_DIR))
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tasks = resolve_tasks(a.tasks)
    out = Path(a.out or ROOT / "cap/runs/rl/multi" / f"{a.policy}_{'+'.join(tasks)}")
    out.mkdir(parents=True, exist_ok=True)
    policy_mod = importlib.import_module(a.policy)
    allres = [run_seed(s, tasks, policy_mod, a, out) for s in range(a.seed, a.seed + a.n_seeds)]
    summ = dict(tasks=tasks, policy=a.policy, per_part_s=a.per_part_s, n_seeds=len(allres),
                all_assembled_rate=round(100 * np.mean([r["all_assembled"] for r in allres]), 1),
                parts_assembled_mean=round(float(np.mean([r["n_ok"] for r in allres])), 2),
                per_task_success={t: round(100 * np.mean([p["success"] for r in allres for p in r["parts"] if p["task"] == t]), 1)
                                  for t in tasks},
                total_s_mean=round(float(np.mean([r["total_s"] for r in allres])), 1))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
