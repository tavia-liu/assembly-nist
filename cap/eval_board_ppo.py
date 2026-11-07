"""Evaluate a board-trained PPO checkpoint on the near20 eval bank (all 20 parts physical, one active part).

  PYTHONPATH=.:cap python cap/eval_board_ppo.py --ckpt runs/board_P4/model_01000.pt --task P4 [--deterministic]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from assembly_policy.evaluate import TorchPolicy, env_config
from assembly_policy.sim.tasks import resolve_tasks
from multi_env import MultiPartEnv

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="P4")
    ap.add_argument("--bank-dir", default=str(ROOT / "data/banks_near20"))
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--horizon", type=float, default=15.0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--nthread", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    bank = np.load(Path(a.bank_dir) / f"{a.task}_eval.npz")
    S = bank["states"][:a.n]
    N = len(S)
    pol = TorchPolicy(a.ckpt, [a.task], "cpu", deterministic=a.deterministic)
    cfg = env_config(pol.args, a.nthread)
    cfg.horizon_s = a.horizon
    env = MultiPartEnv(resolve_tasks("all20"), cfg, n_envs=N, active=a.task, physics_of=a.task)
    env.set_states(np.arange(N), S)
    done_at = np.full(N, -1); success = np.zeros(N, bool); safety = np.zeros(N, bool)
    torch.manual_seed(0); t0 = time.time()
    for t in range(env.horizon):
        _, done, info = env.step(pol(env))
        newly = (done_at < 0) & done
        done_at[newly] = t; success[newly] = info["success"][newly]; safety[newly] = info["safety"][newly]
        if (done_at >= 0).all():
            break
    env.close()
    times = [(done_at[k] + 1) * cfg.control_dt for k in range(N) if success[k]]
    res = dict(ckpt=a.ckpt, task=a.task, n=N, n_ok=int(success.sum()), success_rate=round(100 * success.mean(), 1),
               safety=int(safety.sum()), time_mean_s=round(float(np.mean(times)), 2) if times else None,
               horizon_s=a.horizon, deterministic=a.deterministic, iteration=pol.it, wall_s=round(time.time() - t0, 1))
    print(json.dumps(res), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
