"""Evaluate an encoder-feature PPO checkpoint on the near eval bank (single-part scene, cameras -> ResNet-18 features).

  PYTHONPATH=.:cap MUJOCO_GL=egl python cap/eval_feat.py --ckpt runs/feat_P4/model_004000.pt --task P4 [--deterministic]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from assembly_policy.evaluate import env_config
from assembly_policy.rl.obs import PointTables
from assembly_policy.sim.resets import FAMILIES
from encoded_env import FEAT_DIM, EncodedEnv
from feat_nets import FeatActorCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="P4")
    ap.add_argument("--bank-dir", default="data/banks_near")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--horizon", type=float, default=15.0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--nthread", type=int, default=8)
    a = ap.parse_args()
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = d["args"]
    pts = PointTables([a.task], args["n_part"], args["n_fix"], args["n_grip"], device="cpu")
    model = FeatActorCritic(pts.n_pts, FEAT_DIM, max_std=args.get("max_std", 2.0), min_std=args.get("min_std", 1e-3))
    model.load_state_dict(d["model"], strict=False); model.eval()
    cfg = env_config(args, a.nthread); cfg.horizon_s = a.horizon
    z = np.load(Path(a.bank_dir) / f"{a.task}_eval.npz")
    S = z["states"][np.nonzero(z["family"] == FAMILIES.index("near"))[0][:a.n]]
    N = len(S)
    env = EncodedEnv([a.task], np.zeros(N, int), cfg)
    env.set_states(np.arange(N), S, np.zeros(N, int))
    done_at = np.full(N, -1); success = np.zeros(N, bool); safety = np.zeros(N, bool)
    torch.manual_seed(0); t0 = time.time()
    with torch.no_grad():
        for t in range(env.horizon):
            o = env.observe()
            dist = model.dist((None, torch.as_tensor(o["feat"])), torch.as_tensor(o["proprio"].reshape(N, -1)))
            act = (dist.mean if a.deterministic else dist.sample()).numpy()
            _, done, info = env.step(act)
            newly = (done_at < 0) & done
            done_at[newly] = t; success[newly] = info["success"][newly]; safety[newly] = info["safety"][newly]
            if (done_at >= 0).all():
                break
    env.close()
    times = [(done_at[k] + 1) * cfg.control_dt for k in range(N) if success[k]]
    res = dict(ckpt=a.ckpt, iteration=d.get("it"), task=a.task, n=N, n_ok=int(success.sum()), success_rate=round(100 * success.mean(), 1),
               safety=int(safety.sum()), timeouts=int((done_at < 0).sum()), time_mean_s=round(float(np.mean(times)), 2) if times else None,
               deterministic=a.deterministic, wall_s=round(time.time() - t0, 1))
    print(json.dumps(res), flush=True)
    out = Path(a.ckpt).parent / f"eval_{Path(a.ckpt).stem}{'_det' if a.deterministic else ''}.json"
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
