"""Evaluate a symmetric pixel PPO checkpoint (cap/train_pixel.py) on the near eval bank and render the first episodes.

  PYTHONPATH=.:cap MUJOCO_GL=egl python cap/eval_pixel.py --ckpt runs/peg_4mm_pixel_near/model_001000.pt --task P4 [--deterministic]
-> <run>/eval_<ckpt>[_det].json and <run>/videos/<ckpt>_rollouts[_det].mp4 (first --video starts, global | wrist)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from agent_tools import Vis
from assembly_policy.evaluate import env_config
from assembly_policy.sim.resets import FAMILIES
from pixel_env import PixelEnv
from pixel_nets import PixelActorCritic
from record import label

FPS = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="P4")
    ap.add_argument("--bank-dir", default="data/banks_near")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--horizon", type=float, default=15.0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--nthread", type=int, default=8)
    ap.add_argument("--video", type=int, default=8, help="render the first K starts (0 = no video)")
    a = ap.parse_args()
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = d["args"]; res = args.get("pix_res", 96)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PixelActorCritic(res=res, max_std=args.get("max_std", 2.0), min_std=args.get("min_std", 1e-3)).to(dev)
    model.load_state_dict(d["model"]); model.eval()
    cfg = env_config(args, a.nthread); cfg.horizon_s = a.horizon
    z = np.load(Path(a.bank_dir) / f"{a.task}_eval.npz")
    S = z["states"][np.nonzero(z["family"] == FAMILIES.index("near"))[0][:a.n]]
    N = len(S); K = min(a.video, N)
    env = PixelEnv([a.task], np.zeros(N, int), cfg, res=res)
    env.set_states(np.arange(N), S, np.zeros(N, int))
    hist = [env.state[:K].copy()]
    done_at = np.full(N, -1); success = np.zeros(N, bool); safety = np.zeros(N, bool)
    torch.manual_seed(0); t0 = time.time()
    with torch.no_grad():
        for t in range(env.horizon):
            o = env.observe()
            z_ = model.encode(torch.as_tensor(o["img"], device=dev))
            dist = model.dist(z_, torch.as_tensor(o["proprio"].reshape(N, -1), dtype=torch.float32, device=dev))
            act = (dist.mean if a.deterministic else dist.sample()).cpu().numpy()
            _, done, info = env.step(act)
            newly = (done_at < 0) & done
            done_at[newly] = t; success[newly] = info["success"][newly]; safety[newly] = info["safety"][newly]
            hist.append(env.state[:K].copy())
            if (done_at >= 0).all():
                break
    env.close()
    times = [(done_at[k] + 1) * cfg.control_dt for k in range(N) if success[k]]
    tag = "_det" if a.deterministic else ""
    res_ = dict(ckpt=a.ckpt, iteration=d.get("it"), transitions=d.get("transitions"), task=a.task, n=N, n_ok=int(success.sum()),
                success_rate=round(100 * success.mean(), 1), safety=int(safety.sum()), timeouts=int((done_at < 0).sum()),
                time_mean_s=round(float(np.mean(times)), 2) if times else None, deterministic=a.deterministic,
                per_start=[("ok" if success[k] else "safety" if safety[k] else "timeout") for k in range(N)],
                wall_s=round(time.time() - t0, 1))
    print(json.dumps({k: v for k, v in res_.items() if k != "per_start"}), flush=True)
    run = Path(a.ckpt).parent
    (run / f"eval_{Path(a.ckpt).stem}{tag}.json").write_text(json.dumps(res_, indent=1))
    if K == 0:
        return
    hist = np.stack(hist)
    vis = Vis(a.task, w=640, h=480)
    frames = []
    for k in range(K):
        last = int(done_at[k] + 1) if done_at[k] >= 0 else len(hist) - 1
        cam = vis.record.camera_for(vis.m, vis.d, hist[0, k], a.task)
        verdict = "SUCCESS" if success[k] else ("SAFETY STOP" if safety[k] else "TIMEOUT")
        for j in range(last + 1):
            img = Image.fromarray(vis.frame(hist[j, k], cam)).convert("RGBA")
            frames.append(label(img, [f"{a.task}  it{d.get('it')}  start {k}   t={j * cfg.control_dt:4.1f}s", "left: board camera   right: wrist camera",
                                      verdict if j == last else ""],
                                [(235, 235, 235, 255), (190, 200, 215, 255), (120, 255, 120, 255) if success[k] else (255, 150, 120, 255)]))
        frames += [frames[-1]] * FPS
    out = run / "videos" / f"{Path(a.ckpt).stem}_rollouts{tag}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
