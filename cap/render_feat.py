"""Roll out an encoder-feature PPO checkpoint (cap/train_feat.py) from near starts and render a video.

  PYTHONPATH=.:cap MUJOCO_GL=egl python cap/render_feat.py --ckpt runs/feat_P4/model_00400.pt --task P4 --episodes 6
-> <run>/videos/<ckpt>_rollouts.mp4 (global camera | wrist camera, one episode after another, result banner)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from agent_tools import Vis
from assembly_policy.evaluate import env_config
from assembly_policy.rl.obs import PointTables
from assembly_policy.sim.resets import FAMILIES
from encoded_env import FEAT_DIM, EncodedEnv
from feat_nets import FeatActorCritic
from record import label

FPS = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="P4")
    ap.add_argument("--bank-dir", default="data/banks_near")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--horizon", type=float, default=15.0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = d["args"]
    pts = PointTables([a.task], args["n_part"], args["n_fix"], args["n_grip"], device="cpu")
    model = FeatActorCritic(pts.n_pts, FEAT_DIM, max_std=args.get("max_std", 2.0), min_std=args.get("min_std", 1e-3))
    model.load_state_dict(d["model"], strict=False); model.eval()
    cfg = env_config(args, 4); cfg.horizon_s = a.horizon
    z = np.load(Path(a.bank_dir) / f"{a.task}_eval.npz")
    S = z["states"][np.nonzero(z["family"] == FAMILIES.index("near"))[0][:a.episodes]]
    N = len(S)
    env = EncodedEnv([a.task], np.zeros(N, int), cfg)
    env.set_states(np.arange(N), S, np.zeros(N, int))
    hist = [env.state.copy()]
    done_at = np.full(N, -1); success = np.zeros(N, bool); safety = np.zeros(N, bool)
    torch.manual_seed(0)
    with torch.no_grad():
        for t in range(env.horizon):
            o = env.observe()
            feat = torch.as_tensor(o["feat"]); prop = torch.as_tensor(o["proprio"].reshape(N, -1))
            dist = model.dist((None, feat), prop)
            act = (dist.mean if a.deterministic else dist.sample()).numpy()
            _, done, info = env.step(act)
            newly = (done_at < 0) & done
            done_at[newly] = t; success[newly] = info["success"][newly]; safety[newly] = info["safety"][newly]
            hist.append(env.state.copy())
            if (done_at >= 0).all():
                break
    hist = np.stack(hist)
    vis = Vis(a.task, w=640, h=480)
    frames = []
    for k in range(N):
        last = int(done_at[k] + 1) if done_at[k] >= 0 else len(hist) - 1
        cam = vis.record.camera_for(vis.m, vis.d, hist[0, k], a.task)
        verdict = "SUCCESS" if success[k] else ("SAFETY STOP" if safety[k] else "TIMEOUT")
        for j in range(last + 1):
            img = Image.fromarray(vis.frame(hist[j, k], cam)).convert("RGBA")
            frames.append(label(img, [f"{a.task}  it{d.get('it')}  start {k}   t={j * cfg.control_dt:4.1f}s", "left: board camera   right: wrist camera",
                                      verdict if j == last else ""],
                                [(235, 235, 235, 255), (190, 200, 215, 255), (120, 255, 120, 255) if success[k] else (255, 150, 120, 255)]))
        frames += [frames[-1]] * FPS
    out = Path(a.out or Path(a.ckpt).parent / "videos" / f"{Path(a.ckpt).stem}_rollouts.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    print(f"wrote {out}: {int(success.sum())}/{N} success, {int(safety.sum())} safety stops, "
          f"lengths {[(int(done_at[k]) + 1 if done_at[k] >= 0 else env.horizon) for k in range(N)]}", flush=True)


if __name__ == "__main__":
    main()
