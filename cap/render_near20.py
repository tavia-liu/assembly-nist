"""Videos of the near20 bank starts and of a board-PPO checkpoint rolling out from them.

  PYTHONPATH=.:cap MUJOCO_GL=egl python cap/render_near20.py --task P4 --starts 8 --ckpt runs/board_P4/checkpoint.pt --episodes 6
Writes cap/runs/perception/near20/<task>_starts.mp4 (+ _starts.png contact sheet) and <task>_<ckpt>_rollouts.mp4.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from assembly_policy.evaluate import TorchPolicy, env_config
from assembly_policy.sim.env import EnvConfig
from assembly_policy.sim.tasks import resolve_tasks
from multi_board import render
from multi_env import MultiPartEnv

ROOT = Path(__file__).resolve().parents[1]
FPS = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="P4")
    ap.add_argument("--bank-dir", default=str(ROOT / "data/banks_near20"))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--horizon", type=float, default=15.0)
    ap.add_argument("--out", default=str(ROOT / "cap/runs/perception/near20"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks = resolve_tasks("all20")
    z = np.load(Path(a.bank_dir) / f"{a.task}_{a.split}.npz")
    S, lev = z["states"], z["level"]
    # ---- the starts: hold each for 1.5 s
    frames, stages = [], []
    for k in range(min(a.starts, len(S))):
        frames += [S[k]] * int(1.5 * FPS)
        stages.append((len(frames) - 1, f"{a.task} near start {k} (level {lev[k]:.2f})", True))
    n = render(tasks, frames, stages, str(out / f"{a.task}_starts.mp4"), 1280, 720, banner_s=1.5)
    print(f"wrote {out / f'{a.task}_starts.mp4'} ({n} frames)", flush=True)
    # contact sheet: first frame of each start, from the video's main camera
    import imageio.v2 as imageio
    rd = imageio.get_reader(str(out / f"{a.task}_starts.mp4"))
    tiles = [rd.get_data(int(1.5 * FPS) * k) for k in range(min(a.starts, len(S)))]
    cols = 4; rows = -(-len(tiles) // cols)
    tiles += [np.zeros_like(tiles[0])] * (rows * cols - len(tiles))
    sheet = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range(rows)], 0)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(out / f"{a.task}_starts.png")
    if not a.ckpt or not Path(a.ckpt).exists():
        print("no checkpoint yet; starts only", flush=True)
        return
    # ---- rollouts of the checkpoint
    pol = TorchPolicy(a.ckpt, [a.task], "cpu", deterministic=False)
    cfg = env_config(pol.args, 8); cfg.horizon_s = a.horizon
    N = min(a.episodes, len(S))
    env = MultiPartEnv(tasks, cfg, n_envs=N, active=a.task, physics_of=a.task)
    env.set_states(np.arange(N), S[:N])
    hist = [env.state.copy()]
    done_at = np.full(N, -1); success = np.zeros(N, bool)
    torch.manual_seed(0)
    for t in range(env.horizon):
        _, done, info = env.step(pol(env))
        newly = (done_at < 0) & done
        done_at[newly] = t; success[newly] = info["success"][newly]
        hist.append(env.state.copy())
        if (done_at >= 0).all():
            break
    env.close()
    hist = np.stack(hist)
    frames, stages = [], []
    for k in range(N):
        last = int(done_at[k] + 1) if done_at[k] >= 0 else len(hist) - 1
        frames += list(hist[:last + 1, k]) + [hist[last, k]] * FPS
        secs = last * cfg.control_dt
        stages.append((len(frames) - 1 - FPS, f"start {k}: {'SUCCESS in %.1f s' % secs if success[k] else 'not assembled (%.0f s)' % secs}", bool(success[k])))
    tag = Path(a.ckpt).stem + (f"_it{pol.it}" if pol.it is not None else "")
    n = render(tasks, frames, stages, str(out / f"{a.task}_{tag}_rollouts.mp4"), 1280, 720, banner_s=1.0)
    print(f"wrote {out / f'{a.task}_{tag}_rollouts.mp4'} ({n} frames): {int(success.sum())}/{N} success", flush=True)


if __name__ == "__main__":
    main()
