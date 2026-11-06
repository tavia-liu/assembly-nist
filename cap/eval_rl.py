"""The AssembleNIST RL checkpoint under the cap benchmark protocol (same starts and metrics as eval_protocol.py).

Every episode is a full assembly from a `board` start (part loose on the board, robot at home, gripper open):
the first --n (default 100) board states of the task's frozen evaluation bank in <data>/banks_board, one episode
per start, success + completion time. Actions are sampled (as in the AlphaNIST evaluation) unless --deterministic.
Output per task: <out>/<TASK>/summary.json, episodes.json and videos/epNNN_<ok|fail>.mp4 for the first --record
episodes (default: all of them).

  PYTHONPATH=.:cap MUJOCO_GL=egl python cap/eval_rl.py --tasks R8 --out cap/runs/rl_eval/e8192_it12500
  PYTHONPATH=.:cap MUJOCO_GL=egl python cap/eval_rl.py --tasks all20 --out cap/runs/rl_eval/e8192_it12500
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image

from assembly_policy.evaluate import TorchPolicy, env_config
from assembly_policy.sim.env import BatchedAssemblyEnv, STATE_SPEC
from assembly_policy.sim.resets import FAMILIES
from assembly_policy.sim.scene import build_xml
from assembly_policy.sim.tasks import resolve_tasks
from eval_protocol import CAP_BANK_DIR, N_EVAL, START_FAMILY
from record import FPS, HOLD_S, camera_for, label

ROOT = Path(__file__).resolve().parents[1]


def run_task(ckpt, task, cfg, bank_dir, n, record_n, deterministic, device, seed=0):
    bank = np.load(Path(bank_dir) / f"{task}_eval.npz")
    idx = np.nonzero(bank["family"] == FAMILIES.index(START_FAMILY))[0][:n]
    N = len(idx)
    pol = TorchPolicy(ckpt, [task], device, deterministic=deterministic)
    env = BatchedAssemblyEnv([task], np.zeros(N, int), cfg)
    env.set_states(np.arange(N), bank["states"][idx], np.zeros(N, int))
    done_at = np.full(N, -1); success = np.zeros(N, bool); safety = np.zeros(N, bool)
    frames = [env.state[:record_n].copy()]
    torch.manual_seed(seed)
    t0 = time.time()
    for t in range(env.horizon):
        _, done, info = env.step(pol(env))
        newly = (done_at < 0) & done                  # finished episodes keep stepping; their outcome is frozen
        done_at[newly] = t; success[newly] = info["success"][newly]; safety[newly] = info["safety"][newly]
        if record_n:
            frames.append(env.state[:record_n].copy())
        if (done_at >= 0).all():
            break
    env.close()
    rows = []
    for k in range(N):
        steps = int(done_at[k] + 1) if done_at[k] >= 0 else int(env.horizon)
        rows.append(dict(trial=k, bank_idx=int(idx[k]), success=bool(success[k]), safety=bool(safety[k]),
                         timeout=bool(done_at[k] < 0), steps=steps,
                         time_s=round(steps * cfg.control_dt, 2) if success[k] else None))
    ok = [r for r in rows if r["success"]]
    times = [r["time_s"] for r in ok]
    summary = dict(task=task, horizon_s=cfg.horizon_s,
                   protocol=f"NIST task board: full assembly from the first {N} {START_FAMILY} starts (part loose on "
                            f"the board, robot at home, gripper open) of the frozen eval bank",
                   n=N, n_ok=len(ok), success_rate=round(100 * len(ok) / max(N, 1), 1),
                   time_mean_s=round(float(np.mean(times)), 2) if times else None,
                   time_median_s=round(float(np.median(times)), 2) if times else None,
                   safety=int(sum(r["safety"] for r in rows)), timeouts=int(sum(r["timeout"] for r in rows)),
                   policy=str(ckpt), iteration=pol.it, deterministic=deterministic, wall_s=round(time.time() - t0, 1))
    print(f"{task}: {len(ok)}/{N} ok ({summary['success_rate']}%)  time mean {summary['time_mean_s']} s  "
          f"safety {summary['safety']}  timeouts {summary['timeouts']}  [{summary['wall_s']} s]", flush=True)
    return summary, rows, (np.stack(frames) if record_n else None), done_at[:record_n], success[:record_n]


def render(task, frames, done_at, success, vdir, width=480, height=360):
    """One mp4 per recorded episode (same camera/labels as cap/record.py)."""
    T, n, _ = frames.shape
    vdir.mkdir(parents=True, exist_ok=True)
    vm = mujoco.MjModel.from_xml_string(build_xml(task, visual=True))
    vd = mujoco.MjData(vm)
    rend = mujoco.Renderer(vm, height, width)
    dt = 0.04
    for i in range(n):
        last = int(done_at[i] + 1) if done_at[i] >= 0 else T - 1
        cam = camera_for(vm, vd, frames[0, i], task)
        ok = bool(success[i])
        video = []
        for f in range(last + 1 + int(HOLD_S * FPS)):
            k = min(f, last)
            mujoco.mj_setState(vm, vd, frames[k, i], STATE_SPEC)
            mujoco.mj_forward(vm, vd)
            rend.update_scene(vd, cam)
            img = Image.fromarray(rend.render()).convert("RGBA")
            video.append(label(img, [f"{task} | RL | seed {i}", f"t={k * dt:4.1f}s",
                                     ("SUCCESS" if ok else "FAIL") if k == last else ""],
                               [(235, 235, 235, 255), (190, 200, 215, 255),
                                (120, 255, 120, 255) if ok else (255, 150, 120, 255)]))
        imageio.mimwrite(vdir / f"ep{i:03d}_{'ok' if ok else 'fail'}.mp4", video, fps=FPS, codec="libx264",
                         quality=7, macro_block_size=8)
    print(f"wrote {n} videos to {vdir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/e8192_it12500.pt"))
    ap.add_argument("--tasks", default="R8")
    ap.add_argument("--bank-dir", default=str(CAP_BANK_DIR))
    ap.add_argument("--n", type=int, default=N_EVAL)
    ap.add_argument("--horizon", type=float, default=40.0, help="episode time limit in seconds")
    ap.add_argument("--record", type=int, default=N_EVAL, help="episodes per task rendered to <out>/<TASK>/videos/")
    ap.add_argument("--deterministic", action="store_true", help="mean actions (default: sampled, as in the paper)")
    ap.add_argument("--nthread", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "cap/runs/rl_eval/e8192_it12500"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    args = torch.load(a.ckpt, map_location="cpu", weights_only=False)["args"]
    cfg = dataclasses.replace(env_config(args, a.nthread), horizon_s=a.horizon)
    print(f"ckpt {a.ckpt}  horizon {a.horizon:g} s  starts {START_FAMILY}  bank {a.bank_dir}", flush=True)
    all_summ = {}
    for task in resolve_tasks(a.tasks):
        summary, rows, frames, done_at, success = run_task(a.ckpt, task, cfg, a.bank_dir, a.n, a.record,
                                                           a.deterministic, a.device)
        tdir = out / task; tdir.mkdir(exist_ok=True)
        (tdir / "summary.json").write_text(json.dumps(summary, indent=1))
        (tdir / "episodes.json").write_text(json.dumps(rows, indent=1))
        all_summ[task] = summary
        if frames is not None:
            try:
                render(task, frames, done_at, success, tdir / "videos")
            except Exception as e:                       # rendering is optional; the numbers are not
                print(f"[warn] no video for {task}: {type(e).__name__}: {e}", flush=True)
    if len(all_summ) > 1:
        rates = [s["success_rate"] for s in all_summ.values()]
        (out / "summary.json").write_text(json.dumps(dict(
            mean_success_rate=round(float(np.mean(rates)), 1), horizon_s=a.horizon, deterministic=a.deterministic,
            per_task={t: dict(success_rate=s["success_rate"], n_ok=s["n_ok"], n=s["n"], time_mean_s=s["time_mean_s"])
                      for t, s in all_summ.items()}), indent=1))
        print(f"mean over {len(rates)} tasks: {np.mean(rates):.1f}%", flush=True)


if __name__ == "__main__":
    main()
