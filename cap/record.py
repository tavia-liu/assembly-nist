"""Render episodes of a cap policy to an mp4.

Episodes are simulated on CPU with frame states recorded, then replayed into the visual
model (original meshes, no collision proxies) offscreen. Offscreen GL needs a GPU here --
EGL has no device on the login node -- so run this under srun/sbatch with --gres=gpu:1.

  MUJOCO_GL=egl python -m cap.record --family near --n 6 --out cap/runs/near.mp4
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import api
from assembly_policy.sim.env import STATE_SPEC, EnvConfig
from assembly_policy.sim.scene import build_xml
from assembly_policy.sim.tasks import BANK_DIR, TASKS, scene_pose

FPS = 25
HOLD_S = 1.0          # freeze on the last frame this long


def camera_for(m, d, state0, task):
    """Frame the part's start position together with its fixture."""
    mujoco.mj_setState(m, d, state0, STATE_SPEC)
    mujoco.mj_forward(m, d)
    part = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "part")].copy()
    fix = scene_pose(TASKS[task].fixture_scene)[0]
    c = mujoco.MjvCamera()
    c.lookat[:] = 0.5 * (part + fix) + [0.0, 0.0, 0.02]
    c.distance = float(np.clip(1.3 * np.linalg.norm(part - fix) + 0.32, 0.38, 0.95))
    c.azimuth, c.elevation = 150.0, -32.0
    return c


def label(img, lines, colors):
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(over)
    w = int(max(dr.textlength(t) for t in lines)) + 12
    dr.rectangle([0, 0, w, 12 + 13 * len(lines)], fill=(15, 17, 22, 140))
    for i, (t, col) in enumerate(zip(lines, colors)):
        dr.text((6, 4 + 13 * i), t, fill=col)
    return np.asarray(Image.alpha_composite(img, over).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="R16")
    ap.add_argument("--family", default="near")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--policy", default="oracle_r16:policy", help="module:attr")
    ap.add_argument("--level", default="grasp")
    ap.add_argument("--out", default="cap/runs/episodes.mp4")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--from-log", default=None,
                    help="fix_loop log.jsonl: render the debug episodes of --turn from the training bank instead of "
                         "generator starts (policy defaults to that turn's vNN.py)")
    ap.add_argument("--turn", type=int, default=None)
    a = ap.parse_args()
    starts, labels = None, None
    if a.from_log:
        recs = [json.loads(l) for l in Path(a.from_log).read_text().splitlines()]
        rec = next(r for r in recs if r.get("turn") == a.turn and "episodes" in r and not r.get("eval"))
        bank = np.load(Path(BANK_DIR) / f"{a.task}_train.npz")["states"]
        starts = [bank[e["bank_idx"]] for e in rec["episodes"]]
        labels = [e["family"] for e in rec["episodes"]]
        a.policy = f"v{a.turn:02d}:policy"
        a.n = len(starts)
        if a.n > 12 and a.width == 480:
            a.width, a.height = 320, 240

    mod, attr = a.policy.split(":")
    policy = getattr(importlib.import_module(mod), attr)

    env = api.make_env(a.task)
    if starts is None:
        starts = api.sample_starts(a.task, a.family, a.n, seed=0)
        labels = [a.family] * len(starts)
    eps = []
    for i, s in enumerate(starts):
        r = api.run_episode(env, s, policy, level=a.level, seed=i, record=True)
        eps.append(r)
        print(f"ep{i}: success={r['success']} d_asm={r['d_asm']*1000:.1f}mm "
              f"frames={len(r['frames'])}", flush=True)

    vm = mujoco.MjModel.from_xml_string(build_xml(a.task, visual=True))
    vd = mujoco.MjData(vm)
    rend = mujoco.Renderer(vm, a.height, a.width)
    cams = [camera_for(vm, vd, e["frames"][0], a.task) for e in eps]

    cols = int(np.ceil(np.sqrt(len(eps) * 4 / 3)))
    rows = int(np.ceil(len(eps) / cols))
    n_frames = max(len(e["frames"]) for e in eps) + int(HOLD_S * FPS)
    dt = EnvConfig().control_dt
    video = []
    for f in range(n_frames):
        tiles = []
        for i, e in enumerate(eps):
            k = min(f, len(e["frames"]) - 1)
            mujoco.mj_setState(vm, vd, e["frames"][k], STATE_SPEC)
            mujoco.mj_forward(vm, vd)
            rend.update_scene(vd, cams[i])
            img = Image.fromarray(rend.render()).convert("RGBA")
            ok = e["success"]
            tiles.append(label(img, [f"{a.task} | {labels[i]} | ep{i}",
                                     f"t={k*dt:4.1f}s  d_asm={e['d_asm']*1000:5.1f}mm",
                                     ("SUCCESS" if ok else "FAIL") if k == len(e["frames"]) - 1 else ""],
                               [(235, 235, 235, 255), (190, 200, 215, 255),
                                (120, 255, 120, 255) if ok else (255, 150, 120, 255)]))
        while len(tiles) < rows * cols:
            tiles.append(np.zeros_like(tiles[0]))
        video.append(np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], 1)
                                     for r in range(rows)], 0))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(a.out, video, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    print(f"wrote {a.out}  ({len(video)} frames, {len(eps)} episodes, "
          f"{sum(e['success'] for e in eps)}/{len(eps)} success)")


if __name__ == "__main__":
    main()
