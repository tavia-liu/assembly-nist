"""The NIST task-board benchmark with several parts physically on the board at once (cap/multi_env.py).

All parts of --tasks lie at random settled board poses (non-overlapping), the robot starts at home with the
gripper open, and a cap policy assembles them one after another in one continuous episode: code picks the next
part (--order greedy = nearest to the hand), the per-part policy (default: the code+RL hybrid) does the rest.
The whole board has --per-part-s x n_parts seconds. Scored at the end: parts seated (assembly distance < 1 mm)
and total time; per-part stage outcomes are logged too.

Output per seed: <out>/seedNN.json and <out>/seedNN.mp4 (first --video-seeds seeds); <out>/summary.json.

  PYTHONPATH=.:cap MUJOCO_GL=egl python cap/multi_board.py --tasks R8+P12+Gm+USB+N8 --n-seeds 10 --video-seeds 2
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import api
from assembly_policy.sim.env import STATE_SPEC, EnvConfig
from assembly_policy.sim.tasks import resolve_tasks
from assembly_policy.viz.demo import NAMES
from eval_protocol import CAP_BANK_DIR
from multi_env import MultiPartEnv
from multi_part import choose_layout
from multi_scene import build_multi_xml

ROOT = Path(__file__).resolve().parents[1]
FPS = 25
PRESETS = {"board5": "R8+P12+Gm+USB+N8",
           "board10": "R8+R16+P8+P16+Gm+N8+N16+BNC+USB+WP",
           "board20": "all20"}


def render(tasks, frames, stages, out, width=1280, height=720, banner_s=3.0):
    """frames: states; stages: [(end_frame_index, text, ok)] -> mp4 with a main view, a wrist close-up, a clock
    and a result banner after each part."""
    vm = mujoco.MjModel.from_xml_string(build_multi_xml(tasks, visual=True))
    vd = mujoco.MjData(vm)
    main = mujoco.Renderer(vm, height, width)
    iw, ih = int(width * 0.38), int(height * 0.38)
    inset = mujoco.Renderer(vm, ih, iw)
    cam = mujoco.MjvCamera(); cam.distance = 0.8; cam.azimuth = 100.0; cam.elevation = -40.0
    right = np.array([math.sin(math.radians(cam.azimuth)), -math.cos(math.radians(cam.azimuth)), 0.0])
    cam.lookat[:] = np.array([0.47, 0.0, 0.05]) + 0.09 * right
    close = mujoco.MjvCamera(); close.distance = 0.2; close.azimuth = 215.0; close.elevation = -20.0
    tcp_site = mujoco.mj_name2id(vm, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    lookat = None
    video, banner = [], None
    stage_at = {end: (text, ok) for end, text, ok in stages}
    for k, s in enumerate(frames):
        mujoco.mj_setState(vm, vd, s, STATE_SPEC)
        mujoco.mj_forward(vm, vd)
        main.update_scene(vd, cam)
        img = Image.fromarray(main.render()).convert("RGBA")
        tcp = vd.site_xpos[tcp_site].copy() - np.array([0.0, 0.0, 0.01])
        lookat = tcp if lookat is None else 0.8 * lookat + 0.2 * tcp
        close.lookat[:] = lookat
        inset.update_scene(vd, close)
        img.paste(Image.fromarray(inset.render()).convert("RGBA"), (width - iw - 12, 12))
        dr = ImageDraw.Draw(img)
        dr.rectangle([width - iw - 14, 10, width - 10, 12 + ih + 2], outline=(255, 255, 255, 255), width=2)
        clock = f"{int(k / FPS) // 60}:{(k / FPS) % 60:04.1f}"
        dr.text((width - 14 - dr.textlength(clock, font=font), height - 38), clock, fill=(255, 255, 255, 255), font=font)
        if k in stage_at:
            banner = (len(video), *stage_at[k])
        if banner is not None and len(video) - banner[0] < banner_s * FPS:
            _, msg, ok = banner
            bw = dr.textlength(msg, font=font) + 24; y0 = height - 52
            dr.rectangle([12, y0, 12 + bw, y0 + 38], fill=(20, 140, 60, 220) if ok else (170, 40, 40, 220))
            dr.text((24, y0 + 7), msg, fill=(255, 255, 255, 255), font=font)
        video.append(np.asarray(img.convert("RGB")))
    for _ in range(2 * FPS):
        video.append(video[-1])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, video, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    return len(video)


def run_seed(seed, tasks, policy_mod, a, out, sp_single):
    rng = np.random.default_rng(seed)
    env = MultiPartEnv(tasks, EnvConfig(nthread=1, horizon_s=a.per_part_s * len(tasks)))
    dt = env.ecfg.control_dt
    layout = choose_layout(tasks, a.bank_dir, sp_single, rng)
    env.reset_board(layout, sp_single)
    frames, stages, results = [env.state[0].copy()], [], []
    remaining, t_wall = list(tasks), time.time()
    random_order = list(rng.permutation(tasks))
    timed_out = False
    for i in range(len(tasks)):
        if a.order == "greedy":                       # code decides: the part nearest to the hand
            tcp = env.tcp_pos()
            t = min(remaining, key=lambda u: np.linalg.norm(env.part_pos(u)[:2] - tcp[:2]))
        else:
            t = next(u for u in random_order if u in remaining)
        remaining.remove(t)
        env.set_active(t)
        t0 = int(env.t[0])
        err, tries = None, 0
        while True:                                   # a safety stop (velocity limit) ends the attempt, not the part
            ap = api.AssemblyAPI(env, rng, level="grasp", record=True)
            try:
                if i > 0 or tries > 0:
                    policy_mod.recover(ap)
                policy_mod.policy(ap)
            except api.EpisodeOver:
                pass
            except Exception as e:                    # a policy bug is a result, not a crash
                err = f"{type(e).__name__}: {e}"
            frames += ap.frames + [env.state[0].copy()]
            out_ = ap.outcome if ap.done else dict(success=False, safety=False, timeout=False)
            tries += 1
            if out_.get("safety") and not out_.get("success") and tries < a.retries + 1:
                print(f"seed {seed} part {i + 1} {t}: safety stop after {(int(env.t[0]) - t0) * dt:.1f} s, retry {tries}", flush=True)
                continue
            break
        d_mm = round(float(env.assembly_distance(t)[0]) * 1000, 2)
        if out_["success"] and hasattr(policy_mod, "finish") and not out_.get("timeout"):
            env.scoring = False                       # the part is seated: press, release, lift (no re-trigger)
            ap2 = api.AssemblyAPI(env, rng, level="grasp", record=True)
            try:
                policy_mod.finish(ap2)
            except api.EpisodeOver:
                pass
            except Exception as e:
                err = f"finish: {type(e).__name__}: {e}"
            env.scoring = True
            frames += ap2.frames + [env.state[0].copy()]
            if ap2.done and ap2.outcome.get("timeout"):
                out_["timeout"] = True
        secs = round((int(env.t[0]) - t0) * dt, 2)
        after = {u: round(float(env.assembly_distance(u)[0]) * 1000, 2) for u in tasks}
        results.append(dict(stage=i, task=t, success=bool(out_["success"]), safety=bool(out_.get("safety", False)),
                            tries=tries, seconds=secs, d_asm_mm=d_mm, d_asm_after_release_mm=after[t],
                            after_stage_mm=after, error=err))
        stages.append((len(frames) - 1, f"{NAMES.get(t, t)}: {'assembled in %.1f s' % secs if out_['success'] else 'not assembled (%.0f s)' % secs}",
                       bool(out_["success"])))
        knocked = [s_["task"] for s_ in results[:-1]        # seated after its own stage, displaced during this one
                   if s_["after_stage_mm"][s_["task"]] < 1.0 and after[s_["task"]] >= 1.0]
        print(f"seed {seed} part {i + 1}/{len(tasks)} {t}: {'assembled' if out_['success'] else 'NOT assembled'} "
              f"in {secs:.1f} s  d_asm {d_mm} -> {after[t]} mm after release"
              f"{'  knocked out: ' + ','.join(knocked) if knocked else ''}{'  ' + err if err else ''}", flush=True)
        if out_.get("timeout"):
            timed_out = True
            break
    final = env.assembled()
    res = dict(seed=seed, tasks=tasks, order=[r["task"] for r in results], stages=results,
               final={t: dict(ok=v["ok"], d_asm_mm=round(v["d_asm"] * 1000, 2)) for t, v in final.items()},
               n_seated=sum(v["ok"] for v in final.values()), n=len(tasks),
               all_seated=all(v["ok"] for v in final.values()), timed_out=timed_out,
               total_s=round(int(env.t[0]) * dt, 2), wall_s=round(time.time() - t_wall, 1))
    (out / f"seed{seed:02d}.json").write_text(json.dumps(res, indent=1))
    print(f"seed {seed}: {res['n_seated']}/{res['n']} seated at the end, {res['total_s']} s  [{res['wall_s']} s wall]", flush=True)
    if seed - a.seed < a.video_seeds:
        n = render(tasks, frames, stages, str(out / f"seed{seed:02d}.mp4"), a.width, a.height)
        print(f"wrote {out / f'seed{seed:02d}.mp4'} ({n} frames)", flush=True)
    env.close()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="board5", help="task list, or board5 / board10 / board20 (= all20)")
    ap.add_argument("--policy", default="hybrid_rl", help="module in cap/ with policy(api) and recover(api)")
    ap.add_argument("--order", default="greedy", choices=["greedy", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--video-seeds", type=int, default=1)
    ap.add_argument("--per-part-s", type=float, default=40.0)
    ap.add_argument("--retries", type=int, default=2, help="extra attempts on a part after a safety stop")
    ap.add_argument("--bank-dir", default=str(CAP_BANK_DIR))
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--tag", default="", help="suffix of the default output folder")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tasks = resolve_tasks(PRESETS.get(a.tasks, a.tasks))
    out = Path(a.out or ROOT / "cap/runs/board" / f"{a.policy}_{a.order}_{'+'.join(tasks)}{a.tag}")
    out.mkdir(parents=True, exist_ok=True)
    policy_mod = importlib.import_module(a.policy)
    e1 = api.make_env(tasks[0]); sp_single = e1.sp.copy(); e1.close()
    allres = [run_seed(s, tasks, policy_mod, a, out, sp_single) for s in range(a.seed, a.seed + a.n_seeds)]
    summ = dict(tasks=tasks, policy=a.policy, order=a.order, per_part_s=a.per_part_s, n_seeds=len(allres),
                all_seated_rate=round(100 * np.mean([r["all_seated"] for r in allres]), 1),
                seated_mean=round(float(np.mean([r["n_seated"] for r in allres])), 2),
                per_task_seated={t: round(100 * np.mean([r["final"][t]["ok"] for r in allres]), 1) for t in tasks},
                total_s_mean=round(float(np.mean([r["total_s"] for r in allres])), 1))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
