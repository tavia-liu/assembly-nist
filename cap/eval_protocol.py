"""Held-out evaluation: the NIST task-board benchmark, one part per episode.

Protocol. The part lies loose on the board at a random settled pose, the robot is at its home configuration with
the gripper open (reset family `board`); the policy has --horizon seconds (default 40) to assemble it. Success:
both of the part's keypoints within 1 mm of their goal, for whichever of the part's mesh symmetries fits best (a
rod seated either way up counts). Starts come from the simulator's reset generator seeded by --seed: seed 1000
(the default) is the held-out set; develop on other seeds. Every episode is rendered to an mp4. Stage verifiers a
policy may carry (api.strict_stages) do not stop an episode here: they are recorded in the trace only.
Reported: success rate, completion time of the successful episodes, timeouts, safety stops, policy errors.

  MUJOCO_GL=egl PYTHONPATH=.:cap python cap/eval_protocol.py --task R8 --policy my_policy.py --out runs/eval/R8
  (cap/fpose_server.py must be listening on $FPOSE_PORT; cap/eval.sbatch starts it on the job's GPU)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

import api
from assembly_policy.sim.env import STATE_SPEC
from assembly_policy.sim.scene import build_xml
from assembly_policy.sim.tasks import TASKS, scene_pose

START_FAMILY = "board"
EVAL_SEED = 1000
N_EVAL = 100
DT = 0.04
FPS = 25


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"policy_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_policy(path: Path):
    return load_module(path).policy


def starts(task, n=N_EVAL, seed=EVAL_SEED):
    """n `board` start states of the task, deterministic in seed."""
    return api.sample_starts(task, START_FAMILY, n, seed=seed)


class Video:
    """Renders an episode's recorded states (api.run_episode(record=True)) to an mp4: the part's start and its
    fixture framed together, a clock and the outcome."""

    def __init__(self, task, width=640, height=480):
        import mujoco
        self.mujoco, self.task = mujoco, task
        self.m = mujoco.MjModel.from_xml_string(build_xml(task, visual=True))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, height, width)
        self.fix = scene_pose(TASKS[task].fixture_scene)[0]

    def camera(self, state0):
        mj = self.mujoco
        mj.mj_setState(self.m, self.d, state0, STATE_SPEC); mj.mj_forward(self.m, self.d)
        part = self.d.xpos[mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "part")].copy()
        c = mj.MjvCamera()
        c.lookat[:] = 0.5 * (part + self.fix) + [0.0, 0.0, 0.02]
        c.distance = float(np.clip(1.3 * np.linalg.norm(part - self.fix) + 0.32, 0.38, 0.95))
        c.azimuth, c.elevation = 150.0, -32.0
        return c

    def write(self, res, out: Path, hold_s=1.0):
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
        frames = res["frames"]
        cam = self.camera(frames[0])
        ok, video = bool(res["success"]), []
        for k, s in enumerate(frames):
            self.mujoco.mj_setState(self.m, self.d, s, STATE_SPEC); self.mujoco.mj_forward(self.m, self.d)
            self.r.update_scene(self.d, cam)
            img = Image.fromarray(self.r.render()).convert("RGBA")
            over = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(over)
            lines = [f"{self.task}  t={k * DT:4.1f}s", ("SUCCESS" if ok else "FAIL") if k == len(frames) - 1 else ""]
            dr.rectangle([0, 0, int(max(dr.textlength(t) for t in lines)) + 12, 12 + 13 * len(lines)], fill=(15, 17, 22, 140))
            dr.text((6, 4), lines[0], fill=(235, 235, 235, 255))
            dr.text((6, 17), lines[1], fill=(120, 255, 120, 255) if ok else (255, 150, 120, 255))
            video.append(np.asarray(Image.alpha_composite(img, over).convert("RGB")))
        video += [video[-1]] * int(hold_s * FPS)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(out, video, fps=FPS, codec="libx264", quality=7, macro_block_size=8)


def evaluate(env, policy, task, level="final", n=N_EVAL, seed=EVAL_SEED, record=True, on_episode=None, verbose=True):
    """Runs the protocol; on_episode(k, result) is called after every episode (result has `frames` when
    record=True) so callers can save videos and traces. Returns (summary, rows)."""
    S = starts(task, n, seed)
    rows, t0 = [], time.time()
    strict, api.STRICT_STAGES = api.STRICT_STAGES, False     # held-out: a stage verifier's FAIL is recorded, not a stop
    try:
        for k, s0 in enumerate(S):
            try:
                r = api.run_episode(env, s0, policy, level=level, seed=20_000 + k, record=record)
                if on_episode is not None:
                    on_episode(k, r)
            except api.LevelError as e:
                r = dict(success=False, safety=False, timeout=False, steps=0, error=str(e), d_asm=float("nan"))
            rows.append(dict(trial=k, success=bool(r["success"]), safety=bool(r.get("safety", False)),
                             timeout=bool(r.get("timeout", False)), steps=int(r.get("steps", 0)),
                             time_s=round(int(r.get("steps", 0)) * DT, 2), error=r.get("error"),
                             d_asm_mm=round(float(r.get("d_asm", float("nan"))) * 1000, 2)))
            if verbose and (k + 1) % 10 == 0:
                print(f"[eval] {k + 1}/{len(S)}: {sum(x['success'] for x in rows)} ok  ({time.time() - t0:.0f}s)", flush=True)
    finally:
        api.STRICT_STAGES = strict
    ok = [x for x in rows if x["success"]]
    summary = dict(task=task, level=level, horizon_s=env.ecfg.horizon_s, seed=seed,
                   protocol=f"NIST task board, one part: {len(S)} `{START_FAMILY}` starts (part loose on the board, robot at "
                            f"home, gripper open) from seed {seed}; success = keypoints within 1 mm (mesh symmetries allowed)",
                   n=len(rows), n_ok=len(ok), success_rate=round(100 * len(ok) / max(len(rows), 1), 1),
                   time_mean_s=round(float(np.mean([x["time_s"] for x in ok])), 2) if ok else None,
                   time_median_s=round(float(np.median([x["time_s"] for x in ok])), 2) if ok else None,
                   safety=sum(x["safety"] for x in rows), errors=sum(bool(x["error"]) for x in rows),
                   timeouts=sum(x["timeout"] for x in rows))
    return summary, rows


def print_summary(s, name):
    print(f"==> {name} on {s['task']}: {s['n_ok']}/{s['n']} = {s['success_rate']}%  "
          f"time mean {s['time_mean_s']} s / median {s['time_median_s']} s  "
          f"(timeouts {s['timeouts']}, safety {s['safety']}, errors {s['errors']})", flush=True)


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return str(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="one of " + ", ".join(TASKS))
    ap.add_argument("--policy", required=True, help="a policy.py defining policy(api)")
    ap.add_argument("--level", default="final", choices=api.LEVELS)
    ap.add_argument("--n", type=int, default=N_EVAL)
    ap.add_argument("--seed", type=int, default=EVAL_SEED, help=f"start-state seed ({EVAL_SEED} = held-out)")
    ap.add_argument("--horizon", type=float, default=40.0, help="episode time limit in seconds")
    ap.add_argument("--out", default=None, help="output folder (default runs/eval/<task>_<policy stem>_s<seed>)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--width", type=int, default=640); ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()
    path = Path(a.policy)
    policy = load_policy(path)
    out = Path(a.out or Path("runs/eval") / f"{a.task}_{path.stem}_s{a.seed}")
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    env = api.make_env(a.task, horizon_s=a.horizon)
    vid = None if a.no_video else Video(a.task, a.width, a.height)

    def save(k, r):
        ep = out / "episodes" / f"e{k:03d}"
        ep.mkdir(exist_ok=True)
        (ep / "outcome.json").write_text(json.dumps(
            {kk: v for kk, v in r.items() if kk not in ("frames", "keyframes")}, indent=1, default=_jsonable))
        if vid is not None and len(r.get("frames", ())):
            vid.write(r, ep / "episode.mp4")

    print(f"policy {path}  task {a.task}  level {a.level}  horizon {a.horizon:g} s  seed {a.seed}  n {a.n}", flush=True)
    summary, rows = evaluate(env, policy, a.task, a.level, n=a.n, seed=a.seed, record=True, on_episode=save)
    summary["policy"] = str(path)
    (out / "eval.json").write_text(json.dumps(dict(summary=summary, episodes=rows), indent=1))
    print_summary(summary, path.stem)
    print(f"wrote {out}/eval.json and {out}/episodes/eNNN/{{outcome.json,episode.mp4}}")


if __name__ == "__main__":
    main()
