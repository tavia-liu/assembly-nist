"""Code-as-policy fix loop: Claude rewrites a policy file, the simulator scores it, repeat.

Each turn the model sees the API spec for its level, its current policy source, and the
outcome + primitive trace of every debug episode. It returns a complete replacement file.
Nothing from the simulator's ground truth (d_asm) is shown; only what observe() exposes
plus success / safety / timeout.

  PYTHONPATH=.:cap python cap/fix_loop.py --task R16 --family near --n 6 --turns 10
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import api

CLAUDE = os.environ.get("CAP_CLAUDE_BIN", os.path.expanduser("~/.local/bin/claude"))

BRIEF = """\
You are writing the control program for a 7-DoF Franka Panda arm with a parallel-jaw gripper
in a physics simulator. One movable part must be brought to a fixed target pose on a fixture.
The episode ends with SUCCESS the moment the part reaches the target pose (both of its
keypoints within 1 mm), with TIMEOUT after 15 s of simulated time, or with a SAFETY failure if
a joint exceeds its velocity limit. Start states vary: the part may already be in the gripper
or lying loose nearby.

You write one Python file defining `policy(api)`. It is run once per episode from the start
state; it returns when it has nothing more to do (the clock then runs out). You may import
numpy and the quaternion helpers from `assembly_policy.sim.robot`
(quat_mul, quat_conj, quat_to_mat, mat_to_quat, quat_apply, axis_angle_quat). You may NOT
import anything else from the simulator, read files, or access ground truth: the robot knows
only what `api.observe()` returns. Exceptions in your code end the episode as a failure.
"""

RULES = """\
Rewrite the policy to raise the success rate on these start states. Reply with ONE fenced
python block containing the COMPLETE new file (not a diff), and at most five lines of prose
before it saying what you changed and why. Keep helper functions you still need. Do not call
methods above the declared API level.
"""


def load_policy(path: Path):
    spec = importlib.util.spec_from_file_location(f"cap_policy_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.policy


def fmt_obs(o):
    r = lambda x, n=4: np.round(np.asarray(x, float), n).tolist()
    return dict(part_pos=r(o["part_pos"]), part_quat=r(o["part_quat"]), tcp_pos=r(o["tcp_pos"]),
                gripper_width=round(float(o["gripper_width"]), 4), gripper_open_cmd=o["gripper_open_cmd"],
                tcp_wrench_ext=r(o["tcp_wrench_ext"], 2), t=o["t"])


def evaluate(env, starts, policy, level, max_trace=60):
    eps = []
    for i, s in enumerate(starts):
        t0 = time.time()
        try:
            r = api.run_episode(env, s, policy, level=level, seed=i)
        except api.LevelError as e:
            r = dict(success=False, safety=False, timeout=False, steps=0, error=f"LevelError: {e}", trace=[])
        env.set_states([0], np.asarray(s)[None, :], task_ids=[0]) if not r.get("steps") else None
        # final observation: what the robot itself can see at the end
        a = api.AssemblyAPI(env, np.random.default_rng(0), level="raw")
        final = fmt_obs(a.observe())
        trace = r["trace"]
        if len(trace) > max_trace:
            trace = trace[: max_trace // 2] + [dict(call="...", skipped=len(trace) - max_trace)] + trace[-max_trace // 2:]
        eps.append(dict(ep=i, success=bool(r["success"]), safety=bool(r.get("safety", False)),
                        timeout=bool(r.get("timeout", False)), steps=int(r.get("steps", 0)),
                        error=r.get("error"), start=fmt_obs(api.AssemblyAPI(_reset(env, s), np.random.default_rng(0), level="raw").observe()),
                        final=final, trace=trace, d_asm_mm=round(float(r.get("d_asm", float("nan"))) * 1000, 2),
                        wall_s=round(time.time() - t0, 1)))
    return eps


def _reset(env, s):
    env.set_states([0], np.asarray(s)[None, :], task_ids=[0])
    return env


def public(ep):
    """The episode as shown to the model: no d_asm."""
    return {k: v for k, v in ep.items() if k not in ("d_asm_mm", "wall_s")}


def build_prompt(level, const, src, eps, history):
    n_ok = sum(e["success"] for e in eps)
    p = [BRIEF, "FIXED VALUES", json.dumps(const), "", api.spec(level), ""]
    if history:
        p.append("PREVIOUS VERSIONS (success / episodes):")
        p += [f"  v{h['turn']:02d}: {h['n_ok']}/{h['n']}  {h['note']}" for h in history]
        p.append("")
    p += ["CURRENT POLICY (the file to rewrite):", "```python", src.rstrip(), "```", "",
          f"RESULT OF THE CURRENT POLICY: {n_ok}/{len(eps)} success. Per-episode outcome, start and",
          "final observation, and the trace of every primitive call (t is episode time in s):"]
    for e in eps:
        p.append(json.dumps(public(e)))
    p += ["", RULES]
    return "\n".join(p)


def ask(prompt, model, timeout=1800):
    t0 = time.time()
    cmd = [CLAUDE, "-p", "--model", model, "--output-format", "text", "--max-turns", "1",
           "--disallowed-tools", "Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch",
           "WebSearch", "Task", "TodoWrite", "NotebookEdit"]
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd="/tmp")
    if r.returncode != 0:
        raise RuntimeError(f"claude exited {r.returncode}: {r.stderr[-600:]}")
    m = re.search(r"```python\s*\n(.*?)```", r.stdout, re.S)
    if not m:
        raise ValueError(f"no python block in reply: {r.stdout[:600]}")
    note = r.stdout[: m.start()].strip().replace("\n", " ")[:300]
    return m.group(1), note, time.time() - t0, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="R16")
    ap.add_argument("--family", default="near")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--level", default="grasp")
    ap.add_argument("--start", default="cap/naive_r16.py")
    ap.add_argument("--model", default="claude-opus-5-5[1m]")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"cap/runs/fixloop_{a.task}_{a.family}")
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")

    env = api.make_env(a.task)
    starts = api.sample_starts(a.task, a.family, a.n, seed=0)
    const = api.AssemblyAPI(_reset(env, starts[0]), np.random.default_rng(0), level=a.level).constants()

    src = Path(a.start).read_text()
    (out / "v00.py").write_text(src)
    history, best = [], None
    for turn in range(a.turns + 1):
        path = out / f"v{turn:02d}.py"
        try:
            policy = load_policy(path)
            eps = evaluate(env, starts, policy, a.level)
        except Exception:
            eps = [dict(ep=i, success=False, safety=False, timeout=False, steps=0,
                        error="import failed: " + traceback.format_exc()[-800:], start={}, final={}, trace=[],
                        d_asm_mm=float("nan"), wall_s=0.0) for i in range(a.n)]
        n_ok = sum(e["success"] for e in eps)
        note = history[-1]["pending_note"] if history else "hand-written start"
        rec = dict(turn=turn, n_ok=n_ok, n=a.n, note=note, file=str(path), episodes=eps)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if history:
            history[-1].pop("pending_note", None)
        history.append(dict(turn=turn, n_ok=n_ok, n=a.n, note=note))
        print(f"[v{turn:02d}] {n_ok}/{a.n} success  "
              f"d_asm_mm={[e['d_asm_mm'] for e in eps]}  errors={sum(bool(e['error']) for e in eps)}  | {note}", flush=True)
        if best is None or n_ok > best[0]:
            best = (n_ok, turn)
        if turn == a.turns:
            break
        prompt = build_prompt(a.level, const, src, eps, history)
        (out / f"prompt{turn:02d}.txt").write_text(prompt)
        try:
            new_src, pnote, dt, raw = ask(prompt, a.model)
        except Exception as e:
            print(f"[v{turn:02d}] model call failed: {e}", flush=True)
            new_src, pnote, dt, raw = src, f"model call failed: {e}", 0.0, ""
        (out / f"reply{turn:02d}.txt").write_text(raw)
        src = new_src
        (out / f"v{turn + 1:02d}.py").write_text(src)
        history[-1]["pending_note"] = f"{pnote} ({dt:.0f}s)"
    print(f"\n==> best v{best[1]:02d}: {best[0]}/{a.n}   logs in {out}", flush=True)


if __name__ == "__main__":
    main()
