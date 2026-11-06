"""Agent-as-policy probe: the agent emits raw action chunks, no primitives at all.

Every second of episode time the agent is shown the state and returns the next 25 actions
(one control step each). It is open loop inside a chunk and closed loop between chunks, so
the question this answers is narrow and sharp:

    with only the action space and the state -- no move_to, no ik, no grasp sampler, no
    distance-to-goal -- does the agent produce a search motion on its own?

What it is given is what a real cell would have: the robot's own joint limits and command
convention (datasheet), its proprioception and TCP Jacobian (encoders plus the robot model --
libfranka publishes both every cycle), the board pose (calibrated once), and a pose for the
part, plus the external joint torques and TCP wrench the robot's own torque sensors report.
The simulator's d_asm is withheld, since no real robot can measure it and the RL baseline is
not given it either.

  python -m cap.agent_control --family near --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

import api

CHUNK = 25                 # actions per agent call = 1.0 s of episode time at 25 Hz
CLAUDE = os.environ.get("CAP_CLAUDE_BIN", "/tmp/claude-bin/claude")

BRIEF = """\
You are controlling a 7-DoF Franka Panda arm with a parallel-jaw gripper in a physics
simulator. One movable part must be brought to a fixed target pose. The episode ends the
moment the part reaches it (both of its keypoints within 1 mm), or after 15 s, or if a joint
exceeds its velocity limit.

ACTION SPACE
  An action is 8 numbers, applied for one 25 Hz control step:
    a[0:7]  relative joint-position targets. The controller is commanded
            q_target = q + action_scale * a, each a[i] clipped to [-1, 1]. Under the
            built-in impedance gains a joint then moves at roughly 0.5 rad/s per unit of a,
            so the target is approached, never snapped to.
    a[7]    gripper: > 0.5 opens, < -0.5 closes, in between holds the current command.

YOUR OUTPUT
  Exactly {chunk} actions, run open loop one after another ({dt:.1f} s of motion), then you
  see the state again and send the next {chunk}. Reply with one fenced json block:

```json
{{"reasoning": "<one short line, for the log only>",
  "actions": [[a0,...,a7], ... {chunk} rows total]}}
```
  Nothing outside the block. Every row must have exactly 8 numbers in [-1, 1].
"""


def fmt_state(o, const):
    def v(x, n=4):
        return "[" + ", ".join(f"{float(y):.{n}f}" for y in np.atleast_1d(x)) + "]"
    return (f"t = {o['t']:.1f} s, time_left = {o['time_left']:.1f} s\n"
            f"  joints          {v(o['joints'], 3)}\n"
            f"  joint_vel       {v(o['joint_vel'], 3)}  rad/s\n"
            f"  tcp_jacobian    6x7, rows = d(tcp linear xyz)/dq then d(tcp angular xyz)/dq\n"
            + "".join(f"      {v(r, 4)}\n" for r in o['tcp_jacobian']) +
            f"  gripper_width   {o['gripper_width']:.4f} m   (open command: {o['gripper_open_cmd']})\n"
            f"  ext_joint_torque {v(o['joint_torque_ext'], 3)}  Nm  (external load felt at the joints)\n"
            f"  ext_tcp_wrench  {v(o['tcp_wrench_ext'], 2)}  N then Nm, world frame (load on the hand)\n"
            f"  tcp_pos         {v(o['tcp_pos'])}\n"
            f"  tcp_quat        {v(o['tcp_quat'])}   (wxyz; tcp +z = palm->fingertips, +y = closing axis)\n"
            f"  part_pos        {v(o['part_pos'])}\n"
            f"  part_quat       {v(o['part_quat'])}\n"
            f"  target_pos      {v(o['goal_pos'])}\n"
            f"  target_quat     {v(o['goal_quat'])}")


def build_prompt(const, history, o):
    p = [BRIEF.format(chunk=CHUNK, dt=CHUNK * 0.04), "", "FIXED VALUES", json.dumps(const), ""]
    if history:
        p.append("WHAT HAS HAPPENED SO FAR")
        for h in history:
            p.append(f"-- chunk {h['i']} -- you said: {h['reasoning']}")
            p.append(f"   state after:\n{h['after']}")
        p.append("")
    p += ["CURRENT STATE", fmt_state(o, const), "",
          f"Send the next {CHUNK} actions."]
    return "\n".join(p)


def ask(prompt, timeout=900):
    # run from a neutral cwd with no tools: the controller must answer from the prompt alone,
    # not go exploring the repository it happens to be launched from
    t0 = time.time()
    cmd = [CLAUDE, "-p", "--output-format", "text", "--max-turns", "1",
           "--disallowed-tools", "Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch",
           "WebSearch", "Task", "TodoWrite", "NotebookEdit"]
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd="/tmp")
    if r.returncode != 0:
        raise RuntimeError(f"claude exited {r.returncode}: {r.stderr[-400:]}")
    m = re.search(r"```json\s*(.*?)\s*```", r.stdout, re.S) or re.search(r"(\{.*\})", r.stdout, re.S)
    if not m:
        raise ValueError(f"no json block in reply: {r.stdout[:400]}")
    d = json.loads(m.group(1))
    a = np.asarray(d["actions"], dtype=float)
    if a.ndim != 2 or a.shape[1] != 8:
        raise ValueError(f"actions has shape {a.shape}, want (n, 8)")
    return np.clip(a, -1.0, 1.0), str(d.get("reasoning", ""))[:200], time.time() - t0


def run(task="R16", family="near", seed=0, max_chunks=15, out=None):
    env = api.make_env(task)
    state = api.sample_starts(task, family, seed + 1, seed=0)[seed]
    env.set_states([0], state[None, :], task_ids=[0])
    a = api.AssemblyAPI(env, np.random.default_rng(seed), level="raw")
    const = a.constants()

    history, log, chunks = [], [], 0
    o = a.observe()                                  # d_asm is not in observe(): no robot can measure it
    try:
        while chunks < max_chunks:
            prompt = build_prompt(const, history, o)
            actions, reasoning, dt = ask(prompt)
            print(f"[chunk {chunks}] {dt:.0f}s  {reasoning}", flush=True)
            for row in actions[:CHUNK]:
                o = a.step(row)                      # raises EpisodeOver when it ends
            log.append(dict(chunk=chunks, reasoning=reasoning, seconds=dt,
                            actions=actions[:CHUNK].round(3).tolist()))
            history.append(dict(i=chunks, reasoning=reasoning, after=fmt_state(o, const)))
            chunks += 1
    except api.EpisodeOver:
        pass

    res = dict(task=task, family=family, seed=seed, chunks=chunks, **a.outcome, log=log)
    print(f"\n==> success={res.get('success')} d_asm={res.get('d_asm', float('nan'))*1000:.1f}mm "
          f"steps={res.get('steps')} agent_calls={chunks}")
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="R16")
    ap.add_argument("--family", default="near")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chunks", type=int, default=15)
    ap.add_argument("--out", default=None)
    sys.exit(0 if run(**vars(ap.parse_args())) else 0)
