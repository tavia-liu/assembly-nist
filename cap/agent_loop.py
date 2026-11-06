"""Code-as-policy fix loop: one coding agent, one session, a sandbox with two commands.

The agent (Claude Code, `claude -p` with Read/Edit/Write and a shell allow-list) works inside
<run>/ where it owns policy.py and may run `evaluate`. It reads the artifacts that command
writes (rounds/rNN/report.md; per seed trace.json, summary.txt, keyframes/, episode.mp4) and
edits policy.py.

Benchmark: the NIST task board in simulation. Every episode is a full assembly from scratch: the
part lies on the board at a random pose, the gripper is open. The debug set is drawn from the
training bank and follows `schedule` (2 -> 5 -> 10 -> 15 -> 20 starts): it moves to the next size
whenever a round passes everything; passing the last size ends the loop. The agent has `rounds`
evaluate calls.
Afterwards the harness scores the best round on the held-out eval bank (eval_protocol.evaluate:
success rate and completion time on 100 unseen starts).

  PYTHONPATH=.:cap MUJOCO_GL=egl python cap/agent_loop.py --task R16 --rounds 10
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

import api
import eval_protocol
from assembly_policy.sim.tasks import BANK_DIR

ROOT = Path(__file__).resolve().parents[1]
CLAUDE = os.environ.get("CAP_CLAUDE_BIN", os.path.expanduser("~/.local/bin/claude"))

# what the agent may run: the two sandbox commands plus read-only inspection of its own directory
ALLOWED_TOOLS = ["Read", "Edit", "Write", "Glob", "Grep",
                 "Bash(evaluate:*)", "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)",
                 "Bash(tail:*)", "Bash(wc:*)", "Bash(grep:*)", "Bash(diff:*)", "Bash(find:*)",
                 "Bash(python3 -c:*)", "Bash(python3 -m py_compile:*)", "Bash(python3 -m json.tool:*)"]

BRIEF = """\
# Task

Write `policy(api)` in policy.py: a 7-DoF Franka arm with a parallel-jaw gripper must pick up one
part ({task}) lying on the board and bring it to its target pose on the fixture, in a physics
simulator. An episode ends with SUCCESS when the part reaches the target pose (both keypoints
within 1 mm), TIMEOUT after 15 s, SAFETY if a joint exceeds its velocity limit, or FAILURE on an
exception in your code.

Each episode starts with the part at a random pose on the board and the gripper open. You are
scored on held-out starts you never see, by success rate and completion time. Write one program
that decides from `api.observe()`; never special-case a start or hard-code numbers taken from a
trace.

# Command

evaluate       run policy.py on all debug seeds (ids in starts.json) = one round. Writes
               rounds/rNN/report.md and, for every seed, attempts/rNN/<id>/ with trace.json,
               summary.txt, keyframes/ (png after each primitive: left global camera, right
               wrist camera) and episode.mp4. You have {rounds} calls. When all
               starts pass, the set grows ({schedule}); passing all {last} ends the loop.

The simulator's distance-to-goal is never shown. Shell is limited to `evaluate` and read-only
inspection; edit policy.py with the Edit tool and keep it importable. Everything outside this
directory is off limits and blocked: the robot knows only what the API and these artifacts tell it.

# API

{spec}

Also allowed: numpy, and quat_mul, quat_conj, quat_to_mat, mat_to_quat, quat_apply,
axis_angle_quat from assembly_policy.sim.robot. Quaternions are (w,x,y,z), positions metres in
the world frame, +z up; tcp +z points palm->fingertips, +y is the finger closing axis.
Constants: {const}

# Procedure

1. `evaluate`.
2. Diagnose before touching code. For every failed seed read summary.txt, trace.json and the
   keyframes (at least the last three and final.png). Write rounds/rNN/diagnosis.md with, per
   seed: what happened step by step, the root cause and the numbers or images that prove it,
   and the one change you will make. `evaluate` refuses the next round until this file exists.
3. Make that change in policy.py, `evaluate` again.
4. Repeat until `evaluate` refuses for good, then write findings.md: causes, fixes with
   evidence, what still fails, best round.
"""


def setup(out: Path, a):
    out.mkdir(parents=True, exist_ok=True)
    (out / "attempts").mkdir(exist_ok=True)
    (out / "rounds").mkdir(exist_ok=True)
    shutil.copy(a.start, out / "policy.py")
    (out / "bin").mkdir(exist_ok=True)
    py = sys.executable
    for f in ("api.py", "agent_tools.py", "record.py"):  # frozen copies: later harness edits must not touch this run
        shutil.copy(ROOT / "cap" / f, out / "bin" / f)
    for cmd in ("evaluate",):
        w = out / "bin" / cmd
        w.write_text(f'#!/bin/bash\nexport PYTHONPATH="{out / "bin"}:{ROOT}:{ROOT}/cap" MUJOCO_GL="${{MUJOCO_GL:-egl}}"\n'
                     f'exec "{py}" "{out / "bin" / "agent_tools.py"}" {cmd} "$@"\n')
        w.chmod(0o755)
    schedule = [int(x) for x in a.schedule.split(",")]
    shutil.copy(ROOT / "cap" / "guard.py", out / "bin" / "guard.py")
    (out / ".claude").mkdir(exist_ok=True)
    (out / ".claude" / "settings.json").write_text(json.dumps(dict(hooks=dict(PreToolUse=[dict(
        matcher="Read|Edit|Write|MultiEdit|Glob|Grep|Bash",
        hooks=[dict(type="command", command=f"python3 {out / 'bin' / 'guard.py'}")])])), indent=1))
    st = dict(task=a.task, level=a.level, bank_dir=str(a.bank_dir), schedule=schedule, stage=0,
              round=0, max_rounds=a.rounds, horizon_s=a.horizon)
    (out / "state.json").write_text(json.dumps(st, indent=1))
    import agent_tools                                    # noqa: E402  (same PYTHONPATH)
    agent_tools.RUN = out                                 # setup only; the agent runs the frozen copy in bin/
    agent_tools.write_starts(st)
    env = api.make_env(a.task, horizon_s=a.horizon)
    s0 = agent_tools.debug_starts(st)[0]["state"]
    env.set_states([0], s0[None, :], task_ids=[0])
    const = api.AssemblyAPI(env, np.random.default_rng(0), level=a.level).constants()
    brief = BRIEF.format(task=a.task, spec=api.spec(a.level), const=json.dumps(const), rounds=a.rounds,
                         schedule=" -> ".join(map(str, schedule)), last=schedule[-1])
    (out / "BRIEF.md").write_text(brief)
    return brief


def run_agent(out: Path, prompt: str, a, session: str, resume: bool):
    cmd = [CLAUDE, "-p", "--model", a.model, "--permission-mode", "acceptEdits",
           "--allowedTools", ",".join(ALLOWED_TOOLS),
           "--disallowed-tools", "WebFetch", "WebSearch", "Task", "Agent", "NotebookEdit",
           "--max-turns", str(a.max_turns), "--output-format", "stream-json", "--verbose"]
    cmd += ["--resume", session] if resume else ["--session-id", session]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PATH"] = f"{out / 'bin'}:{env.get('PATH', '')}"
    env["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
    t0 = time.time()
    with open(out / "agent.log", "a") as log:
        log.write(f"\n===== {'resume' if resume else 'start'} {time.strftime('%Y-%m-%d %H:%M:%S')} session {session}\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=out, env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
        p.stdin.write(prompt)
        p.stdin.close()
        rc = p.wait()
    print(f"[agent] exited rc={rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
    return rc


def rounds_done(out: Path):
    return json.loads((out / "state.json").read_text())["round"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="R16")
    ap.add_argument("--schedule", default="2,5,10,15,20",
                    help="debug-set sizes; the set moves to the next size when a round passes every start")
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--level", default="grasp")
    ap.add_argument("--horizon", type=float, default=60.0, help="episode time limit in seconds")
    ap.add_argument("--start", default=str(ROOT / "cap/naive_r16.py"))
    ap.add_argument("--model", default="claude-opus-5-5[1m]")
    ap.add_argument("--max-turns", type=int, default=800)
    ap.add_argument("--max-resumes", type=int, default=3)
    ap.add_argument("--bank-dir", default=str(BANK_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    out = Path(a.out or ROOT / f"cap/runs/fixloop/{a.task}_agent").resolve()
    if (out / "state.json").exists():
        sys.exit(f"{out} already exists; pick another --out")
    brief = setup(out, a)
    session = str(uuid.uuid4())
    (out / "run.json").write_text(json.dumps(dict(vars(a), session=session,
                                                  started=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    print(f"[run] {out}  session {session}", flush=True)

    run_agent(out, brief, a, session, resume=False)
    for i in range(a.max_resumes):
        done = rounds_done(out)
        if done >= a.rounds:
            break
        print(f"[agent] stopped after {done}/{a.rounds} rounds; resuming ({i + 1}/{a.max_resumes})", flush=True)
        run_agent(out, f"You have used {done} of {a.rounds} evaluate calls. Continue the loop: diagnose the latest "
                       f"report, fix policy.py, evaluate. Write findings.md only when evaluate refuses.",
                  a, session, resume=True)

    if not (out / "log.jsonl").exists():
        sys.exit("[run] no rounds were evaluated")
    rnd, path = eval_protocol.best_round(out)
    print(f"\n==> best round r{rnd:02d}", flush=True)
    for line in (out / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "round" in r:
            print(f"  r{r['round']:02d}: {r['n_ok']}/{r['n']}  mean time {r.get('time_mean_s')} s", flush=True)
    if a.no_eval:
        return
    print(f"[eval] held-out NIST protocol on {path.relative_to(out)}", flush=True)
    policy = eval_protocol.load_policy(path)
    env = api.make_env(a.task, horizon_s=a.horizon)
    summary, rows = eval_protocol.evaluate(env, policy, a.task, a.level, a.bank_dir)
    summary["policy"] = str(path)
    (out / f"eval_r{rnd:02d}.json").write_text(json.dumps(dict(summary=summary, episodes=rows), indent=1))
    eval_protocol.print_summary(summary, f"r{rnd:02d}")


if __name__ == "__main__":
    main()
