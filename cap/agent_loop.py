"""Code-as-policy fix loop, ASPIRE-style: one coding agent, one session, a sandbox with two commands.

The agent (Claude Code, `claude -p` with Read/Edit/Write and a shell allow-list) works inside
<run>/ where it owns policy.py and may run `replay <start-id>` and `evaluate`. It reads the
artifacts those commands write (trace.json, summary.txt, keyframes/, episode.mp4,
rounds/rNN/report.md) and edits policy.py. The seed rhythm is fixed by the harness: the debug set
starts at `per_family` training-bank starts for each of the five reset families, grows by `grow`
per family whenever a round passes everything, and the agent has exactly `rounds` evaluate calls.
Afterwards the harness scores the best round with the official protocol
(eval_protocol.official_eval).

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
                 "Bash(replay:*)", "Bash(evaluate:*)", "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)",
                 "Bash(tail:*)", "Bash(wc:*)", "Bash(grep:*)", "Bash(diff:*)", "Bash(find:*)",
                 "Bash(python3 -c:*)", "Bash(python3 -m py_compile:*)", "Bash(python3 -m json.tool:*)"]

BRIEF = """\
# Fix loop: write one robust assembly program

You are debugging the control program for a 7-DoF Franka Panda arm with a parallel-jaw gripper in
a physics simulator. One movable part ({task}) must be brought to a fixed target pose on a
fixture. An episode ends with SUCCESS the moment the part reaches the target pose (both of its
keypoints within 1 mm), with TIMEOUT after 15 s of simulated time, or with SAFETY if a joint
exceeds its velocity limit. An exception in your code ends the episode as a failure.

## The start state varies. Your code must be robust to it.

Every episode begins from a different, randomly drawn start state: the part may be lying loose,
about to be grasped, already in hand away from the fixture, partly engaged, or held at the
fixture mouth, and within each of those the poses, the grasp offset in the fingers and the
residual misalignment all differ. The part can also shift in the fingers during transport. You
are scored on START STATES YOU HAVE NEVER SEEN: the debug starts here are a sample of this
distribution; the final score is measured by the harness on a separate held-out set. Therefore:

* Write ONE general program that reads `api.observe()` and decides what to do from the current
  state (is the part in hand? where is it relative to the goal? is it stuck?). Branch on the
  observation, never on the start id.
* Never hard-code poses, offsets or thresholds copied from a particular episode's trace.
* Recompute grasp-dependent targets from the current observation each time; do not assume the
  grasp is where it was a second ago.
* A fix that helps one start must not break another. Always re-run `evaluate` to see the whole set.

## Your sandbox

```
policy.py        the file you edit. Must define policy(api). This is the only file that matters.
starts.json      ids of the current debug starts (kind-NN, e.g. loose-00 ... near-04)
replay <id>      run policy.py on one debug start. Writes attempts/rNN/<id>_<k>/ with
                   trace.json    every primitive call: args, result, and the state after it
                                 (tcp, part position, gripper width, external wrench on the hand)
                   summary.txt   outcome, compressed trace, your print() output
                   code.py       the policy.py that was run
                   keyframes/    an image after every primitive call, plus final.png; each is two
                                 views side by side: left a fixed global camera, right the wrist
                                 camera on the hand. Open them with the Read tool.
                   episode.mp4   the whole episode, same two views (for humans; you cannot view it)
                 Limit: {max_replays} replays per start per round. Then move on or evaluate.
evaluate         run policy.py on ALL debug starts = one round. Writes rounds/rNN/report.md
                 (per-kind table, compressed traces of every failed start) and
                 attempts/rNN/<id>_eval/ for each failure. You have exactly {rounds} evaluate
                 calls. When a round passes every start, the set grows by {grow} per kind and the
                 next round scores on the larger set.
findings.md      write this at the end (see below).
```

Both commands print their summary to stdout and also write it to disk. The simulator's own
distance-to-goal is never shown: like a real robot, you judge from the observation. Your shell is
restricted to `replay`, `evaluate` and read-only inspection (ls, cat, head, tail, grep, diff,
python3 -c for arithmetic). Edit files with the Edit tool.

## What policy.py may use

{spec}

You may also import numpy and these quaternion helpers from `assembly_policy.sim.robot`:
quat_mul, quat_conj, quat_to_mat, mat_to_quat, quat_apply, axis_angle_quat. Quaternions are
(w, x, y, z); positions are metres in the world frame; +z is up. `tcp` is the fingertip centre:
its +z axis points from palm to fingertips (the approach direction), +y is the finger closing
axis. The fixed numbers of the task: {const}

## Hard rules

* Edit only policy.py. Do not create other modules, do not import anything from the simulator
  except the helpers above, do not read or run anything outside this directory. The only way to
  execute the policy is `replay` and `evaluate`.
* No special-casing of debug starts, no reading of ground truth. Breaking this makes the result
  worthless.
* Keep policy.py importable at all times: a syntax error fails every start of a round.

## How to work (follow this order)

1. Read starts.json. Run `evaluate` once on the given policy.py to get round 1.
2. Read rounds/r01/report.md. For failed starts, read attempts/r01/<id>_eval/summary.txt and
   trace.json; look at keyframes/final.png and the keyframe after the last primitive. Diagnose:
   grasp missed or slipped (gripper width), target unreachable (move_to FAIL unreachable),
   jam at the fixture mouth (tcp stops short of target while wrench grows), wrong orientation,
   time ran out. Group failures by cause across kinds before changing anything.
3. Fix policy.py for one cause. Test with `replay <id>` on one or two affected starts (at most
   {max_replays} per start). Then `evaluate` to confirm nothing else broke. Prefer one real
   diagnosis-driven change per round over many speculative ones.
4. Repeat until the evaluate budget is used. Use every evaluate call: if all starts pass, the set
   grows and new starts will expose what is still fragile.
5. When the budget is used (evaluate refuses), write findings.md: root causes observed (which
   starts), what fixed them (evidence: which starts flipped), what is still failing and why, and
   which round you consider best. Then stop.

Be concrete and economical. The trace numbers are the evidence; reason from them. Start now.
"""


def setup(out: Path, a):
    out.mkdir(parents=True, exist_ok=True)
    (out / "attempts").mkdir(exist_ok=True)
    (out / "rounds").mkdir(exist_ok=True)
    shutil.copy(a.start, out / "policy.py")
    (out / "bin").mkdir(exist_ok=True)
    py = sys.executable
    for cmd in ("replay", "evaluate"):
        w = out / "bin" / cmd
        w.write_text(f'#!/bin/bash\nexport PYTHONPATH="{ROOT}:{ROOT}/cap" MUJOCO_GL="${{MUJOCO_GL:-egl}}"\n'
                     f'exec "{py}" "{ROOT}/cap/agent_tools.py" {cmd} "$@"\n')
        w.chmod(0o755)
    st = dict(task=a.task, level=a.level, bank_dir=str(a.bank_dir), per_family=a.per_family, grow=a.grow,
              round=0, max_rounds=a.rounds, max_replays_per_start=a.max_replays, replays={})
    (out / "state.json").write_text(json.dumps(st, indent=1))
    import agent_tools                                    # noqa: E402  (same PYTHONPATH)
    agent_tools.RUN = out
    agent_tools.write_starts(st)
    env = api.make_env(a.task)
    s0 = agent_tools.debug_starts(st)[0]["state"]
    env.set_states([0], s0[None, :], task_ids=[0])
    const = api.AssemblyAPI(env, np.random.default_rng(0), level=a.level).constants()
    brief = BRIEF.format(task=a.task, spec=api.spec(a.level), const=json.dumps(const),
                         rounds=a.rounds, grow=a.grow, max_replays=a.max_replays)
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


def best_round(out: Path):
    best = None
    if not (out / "log.jsonl").exists():
        return None
    for line in (out / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "round" not in r:
            continue
        key = (r["n_ok"] / max(r["n"], 1), r["n"], r["round"])
        if best is None or key > best[0]:
            best = (key, r["round"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="R16")
    ap.add_argument("--per-family", type=int, default=5)
    ap.add_argument("--grow", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--max-replays", type=int, default=3)
    ap.add_argument("--level", default="grasp")
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
                       f"report, fix policy.py, replay, evaluate. Write findings.md only when evaluate refuses.",
                  a, session, resume=True)

    best = best_round(out)
    if best is None:
        sys.exit("[run] no rounds were evaluated")
    (rate, n, _), rnd = best
    print(f"\n==> best round r{rnd:02d}: {rate * 100:.0f}% on {n} debug starts", flush=True)
    for line in (out / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "round" in r:
            print(f"  r{r['round']:02d}: {r['n_ok']}/{r['n']}  {r['by_family']}", flush=True)
    if a.no_eval:
        return
    print(f"[eval] official protocol on rounds/r{rnd:02d}/policy.py", flush=True)
    import agent_tools
    policy = agent_tools.load_policy(out / "rounds" / f"r{rnd:02d}" / "policy.py")
    env = api.make_env(a.task)
    summary, rows = eval_protocol.official_eval(env, policy, a.task, a.level, a.bank_dir)
    summary["policy"] = str(out / "rounds" / f"r{rnd:02d}" / "policy.py")
    (out / f"eval_r{rnd:02d}.json").write_text(json.dumps(dict(summary=summary, episodes=rows), indent=1))
    eval_protocol.print_summary(summary, f"r{rnd:02d}")


if __name__ == "__main__":
    main()
