"""Code-as-policy fix loop on the multi-part NIST board, with the pretrained RL insertion policy as a primitive.

Same machinery as agent_loop.py (one `claude -p` session in a sandbox with `evaluate`, a path guard and a mandatory
per-round diagnosis.md); the task is the whole board: all parts of --tasks physically on the board at once, the
robot at home, one continuous episode, scored by parts seated at the end and total time. The debug set is a list
of board layouts (seeds) that grows with `schedule` (e.g. 2 -> 4) whenever every seed is fully assembled.
policy.py starts from cap/hybrid_rl.py.

  PYTHONPATH=.:cap MUJOCO_GL=egl python cap/board_loop.py --tasks board10 --rounds 10
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

import api
import agent_loop
import eval_protocol
from assembly_policy.sim.tasks import resolve_tasks
from multi_board import PRESETS

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ("api.py", "agent_tools.py", "board_tools.py", "record.py", "multi_env.py", "multi_scene.py", "multi_part.py",
          "rl_skill.py", "eval_protocol.py")

BRIEF = """\
# Task

Write policy.py for a 7-DoF Franka arm with a parallel-jaw gripper that assembles the NIST task board in a physics
simulator: {n} parts ({tasks}) lie loose at random poses on the board, all physical at the same time, and must be
brought one after another to their target poses on their fixtures. The robot starts at home with the gripper open;
the run is one continuous episode with a budget of {budget:.0f} s. The score is the number of parts seated (within
1 mm of the target) WHEN THE RUN ENDS, and the total time. A part that is seated and later knocked out counts as
not seated. Parts not being worked on are obstacles.

You are scored on board layouts you never see; write one program that decides from `api.observe()`, never
special-case a seed or hard-code numbers read off a trace.

# policy.py contract

The harness calls your functions once per part on a fresh `api` whose active part is the one to assemble:
  policy(api)          assemble the active part from wherever the robot is.
  recover(api)         called before policy() for every part but the first, and after a safety stop (joint
                       velocity limit): put the hand in a state from which policy() can start.
  finish(api)          optional; called right after the active part is reported seated, while it is still in the
                       fingers (success cannot re-trigger during finish): e.g. press it home, release, lift away.
  choose_next(remaining, info)   optional; returns the name of the next part. info = dict(tcp_pos,
                       part_pos={{name: pos}}). Default: the part nearest to the hand.
The episode of a part ends (api.EpisodeOver) when it is seated, on a safety stop or when the board budget runs out.
After a safety stop the harness calls recover() and policy() again (up to {retries} extra attempts).

# Primitives

{spec}

A pretrained RL insertion policy for the active part is available as a primitive:
  from rl_skill import RLSkill
  rl = RLSkill(api)            # once per policy() call (loads the network for the active part)
  rl.run(max_s=12.0)           # drives the robot through api.step() for up to max_s seconds, returns when the
                               # part is seated (api.EpisodeOver propagates) or the budget is used; may be called again
It was trained on single parts: it grasps, carries and inserts the active part but is blind to the other parts and
fixtures, stops the instant the 1 mm threshold is crossed (parts may spring back when released) and can trip the
joint velocity limit. Everything around it -- where to start it, how long to let it run, what to do when it fails,
how to release, how to move between parts, in which order -- is yours to write.

Also allowed: numpy, and quat_mul, quat_conj, quat_to_mat, mat_to_quat, quat_apply, axis_angle_quat from
assembly_policy.sim.robot. Quaternions are (w,x,y,z), positions metres in the world frame, +z up; tcp +z points
palm->fingertips, +y is the finger closing axis. Constants: {const}

# Command

evaluate       run policy.py on every debug seed (ids in starts.json; a seed is a board layout) = one round.
               Writes rounds/rNN/report.md and per seed attempts/rNN/<id>/: summary.txt (per part outcome, time,
               whether it stayed seated after release, which earlier parts it displaced), episode.mp4 (whole run),
               stages/<k>_<part>/trace.json and keyframes/ (png after every primitive call: left a fixed board
               camera, right the wrist camera). You have {rounds} calls. When every seed is fully assembled the
               set grows ({schedule}); passing all {last} seeds ends the loop.

The simulator's distance-to-goal is never shown. Shell is limited to `evaluate` and read-only inspection; edit
policy.py with the Edit tool and keep it importable. Everything outside this directory is off limits and blocked;
bin/ holds the frozen harness sources you may read (api.py, rl_skill.py).

# Procedure

1. `evaluate`.
2. Diagnose before touching code. For every part that is not seated at the end (failed, not seated after release,
   or displaced later) read summary.txt, the stage's trace.json and keyframes (at least the last three and
   final.png). Write rounds/rNN/diagnosis.md with, per such part: what happened step by step, the root cause and the
   numbers or images that prove it, and the one change you will make. `evaluate` refuses the next round until this
   file exists.
3. Make that change in policy.py, `evaluate` again.
4. Repeat until `evaluate` refuses for good, then write findings.md: causes, fixes with evidence, what still fails,
   best round.
"""


def setup(out: Path, a, tasks):
    out.mkdir(parents=True, exist_ok=True)
    (out / "attempts").mkdir(exist_ok=True)
    (out / "rounds").mkdir(exist_ok=True)
    shutil.copy(a.start, out / "policy.py")
    (out / "bin").mkdir(exist_ok=True)
    py = sys.executable
    for f in FROZEN:
        shutil.copy(ROOT / "cap" / f, out / "bin" / f)
    w = out / "bin" / "evaluate"
    w.write_text(f'#!/bin/bash\nexport PYTHONPATH="{out / "bin"}:{ROOT}:{ROOT}/cap" MUJOCO_GL="${{MUJOCO_GL:-egl}}"\n'
                 f'exec "{py}" "{out / "bin" / "board_tools.py"}" evaluate "$@"\n')
    w.chmod(0o755)
    shutil.copy(ROOT / "cap" / "guard.py", out / "bin" / "guard.py")
    (out / ".claude").mkdir(exist_ok=True)
    (out / ".claude" / "settings.json").write_text(json.dumps(dict(hooks=dict(PreToolUse=[dict(
        matcher="Read|Edit|Write|MultiEdit|Glob|Grep|Bash",
        hooks=[dict(type="command", command=f"python3 {out / 'bin' / 'guard.py'}")])])), indent=1))
    schedule = [int(x) for x in a.schedule.split(",")]
    e1 = api.make_env(tasks[0]); sp_single = e1.sp.tolist(); e1.close()
    st = dict(tasks=tasks, bank_dir=str(a.bank_dir), schedule=schedule, stage=0, round=0, max_rounds=a.rounds,
              per_part_s=a.per_part_s, retries=a.retries, sp_single=sp_single, require_diagnosis=True)
    (out / "state.json").write_text(json.dumps(st, indent=1))
    (out / "starts.json").write_text(json.dumps([f"s{j:02d}" for j in range(schedule[0])]))
    env = api.make_env(tasks[0])
    import numpy as np
    z = np.load(Path(a.bank_dir) / f"{tasks[0]}_eval.npz")
    env.set_states([0], z["states"][:1], task_ids=[0])
    const = api.AssemblyAPI(env, np.random.default_rng(0), level="grasp").constants()
    const.pop("task", None); const.pop("family", None); const["horizon_s"] = a.per_part_s * len(tasks)
    env.close()
    brief = BRIEF.format(n=len(tasks), tasks=", ".join(tasks), budget=a.per_part_s * len(tasks), retries=a.retries,
                         spec=api.spec("grasp"), const=json.dumps(const), rounds=a.rounds,
                         schedule=" -> ".join(map(str, schedule)), last=schedule[-1])
    (out / "BRIEF.md").write_text(brief)
    return brief


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="board10")
    ap.add_argument("--schedule", default="2,4", help="debug seeds (board layouts); grows when every seed is fully assembled")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--per-part-s", type=float, default=60.0, help="board budget = this x n_parts")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--start", default=str(ROOT / "cap/hybrid_rl.py"))
    ap.add_argument("--model", default="claude-opus-5-5[1m]")
    ap.add_argument("--max-turns", type=int, default=800)
    ap.add_argument("--max-resumes", type=int, default=3)
    ap.add_argument("--bank-dir", default=str(eval_protocol.CAP_BANK_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--setup-only", action="store_true", help="create the sandbox and stop (no agent)")
    a = ap.parse_args()
    tasks = resolve_tasks(PRESETS.get(a.tasks, a.tasks))
    out = Path(a.out or ROOT / f"cap/runs/board_loop/{a.tasks}_agent").resolve()
    if (out / "state.json").exists():
        sys.exit(f"{out} already exists; pick another --out")
    brief = setup(out, a, tasks)
    session = str(uuid.uuid4())
    (out / "run.json").write_text(json.dumps(dict(vars(a), tasks=tasks, session=session,
                                                  started=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    print(f"[run] {out}  session {session}", flush=True)
    if a.setup_only:
        return
    agent_loop.run_agent(out, brief, a, session, resume=False)
    for i in range(a.max_resumes):
        done = agent_loop.rounds_done(out)
        if done >= a.rounds:
            break
        print(f"[agent] stopped after {done}/{a.rounds} rounds; resuming ({i + 1}/{a.max_resumes})", flush=True)
        agent_loop.run_agent(out, f"You have used {done} of {a.rounds} evaluate calls. Continue the loop: diagnose the "
                                  f"latest report, fix policy.py, evaluate. Write findings.md only when evaluate refuses.",
                             a, session, resume=True)
    if not (out / "log.jsonl").exists():
        sys.exit("[run] no rounds were evaluated")
    best = None
    for line in (out / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "round" in r:
            print(f"  r{r['round']:02d}: seeds fully assembled {r['n_all']}/{r['n']}  seated {r['seated']}  "
                  f"mean board time {r['total_s_mean']} s", flush=True)
            key = (r["n_all"], sum(r["seated"]), -r["total_s_mean"])
            if best is None or key > best[0]:
                best = (key, r["round"])
    print(f"\n==> best round r{best[1]:02d}  (rounds/r{best[1]:02d}/policy.py)", flush=True)


if __name__ == "__main__":
    main()
