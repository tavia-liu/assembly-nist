"""Code-as-policy fix loop: Claude rewrites a policy file, the simulator scores it, repeat.

Debug starts come from the task's TRAINING reset bank, the same distribution the RL baseline
trains on: `per_family` states from each of the five reset families. When a version passes
every debug start, the set grows by `grow` per family for the next turns, so a policy that
merely fits the current starts is exposed. After the last turn the best version is scored
with the official protocol (eval_protocol.official_eval: 5 seeds x 32 trials from the frozen
evaluation bank).

Each turn the model sees the API spec for its level, its current policy source, and the
outcome + primitive trace of every debug episode. It returns a complete replacement file.
Nothing from the simulator's ground truth (d_asm) is shown; only what observe() exposes
plus success / safety / timeout.

  PYTHONPATH=.:cap python cap/fix_loop.py --task R16 --per-family 5 --grow 5 --turns 10
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

import api
import eval_protocol
from assembly_policy.sim.resets import FAMILIES
from assembly_policy.sim.tasks import BANK_DIR

CLAUDE = os.environ.get("CAP_CLAUDE_BIN", os.path.expanduser("~/.local/bin/claude"))

BRIEF = """\
You are writing the control program for a 7-DoF Franka Panda arm with a parallel-jaw gripper
in a physics simulator. One movable part must be brought to a fixed target pose on a fixture.
The episode ends with SUCCESS the moment the part reaches the target pose (both of its
keypoints within 1 mm), with TIMEOUT after 15 s of simulated time, or with a SAFETY failure if
a joint exceeds its velocity limit.

THE START STATE VARIES, AND YOUR CODE MUST HANDLE ALL OF IT. Episodes begin from five kinds of
start state, drawn at random:
  loose    the part lies loose on the board, gripper open, arm somewhere above
  pickup   the gripper is open and positioned around the part, about to grasp it
  inhand   the part is already grasped, the arm is somewhere away from the fixture
  partial  the part is grasped and already partly engaged with the fixture
  near     the part is grasped and held just above / at the mouth of the fixture
Exact poses, grasp offsets and part orientation differ in every episode. You are scored on
START STATES YOU HAVE NOT SEEN: the debug episodes below are a sample of the same distribution,
and the final score is measured on a separate held-out set. Write ONE general program that
reads the current observation and decides what to do -- do not special-case the episodes you
are shown, and do not hard-code poses taken from their traces. Whenever every debug episode
passes, the debug set is enlarged with new start states.

You write one Python file defining `policy(api)`. It is run once per episode from the start
state; it returns when it has nothing more to do (the clock then runs out). You may import
numpy and the quaternion helpers from `assembly_policy.sim.robot`
(quat_mul, quat_conj, quat_to_mat, mat_to_quat, quat_apply, axis_angle_quat). You may NOT
import anything else from the simulator, read files, or access ground truth: the robot knows
only what `api.observe()` returns. Exceptions in your code end the episode as a failure.
"""

RULES = """\
Rewrite the policy to raise the success rate across ALL start-state kinds, not only the ones
that failed most. Reply with ONE fenced python block containing the COMPLETE new file (not a
diff), and at most five lines of prose before it saying what you changed and why. Keep helper
functions you still need. Do not call methods above the declared API level.
"""


def load_policy(path: Path):
    spec = importlib.util.spec_from_file_location(f"cap_policy_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.policy


# ---------------------------------------------------------------- debug set from the training bank
class DebugBank:
    """Fixed, growable sample of the training bank: the first k of a per-family shuffle, so a larger
    set always contains the smaller one."""

    def __init__(self, task, bank_dir, seed=0):
        z = np.load(Path(bank_dir) / f"{task}_train.npz")
        self.states, fam = z["states"], z["family"]
        rng = np.random.default_rng(seed)
        self.order = {f: rng.permutation(np.nonzero(fam == i)[0]) for i, f in enumerate(FAMILIES)}

    def starts(self, per_family):
        out = []
        for f in FAMILIES:
            for i in self.order[f][:per_family]:
                out.append((f, int(i), self.states[i]))
        return out


def fmt_obs(o):
    r = lambda x, n=4: np.round(np.asarray(x, float), n).tolist()
    return dict(part_pos=r(o["part_pos"]), part_quat=r(o["part_quat"]), tcp_pos=r(o["tcp_pos"]),
                gripper_width=round(float(o["gripper_width"]), 4), gripper_open_cmd=o["gripper_open_cmd"],
                tcp_wrench_ext=r(o["tcp_wrench_ext"], 2), t=o["t"])


def _reset(env, s):
    env.set_states([0], np.asarray(s)[None, :], task_ids=[0])
    return env


def evaluate(env, starts, policy, level, max_trace=60):
    eps = []
    for j, (fam, idx, s) in enumerate(starts):
        t0 = time.time()
        start = fmt_obs(api.AssemblyAPI(_reset(env, s), np.random.default_rng(0), level="raw").observe())
        try:
            r = api.run_episode(env, s, policy, level=level, seed=j)
        except api.LevelError as e:
            r = dict(success=False, safety=False, timeout=False, steps=0, error=f"LevelError: {e}", trace=[])
        final = fmt_obs(api.AssemblyAPI(env, np.random.default_rng(0), level="raw").observe())
        trace = r["trace"]
        if len(trace) > max_trace:
            trace = trace[: max_trace // 2] + [dict(call="...", skipped=len(trace) - max_trace)] + trace[-max_trace // 2:]
        eps.append(dict(ep=j, family=fam, bank_idx=idx, success=bool(r["success"]),
                        safety=bool(r.get("safety", False)), timeout=bool(r.get("timeout", False)),
                        steps=int(r.get("steps", 0)), error=r.get("error"), start=start, final=final, trace=trace,
                        d_asm_mm=round(float(r.get("d_asm", float("nan"))) * 1000, 2),
                        wall_s=round(time.time() - t0, 1)))
    return eps


def public(ep):
    """The episode as shown to the model: no d_asm, no bank bookkeeping."""
    return {k: v for k, v in ep.items() if k not in ("d_asm_mm", "wall_s", "bank_idx")}


def by_family(eps):
    return {f: f"{sum(e['success'] for e in eps if e['family'] == f)}/{sum(e['family'] == f for e in eps)}"
            for f in FAMILIES}


def build_prompt(level, const, src, eps, history):
    n_ok = sum(e["success"] for e in eps)
    p = [BRIEF, "FIXED VALUES", json.dumps(const), "", api.spec(level), ""]
    if history:
        p.append("PREVIOUS VERSIONS (success / debug episodes, then per start kind):")
        p += [f"  v{h['turn']:02d}: {h['n_ok']}/{h['n']}  {json.dumps(h['fam'])}  {h['note']}" for h in history]
        p.append("")
    p += ["CURRENT POLICY (the file to rewrite):", "```python", src.rstrip(), "```", "",
          f"RESULT OF THE CURRENT POLICY: {n_ok}/{len(eps)} success; per start kind {json.dumps(by_family(eps))}.",
          "Per-episode outcome, start and final observation, and the trace of every primitive call",
          "(t is episode time in s):"]
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
    ap.add_argument("--per-family", type=int, default=5, help="debug starts per reset family at the start")
    ap.add_argument("--grow", type=int, default=5, help="add this many per family whenever all debug starts pass")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--level", default="grasp")
    ap.add_argument("--start", default="cap/naive_r16.py")
    ap.add_argument("--model", default="claude-opus-5-5[1m]")
    ap.add_argument("--bank-dir", default=str(BANK_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-eval", action="store_true", help="skip the official eval of the best version at the end")
    a = ap.parse_args()
    out = Path(a.out or f"cap/runs/fixloop/{a.task}_bank")
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")

    env = api.make_env(a.task)
    bank = DebugBank(a.task, a.bank_dir, seed=0)
    k = a.per_family
    const = api.AssemblyAPI(_reset(env, bank.starts(1)[0][2]), np.random.default_rng(0), level=a.level).constants()

    src = Path(a.start).read_text()
    (out / "v00.py").write_text(src)
    history, best = [], None
    for turn in range(a.turns + 1):
        path = out / f"v{turn:02d}.py"
        starts = bank.starts(k)
        try:
            policy = load_policy(path)
            eps = evaluate(env, starts, policy, a.level)
        except Exception:
            eps = [dict(ep=j, family=f, bank_idx=i, success=False, safety=False, timeout=False, steps=0,
                        error="import failed: " + traceback.format_exc()[-800:], start={}, final={}, trace=[],
                        d_asm_mm=float("nan"), wall_s=0.0) for j, (f, i, _) in enumerate(starts)]
        n_ok, n = sum(e["success"] for e in eps), len(eps)
        note = history[-1].pop("pending_note") if history else "hand-written start"
        fam = by_family(eps)
        log.write(json.dumps(dict(turn=turn, n_ok=n_ok, n=n, per_family=k, by_family=fam, note=note,
                                  file=str(path), episodes=eps)) + "\n"); log.flush()
        history.append(dict(turn=turn, n_ok=n_ok, n=n, fam=fam, note=note))
        print(f"[v{turn:02d}] {n_ok}/{n} success on {k}/family  {json.dumps(fam)}  "
              f"errors={sum(bool(e['error']) for e in eps)}  | {note[:120]}", flush=True)
        key = (n_ok / n, n, turn)
        if best is None or key > best[0]:
            best = (key, turn)
        if n_ok == n:
            k += a.grow
            print(f"[v{turn:02d}] all debug starts pass -> debug set grows to {k}/family", flush=True)
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

    bt = best[1]
    print(f"\n==> best v{bt:02d} ({best[0][0] * 100:.0f}% on {best[0][1]} debug starts)   logs in {out}", flush=True)
    if not a.no_eval:
        print(f"[eval] official protocol on v{bt:02d}", flush=True)
        policy = load_policy(out / f"v{bt:02d}.py")
        summary, rows = eval_protocol.official_eval(env, policy, a.task, a.level, a.bank_dir)
        summary["policy"] = str(out / f"v{bt:02d}.py")
        (out / f"eval_v{bt:02d}.json").write_text(json.dumps(dict(summary=summary, episodes=rows), indent=1))
        log.write(json.dumps(dict(eval=True, turn=bt, summary=summary)) + "\n"); log.flush()
        eval_protocol.print_summary(summary, f"v{bt:02d}")


if __name__ == "__main__":
    main()
