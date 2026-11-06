"""The two commands the fix-loop agent may run from its sandbox: `replay` and `evaluate`.

    replay <start-id>        run policy.py on one debug start; writes attempts/rNN/<id>_<k>/
                             {trace.json, summary.txt, code.py, keyframes/*.png, episode.mp4}
    evaluate                 run policy.py on every debug start (one "round"); writes
                             rounds/rNN/{policy.py, results.json, report.md}; failed starts also get
                             attempts/rNN/<id>_eval/. When everything passes the debug set grows to
                             the next size in state.json's schedule; passing the last size completes the loop.

State lives in state.json; the harness (agent_loop.py) creates it and reads log.jsonl afterwards.
The simulator's assembly distance is never written anywhere the agent reads.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np

import api
from assembly_policy.sim.env import STATE_SPEC
from assembly_policy.sim.resets import FAMILIES
from assembly_policy.sim.scene import build_xml
from assembly_policy.sim.tasks import BANK_DIR

RUN = Path.cwd()


# ---------------------------------------------------------------- state / starts
def load_state():
    return json.loads((RUN / "state.json").read_text())


def save_state(st):
    (RUN / "state.json").write_text(json.dumps(st, indent=1))


def debug_starts(st):
    """First n of a fixed shuffle of the training bank's loose starts, so a larger set contains the smaller."""
    z = np.load(Path(st["bank_dir"]) / f"{st['task']}_train.npz")
    n = st["schedule"][st["stage"]]
    idx = np.random.default_rng(0).permutation(np.nonzero(z["family"] == FAMILIES.index("loose"))[0])[:n]
    return [dict(id=f"s{j:02d}", bank_idx=int(b), state=z["states"][b]) for j, b in enumerate(idx)]


def write_starts(st):
    (RUN / "starts.json").write_text(json.dumps([s["id"] for s in debug_starts(st)]))


def load_policy(path: Path):
    spec = importlib.util.spec_from_file_location("cap_policy_" + hashlib.sha1(path.read_bytes()).hexdigest()[:8], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.policy


# ---------------------------------------------------------------- one episode
def run_one(env, start, policy, level, seed):
    """Run policy on one start with stdout captured. Returns the api.run_episode result + stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = api.run_episode(env, start["state"], policy, level=level, seed=seed, record=True)
    r["stdout"] = buf.getvalue()[-4000:]
    return r


def outcome_line(r):
    if r.get("error"):
        why = "ERROR"
    elif r.get("success"):
        why = "SUCCESS"
    elif r.get("safety"):
        why = "SAFETY"
    else:
        why = "TIMEOUT"
    return f"{why}  steps={r.get('steps', 0)} ({r.get('steps', 0) * 0.04:.1f}s)" + (f"  error={r['error']}" if r.get("error") else "")


def compress_trace(trace):
    """One line per primitive; consecutive identical (call, ok, why) entries merged with a count."""
    lines, prev, n = [], None, 0

    def flush():
        if prev is not None:
            lines.append(prev + (f"  x{n}" if n > 1 else ""))

    for e in trace:
        if e["call"] == "set_gripper":
            core = f"set_gripper({'open' if e.get('open') else 'close'}) -> width {e.get('width')}"
        elif e["call"] in ("move_to", "move_to_joints"):
            core = f"{e['call']} -> {'ok' if e.get('ok') else 'FAIL:' + str(e.get('why', ''))}" + \
                   (f" err={e['err']}" if "err" in e else "")
        else:
            core = e["call"]
        line = f"t={e['t']:5.2f}  {core}  tcp={e.get('tcp')} part={e.get('part')} grip={e.get('grip')} wrench={e.get('wrench')}"
        key = (e["call"], e.get("ok"), e.get("why"), e.get("open"))
        if prev is not None and key == pkey:
            n += 1
            prev = line                                       # keep the last state of the run
        else:
            flush(); prev, pkey, n = line, key, 1
    flush()
    return lines


# ---------------------------------------------------------------- rendering (EGL; GPU node)
def _with_wrist_camera(xml: str) -> str:
    """Add robosuite's eye_in_hand camera to the Franka hand body (the LIBERO / ASPIRE wrist view):
    5 cm off the palm centre, looking straight along the hand's +z axis toward the fingertips."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    hand = root.find(".//body[@name='hand']")
    ET.SubElement(hand, "camera", name="wrist", mode="fixed", pos="0.05 0 0", quat="0 0.707108 0.707108 0", fovy="75")
    return ET.tostring(root, encoding="unicode")


class Vis:
    """Two views per frame: a fixed global camera framing part + fixture, and the wrist camera."""

    def __init__(self, task, w=400, h=304):
        self.task = task
        self.m = mujoco.MjModel.from_xml_string(_with_wrist_camera(build_xml(task, visual=True)))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w)
        import record
        self.record = record

    def frame(self, state, cam):
        """Global view | wrist view, side by side."""
        mujoco.mj_setState(self.m, self.d, state, STATE_SPEC)
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, cam)
        g = self.r.render().copy()
        self.r.update_scene(self.d, camera="wrist")
        w = self.r.render().copy()
        return np.concatenate([g, w], axis=1)

    def keyframes(self, res, out: Path):
        from PIL import Image
        out.mkdir(parents=True, exist_ok=True)
        cam = self.record.camera_for(self.m, self.d, res["frames"][0], self.task)
        kf = res.get("keyframes")
        trace = res["trace"]
        if kf is not None and len(kf):
            for e in trace:
                if "key" in e:
                    img = self.record.label(Image.fromarray(self.frame(kf[e["key"]], cam)).convert("RGBA"),
                                            [f"t={e['t']:.2f}s  after {e['call']}", "left: global   right: wrist camera"],
                                            [(235, 235, 235, 255), (190, 200, 215, 255)])
                    Image.fromarray(img).save(out / f"k{e['key']:02d}_t{e['t']:05.2f}_{e['call']}.png")
        img = self.record.label(Image.fromarray(self.frame(res["frames"][-1], cam)).convert("RGBA"),
                                ["final   left: global   right: wrist camera", outcome_line(res)],
                                [(235, 235, 235, 255), (255, 200, 120, 255)])
        Image.fromarray(img).save(out / "final.png")

    def video(self, res, out: Path, fps=25):
        import imageio.v2 as imageio
        cam = self.record.camera_for(self.m, self.d, res["frames"][0], self.task)
        frames = [self.frame(s, cam) for s in res["frames"][::1]]
        imageio.mimwrite(out, frames + [frames[-1]] * fps, fps=fps, codec="libx264", quality=7, macro_block_size=8)


def write_attempt(adir: Path, start, r, vis: Vis | None, video=False):
    adir.mkdir(parents=True, exist_ok=True)
    shutil.copy(RUN / "policy.py", adir / "code.py")
    (adir / "trace.json").write_text(json.dumps(r["trace"], indent=0))
    summ = [f"start {start['id']}", outcome_line(r), "",
            "TRACE (compressed; full detail in trace.json):", *compress_trace(r["trace"]), "",
            "POLICY STDOUT/STDERR:", r.get("stdout", "").rstrip() or "(none)"]
    (adir / "summary.txt").write_text("\n".join(summ) + "\n")
    if vis is not None:
        try:
            vis.keyframes(r, adir / "keyframes")
            if video:
                vis.video(r, adir / "episode.mp4")
        except Exception as e:                                # rendering must never break the loop
            (adir / "render_error.txt").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    return summ


def make_vis(task):
    try:
        return Vis(task)
    except Exception as e:
        print(f"[warn] rendering unavailable ({type(e).__name__}: {e}); no keyframes", file=sys.stderr)
        return None


# ---------------------------------------------------------------- commands
def cmd_replay(a):
    st = load_state()
    starts = {s["id"]: s for s in debug_starts(st)}
    if a.start_id not in starts:
        sys.exit(f"unknown start id {a.start_id!r}; see starts.json ({len(starts)} ids)")
    used = st.setdefault("replays", {}).get(a.start_id, 0)
    if used >= st["max_replays_per_start"]:
        sys.exit(f"{a.start_id}: {used} replays already used this round (limit {st['max_replays_per_start']}). "
                 f"Move on to another start, or run `evaluate` to begin a new round.")
    st["replays"][a.start_id] = used + 1
    save_state(st)
    try:
        policy = load_policy(RUN / "policy.py")
    except Exception:
        sys.exit("policy.py failed to import:\n" + traceback.format_exc()[-1500:])
    env = api.make_env(st["task"])
    r = run_one(env, starts[a.start_id], policy, st["level"], seed=used)
    adir = RUN / "attempts" / f"r{st['round']:02d}" / f"{a.start_id}_{used + 1}"
    summ = write_attempt(adir, starts[a.start_id], r, make_vis(st["task"]), video=True)
    print("\n".join(summ))
    print(f"\nartifacts: {adir.relative_to(RUN)}/  (trace.json, summary.txt, code.py, keyframes/, episode.mp4)")


def cmd_evaluate(a):
    st = load_state()
    if st["round"] >= st["max_rounds"]:
        sys.exit(f"evaluate budget exhausted ({st['max_rounds']} rounds). Write findings.md and stop.")
    st["round"] += 1
    rnd = st["round"]
    rdir = RUN / "rounds" / f"r{rnd:02d}"
    rdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(RUN / "policy.py", rdir / "policy.py")
    starts = debug_starts(st)
    vis = make_vis(st["task"])
    env = api.make_env(st["task"])
    try:
        policy = load_policy(RUN / "policy.py")
        import_error = None
    except Exception:
        policy, import_error = None, traceback.format_exc()[-1500:]
    results = []
    t0 = time.time()
    for j, s in enumerate(starts):
        if import_error:
            r = dict(success=False, safety=False, timeout=False, steps=0, error="import failed: " + import_error,
                     trace=[], stdout="", d_asm=float("nan"))
        else:
            r = run_one(env, s, policy, st["level"], seed=1000 * rnd + j)
        rec = dict(id=s["id"], bank_idx=s["bank_idx"], success=bool(r["success"]),
                   safety=bool(r.get("safety", False)), timeout=bool(r.get("timeout", False)),
                   steps=int(r.get("steps", 0)), error=r.get("error"), d_asm_mm=round(float(r.get("d_asm", float("nan"))) * 1000, 2),
                   trace=r.get("trace", []))
        if not rec["success"] and not import_error:
            write_attempt(RUN / "attempts" / f"r{rnd:02d}" / s["id"], s, r, vis, video=True)
        results.append(rec)
        print(f"  {s['id']:12s} {outcome_line(r)}", flush=True)
    n_ok, n = sum(r["success"] for r in results), len(results)
    grew = done = False
    if n_ok == n:
        if st["stage"] + 1 >= len(st["schedule"]):
            done = True                                   # everything passes at the last stage: loop complete
            st["round"] = st["max_rounds"]
        else:
            st["stage"] += 1
            grew = True
            write_starts(st)
    save_state(st)

    # report for the agent (no d_asm)
    ok_t = [r["steps"] * 0.04 for r in results if r["success"]]
    rep = [f"# Round {rnd:02d} of {st['max_rounds']}: {n_ok}/{n} starts succeeded"
           + (f", mean time {np.mean(ok_t):.1f} s" if ok_t else ""), "",
           " ".join(f"{r['id']}:{'ok' if r['success'] else ('ERR' if r['error'] else ('SAF' if r['safety'] else 'TO'))}"
                    for r in results), ""]
    if grew:
        rep.append(f"ALL PASSED. The debug set grew to {st['schedule'][st['stage']]} starts; new ids are in "
                   f"starts.json. The next round scores on all of them.")
        rep.append("")
    if done:
        rep.append(f"ALL {n} STARTS PASSED AT THE MAXIMUM SET SIZE. The loop is complete: write findings.md and stop.")
        rep.append("")
    failed = [r for r in results if not r["success"]]
    if failed:
        rep.append("## Failed starts")
        for r in failed:
            rep.append(f"### {r['id']}  {'ERROR ' + str(r['error']) if r['error'] else ('SAFETY' if r['safety'] else 'TIMEOUT')}"
                       f"  steps={r['steps']}")
            rep += ["```", *compress_trace(r["trace"])[-25:], "```",
                    f"details: attempts/r{rnd:02d}/{r['id']}/ (trace.json, summary.txt, keyframes/, episode.mp4)", ""]
    left = st["max_rounds"] - rnd
    if not done:
        rep.append(f"Evaluate calls left: {left}." + (" This was the last one: write findings.md and stop." if left == 0 else ""))
    (rdir / "report.md").write_text("\n".join(rep) + "\n")
    (rdir / "results.json").write_text(json.dumps(results, indent=0))
    with open(RUN / "log.jsonl", "a") as f:
        f.write(json.dumps(dict(round=rnd, n_ok=n_ok, n=n, grew=grew, done=done,
                                time_mean_s=round(float(np.mean(ok_t)), 2) if ok_t else None,
                                policy_sha=hashlib.sha1((rdir / "policy.py").read_bytes()).hexdigest()[:10],
                                wall_s=round(time.time() - t0, 1),
                                episodes=[{k: v for k, v in r.items() if k != "trace"} for r in results])) + "\n")
    print("\n" + "\n".join(rep))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("replay"); r.add_argument("start_id")       # kept for manual use; not exposed to the agent
    sp.add_parser("evaluate")
    a = ap.parse_args()
    {"replay": cmd_replay, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
