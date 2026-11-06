"""Sandbox command of the board loop: `evaluate` runs policy.py on the debug seeds of the multi-part board.

A seed = one board layout (all parts of the run at random settled poses, robot at home, gripper open). The policy
assembles the parts one after another in one continuous episode (see multi_env.py). Per seed the command writes
attempts/rNN/<seed>/: summary.txt, episode.mp4, code.py and stages/<k>_<part>/ with trace.json and keyframes/
(png after every primitive call: left a fixed board camera, right the wrist camera). rounds/rNN/report.md
summarises the round. Scoring: parts seated at the end of the run (within 1 mm) and the total time.
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
from agent_tools import _with_wrist_camera, compress_trace, load_state, save_state
from assembly_policy.sim.env import STATE_SPEC, EnvConfig
from multi_env import MultiPartEnv
from multi_part import choose_layout
from multi_scene import build_multi_xml

RUN = Path.cwd()
DT = 0.04


# ---------------------------------------------------------------- seeds / policy
def seeds_of(st):
    return [dict(id=f"s{j:02d}", seed=j) for j in range(st["schedule"][st["stage"]])]


def write_starts(st):
    (RUN / "starts.json").write_text(json.dumps([s["id"] for s in seeds_of(st)]))


def load_policy_module(path: Path):
    spec = importlib.util.spec_from_file_location("cap_policy_" + hashlib.sha1(path.read_bytes()).hexdigest()[:8], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("policy", "recover"):
        if not callable(getattr(mod, fn, None)):
            raise AttributeError(f"policy.py must define {fn}(api)")
    return mod


def stage_outcome(s):
    if s.get("error"):
        return "ERROR"
    if s["success"]:
        return "SUCCESS"
    if s.get("safety"):
        return "SAFETY"
    if s.get("timeout"):
        return "TIMEOUT"
    return "NOT ASSEMBLED"


# ---------------------------------------------------------------- one board run
def _collect(ap, stage):
    """Append one AssemblyAPI's trace / keyframes / frames to the stage record."""
    koff = len(stage["keys"])
    for e in ap.trace:
        if "key" in e:
            e["key"] += koff
    stage["trace"] += ap.trace
    stage["keys"] += list(ap.key_states)
    stage["frames"] += ap.frames


def run_board(st, policy_mod, seed, rng):
    tasks = list(st["tasks"])
    thr = EnvConfig().success_threshold
    env = MultiPartEnv(tasks, EnvConfig(nthread=1, horizon_s=st["per_part_s"] * len(tasks)))
    sp1 = np.asarray(st["sp_single"])
    layout = choose_layout(tasks, st["bank_dir"], sp1, np.random.default_rng(seed))
    env.reset_board(layout, sp1)
    frames, stages, remaining = [env.state[0].copy()], [], list(tasks)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        for i in range(len(tasks)):
            tcp = env.tcp_pos()
            info = dict(tcp_pos=tcp, part_pos={u: env.part_pos(u) for u in remaining})
            t = None
            if callable(getattr(policy_mod, "choose_next", None)):
                try:
                    t = policy_mod.choose_next(list(remaining), info)
                except Exception as e:
                    print(f"[harness] choose_next failed ({type(e).__name__}: {e}); nearest part instead", flush=True)
            if t not in remaining:
                t = min(remaining, key=lambda u: np.linalg.norm(info["part_pos"][u][:2] - tcp[:2]))
            remaining.remove(t)
            env.set_active(t)
            print(f"[harness] part {i + 1}/{len(tasks)}: {t}  t={env.t[0] * DT:.1f}s", flush=True)
            before = {u: bool(env.assembly_distance(u)[0] < thr) for u in tasks}
            stage = dict(stage=i, task=t, trace=[], keys=[], frames=[], tries=0, error=None, t_start=int(env.t[0]))
            while True:
                ap = api.AssemblyAPI(env, rng, level="grasp", record=True)
                try:
                    if i > 0 or stage["tries"] > 0:
                        policy_mod.recover(ap)
                    policy_mod.policy(ap)
                except api.EpisodeOver:
                    pass
                except Exception as e:                         # a policy bug is a result, not a crash
                    stage["error"] = f"{type(e).__name__}: {e}"
                    print(traceback.format_exc()[-1500:], flush=True)
                _collect(ap, stage)
                out_ = ap.outcome if ap.done else dict(success=False, safety=False, timeout=False)
                stage["tries"] += 1
                if out_.get("safety") and not out_.get("success") and stage["tries"] <= st.get("retries", 2):
                    print(f"[harness] safety stop (velocity limit) on {t}; retry {stage['tries']}", flush=True)
                    continue
                break
            if out_["success"] and callable(getattr(policy_mod, "finish", None)) and not out_.get("timeout"):
                env.scoring = False                            # part seated: finishing moves cannot re-trigger success
                ap2 = api.AssemblyAPI(env, rng, level="grasp", record=True)
                try:
                    policy_mod.finish(ap2)
                except api.EpisodeOver:
                    pass
                except Exception as e:
                    stage["error"] = f"finish: {type(e).__name__}: {e}"
                    print(traceback.format_exc()[-1500:], flush=True)
                env.scoring = True
                _collect(ap2, stage)
                if ap2.done and ap2.outcome.get("timeout"):
                    out_["timeout"] = True
            frames += stage["frames"] + [env.state[0].copy()]
            after = {u: bool(env.assembly_distance(u)[0] < thr) for u in tasks}
            stage.update(success=bool(out_["success"]), safety=bool(out_.get("safety", False)),
                         timeout=bool(out_.get("timeout", False)), seconds=round((int(env.t[0]) - stage["t_start"]) * DT, 2),
                         seated_after_release=after[t], displaced=[u for u in tasks if before[u] and not after[u]])
            stages.append(stage)
            if out_.get("timeout"):
                print("[harness] board time budget exhausted", flush=True)
                break
    final = {u: bool(env.assembly_distance(u)[0] < thr) for u in tasks}
    total_s = round(int(env.t[0]) * DT, 2)
    env.close()
    return dict(seed=seed, order=[s["task"] for s in stages], stages=stages, final=final,
                n_seated=int(sum(final.values())), n=len(tasks), all_seated=all(final.values()),
                total_s=total_s, frames=np.asarray(frames), stdout=buf.getvalue()[-6000:])


# ---------------------------------------------------------------- rendering (EGL; GPU node)
class BoardVis:
    """Fixed board camera | wrist camera."""

    def __init__(self, tasks, w=400, h=304):
        self.m = mujoco.MjModel.from_xml_string(_with_wrist_camera(build_multi_xml(tasks, visual=True)))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.75, 100.0, -42.0
        self.cam.lookat[:] = [0.5, 0.0, 0.03]
        import record
        self.record = record

    def frame(self, state):
        mujoco.mj_setState(self.m, self.d, state, STATE_SPEC)
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, self.cam)
        g = self.r.render().copy()
        self.r.update_scene(self.d, camera="wrist")
        w = self.r.render().copy()
        return np.concatenate([g, w], axis=1)

    def keyframes(self, stage, out: Path):
        from PIL import Image
        out.mkdir(parents=True, exist_ok=True)
        for e in stage["trace"]:
            if "key" in e and e["key"] < len(stage["keys"]):
                img = self.record.label(Image.fromarray(self.frame(stage["keys"][e["key"]])).convert("RGBA"),
                                        [f"{stage['task']}  t={e['t']:.2f}s  after {e['call']}", "left: board camera   right: wrist camera"],
                                        [(235, 235, 235, 255), (190, 200, 215, 255)])
                Image.fromarray(img).save(out / f"k{e['key']:02d}_t{e['t']:06.2f}_{e['call']}.png")
        if stage["frames"]:
            img = self.record.label(Image.fromarray(self.frame(stage["frames"][-1])).convert("RGBA"),
                                    [f"{stage['task']}  end of stage: {stage_outcome(stage)}", "left: board camera   right: wrist camera"],
                                    [(235, 235, 235, 255), (255, 200, 120, 255)])
            Image.fromarray(img).save(out / "final.png")

    def video(self, frames, out: Path, fps=25):
        import imageio.v2 as imageio
        fr = [self.frame(s) for s in frames]
        imageio.mimwrite(out, fr + [fr[-1]] * fps, fps=fps, codec="libx264", quality=7, macro_block_size=8)


def make_vis(tasks):
    try:
        return BoardVis(tasks)
    except Exception as e:
        print(f"[warn] rendering unavailable ({type(e).__name__}: {e}); no keyframes", file=sys.stderr)
        return None


# ---------------------------------------------------------------- artifacts
def seed_summary(res):
    lines = [f"seed {res['seed']:02d}: {res['n_seated']}/{res['n']} parts seated at the end, total time {res['total_s']} s",
             f"order: {' -> '.join(res['order'])}", ""]
    for s in res["stages"]:
        extra = []
        if s["success"] and not s["seated_after_release"]:
            extra.append("NOT SEATED after release")
        if s["displaced"]:
            extra.append("displaced earlier parts: " + ", ".join(s["displaced"]))
        if s["tries"] > 1:
            extra.append(f"{s['tries']} attempts (safety stops)")
        if s["error"]:
            extra.append("error: " + s["error"])
        lines.append(f"  part {s['stage'] + 1} {s['task']:5s} {stage_outcome(s):13s} {s['seconds']:6.1f} s" + ("   " + "; ".join(extra) if extra else ""))
    lines += ["", "seated at the end: " + ", ".join(f"{u}:{'yes' if ok else 'NO'}" for u, ok in res["final"].items())]
    return lines


def write_seed(adir: Path, res, vis: BoardVis | None, video=True):
    adir.mkdir(parents=True, exist_ok=True)
    shutil.copy(RUN / "policy.py", adir / "code.py")
    lines = seed_summary(res) + [""]
    for s in res["stages"]:
        sdir = adir / "stages" / f"{s['stage']}_{s['task']}"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "trace.json").write_text(json.dumps(s["trace"], indent=0))
        lines += [f"== part {s['stage'] + 1} {s['task']} ({stage_outcome(s)}) trace (compressed; full detail in stages/{s['stage']}_{s['task']}/trace.json):",
                  *compress_trace(s["trace"]), ""]
        if vis is not None:
            try:
                vis.keyframes(s, sdir / "keyframes")
            except Exception as e:
                (sdir / "render_error.txt").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    lines += ["POLICY STDOUT/STDERR (tail):", res["stdout"].rstrip() or "(none)"]
    (adir / "summary.txt").write_text("\n".join(lines) + "\n")
    if vis is not None and video:
        try:
            vis.video(res["frames"], adir / "episode.mp4")
        except Exception as e:
            (adir / "render_error.txt").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------- evaluate
def cmd_evaluate(a):
    st = load_state()
    if st["round"] >= st["max_rounds"]:
        sys.exit(f"evaluate budget exhausted ({st['max_rounds']} rounds). Write findings.md and stop.")
    if st["round"] >= 1 and st.get("require_diagnosis", True):
        prev = RUN / "rounds" / f"r{st['round']:02d}"
        diag = prev / "diagnosis.md"
        failed_prev = [r["id"] for r in json.loads((prev / "results.json").read_text()) if not r["all_seated"]]
        if failed_prev and (not diag.exists() or len(diag.read_text().strip()) < 400):
            sys.exit(f"Refused: write {diag.relative_to(RUN)} first (seeds with parts not seated: {', '.join(failed_prev)}): "
                     f"per failed part what happened step by step, the root cause with the trace numbers or keyframes that "
                     f"prove it, and the one change you will make. Then run evaluate again.")
    st["round"] += 1
    rnd = st["round"]
    rdir = RUN / "rounds" / f"r{rnd:02d}"
    rdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(RUN / "policy.py", rdir / "policy.py")
    seeds = seeds_of(st)
    vis = make_vis(st["tasks"])
    try:
        policy_mod = load_policy_module(RUN / "policy.py")
        import_error = None
    except Exception:
        policy_mod, import_error = None, traceback.format_exc()[-1500:]
    results, t0 = [], time.time()
    for s in seeds:
        if import_error:
            results.append(dict(id=s["id"], seed=s["seed"], all_seated=False, n_seated=0, n=len(st["tasks"]), total_s=0,
                                order=[], stages=[], final={}, error="import failed: " + import_error))
            print(f"  {s['id']}: policy.py failed to import", flush=True)
            continue
        res = run_board(st, policy_mod, s["seed"], np.random.default_rng(1000 * rnd + s["seed"]))
        write_seed(RUN / "attempts" / f"r{rnd:02d}" / s["id"], res, vis, video=True)
        results.append(dict(id=s["id"], seed=s["seed"], all_seated=res["all_seated"], n_seated=res["n_seated"], n=res["n"],
                            total_s=res["total_s"], order=res["order"], final=res["final"],
                            stages=[{k: v for k, v in sg.items() if k not in ("trace", "keys", "frames")} for sg in res["stages"]],
                            error=None))
        print(f"  {s['id']}: {res['n_seated']}/{res['n']} seated, {res['total_s']} s", flush=True)
    n_all, n = sum(r["all_seated"] for r in results), len(results)
    grew = done = False
    if n_all == n:
        if st["stage"] + 1 >= len(st["schedule"]):
            done = True
            st["round"] = st["max_rounds"]
        else:
            st["stage"] += 1
            grew = True
            write_starts(st)
    save_state(st)
    # report (outcomes only; no simulator distances)
    seated = [r["n_seated"] for r in results]
    rep = [f"# Round {rnd:02d} of {st['max_rounds']}: {n_all}/{n} seeds fully assembled; parts seated at the end "
           f"{sum(seated)}/{sum(r['n'] for r in results)}; mean board time {np.mean([r['total_s'] for r in results]):.1f} s", ""]
    for r in results:
        if r["error"]:
            rep += [f"## {r['id']}: {r['error']}", ""]
            continue
        rep.append(f"## {r['id']}: {r['n_seated']}/{r['n']} seated, {r['total_s']} s   order {' -> '.join(r['order'])}")
        for sg in r["stages"]:
            flags = []
            if sg["success"] and not sg["seated_after_release"]:
                flags.append("NOT SEATED after release")
            if sg["displaced"]:
                flags.append("displaced " + ",".join(sg["displaced"]))
            if sg["tries"] > 1:
                flags.append(f"{sg['tries']} attempts")
            if sg["error"]:
                flags.append("error")
            rep.append(f"  part {sg['stage'] + 1} {sg['task']:5s} {stage_outcome(sg):13s} {sg['seconds']:6.1f} s" + ("   " + "; ".join(flags) if flags else ""))
        missing = [u for u, ok in r["final"].items() if not ok]
        rep.append("  not seated at the end: " + (", ".join(missing) if missing else "none"))
        rep.append(f"  artifacts: attempts/r{rnd:02d}/{r['id']}/ (summary.txt, episode.mp4, stages/<k>_<part>/trace.json + keyframes/)")
        rep.append("")
    if grew:
        rep += [f"ALL SEEDS FULLY ASSEMBLED. The debug set grew to {st['schedule'][st['stage']]} seeds; ids in starts.json.", ""]
    if done:
        rep += [f"ALL {n} SEEDS FULLY ASSEMBLED AT THE MAXIMUM SET SIZE. The loop is complete: write findings.md and stop.", ""]
    left = st["max_rounds"] - rnd
    if not done:
        rep.append(f"Evaluate calls left: {left}." + (" This was the last one: write findings.md and stop." if left == 0 else ""))
    (rdir / "report.md").write_text("\n".join(rep) + "\n")
    (rdir / "results.json").write_text(json.dumps(results, indent=0))
    with open(RUN / "log.jsonl", "a") as f:
        f.write(json.dumps(dict(round=rnd, n_all=n_all, n=n, seated=seated, grew=grew, done=done,
                                total_s_mean=round(float(np.mean([r["total_s"] for r in results])), 2),
                                policy_sha=hashlib.sha1((rdir / "policy.py").read_bytes()).hexdigest()[:10],
                                wall_s=round(time.time() - t0, 1), seeds=results)) + "\n")
    print("\n" + "\n".join(rep))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("evaluate")
    a = ap.parse_args()
    {"evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
