"""Minimal AutoRobo-style loop on the AssembleNIST benchmark: a staged coder with self-designed experiments, an
independent auditor, and the harness driving every rollout.

Per round: the harness assembles policy.py from plan.md + stages/NN_<name>.py and runs it on the debug set (first n
`board` starts of the training bank, n following --schedule; a round that passes everything runs the next size in
the same round, passing the last size ends the run). Every failed start is judged by a fresh auditor session
(claude -p) from its keyframes, sheet and trace; the per-stage grid, the auditor's notes and the focus stage go to
the coder (one persistent claude -p session, fenced to the run directory) as rounds/rNN/report.md. The coder
replies READY (stage files revised) or EXPERIMENT (experiments/NN_<slug>/ to run on starts it names, up to
--max-experiments per round). Afterwards the best round is scored on the held-out eval bank (eval_protocol).

  PYTHONPATH=.:cap:cap/auto MUJOCO_GL=egl python cap/auto/loop.py --task R8 --rounds 25
"""
from __future__ import annotations

from assembly_policy.sim.tasks import task_name
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "cap", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                      # noqa: E402

import agent_tools as T                 # noqa: E402  (debug_starts, run_one, write_attempt, Vis, stage_verdicts)
import api                              # noqa: E402
import eval_protocol                    # noqa: E402
import prompts as P                     # noqa: E402

CLAUDE = os.environ.get("CAP_CLAUDE_BIN", os.path.expanduser("~/.local/bin/claude"))
READ_ONLY = ["Read", "Glob", "Grep", "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)", "Bash(tail:*)", "Bash(wc:*)",
             "Bash(grep:*)", "Bash(diff:*)", "Bash(find:*)", "Bash(python3 -c:*)", "Bash(python3 -m py_compile:*)",
             "Bash(python3 -m json.tool:*)"]
CODER_TOOLS = ["Edit", "Write"] + READ_ONLY
DISALLOWED = ["WebFetch", "WebSearch", "Task", "Agent", "NotebookEdit"]
_LIMIT_RE = re.compile(r"(session|usage|weekly|rate|hour)\s+limit|limit\s+reached|hit your|resets \d", re.I)
CONFIG_DIR = Path(os.environ.get("CLAUDE_CONFIG_DIR", os.path.expanduser("~/.claude")))
CREDS_DIR = CONFIG_DIR / "creds"     # <account>.json slots, `order` (one name per line, first = preferred), `current`, limits.json


def _reset_epoch(msg):
    """Epoch when a limit lifts, from 'resets 6pm (America/Chicago)' or 'resets Nov 12, 3pm (America/Chicago)'."""
    m = re.search(r"resets?\s+(?:at\s+|on\s+)?([^()]+?)\s*\(([\w/_+-]+)\)", msg)
    if not m:
        return None
    txt, tz = m.group(1), m.group(2)
    try:
        zone = ZoneInfo(tz)
    except Exception:
        zone = ZoneInfo("America/Chicago")
    now = datetime.now(zone)
    tm = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", txt, re.I)
    if not tm:
        return None
    h = int(tm.group(1)) % 12 + (12 if tm.group(3).lower() == "pm" else 0)
    t = now.replace(hour=h, minute=int(tm.group(2) or 0), second=0, microsecond=0)
    dm = re.search(r"([A-Z][a-z]{2})[a-z]*\.?\s+(\d{1,2})", txt)
    if dm:
        t = t.replace(month=datetime.strptime(dm.group(1), "%b").month, day=int(dm.group(2)))
        if t < now - timedelta(days=30):
            t = t.replace(year=t.year + 1)
    elif t <= now:
        t += timedelta(days=1)
    return t.timestamp()


def _next_weekly(day, clock, now):
    """Next `Tue 5pm` / `Fri 14:00` after `now` (tz-aware)."""
    wd = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"].index(day[:3].lower())
    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", clock, re.I)
    h = int(m.group(1)) % 12 + (12 if (m.group(3) or "").lower() == "pm" else 0) if m.group(3) else int(m.group(1))
    t = now.replace(hour=h, minute=int(m.group(2) or 0), second=0, microsecond=0) + timedelta(days=(wd - now.weekday()) % 7)
    return t if t > now else t + timedelta(days=7)


class Accounts:
    """Saved Claude accounts (bash cap/auto/save_creds.sh <name>). pick() makes the first account in `order` whose
    limit has lifted the live .credentials.json; limited() records a limit message against the live account."""

    def __init__(self, log):
        self.log, self.live = log, CONFIG_DIR / ".credentials.json"

    def names(self, now=None):
        """Priority order. `order` lines: `name` (fixed, listed order first) or `name weekly Tue 5pm` (these come after
        the fixed ones, the account whose weekly reset is soonest first: spend quota before it expires)."""
        have = {p.stem for p in CREDS_DIR.glob("*.json")} - {"limits"} if CREDS_DIR.is_dir() else set()
        f = CREDS_DIR / "order"
        fixed, weekly = [], []
        now = now or datetime.now(ZoneInfo("America/Chicago"))
        for line in (f.read_text().splitlines() if f.exists() else []):
            parts = line.split()
            if not parts or parts[0] not in have:
                continue
            if len(parts) >= 4 and parts[1].lower() == "weekly":
                weekly.append((_next_weekly(parts[2], parts[3], now), parts[0]))
            else:
                fixed.append(parts[0])
        ordered = fixed + [n for _, n in sorted(weekly)]
        return ordered + sorted(have - set(ordered))

    def limits(self):
        f = CREDS_DIR / "limits.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def current(self):
        f = CREDS_DIR / "current"
        return f.read_text().strip() if f.exists() else ""

    def limited(self, msg):
        cur = self.current()
        if not cur:
            return
        lim = self.limits()
        lim[cur] = _reset_epoch(msg) or time.time() + (24 * 3600 if "week" in msg.lower() else 5 * 3600)
        (CREDS_DIR / "limits.json").write_text(json.dumps(lim, indent=1))
        self.log(f"credentials: {cur} limited until {time.strftime('%m-%d %H:%M', time.localtime(lim[cur]))}: {msg[:120]!r}")

    def pick(self):
        """Returns seconds to wait before calling claude (0 when an account is available now)."""
        names = self.names()
        if not names or not self.live.exists():
            return 0
        with open(CREDS_DIR / "lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            lim, now, cur = self.limits(), time.time(), self.current()
            avail = [n for n in names if lim.get(n, 0) <= now]
            target = avail[0] if avail else min(names, key=lambda n: lim.get(n, 0))
            if target != cur:
                if cur in names:
                    shutil.copy(self.live, CREDS_DIR / f"{cur}.json")       # keep refreshed tokens
                shutil.copy(CREDS_DIR / f"{target}.json", self.live)
                (CREDS_DIR / "current").write_text(target)
                self.log(f"credentials: {cur or '?'} -> {target}")
        return 0 if avail else max(0.0, lim[target] - now)

PROGRAM = '''# Assembled by the loop from plan.md + stages/NN_<name>.py{extra} -- edit the stage files, not this file.
import traceback

import numpy as np

import api as _api

STAGES = {stages!r}   # (name, file, source, strict): a strict stage stops the policy on FAIL


class StageFailed(Exception):
    pass


def policy(api):
    ns = dict(api=api, np=np)
    cur = [None, True]

    def stage_done(ok, evidence=""):
        api._rec("stage_done", stage=cur[0], ok=bool(ok), evidence=str(evidence)[:200])
        print(f"[STAGE {{cur[0]}}] {{'PASS' if ok else 'FAIL'}} {{evidence}}", flush=True)
        if not ok and cur[1] and getattr(api, "strict_stages", True):
            raise StageFailed(cur[0])

    ns["stage_done"] = stage_done
    try:
        for name, fname, src, strict in STAGES:
            cur[0], cur[1] = name, strict
            print(f"=== {{name}} ({{fname}}) ===", flush=True)
            try:
                exec(compile(src, fname, "exec"), ns)
            except (StageFailed, _api.EpisodeOver, _api.LevelError):
                raise
            except Exception as e:
                traceback.print_exc()
                api._rec("stage_done", stage=name, ok=False, evidence=f"exception {{type(e).__name__}}: {{e}}"[:200])
                print(f"[STAGE {{name}}] FAIL exception {{type(e).__name__}}: {{e}}", flush=True)
                raise StageFailed(name)
    except StageFailed:
        return                        # the episode goes on without the policy
'''


# ---------------------------------------------------------------- programs
def stage_files(run: Path):
    d = run / "stages"
    return sorted(d.glob("[0-9][0-9]_*.py")) if d.is_dir() else []


def build_policy(run: Path):
    """policy.py from the stage files; returns the stage names in order."""
    stages = [(f.stem.split("_", 1)[1], f"stages/{f.name}", f.read_text(), True) for f in stage_files(run)]
    (run / "policy.py").write_text(PROGRAM.format(stages=stages, extra=""))
    return [s[0] for s in stages]


def build_experiment(run: Path, name: str):
    """experiments/<name>/program.py = prefix stages (FAIL does not stop) + experiment.py. Returns (path, ids, fields)."""
    d = run / "experiments" / name
    fields = {}
    if (d / "hypothesis.md").exists():
        for line in (d / "hypothesis.md").read_text().splitlines():
            m = re.match(r"^(Hypothesis|Prediction|Prefix|Starts):\s*(.*)$", line.strip())
            if m:
                fields.setdefault(m.group(1), m.group(2).strip())
    by = {f.stem.split("_", 1)[1]: f for f in stage_files(run)}
    prefix = [s.strip() for s in fields.get("Prefix", "").split("->")
              if s.strip() and not re.match(r"\(?\s*(none|n/a|no prefix|-)", s.strip(), re.I)]
    missing = [s for s in prefix if s not in by]
    if missing:
        raise ValueError(f"Prefix names stages without a file: {missing} (have {sorted(by)})")
    stages = [(s, f"stages/{by[s].name}", by[s].read_text(), False) for s in prefix]
    stages.append(("experiment", f"experiments/{name}/experiment.py", (d / "experiment.py").read_text(), True))
    (d / "program.py").write_text(PROGRAM.format(stages=stages, extra=f" + experiments/{name}/experiment.py"))
    return d / "program.py", re.findall(r"s\d\d", fields.get("Starts", "")), fields


def last_word(reply: str) -> str:
    words = re.findall(r"[A-Z]+", (reply or "").strip()[-200:])
    return words[-1] if words else ""


def parse_entries(reply: str) -> dict:
    """{id: entry} from the auditor's JSON list (tolerates prose or code fences around it)."""
    m = re.search(r"\[.*\]", reply or "", re.DOTALL)
    try:
        entries = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        entries = []
    return {str(e.get("id")): e for e in entries if isinstance(e, dict)}


def outcome(r) -> str:
    return "ERROR" if r.get("error") else "SUCCESS" if r["success"] else "SAFETY" if r.get("safety") else "TIMEOUT"


# ---------------------------------------------------------------- the loop
class Loop:
    def __init__(self, a, run: Path):
        self.a, self.run = a, run
        self.accounts = Accounts(self.log)
        self.resumed = bool(a.resume)                         # only the first turn after --resume may pick up a pending experiment
        self.st = json.loads((run / "state.json").read_text()) if (run / "state.json").exists() else None
        if self.st is None:
            self.setup()
        self.schedule = self.st["schedule"]
        T.RUN = run
        if not (run / "policy.py").exists():
            build_policy(run)                                 # an empty policy.py until the coder writes stages
        self.env = api.make_env(self.st["task"], horizon_s=self.st["horizon_s"])
        self.vis = T.make_vis(self.st["task"])
        self.dt = self.env.ecfg.control_dt

    # ----- setup / state
    def setup(self):
        a, run = self.a, self.run
        for d in ("attempts", "rounds", "experiments", "stages", "log", "bin", ".claude"):
            (run / d).mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "cap" / "guard.py", run / "bin" / "guard.py")
        (run / ".claude" / "settings.json").write_text(json.dumps(dict(hooks=dict(PreToolUse=[dict(
            matcher="Read|Edit|Write|MultiEdit|Glob|Grep|Bash",
            hooks=[dict(type="command", command=f"python3 {run / 'bin' / 'guard.py'}")])])), indent=1))
        schedule = [int(x) for x in a.schedule.split(",")]
        self.st = dict(task=a.task, level=a.level, perception=api.PERCEPTION, bank_dir=str(a.bank_dir), schedule=schedule, stage=0, round=0,
                       max_rounds=a.rounds, horizon_s=a.horizon, phase="brief", session=None,
                       model=a.model, audit_model=a.audit_model, started=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.save()
        T.RUN = run
        T.write_starts(self.st)
        env = api.make_env(a.task, horizon_s=a.horizon)
        env.set_states([0], T.debug_starts(self.st)[0]["state"][None, :], task_ids=[0])
        const = api.AssemblyAPI(env, np.random.default_rng(0), level=a.level).constants()
        (run / "BRIEF.md").write_text(P.BRIEF.format(
            task=a.task, horizon=a.horizon, schedule=" -> ".join(map(str, schedule)), last=schedule[-1],
            rounds=a.rounds, spec=api.spec(a.level), const=json.dumps(const), max_exp=a.max_experiments,
            tracker=", or api.locate_part(), the tracker built on them" if api.LEVELS.index(a.level) >= api.LEVELS.index(api._LEVEL_OF["locate_part"]) else ""))

    def save(self, **kw):
        self.st.update(kw)
        (self.run / "state.json").write_text(json.dumps(self.st, indent=1))

    def log(self, msg):
        print(f"[auto {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def starts(self, stage=None):
        return T.debug_starts({**self.st, "stage": self.st["stage"] if stage is None else stage})

    # ----- claude
    def claude(self, prompt, model, tools, max_turns, tag, session=None):
        """One `claude -p` call in the run directory; returns (reply, session). Resumes `session` when given."""
        log = self.run / "log"
        (log / f"{tag}.prompt.md").write_text(prompt)
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
        attempt = 0
        while attempt < 24:
            attempt += 1
            wait = self.accounts.pick()
            if wait > 0:
                self.log(f"{tag}: every account is limited; sleeping {min(wait + 30, 3600):.0f} s")
                time.sleep(min(wait + 30, 3600))
                attempt -= 1
                continue
            sid = session or str(uuid.uuid4())
            cmd = [CLAUDE, "-p", "--model", model, "--permission-mode", "acceptEdits",
                   "--allowedTools", ",".join(tools), "--disallowed-tools", ",".join(DISALLOWED),
                   "--max-turns", str(max_turns), "--output-format", "json"]
            cmd += ["--resume", sid] if session else ["--session-id", sid]
            t0 = time.time()
            try:
                p = subprocess.run(cmd, cwd=self.run, env=env, input=prompt, capture_output=True, text=True,
                                   timeout=self.a.call_timeout)
            except subprocess.TimeoutExpired:
                self.log(f"{tag}: claude call exceeded {self.a.call_timeout} s; retrying")
                continue
            with open(log / "cli.jsonl", "a") as f:
                f.write(json.dumps(dict(tag=tag, attempt=attempt, rc=p.returncode, wall_s=round(time.time() - t0),
                                        stdout=p.stdout[-200_000:], stderr=p.stderr[-5000:])) + "\n")
            try:
                out = json.loads(p.stdout)
            except json.JSONDecodeError:
                out = {}
            result = out.get("result") or ""
            if p.returncode != 0 and not out:
                self.log(f"{tag}: claude rc={p.returncode} {p.stderr[-300:]!r}; retrying in 60 s")
                time.sleep(60)
                continue
            if out.get("is_error") and _LIMIT_RE.search(result):
                attempt -= 1                                  # usage limits do not count against the retry budget
                if self.accounts.names():
                    self.accounts.limited(result)             # pick() at the top switches or waits
                else:
                    self.log(f"{tag}: {result[:160]!r}; sleeping 1800 s")
                    time.sleep(1800)
                continue
            if out.get("is_error") and out.get("api_error_status"):
                self.log(f"{tag}: {result[:160]!r}; sleeping 120 s")
                time.sleep(120)
                continue
            (log / f"{tag}.reply.md").write_text(result)
            self.log(f"{tag}: {(time.time() - t0) / 60:.1f} min, {out.get('num_turns')} turns, ${out.get('total_cost_usd', 0):.2f}")
            return result, sid
        raise RuntimeError(f"{tag}: claude CLI kept failing")

    def coder(self, prompt, tag):
        reply, sid = self.claude(prompt, self.st["model"], CODER_TOOLS, self.a.coder_turns, tag, self.st["session"])
        self.save(session=sid)
        return reply

    def coder_turn(self, prompt, tag):
        """Send `prompt`; run experiments until the coder replies READY."""
        reply, n_exp, nudges = self.coder(prompt, tag), 0, 0
        for k in range(self.a.max_experiments + 4):
            word = last_word(reply)
            if word == "READY" and (stage_files(self.run) or tag == "final"):
                return
            if word == "READY":
                msg = P.NO_STAGES
            elif word == "EXPERIMENT":
                name = self.unrun_experiment()
                if name is None:
                    msg = P.NO_EXPERIMENT
                elif n_exp >= self.a.max_experiments:
                    msg = P.EXPERIMENT_BUDGET.format(n=self.a.max_experiments)
                else:
                    msg, n_exp = self.run_experiment(name), n_exp + 1
            else:
                nudges += 1
                if nudges > 1:
                    self.log(f"{tag}: reply ended with {word or 'nothing'} twice; carrying on")
                    return
                msg = P.NOT_READY
            reply = self.coder(msg, f"{tag}_x{k + 1}")

    # ----- episodes
    def episode(self, start, policy, seed, adir: Path, code: Path):
        r = T.run_one(self.env, start, policy, self.st["level"], seed)
        T.write_attempt(adir, start, r, self.vis, video=True)
        shutil.copy(code, adir / "code.py")
        self.sheet(r, adir / "sheet.png")
        return r

    def sheet(self, r, path: Path, n=16, cols=4):
        """n frames spanning the episode (global | wrist), tiled, each labelled with its time."""
        frames = r.get("frames")
        if self.vis is None or frames is None or not len(frames):
            return
        try:
            from PIL import Image
            cam = self.vis.record.camera_for(self.vis.m, self.vis.d, frames[0], self.st["task"])
            idx = np.unique(np.linspace(0, len(frames) - 1, n).round().astype(int))
            tiles = [Image.fromarray(self.vis.record.label(
                Image.fromarray(self.vis.frame(frames[i], cam)).convert("RGBA"),
                [f"t={i * self.dt:.1f}s"], [(235, 235, 235, 255)])) for i in idx]
            w, h = tiles[0].size
            rows = -(-len(tiles) // cols)
            out = Image.new("RGB", (cols * w, rows * h))
            for k, t in enumerate(tiles):
                out.paste(t, ((k % cols) * w, (k // cols) * h))
            out.resize((cols * w // 2, rows * h // 2)).save(path)
        except Exception as e:                                # rendering must never break the loop
            (path.parent / "render_error.txt").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def load(self, path: Path):
        try:
            return T.load_policy(path), None
        except Exception:
            return None, traceback.format_exc()[-1500:]

    # ----- experiments
    def unrun_experiment(self):
        dirs = sorted(d for d in (self.run / "experiments").iterdir() if (d / "experiment.py").is_file())
        for d in reversed(dirs):
            if not (d / "result.txt").exists():
                return d.name
        return None

    def run_experiment(self, name):
        d = self.run / "experiments" / name
        try:
            prog, ids, fields = build_experiment(self.run, name)
        except Exception as e:
            text = (f"Experiment {name} could not be assembled: {e}. Fix experiments/{name}/hypothesis.md (Prefix: (none) "
                    f"or existing stage names; Starts: ids from starts.json) or experiment.py and end your reply with "
                    f"EXPERIMENT again, or revise the stage files and end with READY.")
            (d / "result.txt").write_text(text + "\n")
            return text
        known = {s["id"]: s for s in self.starts(len(self.schedule) - 1)}
        ids = [i for i in ids if i in known] or [self.starts()[0]["id"]]
        policy, err = self.load(prog)
        lines = [f"Experiment {name} ran on {', '.join(ids)}. Hypothesis: {fields.get('Hypothesis', '(none)')}", ""]
        if err:
            lines += ["program.py failed to import:", err]
        for k, i in enumerate(ids if not err else []):
            self.log(f"experiment {name} on {i}")
            r = self.episode(known[i], policy, 7000 + k, d / i, prog)
            lines += [f"## {i}: {T.outcome_line(r)}", "```", *T.compress_trace(r["trace"])[-30:], "```",
                      "stdout:", "```", r.get("stdout", "").rstrip()[-3000:] or "(none)", "```",
                      f"artifacts: experiments/{name}/{i}/ (trace.json, summary.txt, keyframes/, sheet.png)", ""]
        text = "\n".join(lines) + "\n" + P.EXP_RESULT_TAIL + "\n"
        (d / "result.txt").write_text(text)
        return text

    # ----- one round: rollout, audit, report
    def rollout(self, rnd):
        names = build_policy(self.run)
        rdir = self.run / "rounds" / f"r{rnd:02d}"
        rdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.run / "policy.py", rdir / "policy.py")
        shutil.copytree(self.run / "stages", rdir / "stages", dirs_exist_ok=True)
        if (self.run / "plan.md").exists():
            shutil.copy(self.run / "plan.md", rdir / "plan.md")
        policy, err = self.load(self.run / "policy.py")
        results, done, stage, accepted, t0 = [], set(), self.st["stage"], False, time.time()
        while True:
            for s in [s for s in self.starts(stage) if s["id"] not in done]:
                if err:
                    r = dict(success=False, safety=False, timeout=False, steps=0, error="import failed: " + err,
                             trace=[], stdout="", d_asm=float("nan"))
                else:
                    r = self.episode(s, policy, 1000 * rnd + len(results), self.run / "attempts" / f"r{rnd:02d}" / s["id"],
                                     self.run / "policy.py")
                results.append(dict(id=s["id"], bank_idx=s["bank_idx"], success=bool(r["success"]),
                                    safety=bool(r.get("safety", False)), timeout=bool(r.get("timeout", False)),
                                    steps=int(r.get("steps", 0)), error=r.get("error"),
                                    d_asm_mm=round(float(r.get("d_asm", float("nan"))) * 1000, 2),
                                    stages=T.stage_verdicts(r.get("trace", []), names)))
                done.add(s["id"])
                self.log(f"r{rnd:02d} {s['id']} {T.outcome_line(r)}")
            if not all(r["success"] for r in results):
                break
            if stage + 1 >= len(self.schedule):
                accepted = True
                break
            stage += 1                                        # all passed: the next size runs in this round
        grew = stage > self.st["stage"]
        self.save(stage=stage)
        T.write_starts(self.st)
        return names, results, accepted, grew, time.time() - t0

    def audit(self, rnd, names, results):
        """grid[id][stage] = (pass, evidence), notes[id]. Successes are not audited; failures go to the auditor."""
        grid, notes = {}, {}
        for r in results:
            if r["success"]:
                bad = [n for n in names if r["stages"][n] != "ok"]
                grid[r["id"]] = {n: (True, "" if n not in bad else f"your verifier said {r['stages'][n]!r}") for n in names}
                notes[r["id"]] = "" if not bad else ("SUCCESS although your verifier failed " + ", ".join(bad)
                                                     + ": the part reached the goal, so fix the verifier, not the motion.")
        failed = [r for r in results if not r["success"]]
        plan = (self.run / "plan.md").read_text().strip() if (self.run / "plan.md").exists() else "(no plan.md)"
        for c in range(0, len(failed), self.a.audit_chunk):
            chunk = failed[c:c + self.a.audit_chunk]
            digest = "\n".join(f"  {r['id']}: {outcome(r)} after {r['steps'] * self.dt:.1f} s"
                               + (f" ({r['error']})" if r["error"] else "") + "; the policy's own verdicts: "
                               + "  ".join(f"{n}:{r['stages'][n]}" for n in names) for r in chunk)
            prompt = P.AUDIT.format(task=self.st["task"], plan=" -> ".join(names), plan_text=plan, rnd=rnd, digest=digest)
            reply, _ = self.claude(prompt, self.st["audit_model"], READ_ONLY, self.a.audit_turns,
                                   f"audit_r{rnd:02d}_{c // self.a.audit_chunk + 1}")
            entries = parse_entries(reply)
            for r in chunk:
                e = entries.get(r["id"], {})
                row = {}
                for n in names:
                    v = (e.get("stages") or {}).get(n)
                    if isinstance(v, dict) and "pass" in v:
                        ok = v["pass"] if isinstance(v["pass"], bool) else str(v["pass"]).lower() in ("true", "pass", "yes")
                        row[n] = (ok, str(v.get("evidence", "")).strip())
                    elif isinstance(v, bool):
                        row[n] = (v, "")
                    else:
                        row[n] = (r["stages"][n] == "ok", "(no auditor verdict; the policy's own verdict is used)")
                if names and all(ok for ok, _ in row.values()):   # a failed start with every stage passing: the last stage
                    ok, ev = row[names[-1]]
                    row[names[-1]] = (False, (ev + " | " if ev else "") + "every stage passed per the auditor, yet the "
                                      "episode did not end in SUCCESS: the final state is not the goal")
                grid[r["id"]], notes[r["id"]] = row, str(e.get("note", "")).strip()
        return grid, notes

    def report(self, rnd, names, results, accepted, grew, grid, notes):
        n_ok, n = sum(r["success"] for r in results), len(results)
        ok_t = [r["steps"] * self.dt for r in results if r["success"]]
        focus = next((nm for nm in names if any(not grid[r["id"]][nm][0] for r in results)), None)
        left = self.st["max_rounds"] - rnd
        rep = [f"# Round {rnd:02d} of {self.st['max_rounds']}: {n_ok}/{n} starts succeeded"
               + (f", mean time {np.mean(ok_t):.1f} s" if ok_t else ""), ""]
        if grew:
            rep += [f"Every start passed, so the debug set grew to {self.schedule[self.st['stage']]} starts and the new "
                    f"ones ran in this round too (ids in starts.json).", ""]
        if accepted:
            rep += [f"ALL {n} STARTS PASSED AT THE LARGEST SET SIZE. The run is complete.", ""]
        rep += ["## Verdicts per start (the auditor's for failed starts, yours for successes)"]
        for r in results:
            rep.append(f"{r['id']} ({outcome(r)} {r['steps'] * self.dt:.1f} s" + (f", {r['error']}" if r["error"] else "")
                       + "): " + "  ".join(f"{nm}:{'PASS' if grid[r['id']][nm][0] else 'FAIL'}" for nm in names))
            for nm in names:
                ok, ev = grid[r["id"]][nm]
                if ev and (not ok or r["success"]):
                    rep.append(f"    {nm}: {ev}")
            if notes.get(r["id"]):
                rep.append(f"    note: {notes[r['id']]}")
        rep += ["", f"Artifacts: attempts/r{rnd:02d}/<id>/ (trace.json, summary.txt, keyframes/, sheet.png, episode.mp4)", ""]
        if not accepted:
            rep.append(P.FINAL if left == 0 else P.CLOSING.format(focus=focus or (names[-1] if names else "(no stages)"), left=left))
        text = "\n".join(rep) + "\n"
        rdir = self.run / "rounds" / f"r{rnd:02d}"
        (rdir / "report.md").write_text(text)
        (rdir / "results.json").write_text(json.dumps(results, indent=0))
        (rdir / "audit.json").write_text(json.dumps({i: dict(note=notes.get(i, ""), stages={nm: dict(ok=ok, evidence=ev)
                                                     for nm, (ok, ev) in row.items()}) for i, row in grid.items()}, indent=1))
        return text, focus

    def one_round(self, rnd):
        names, results, accepted, grew, wall = self.rollout(rnd)
        grid, notes = self.audit(rnd, names, results)
        text, focus = self.report(rnd, names, results, accepted, grew, grid, notes)
        n_ok = sum(r["success"] for r in results)
        ok_t = [r["steps"] * self.dt for r in results if r["success"]]
        with open(self.run / "log.jsonl", "a") as f:
            f.write(json.dumps(dict(round=rnd, n_ok=n_ok, n=len(results), grew=grew, done=accepted, focus=focus,
                                    time_mean_s=round(float(np.mean(ok_t)), 2) if ok_t else None,
                                    policy_sha=hashlib.sha1((self.run / "policy.py").read_bytes()).hexdigest()[:10],
                                    wall_s=round(wall, 1), episodes=results)) + "\n")
        self.log(f"round {rnd}: {n_ok}/{len(results)} focus {focus} accepted={accepted}")
        return text, accepted

    # ----- the whole run
    def pending_turn(self, text, tag):
        """A resumed run first finishes an experiment the coder asked for; otherwise it (re)sends `text`."""
        name = self.unrun_experiment() if self.resumed and self.st["session"] else None
        self.resumed = False
        if name:
            self.log(f"resume: running the pending experiment {name}")
            self.coder_turn(self.run_experiment(name), f"{tag}_resume")
        else:
            self.coder_turn(text, tag)

    def run_loop(self):
        if self.st["phase"] == "brief":
            self.pending_turn((self.run / "BRIEF.md").read_text(), "brief")
            self.save(phase="rollout")
        while self.st["phase"] != "done":
            if self.st["phase"] == "rollout":
                rnd = self.st["round"] + 1
                text, accepted = self.one_round(rnd)
                self.save(round=rnd, phase="done" if accepted else "coder")
            else:                                             # "coder": the report of the last round is pending
                rnd = self.st["round"]
                last = rnd >= self.st["max_rounds"]
                self.pending_turn((self.run / "rounds" / f"r{rnd:02d}" / "report.md").read_text(),
                                  "final" if last else f"r{rnd:02d}")
                self.save(phase="done" if last else "rollout")

    def evaluate(self):
        rnd, path = eval_protocol.best_round(self.run)
        self.log(f"best round r{rnd:02d}; held-out eval on {path.relative_to(self.run)}")
        policy = eval_protocol.load_policy(path)
        edir = self.run / f"eval_r{rnd:02d}"                 # one attempt dir (video, trace, summary) per held-out start

        def save(k, bank_idx, r):
            start = dict(id=f"e{k:03d}", bank_idx=bank_idx)
            r["stdout"] = ""
            T.write_attempt(edir / start["id"], start, r, self.vis, video=True)
            shutil.copy(path, edir / start["id"] / "code.py")
            r.pop("frames", None), r.pop("keyframes", None)

        summary, rows = eval_protocol.evaluate(self.env, policy, self.st["task"], self.st["level"], self.st["bank_dir"],
                                               record=True, on_episode=save)
        summary["policy"] = str(path)
        summary["attempts"] = str(edir)
        (self.run / f"eval_r{rnd:02d}.json").write_text(json.dumps(dict(summary=summary, episodes=rows), indent=1))
        eval_protocol.print_summary(summary, f"r{rnd:02d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="R8")
    ap.add_argument("--schedule", default="2,5,10,15,20", help="debug-set sizes; next size when a round passes every start")
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--level", default="final")
    ap.add_argument("--horizon", type=float, default=40.0, help="episode time limit in seconds")
    ap.add_argument("--model", default="claude-opus-5-5[1m]", help="coder")
    ap.add_argument("--audit-model", default="claude-opus-5-5[1m]", help="auditor (fresh session per audit)")
    ap.add_argument("--audit-chunk", type=int, default=5, help="failed starts per auditor session")
    ap.add_argument("--max-experiments", type=int, default=3, help="per round")
    ap.add_argument("--coder-turns", type=int, default=300, help="claude --max-turns per coder call")
    ap.add_argument("--audit-turns", type=int, default=150)
    ap.add_argument("--call-timeout", type=int, default=4 * 3600, help="seconds per claude call")
    ap.add_argument("--bank-dir", default=str(eval_protocol.CAP_BANK_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true", help="continue an existing run from its state.json")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="no claude: a trivial stage file, one round without audit, one experiment; checks the plumbing")
    a = ap.parse_args()
    run = Path(a.out or ROOT / f"cap/runs/autoloop/{task_name(a.task)}_auto").resolve()
    if (run / "state.json").exists() and not a.resume:
        sys.exit(f"{run} exists; pass --resume to continue it or another --out")
    loop = Loop(a, run)
    print(f"[run] {run}", flush=True)
    if a.smoke:
        return smoke(loop)
    loop.run_loop()
    if not a.no_eval and (run / "log.jsonl").exists():
        loop.evaluate()


def smoke(loop: Loop):
    run = loop.run
    (run / "plan.md").write_text("Stages: open -> reach\nopen: gripper open. reach: tcp 10 cm above the part estimate.\n")
    (run / "stages" / "01_open.py").write_text("api.set_gripper(True)\nstage_done(api.observe()['gripper_width'] > 0.07, 'open')\n")
    (run / "stages" / "02_reach.py").write_text(
        "p, q, conf = api.locate_part()\nprint('part', p, conf)\n"
        "ok = p is not None and api.move_to(p + np.array([0, 0, 0.10]), api.observe()['tcp_quat'])\n"
        "stage_done(ok, f'above part {p}')\n")
    loop.audit = lambda rnd, names, results: ({r["id"]: {n: (r["stages"][n] == "ok", "smoke") for n in names} for r in results},
                                              {r["id"]: "smoke: no auditor" for r in results})
    text, _ = loop.one_round(1)
    print(text)
    d = run / "experiments" / "01_probe"
    d.mkdir(parents=True)
    (d / "hypothesis.md").write_text("Hypothesis: the tracker sees the part from home\nPrediction: n_points > 0\n"
                                     "Prefix: open\nStarts: s00\n")
    (d / "experiment.py").write_text("p, q, conf = api.locate_part()\nprint('probe', p, conf)\nassert conf['n_points'] > 0\n")
    print(loop.run_experiment("01_probe"))
    print("files:", sorted(str(p.relative_to(run)) for p in (run / "attempts" / "r01").rglob("*") if p.is_file())[:12])


if __name__ == "__main__":
    main()
