"""The robot cell as a service. The simulator runs here; a client (robot.py) sees only what the API returns.

  PYTHONPATH=.:benchmark MUJOCO_GL=egl SERVER_TOKEN=... python benchmark/server.py --port 8300 --out runs/server/<id>

One connection per client session (multiprocessing.connection, pickle). Requests are dicts with `cmd`:
  spec | methods | tasks
  start(task, seed, horizon, family)         -> ep        one part; family = board (loose on the board, default) | loose |
                                                         pickup | inhand | partial | near (the reset families of the simulator)
  board(seed, tasks, per_part_s)             -> ep, parts all parts physically on the board, one continuous episode
  select(ep, part) | free(ep)                            board: make `part` the active (scored) one / unscored motion
  call(ep, method, args, kwargs)             -> result, or episode_over + outcome
  outcome(ep) | close(ep)
  save_video(ep, name)                                   evaluation sessions only: renders the episode to <out>/<name>.mp4
Seeds >= RESERVED are the evaluation starts and need the token. Outcomes carry success/safety/timeout/steps only.
"""
from __future__ import annotations

import argparse
import os
import socket
import threading
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

import api
import eval_protocol
from perception import on_render_thread
from assembly_policy.sim.env import EnvConfig
from assembly_policy.sim.resets import FAMILIES
from assembly_policy.sim.tasks import TASKS, resolve_tasks
from multi_board import PRESETS, choose_layout, render as render_board
from multi_env import MultiPartEnv

RESERVED = 1000
TOKEN = os.environ.get("SERVER_TOKEN", "")
MAX_EPISODES = int(os.environ.get("SERVER_MAX_EPISODES", "16"))
DT = 0.04
_SLOTS = threading.BoundedSemaphore(MAX_EPISODES)
_BUILD = threading.Lock()              # model compilation / renderer creation one at a time


def public(o):
    return dict(success=bool(o.get("success", False)), safety=bool(o.get("safety", False)),
                timeout=bool(o.get("timeout", False)), steps=int(o.get("steps", 0)))


class Single:
    def __init__(self, task, seed, horizon, record, family="board"):
        if family not in FAMILIES:
            raise ValueError(f"unknown start family {family!r}; one of {FAMILIES}")
        with _BUILD:
            self.env = api.make_env(task, horizon_s=horizon)
            s0 = api.sample_starts(task, family, 1, seed=seed)[0]
            self.env.set_states([0], s0[None], task_ids=[0])
            self.api = api.AssemblyAPI(self.env, np.random.default_rng(seed), record=record)
        self.task, self.record = task, record

    def outcome(self):
        return dict(done=True, **public(self.api.outcome)) if self.api.done else dict(done=False, steps=int(self.env.t[0]))

    def save_video(self, path: Path):
        frames = np.asarray(self.api.frames + [self.env.state[0].copy()])
        on_render_thread(lambda: eval_protocol.Video(self.task).write(dict(frames=frames, success=self.api.outcome.get("success", False)), path))

    def close(self):
        self.env.close()


class Board:
    def __init__(self, seed, tasks, per_part_s, record):
        self.tasks, self.record = tasks, record
        self.rng = np.random.default_rng(seed)
        with _BUILD:
            self.env = MultiPartEnv(tasks, EnvConfig(nthread=1, horizon_s=per_part_s * len(tasks)))
            e1 = api.make_env(tasks[0]); sp = e1.sp.copy(); e1.close()
            self.env.reset_board(choose_layout(tasks, sp, self.rng), sp)
        self.api, self.frames, self.marks = None, [self.env.state[0].copy()], []

    def _flush(self):
        if self.api is not None and self.record:
            self.frames += self.api.frames + [self.env.state[0].copy()]

    def select(self, part):
        if part not in self.tasks:
            raise ValueError(f"unknown part {part!r}; parts: {self.tasks}")
        self._flush()
        self.env.scoring = True
        self.env.set_active(part)
        with _BUILD:
            self.api = api.AssemblyAPI(self.env, self.rng, record=self.record)
        self.marks.append((len(self.frames), part))

    def free(self):
        self._flush()
        self.env.scoring = False
        with _BUILD:
            self.api = api.AssemblyAPI(self.env, self.rng, record=self.record)

    def outcome(self):
        final = self.env.assembled()
        t = int(self.env.t[0])
        return dict(seated={k: bool(v["ok"]) for k, v in final.items()}, n_seated=sum(v["ok"] for v in final.values()),
                    n=len(self.tasks), time_s=round(t * DT, 2), timeout=bool(t >= self.env.horizon),
                    active=public(self.api.outcome) if self.api is not None and self.api.done else None)

    def save_video(self, path: Path):
        self._flush()
        final = self.env.assembled()
        stages = [(i - 1, f"{p}: {'seated' if final[p]['ok'] else 'not seated'}", bool(final[p]["ok"]))
                  for i, p in self.marks[1:]] + [(len(self.frames) - 1, f"{sum(v['ok'] for v in final.values())}/{len(self.tasks)} seated", True)]
        on_render_thread(render_board, self.tasks, self.frames, stages, str(path))

    def close(self):
        self.env.close()


class Session(threading.Thread):
    def __init__(self, conn, out: Path, log):
        super().__init__(daemon=True)
        self.conn, self.out, self.log, self.eps, self.n = conn, out, log, {}, 0

    def handle(self, r):
        cmd = r["cmd"]
        if cmd == "spec":
            return api.spec()
        if cmd == "methods":
            return list(api.METHODS)
        if cmd == "tasks":
            return list(TASKS)
        if cmd in ("start", "board"):
            seed = int(r.get("seed", 0))
            priv = bool(TOKEN) and r.get("token") == TOKEN
            if seed >= RESERVED and not priv:
                raise PermissionError(f"seeds >= {RESERVED} are reserved for evaluation")
            if not _SLOTS.acquire(timeout=600):
                raise RuntimeError("server busy: no episode slot free")
            self.n += 1
            ep = f"ep{self.n}"
            try:
                if cmd == "start":
                    self.eps[ep] = Single(str(r["task"]), seed, float(r.get("horizon", 40.0)), priv, str(r.get("family", "board")))
                    return dict(ep=ep)
                tasks = resolve_tasks(PRESETS.get(r.get("tasks", "board20"), r.get("tasks", "board20")))
                self.eps[ep] = Board(seed, tasks, float(r.get("per_part_s", 40.0)), priv)
                return dict(ep=ep, parts=list(tasks))
            except Exception:
                _SLOTS.release()
                raise
        e = self.eps[r["ep"]]
        if cmd == "call":
            if r["method"] not in api.METHODS:
                raise AttributeError(f"api has no method {r['method']!r}")
            a = e.api
            if a is None:
                raise RuntimeError("board episode: call select(part) or free() first")
            try:
                return dict(result=getattr(a, r["method"])(*r.get("args", ()), **r.get("kwargs", {})))
            except api.EpisodeOver:
                return dict(episode_over=True, outcome=public(a.outcome))
        if cmd == "select":
            e.select(str(r["part"])); return dict(ok=True)
        if cmd == "free":
            e.free(); return dict(ok=True)
        if cmd == "outcome":
            return e.outcome()
        if cmd == "save_video":
            if not e.record:
                raise PermissionError("videos are saved for evaluation sessions only")
            p = self.out / f"{r['name']}.mp4"
            p.parent.mkdir(parents=True, exist_ok=True)
            e.save_video(p); return dict(path=str(p))
        if cmd == "close":
            self.eps.pop(r["ep"]).close(); _SLOTS.release(); return dict(ok=True)
        raise ValueError(f"unknown command {cmd!r}")

    def run(self):
        try:
            while True:
                try:
                    r = self.conn.recv()
                except (EOFError, OSError):
                    break
                t0 = time.perf_counter()
                try:
                    self.conn.send(self.handle(r))
                except Exception as ex:
                    self.conn.send(dict(error=f"{type(ex).__name__}: {ex}"))
                    if not isinstance(ex, (PermissionError, ValueError, AttributeError, RuntimeError, KeyError)):
                        self.log(traceback.format_exc()[-800:])
                if r.get("cmd") in ("start", "board", "close", "save_video"):
                    self.log(f"{r['cmd']} {r.get('task', r.get('tasks', ''))} {r.get('family', '')} seed={r.get('seed', '')} {time.perf_counter() - t0:.1f}s")
        finally:
            for e in self.eps.values():
                try:
                    e.close()
                finally:
                    _SLOTS.release()
            self.conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--out", required=True, help="videos of evaluation sessions and server.addr go here")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "server.log", "a")

    def log(msg):
        logf.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n"); logf.flush()

    host = socket.gethostname()
    listener = Listener(("0.0.0.0", a.port), authkey=b"robot", backlog=64)
    (out / "server.addr").write_text(f"{host}:{a.port}\n")
    print(f"[server] listening on {host}:{a.port}  (eval token {'set' if TOKEN else 'NOT set'})", flush=True)
    log(f"listening on {host}:{a.port}")
    while True:
        Session(listener.accept(), out, log).start()


if __name__ == "__main__":
    main()
