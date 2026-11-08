"""The robot cell: a Franka arm, two fixed RGB-D cameras and a wrist camera in front of the NIST task board.

    import robot
    print(robot.spec())                        # the methods of `api`
    api = robot.start("R8", seed=0)            # one part lying on the board; 40 s
    pos, quat, conf = api.locate_part()
    ...                                        # api.EpisodeOver is raised when the episode ends
    print(api.outcome())                       # success / safety / timeout / steps

    b = robot.board(seed=0)                    # all 20 parts on the board, 40 s per part, scored at the end
    for part in b.parts:
        api = b.select(part)                   # the part now being scored
        ...
    free = b.free()                            # unscored motion (release, lift) between parts
    print(b.outcome())                         # seated per part, time

Tasks: robot.tasks(). Episodes run on the cell server (ROBOT_ADDR); several may run at once.
"""
from __future__ import annotations

import os
from multiprocessing.connection import Client
from pathlib import Path


def _addr():
    f = Path(__file__).with_name("ROBOT_ADDR")
    s = os.environ.get("ROBOT_ADDR") or (f.read_text().strip() if f.exists() else "127.0.0.1:8300")
    host, port = s.rsplit(":", 1)
    return host, int(port)


class EpisodeOver(Exception):
    """The episode ended: success, a safety stop, or the time limit."""


class _Conn:
    def __init__(self):
        self.c = Client(_addr(), authkey=b"robot")

    def rpc(self, **req):
        self.c.send(req)
        r = self.c.recv()
        if isinstance(r, dict) and "error" in r:
            raise RuntimeError(r["error"])
        return r

    def close(self):
        self.c.close()


_METHODS = None


def _methods(conn):
    global _METHODS
    if _METHODS is None:
        _METHODS = tuple(conn.rpc(cmd="methods"))
    return _METHODS


class Api:
    """The methods listed by robot.spec(), plus outcome()."""
    EpisodeOver = EpisodeOver

    def __init__(self, conn, ep):
        self._conn, self._ep = conn, ep
        self._names = _methods(conn)

    def __getattr__(self, name):
        if name.startswith("_") or name not in self._names:
            raise AttributeError(f"api has no attribute {name!r}; see robot.spec()")

        def call(*args, **kwargs):
            r = self._conn.rpc(cmd="call", ep=self._ep, method=name, args=args, kwargs=kwargs)
            if r.get("episode_over"):
                raise EpisodeOver(r["outcome"])
            return r["result"]
        return call

    def __dir__(self):
        return sorted(self._names) + ["outcome", "EpisodeOver"]

    def outcome(self):
        return self._conn.rpc(cmd="outcome", ep=self._ep)


class Episode(Api):
    def close(self):
        self._conn.rpc(cmd="close", ep=self._ep); self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class Board:
    def __init__(self, seed, tasks, per_part_s, token):
        self._conn = _Conn()
        r = self._conn.rpc(cmd="board", seed=seed, tasks=tasks, per_part_s=per_part_s, token=token)
        self._ep, self.parts = r["ep"], list(r["parts"])

    def select(self, part) -> Api:
        self._conn.rpc(cmd="select", ep=self._ep, part=part)
        return Api(self._conn, self._ep)

    def free(self) -> Api:
        self._conn.rpc(cmd="free", ep=self._ep)
        return Api(self._conn, self._ep)

    def outcome(self):
        return self._conn.rpc(cmd="outcome", ep=self._ep)

    def close(self):
        self._conn.rpc(cmd="close", ep=self._ep); self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def spec() -> str:
    c = _Conn()
    try:
        return c.rpc(cmd="spec")
    finally:
        c.close()


def tasks() -> list:
    c = _Conn()
    try:
        return c.rpc(cmd="tasks")
    finally:
        c.close()


def start(task, seed=0, horizon=40.0, token=None) -> Episode:
    c = _Conn()
    r = c.rpc(cmd="start", task=task, seed=int(seed), horizon=float(horizon), token=token or os.environ.get("ROBOT_TOKEN"))
    return Episode(c, r["ep"])


def board(seed=0, tasks="board20", per_part_s=40.0, token=None) -> Board:
    return Board(int(seed), tasks, float(per_part_s), token or os.environ.get("ROBOT_TOKEN"))
