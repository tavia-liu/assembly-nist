"""BatchedAssemblyEnv whose observation carries the camera-tracked part pose instead of the simulator's.

For every environment, each control step renders the two cameras (cap/perception.py) and asks the FoundationPose
server (cap/fpose_server.py, one track per environment) for the part pose; observe() returns that estimate as
part_p / part_q. The simulator's pose is used only for the reward. New episodes re-register. The estimate is
refreshed every `every` control steps (a real tracker runs at ~20 Hz against the 25 Hz policy).

    env = PerceivedEnv([task], np.zeros(N, int), ecfg, every=2)      # drop-in for BatchedAssemblyEnv
"""
from __future__ import annotations

import time
from multiprocessing.connection import Client

import numpy as np

from assembly_policy.sim.env import BatchedAssemblyEnv
from assembly_policy.sim.scene import build_xml
from assembly_policy.sim.tasks import MESH_DIR, TASKS
from perception import FPOSE_PORT, FPoseTracker, PartPerceiver

NO_POSE = (np.array([0.5, 0.0, -1.0], np.float32), np.array([1.0, 0.0, 0.0, 0.0], np.float32))


class PerceivedEnv(BatchedAssemblyEnv):
    def __init__(self, task_names, task_of_env, ecfg=None, pcfg=None, every=2, port=FPOSE_PORT):
        super().__init__(task_names, task_of_env, ecfg, pcfg)
        assert len(self.task_names) == 1, "one part per perceived env"
        task = TASKS[self.task_names[0]]
        self.perc = PartPerceiver(build_xml(task.name, visual=True), "part", task.part_asset)
        conn = Client(("127.0.0.1", port), authkey=b"fpose")
        axial = task.part_asset.startswith(("round_peg", "rectangular_peg"))
        self.trackers = [FPoseTracker(MESH_DIR / f"{task.part_asset}.obj", track=i, conn=conn, axial=axial) for i in range(self.N)]
        self.every = every
        self.est_p = np.tile(NO_POSE[0], (self.N, 1)); self.est_q = np.tile(NO_POSE[1], (self.N, 1))
        self.err = []                      # perception error (m) of the refreshed estimates, for logging
        self.stats = dict(calls=0, time=0.0)

    def _refresh(self, ids):
        t0 = time.perf_counter()
        fk = self.models[0].fk.forward(self.state[ids][:, self.sq])
        for j, i in enumerate(ids):
            views = self.perc.capture(self.state[i])
            T, conf = self.trackers[i].estimate(views, approach=fk["tcp_R"][j][:, 2])
            if T is not None:
                self.est_p[i], self.est_q[i] = T[:3, 3], _mat_to_quat(T[:3, :3])
                self.err.append(float(np.linalg.norm(self.est_p[i] - self.state[i, self.sp[:3]])))
        self.stats["calls"] += len(ids); self.stats["time"] += time.perf_counter() - t0

    def set_states(self, env_ids, states, task_ids=None):
        super().set_states(env_ids, states, task_ids)
        for i in np.atleast_1d(env_ids):
            self.trackers[i].T_world_part, self.trackers[i].cam, self.trackers[i].stale, self.trackers[i].lost = None, None, 0, 0
            self.est_p[i], self.est_q[i] = NO_POSE
        self._refresh(np.atleast_1d(env_ids))

    def step(self, action):
        out = super().step(action)
        due = np.nonzero(self.t % self.every == 0)[0]
        self._refresh(due)
        return out

    def observe(self):
        o = super().observe()
        o["part_p"], o["part_q"] = self.est_p.astype(np.float32).copy(), self.est_q.astype(np.float32).copy()
        return o


def _mat_to_quat(R):
    from assembly_policy.sim.robot import mat_to_quat
    return mat_to_quat(np.asarray(R)[None])[0]
