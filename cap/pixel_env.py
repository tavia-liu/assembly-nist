"""BatchedAssemblyEnv whose observation carries the raw images of the two cameras (no part pose for the policy).

Each control step renders the fixed `global` camera and the `wrist` camera at low resolution (RGB, uint8);
observe() adds `img` = uint8 [N, 2, RES, RES, 3]. Nothing is encoded here: the policy's encoder is learned
(cap/pixel_nets.py). The simulator's part pose stays in the obs dict for logging only; cap/train_pixel.py never
feeds it to the actor or the critic.

    env = PixelEnv([task], np.zeros(N, int), ecfg)      # drop-in for BatchedAssemblyEnv
"""
from __future__ import annotations

import os
import time

import mujoco
import numpy as np

from assembly_policy.sim.env import STATE_SPEC, BatchedAssemblyEnv
from assembly_policy.sim.scene import build_xml
from perception import add_cameras

RES = int(os.environ.get("CAP_PIX_RES", "96"))
CAMS = ("global", "wrist")


class PixelEnv(BatchedAssemblyEnv):
    def __init__(self, task_names, task_of_env, ecfg=None, pcfg=None, res=RES):
        super().__init__(task_names, task_of_env, ecfg, pcfg)
        assert len(self.task_names) == 1
        self.vm = mujoco.MjModel.from_xml_string(add_cameras(build_xml(self.task_names[0], visual=True)))
        self.vd = mujoco.MjData(self.vm)
        self.res = res
        self.r = mujoco.Renderer(self.vm, res, res)
        self.cams = [mujoco.mj_name2id(self.vm, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in CAMS]
        self.img = np.zeros((self.N, len(CAMS), res, res, 3), np.uint8)
        self.stats = dict(render=0.0, steps=0)

    def _refresh(self, ids):
        ids = np.atleast_1d(ids)
        if len(ids) == 0:
            return
        t0 = time.perf_counter()
        for i in ids:
            mujoco.mj_setState(self.vm, self.vd, self.state[i], STATE_SPEC)
            mujoco.mj_forward(self.vm, self.vd)
            for c, cam in enumerate(self.cams):
                self.r.update_scene(self.vd, camera=cam)
                self.img[i, c] = self.r.render()
        self.stats["render"] += time.perf_counter() - t0; self.stats["steps"] += len(ids)

    def set_states(self, env_ids, states, task_ids=None):
        super().set_states(env_ids, states, task_ids)
        self._refresh(env_ids)

    def step(self, action):
        out = super().step(action)
        self._refresh(np.arange(self.N))
        return out

    def observe(self):
        o = super().observe()
        o["img"] = self.img.copy()
        return o
