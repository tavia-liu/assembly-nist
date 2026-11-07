"""The AssembleNIST RL checkpoint as a skill a cap policy can call.

    from rl_skill import RLSkill
    rl = RLSkill(api)                 # loads checkpoints/e8192_it12500.pt for api's task
    rl.run(max_s=10.0)                # RL drives the robot for up to max_s seconds (or until the episode ends)

The skill steps the same episode through api.step(), so the trace, keyframes and time budget are shared with
the code around it. Actions are sampled (as in the AlphaNIST evaluation) unless deterministic=True.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

import assembly_policy
from assembly_policy.evaluate import TorchPolicy

ROOT = Path(assembly_policy.__file__).resolve().parents[1]          # the repo, also when this file is a frozen copy
DEFAULT_CKPT = Path(os.environ.get("CAP_RL_CKPT", ROOT / "checkpoints/e8192_it12500.pt"))


class _EnvView:
    """env.observe() with the part pose the policy may know (the camera tracker), for TorchPolicy."""

    def __init__(self, api):
        self.api = api

    def observe(self):
        o = self.api.env.observe()
        if self.api._perc is not None:
            pp, pq, _ = self.api._part_pose()
            if not np.isfinite(pp).all():            # never seen: no pose at all (never the simulator's)
                pp, pq = np.array([0.5, 0.0, -1.0]), np.array([1.0, 0.0, 0.0, 0.0])
            o["part_p"] = pp.astype(np.float32)[None]
            o["part_q"] = pq.astype(np.float32)[None]
        return o


class RLSkill:
    def __init__(self, api, ckpt=DEFAULT_CKPT, deterministic=False, seed=0):
        self.api = api
        self.view = _EnvView(api)
        self.pol = TorchPolicy(str(ckpt), [api.task.name], "cpu", deterministic=deterministic)
        torch.manual_seed(seed)

    def run(self, max_s=10.0, stop=None):
        """Run the RL policy for up to max_s seconds. `stop(obs) -> bool` ends it early. Returns the observation."""
        api = self.api
        n = max(1, int(round(max_s / api.env.ecfg.control_dt)))
        o = None
        try:
            for i in range(n):
                a = self.pol(self.view)[0]
                o = api.step(a)
                if stop is not None and stop(o):
                    break
        finally:
            # the RL policy drove the gripper through action[7]; resync the kin primitives' gripper state
            api.grip_open = bool(api.env.grip_cmd[0] > 0)
            api._a7 = 1.0 if api.grip_open else -1.0
            if not api.done:
                api._rec("rl_skill", steps=i + 1, max_s=max_s)
        return o
