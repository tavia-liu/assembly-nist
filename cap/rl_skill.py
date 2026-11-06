"""The AssembleNIST RL checkpoint as a skill a cap policy can call.

    from rl_skill import RLSkill
    rl = RLSkill(api)                 # loads checkpoints/e8192_it12500.pt for api's task
    rl.run(max_s=10.0)                # RL drives the robot for up to max_s seconds (or until the episode ends)

The skill steps the same episode through api.step(), so the trace, keyframes and time budget are shared with
the code around it. Actions are sampled (as in the AlphaNIST evaluation) unless deterministic=True.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from assembly_policy.evaluate import TorchPolicy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "checkpoints/e8192_it12500.pt"


class RLSkill:
    def __init__(self, api, ckpt=DEFAULT_CKPT, deterministic=False, seed=0):
        self.api = api
        self.pol = TorchPolicy(str(ckpt), [api.task.name], "cpu", deterministic=deterministic)
        torch.manual_seed(seed)

    def run(self, max_s=10.0, stop=None):
        """Run the RL policy for up to max_s seconds. `stop(obs) -> bool` ends it early. Returns the observation."""
        api = self.api
        n = max(1, int(round(max_s / api.env.ecfg.control_dt)))
        o = None
        try:
            for i in range(n):
                a = self.pol(api.env)[0]
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
