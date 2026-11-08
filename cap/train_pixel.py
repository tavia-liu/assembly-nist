"""PPO from pixels, symmetric: actor and critic share one learned conv encoder over the two cameras + proprioception.
No oracle part pose and no point clouds anywhere (cap/pixel_env.py, cap/pixel_nets.py). Same trainer as
assembly_policy.train otherwise (same reward, PPO hyper-parameters, near starts).

  PYTHONPATH=.:cap MUJOCO_GL=egl python cap/train_pixel.py --task P4 --run-dir runs/peg_4mm_pixel_near --envs 128 --iters 40000
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import assembly_policy.train as T
from assembly_policy.rl import vecenv
from assembly_policy.sim.env import EnvConfig
from pixel_env import RES, PixelEnv
from pixel_nets import PixelActorCritic

vecenv.OBS_KEYS = vecenv.OBS_KEYS + ("img",)
T.OBS = T.OBS + ("img",)


class PixelLocalVecEnv(vecenv.LocalVecEnv):
    def __init__(self, task_names, n_envs, ecfg: EnvConfig, provider, t0=None):
        self.provider = provider
        self.env = PixelEnv(task_names, np.zeros(n_envs, dtype=np.int64), ecfg)
        self.N, self.horizon = n_envs, self.env.horizon
        e, s, t = provider(n_envs)
        self.entry = e
        self.env.set_states(np.arange(n_envs), s, t)
        self.next_entry, self.next_state, self.next_task = provider(n_envs)
        if t0 is not None:
            self.env.t[:] = t0


class PixelTrainer(T.Trainer):
    def __init__(self, a):
        self._swapped = False                      # base __init__ must not resume into the point-cloud model
        super().__init__(a)
        self.model = PixelActorCritic(res=a.pix_res, init_std=a.init_std, max_std=a.max_std, min_std=a.min_std).to(self.dev)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=a.lr)
        n = sum(p.numel() for p in self.model.parameters())
        print(f"[train_pixel] {n / 1e6:.2f}M parameters, encoder {sum(p.numel() for p in self.model.encoder.parameters()) / 1e6:.2f}M", flush=True)
        self._swapped = True
        self._maybe_resume()

    def _maybe_resume(self):
        if self._swapped:
            super()._maybe_resume()

    def _obs(self, o, ids=None):
        ob = super()._obs(o, ids)
        sel = (lambda x: x) if ids is None else (lambda x: x[ids])
        ob["img"] = torch.as_tensor(sel(o["img"]), device=self.dev)      # uint8, encoded on demand
        return ob

    def _points(self, ob):
        # the "points" slot carries the shared latent; computed once per (micro-)batch, gradients flow from both
        # the policy and the value loss into the encoder
        return self.model.encode(ob["img"])


def main():
    argv = sys.argv[1:]
    if "--task" in argv:
        i = argv.index("--task"); task = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    else:
        task = "P4"
    argv = ["--tasks", task, "--bank-dir", "data/banks_near", "--sampler", "uniform", "--no-eval"] + argv
    T.LocalVecEnv = PixelLocalVecEnv
    a = T.parse(argv)
    a.pix_res = RES
    print(f"[train_pixel] task {task}, symmetric pixel PPO, {a.envs} envs, {RES}x{RES} x 2 cameras", flush=True)
    tr = PixelTrainer(a)
    tr.train()
    for p in tr._children:
        p.wait()


if __name__ == "__main__":
    main()
