"""PPO with image-feature observations: actor sees frozen ResNet-18 features of the two cameras (cap/encoded_env.py),
critic sees the privileged point clouds. Same trainer as assembly_policy.train otherwise.

  local:   PYTHONPATH=.:cap python cap/train_feat.py --task P4 --run-dir runs/feat_P4 --envs 64 --iters 3000
  remote:  cap/feat_learner.sbatch + cap/feat_worker.sbatch (workers run cap/encoded_worker.py)
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import assembly_policy.train as T
from assembly_policy.rl import remote, vecenv
from assembly_policy.sim.env import EnvConfig
from encoded_env import FEAT_DIM, EncodedEnv
from feat_nets import FeatActorCritic

vecenv.OBS_KEYS = vecenv.OBS_KEYS + ("feat",)
remote.OBS_KEYS = vecenv.OBS_KEYS
T.OBS = T.OBS + ("feat",)


class FeatLocalVecEnv(vecenv.LocalVecEnv):
    def __init__(self, task_names, n_envs, ecfg: EnvConfig, provider, t0=None):
        self.provider = provider
        self.env = EncodedEnv(task_names, np.zeros(n_envs, dtype=np.int64), ecfg)
        self.N, self.horizon = n_envs, self.env.horizon
        e, s, t = provider(n_envs)
        self.entry = e
        self.env.set_states(np.arange(n_envs), s, t)
        self.next_entry, self.next_state, self.next_task = provider(n_envs)
        if t0 is not None:
            self.env.t[:] = t0


class FeatTrainer(T.Trainer):
    def __init__(self, a):
        super().__init__(a)
        self.model = FeatActorCritic(self.pts.n_pts, FEAT_DIM, init_std=a.init_std, max_std=a.max_std, min_std=a.min_std).to(self.dev)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=a.lr)
        self._maybe_resume()

    def _obs(self, o, ids=None):
        ob = super()._obs(o, ids)
        sel = (lambda x: x) if ids is None else (lambda x: x[ids])
        ob["feat"] = torch.as_tensor(sel(o["feat"]), dtype=torch.float32, device=self.dev)
        return ob

    def _points(self, ob):
        return (super()._points(ob), ob["feat"])


def main():
    argv = sys.argv[1:]
    if "--task" in argv:
        i = argv.index("--task"); task = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    else:
        task = "P4"
    argv = ["--tasks", task, "--bank-dir", "data/banks_near", "--sampler", "uniform", "--no-eval"] + argv
    T.LocalVecEnv = FeatLocalVecEnv
    a = T.parse(argv)
    print(f"[train_feat] task {task}, actor on image features, {a.envs} envs, remote workers {a.remote_workers}", flush=True)
    tr = FeatTrainer(a)
    tr.train()
    for p in tr._children:
        p.wait()


if __name__ == "__main__":
    main()
