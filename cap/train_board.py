"""PPO from scratch on the full board: all 20 parts physical, one active part, episodes from the near20 bank.

Same trainer as assembly_policy.train (point-cloud observation of the active part, binary reward, beta reset sampler);
only the simulator is the multi-part scene (cap/multi_env.py) and the bank is data/banks_near20/<task>_train.npz.

  PYTHONPATH=.:cap python cap/train_board.py --task P4 --run-dir runs/board_P4 --envs 512 --iters 3000
"""
from __future__ import annotations

import sys

import numpy as np

import assembly_policy.train as T
from assembly_policy.rl import vecenv
from assembly_policy.sim.env import EnvConfig
from assembly_policy.sim.tasks import resolve_tasks
from multi_env import MultiPartEnv

SCENE = "all20"


class BoardVecEnv(vecenv.LocalVecEnv):
    def __init__(self, task_names, n_envs, ecfg: EnvConfig, provider, t0=None):
        assert len(task_names) == 1, "one active part per training run"
        self.provider = provider
        self.env = MultiPartEnv(resolve_tasks(SCENE), ecfg, n_envs=n_envs, active=task_names[0], physics_of=task_names[0])
        self.N, self.horizon = n_envs, self.env.horizon
        e, s, t = provider(n_envs)
        self.entry = e
        self.env.set_states(np.arange(n_envs), s, t)
        self.next_entry, self.next_state, self.next_task = provider(n_envs)
        if t0 is not None:
            self.env.t[:] = t0


def main():
    argv = sys.argv[1:]
    if "--task" in argv:
        i = argv.index("--task"); task = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    else:
        task = "P4"
    argv = ["--tasks", task, "--bank-dir", "data/banks_near20", "--no-eval", "--remote-workers", "0"] + argv
    T.LocalVecEnv = BoardVecEnv
    a = T.parse(argv)
    print(f"[train_board] scene {SCENE}, active part {task}, {a.envs} envs", flush=True)
    tr = T.Trainer(a)
    tr.train()
    for p in tr._children:
        p.wait()


if __name__ == "__main__":
    main()
