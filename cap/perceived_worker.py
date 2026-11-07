"""Remote simulation worker whose environments see the part only through the cameras (cap/perceived_env.py).

Started on a GPU node next to a FoundationPose server (cap/perceived_worker.sbatch); connects to the learner of
--run-dir exactly like assembly_policy.rl.remote's worker, with PerceivedEnv in place of BatchedAssemblyEnv.
"""
import argparse
import os

from assembly_policy.rl import remote
from perceived_env import PerceivedEnv

EVERY = int(os.environ.get("CAP_TRACK_EVERY", "2"))


def make_env(task_names, task_of_env, ecfg):
    return PerceivedEnv(task_names, task_of_env, ecfg, every=EVERY)


remote.BatchedAssemblyEnv = make_env

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--nthread", type=int, default=8)
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--wait", type=float, default=7200.0)
    a = ap.parse_args()
    remote.worker_main(a.run_dir, a.nthread, a.wait, a.poll)
