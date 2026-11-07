"""Remote simulation worker with image-feature observations (cap/encoded_env.py) for cap/train_feat.py."""
import argparse

from assembly_policy.rl import remote, vecenv
from encoded_env import EncodedEnv

vecenv.OBS_KEYS = vecenv.OBS_KEYS + ("feat",)
remote.OBS_KEYS = vecenv.OBS_KEYS
remote.BatchedAssemblyEnv = lambda task_names, task_of_env, ecfg: EncodedEnv(task_names, task_of_env, ecfg)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--nthread", type=int, default=8)
    a = ap.parse_args()
    remote.worker_main(a.run_dir, a.nthread, 7200.0, 15.0)
