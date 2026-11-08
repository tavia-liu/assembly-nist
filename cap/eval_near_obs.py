"""The existing checkpoint on near starts under three observation contracts:
  oracle  - the simulator's part pose (training condition)
  anchor  - FoundationPose once at the start (part in hand, in view), then kinematic propagation with the hand
  track   - FoundationPose tracking every 2 control steps
  PYTHONPATH=.:cap MUJOCO_GL=egl CAP_TRACKER=fpose python cap/eval_near_obs.py --task P4 --n 128 --modes oracle,anchor,track
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from assembly_policy.evaluate import TorchPolicy, env_config
from assembly_policy.sim.env import BatchedAssemblyEnv
from assembly_policy.sim.resets import FAMILIES

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/e8192_it12500.pt")
ap.add_argument("--task", default="P4")
ap.add_argument("--bank-dir", default="data/banks_near")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--horizon", type=float, default=15.0)
ap.add_argument("--modes", default="oracle,anchor,track")
ap.add_argument("--deterministic", action="store_true")
ap.add_argument("--nthread", type=int, default=8)
a = ap.parse_args()
z = np.load(Path(a.bank_dir) / f"{a.task}_eval.npz")
S = z["states"][np.nonzero(z["family"] == FAMILIES.index("near"))[0][:a.n]]
N = len(S)
pol = TorchPolicy(a.ckpt, [a.task], "cpu", deterministic=a.deterministic)
cfg = env_config(pol.args, a.nthread); cfg.horizon_s = a.horizon
out = {}
for mode in a.modes.split(","):
    if mode == "oracle":
        env = BatchedAssemblyEnv([a.task], np.zeros(N, int), cfg)
    else:
        from perceived_env import PerceivedEnv
        env = PerceivedEnv([a.task], np.zeros(N, int), cfg, every=2, mode=mode)
    t0 = time.time()
    env.set_states(np.arange(N), S, np.zeros(N, int))
    done_at = np.full(N, -1); success = np.zeros(N, bool); safety = np.zeros(N, bool)
    torch.manual_seed(0)
    for t in range(env.horizon):
        _, done, info = env.step(pol(env))
        newly = (done_at < 0) & done
        done_at[newly] = t; success[newly] = info["success"][newly]; safety[newly] = info["safety"][newly]
        if (done_at >= 0).all():
            break
    err = np.array(getattr(env, "err", [])) * 1000
    env.close()
    times = [(done_at[k] + 1) * cfg.control_dt for k in range(N) if success[k]]
    res = dict(mode=mode, n=N, n_ok=int(success.sum()), success_rate=round(100 * success.mean(), 1), safety=int(safety.sum()),
               timeouts=int((done_at < 0).sum()), time_mean_s=round(float(np.mean(times)), 2) if times else None,
               pose_err_mm=(dict(mean=round(float(err.mean()), 2), p90=round(float(np.percentile(err, 90)), 2), max=round(float(err.max()), 1)) if len(err) else None),
               wall_s=round(time.time() - t0, 1))
    out[mode] = res
    print(json.dumps(res), flush=True)
Path("cap/runs").mkdir(exist_ok=True)
Path(f"cap/runs/perception/eval_near_obs_{a.task}{'_det' if a.deterministic else ''}.json").write_text(json.dumps(out, indent=1))
