"""Throughput and accuracy of PerceivedEnv: N envs from the near bank driven by the pretrained policy for K steps."""
import sys, time
import numpy as np
from assembly_policy.evaluate import TorchPolicy, env_config
from perceived_env import PerceivedEnv
task, N, K = (sys.argv[1] if len(sys.argv) > 1 else "P4"), int(sys.argv[2] if len(sys.argv) > 2 else 8), int(sys.argv[3] if len(sys.argv) > 3 else 60)
print("[bench] start", flush=True)
z = np.load(f"data/banks_near/{task}_train.npz")
pol = TorchPolicy("checkpoints/e8192_it12500.pt", [task], "cpu"); print("[bench] policy loaded", flush=True)
cfg = env_config(pol.args, 8)
t0 = time.time(); env = PerceivedEnv([task], np.zeros(N, int), cfg, every=2); print(f"[bench] env + cameras + server connection: {time.time()-t0:.1f}s", flush=True)
t0 = time.time(); env.set_states(np.arange(N), z["states"][:N], np.zeros(N, int)); print(f"reset+register {N} envs: {time.time()-t0:.1f}s  err so far {1000*np.mean(env.err) if env.err else -1:.1f} mm", flush=True)
t0 = time.time(); succ = 0
for k in range(K):
    _, done, info = env.step(pol(env))
    if k % 10 == 0: print(f"[bench] step {k}: {(time.time()-t0)/(k+1)*1000:.0f} ms/step so far", flush=True)
    succ += int(info["success"].sum())
    if done.any():
        ids = np.nonzero(done)[0]; env.set_states(ids, z["states"][N + ids], np.zeros(len(ids), int))
dt = time.time() - t0
e = np.array(env.err) * 1000
print(f"{N} envs x {K} steps: {dt:.1f}s = {dt/K*1000:.0f} ms/step, {N*K/dt:.0f} env-steps/s; tracker calls {env.stats['calls']} "
      f"({1000*env.stats['time']/max(env.stats['calls'],1):.0f} ms each); successes {succ}; perception err mm: mean {e.mean():.1f} p90 {np.percentile(e,90):.1f} max {e.max():.1f} (n={len(e)})", flush=True)
