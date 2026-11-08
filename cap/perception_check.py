"""Check the camera tracker against the simulator: per-episode error stats, timing, hybrid policy success."""
import sys, time
from pathlib import Path
import numpy as np
import api
from assembly_policy.sim.resets import FAMILIES
from eval_protocol import CAP_BANK_DIR, START_FAMILY, load_policy

task = sys.argv[1] if len(sys.argv) > 1 else "R8"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
bank = np.load(Path(CAP_BANK_DIR) / f"{task}_eval.npz")
idx = np.nonzero(bank["family"] == FAMILIES.index(START_FAMILY))[0][:n]
env = api.make_env(task, horizon_s=40.0)
env.set_states([0], bank["states"][idx[:1]], task_ids=[0])
ap = api.AssemblyAPI(env, np.random.default_rng(0), level="grasp")
views = ap.camera()
for name, v in views.items():
    print(f"{name}: rgb {v['rgb'].shape} depth {np.nanmin(v['depth']):.2f}..{np.nanmax(v['depth']):.2f} m  part pixels {int(v['mask'].sum())}", flush=True)
pp, pq, conf = ap.locate_part()
s = env.state[0]
print("static: est", None if pp is None else np.round(pp, 4), "true", np.round(s[env.sp[:3]], 4), "conf", conf, flush=True)
t0 = time.time()
for _ in range(20):
    ap._perc_cache = ap._views_cache = None; ap.locate_part()
print(f"capture+estimate {(time.time() - t0) / 20 * 1000:.1f} ms per step", flush=True)
try:
    from PIL import Image
    Image.fromarray(np.concatenate([views["global"]["rgb"], views["wrist"]["rgb"]], 1)).save(f"cap/runs/perception/perc_{task}_rgb.png")
    m = np.concatenate([views["global"]["mask"], views["wrist"]["mask"]], 1)
    Image.fromarray((m * 255).astype(np.uint8)).save(f"cap/runs/perception/perc_{task}_mask.png")
except Exception as e:
    print("no debug image:", e)
policy = load_policy(Path("cap/hybrid_rl.py"))
for k, i in enumerate(idx):
    t0 = time.time()
    r = api.run_episode(env, bank["states"][i], policy, level="grasp", seed=k)
    print(f"{task} start {i}: success={r['success']} steps={r['steps']} wall={time.time() - t0:.1f}s "
          f"perception_err_mm={r.get('perception_err_mm')}", flush=True)
