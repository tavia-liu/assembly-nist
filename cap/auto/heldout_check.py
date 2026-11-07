"""Replay the R8 r24 policy on the first N held-out starts with video, and compare train vs eval start poses."""
import json, sys
from pathlib import Path
import numpy as np
sys.path[:0] = [".", "cap", "cap/auto"]
import api, agent_tools as T, eval_protocol
from assembly_policy.sim.resets import FAMILIES

TASK, RND, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
run = Path(f"cap/runs/autoloop/{TASK}_auto"); T.RUN = run
env = api.make_env(TASK, horizon_s=40.0); vis = T.make_vis(TASK)
sp = np.asarray(env.sp)

def poses(path):
    z = np.load(path); idx = np.nonzero(z["family"] == FAMILIES.index("board"))[0]
    S = z["states"][idx]; p = S[:, sp[:3]]; q = S[:, sp[3:7]]           # wxyz
    w, x, y, zz = q.T
    axis_z = np.stack([2 * (x * zz + w * y), 2 * (y * zz - w * x), 1 - 2 * (x * x + y * y)], 1)   # part +z in world
    return idx, p, axis_z

for split in ("train", "eval"):
    idx, p, az = poses(f"data/banks_board/{TASK}_{split}.npz")
    print(f"{split}: n={len(idx)} x[{p[:,0].min():.3f},{p[:,0].max():.3f}] y[{p[:,1].min():.3f},{p[:,1].max():.3f}] "
          f"z mean {p[:,2].mean()*1000:.1f}mm  |axis_z| mean {np.abs(az[:,2]).mean():.2f} max {np.abs(az[:,2]).max():.2f}  "
          f"axis +z up frac {(az[:,2]>0.3).mean():.2f} down frac {(az[:,2]<-0.3).mean():.2f}", flush=True)
# the debug set the loop used
st = json.loads((run / "state.json").read_text())
dbg = T.debug_starts(st); z = np.load(f"data/banks_board/{TASK}_train.npz")
_, p, az = poses(f"data/banks_board/{TASK}_train.npz")
dbg_idx = [s["bank_idx"] for s in dbg]
print("debug set bank_idx", dbg_idx, flush=True)

policy = eval_protocol.load_policy(run / f"rounds/{RND}/policy.py")
bank = np.load(f"data/banks_board/{TASK}_eval.npz"); idx = np.nonzero(bank["family"] == FAMILIES.index("board"))[0][:N]
out = run / f"heldout_check_{RND}"; out.mkdir(exist_ok=True)
names = [s[0] for s in __import__("importlib").util.spec_from_file_location("p", run / f"rounds/{RND}/policy.py") and []] or None
rows = []
for k, i in enumerate(idx):
    start = dict(id=f"e{k:02d}", bank_idx=int(i), state=bank["states"][i])
    r = T.run_one(env, start, policy, st["level"], 20_000 + k)
    T.write_attempt(out / start["id"], start, r, vis, video=True)
    sd = [(e["stage"], bool(e.get("ok")), str(e.get("evidence", ""))[:90]) for e in r["trace"] if e.get("call") == "stage_done"]
    s0 = bank["states"][i]; pz = s0[sp[3:7]]
    w, x, y, zz = pz; axz = 1 - 2 * (x * x + y * y)
    line = f"{start['id']} bank{i} {T.outcome_line(r)} d_asm={r.get('d_asm', float('nan'))*1000:.1f}mm start_axis_z={axz:+.2f} stages={sd}"
    print(line, flush=True); rows.append(line)
(out / "summary.txt").write_text("\n".join(rows) + "\n")
