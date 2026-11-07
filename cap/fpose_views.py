"""Save the two camera views (RGB with the part mask tinted) for a few near starts, cropped around the part."""
import sys
import numpy as np
from PIL import Image
from assembly_policy.sim.scene import build_xml
from assembly_policy.sim.tasks import TASKS
from perception import PartPerceiver
import api

task = sys.argv[1] if len(sys.argv) > 1 else "P4"
starts = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2, 3]
z = np.load(f"data/banks_near/{task}_train.npz")
env = api.make_env(task, horizon_s=15.0)
perc = PartPerceiver(build_xml(task, visual=True), "part", TASKS[task].part_asset)
tiles = []
for k in starts:
    env.set_states([0], z["states"][k][None], task_ids=[0])
    views = perc.capture(env.state[0])
    row = []
    for name in ("global", "wrist"):
        v = views[name]
        img = v["rgb"].copy()
        img[v["mask"]] = (0.4 * img[v["mask"]] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
        ys, xs = np.nonzero(v["mask"])
        cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (img.shape[0] // 2, img.shape[1] // 2)
        r = 120 if name == "global" else 110
        y0, x0 = max(0, cy - r), max(0, cx - r)
        crop = img[y0:y0 + 2 * r, x0:x0 + 2 * r]
        crop = np.array(Image.fromarray(crop).resize((300, 300), Image.NEAREST))
        row.append(crop)
        print(f"start {k} {name}: {int(v['mask'].sum())} px, mask bbox x {xs.min() if len(xs) else -1}-{xs.max() if len(xs) else -1} y {ys.min() if len(ys) else -1}-{ys.max() if len(ys) else -1}", flush=True)
    tiles.append(np.concatenate(row, 1))
Image.fromarray(np.concatenate(tiles, 0)).save(f"cap/runs/fpose_views_{task}.png")
print("wrote", f"cap/runs/fpose_views_{task}.png")
