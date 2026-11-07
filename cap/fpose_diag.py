"""Why does the tracker fail on near starts? Per start: mask pixels per camera, register from each camera alone vs
the truth, then 20 tracking steps while holding still, then with the camera-switch seeding. Also a self-check of the
camera convention: the truth projected into the camera must land on the mask."""
import sys
import numpy as np
import mujoco
from assembly_policy.sim.env import EnvConfig, STATE_SPEC
from assembly_policy.sim.robot import quat_to_mat
from assembly_policy.sim.scene import build_xml
from assembly_policy.sim.tasks import MESH_DIR, TASKS
from perception import FPoseTracker, PartPerceiver
import api

task = sys.argv[1] if len(sys.argv) > 1 else "P4"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
z = np.load(f"data/banks_near/{task}_train.npz")
env = api.make_env(task, horizon_s=15.0)
perc = PartPerceiver(build_xml(task, visual=True), "part", TASKS[task].part_asset)
mesh = MESH_DIR / f"{TASKS[task].part_asset}.obj"


def err(T, s):
    return 1000 * np.linalg.norm(T[:3, 3] - s[env.sp[:3]])


def project(view, p):
    """pixel (u, v) of a world point via the CV convention used by the tracker."""
    T_wc, T_cw = FPoseTracker.cv_pose(view)
    pc = T_cw[:3, :3] @ p + T_cw[:3, 3]
    K = view["K"]
    return K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2], pc[2]


for k in range(n):
    s = z["states"][k]
    env.set_states([0], s[None], task_ids=[0])
    views = perc.capture(env.state[0])
    truth = s[env.sp[:3]]
    line = [f"start {k}:"]
    for name, v in views.items():
        px = int(v["mask"].sum())
        u, vv, d = project(v, truth)
        ys, xs = np.nonzero(v["mask"])
        cen = (xs.mean(), ys.mean()) if px else (np.nan, np.nan)
        line.append(f"{name}: {px}px extent {FPoseTracker.extent(v['mask']):.0f}, truth projects to ({u:.0f},{vv:.0f}) depth {d:.3f}, mask centroid ({cen[0]:.0f},{cen[1]:.0f})")
        if False:
            tr = FPoseTracker(mesh, track=100 + k)
            T_wc, _ = FPoseTracker.cv_pose(v)
            depth = np.where(v["depth"] < 2.5, v["depth"], 0.0).astype(np.float32)
            T_cp = tr._call(cmd="register", K=v["K"], rgb=v["rgb"], depth=depth, mask=v["mask"])
            T = T_wc @ np.asarray(T_cp)
            line.append(f"register[{name}] err {err(T, s):.1f} mm")
            # hold still and track 10 steps on the same camera
            errs = []
            for _ in range(6):
                env.step(np.array([[0, 0, 0, 0, 0, 0, 0, -1.0]]))
                v2 = perc.capture(env.state[0])[name]
                depth2 = np.where(v2["depth"] < 2.5, v2["depth"], 0.0).astype(np.float32)
                T_cp = tr._call(cmd="track", K=v2["K"], rgb=v2["rgb"], depth=depth2)
                T_wc2, _ = FPoseTracker.cv_pose(v2)
                errs.append(err(T_wc2 @ np.asarray(T_cp), env.state[0]))
            line.append(f"track[{name}] errs {np.round(errs, 1).tolist()}")
            env.set_states([0], s[None], task_ids=[0])
    print("\n  ".join(line), flush=True)
# full tracker (camera choice, axial anchoring) over 12 hold steps, every start
for k in range(n):
    tr = FPoseTracker(mesh, track=900 + k, axial=True)
    env.set_states([0], z["states"][k][None], task_ids=[0])
    errs, cams = [], []
    for t in range(12):
        appr = env.models[0].fk.forward(env.state[:1, env.sq])["tcp_R"][0][:, 2]
        T, conf = tr.estimate(perc.capture(env.state[0]), approach=appr)
        errs.append(err(T, env.state[0]) if T is not None else -1); cams.append(conf.get("cam"))
        env.step(np.array([[0, 0, 0, 0, 0, 0, 0, -1.0]]))
    print(f"full tracker start {k}: errs {np.round(errs, 1).tolist()} cam {cams[0]}->{cams[-1]}", flush=True)
