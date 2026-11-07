"""FoundationPose as a local service (run inside the `fpose` conda env; the simulator talks to it over a socket).

    conda run -n fpose python cap/fpose_server.py --port 8130

Protocol (multiprocessing.connection, pickle): request dict(mesh=<obj path>, cmd="register"|"track", K, rgb, depth
(metres, 0 = invalid), mask (register only), init_pose (track only, optional 4x4 cam<-mesh to seed the track))
-> dict(pose=4x4 cam<-mesh, CV camera convention) or dict(error=...). One estimator per mesh, kept in memory.
"""
import argparse
import logging
import os
import sys
from multiprocessing.connection import Listener

FP = os.environ.get("FOUNDATIONPOSE_DIR", "/engrfs/project/chongjie/buzhao/FoundationPose")
sys.path.insert(0, FP)
os.chdir(FP)
import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
import trimesh                                                # noqa: E402
import nvdiffrast.torch as dr                                 # noqa: E402
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, set_logging_format, set_seed   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--est-iters", type=int, default=5)
    ap.add_argument("--track-iters", type=int, default=2)
    a = ap.parse_args()
    set_logging_format(); logging.getLogger().setLevel(logging.WARNING); set_seed(0)
    scorer, refiner, glctx = ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()
    ests = {}

    def get(mesh_path):
        if mesh_path not in ests:
            mesh = trimesh.load(mesh_path, force="mesh")
            ests[mesh_path] = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                                             scorer=scorer, refiner=refiner, glctx=glctx, debug=0, debug_dir="/tmp/fpose_debug")
            print(f"[fpose] loaded {mesh_path}: {len(mesh.vertices)} vertices, diameter {ests[mesh_path].diameter:.4f} m", flush=True)
        return ests[mesh_path]

    listener = Listener(("127.0.0.1", a.port), authkey=b"fpose")
    print(f"[fpose] listening on {a.port}", flush=True)
    while True:
        conn = listener.accept()
        while True:
            try:
                req = conn.recv()
            except EOFError:
                break
            try:
                est = get(req["mesh"])
                K = np.asarray(req["K"], dtype=np.float64)
                rgb = np.ascontiguousarray(req["rgb"])
                depth = np.ascontiguousarray(req["depth"], dtype=np.float32)
                if req["cmd"] == "register":
                    pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=np.asarray(req["mask"]).astype(bool), iteration=a.est_iters)
                elif req["cmd"] == "track":
                    if req.get("init_pose") is not None:          # seed: cam<-original mesh -> cam<-centred mesh
                        T = torch.as_tensor(np.asarray(req["init_pose"]), device="cuda", dtype=torch.float).reshape(4, 4)
                        est.pose_last = T @ torch.linalg.inv(est.get_tf_to_centered_mesh())
                    pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=a.track_iters)
                else:
                    raise ValueError(req["cmd"])
                conn.send(dict(pose=np.asarray(pose, dtype=np.float64).reshape(4, 4)))
            except Exception as e:
                conn.send(dict(error=f"{type(e).__name__}: {e}"))
        conn.close()


if __name__ == "__main__":
    main()
