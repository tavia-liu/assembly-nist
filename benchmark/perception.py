"""Cameras and a camera-based part pose tracker: what a real cell has instead of the simulator's part pose.

Three cameras -- two fixed over the board (front-right, front-left) and the wrist camera (robosuite eye-in-hand) --
give RGB, metric depth, an instance mask of the part (the simulator's segmentation stands in for a segmentation
model), intrinsics and the camera pose. `estimate()` runs FoundationPose (benchmark/fpose_server.py, a local service on
the GPU) on the camera that sees the part best: registered from RGB-D + mask + the part's CAD mesh on the first
sighting, tracked frame to frame afterwards. Only images, the estimate and its confidence reach the policy.

    p = PartPerceiver(xml_with_meshes, part_body="part", part_mesh="round_peg_8mm")
    views = p.capture(state)                 # {"global": dict(rgb, depth, mask, K, cam_pos, cam_R), "global2": ..., "wrist": ...}
    pos, quat, conf = p.estimate(state)      # conf: n_points, rms_mm (None), stale, views, cam
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
from multiprocessing.connection import Client

import mujoco
import numpy as np

from assembly_policy.sim.env import STATE_SPEC
from assembly_policy.sim.robot import mat_to_quat
from assembly_policy.sim.tasks import MESH_DIR

GLOBAL_CAM = dict(pos=(0.80, -0.42, 0.52), lookat=(0.50, 0.0, 0.02), fovy=34.0)   # front-right, the board fills the frame
GLOBAL_CAM2 = dict(pos=(0.80, 0.42, 0.52), lookat=(0.50, 0.0, 0.02), fovy=34.0)   # front-left: sees what the hand hides from the right
WRIST_CAM = dict(pos="0.05 0 0", quat="0 0.707108 0.707108 0", fovy="75")           # robosuite eye_in_hand
W, H = (int(v) for v in os.environ.get("CAP_CAM_RES", "1280x960").split("x"))            # fixed camera
WW, WH = (int(v) for v in os.environ.get("CAP_WRIST_RES", "640x480").split("x"))         # wrist camera
FPOSE_PORT = int(os.environ.get("FPOSE_PORT", "8130"))
_RENDERERS = {}        # model hash -> (model, data, renderer): one EGL context per model per process
# An EGL context can be current in one thread only, so every GL call of the process (renderer creation, capture,
# video rendering) runs on this single thread; other threads wait for their result.
_RENDER_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")


def on_render_thread(fn, *args, **kwargs):
    """Run fn on the process's render thread and return its result (re-raises its exception)."""
    if threading.current_thread().name.startswith("render"):
        return fn(*args, **kwargs)
    return _RENDER_THREAD.submit(fn, *args, **kwargs).result()


def add_cameras(xml: str) -> str:
    root = ET.fromstring(xml)
    hand = root.find(".//body[@name='hand']")
    if hand.find("camera[@name='wrist']") is None:
        ET.SubElement(hand, "camera", name="wrist", mode="fixed", pos=WRIST_CAM["pos"], quat=WRIST_CAM["quat"],
                      fovy=WRIST_CAM["fovy"])
    world = root.find("worldbody")
    for name, cam in (("global", GLOBAL_CAM), ("global2", GLOBAL_CAM2)):
        if world.find(f"camera[@name='{name}']") is not None:
            continue
        pos, look = np.array(cam["pos"]), np.array(cam["lookat"])
        z = pos - look; z /= np.linalg.norm(z)                 # a MuJoCo camera looks along its -z
        x = np.cross([0.0, 0.0, 1.0], z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        ET.SubElement(world, "camera", name=name, mode="fixed", pos=" ".join(f"{v:.6f}" for v in pos),
                      xyaxes=" ".join(f"{v:.6f}" for v in np.r_[x, y]), fovy=str(cam["fovy"]))
    g = root.find("visual/global")
    if g is None:
        vis = root.find("visual") if root.find("visual") is not None else ET.SubElement(root, "visual")
        g = ET.SubElement(vis, "global")
    g.set("offwidth", str(max(int(g.get("offwidth", "0")), W))); g.set("offheight", str(max(int(g.get("offheight", "0")), H)))
    return ET.tostring(root, encoding="unicode")


class FPoseTracker:
    """Client of benchmark/fpose_server.py (FoundationPose in its own conda env): register on the first sighting with
    the mask, then track_one per step; the track follows the camera that sees the part best (wrist when it sees
    enough of it, else the fixed camera) and is re-seeded in the other camera's frame on a switch."""
    F = np.diag([1.0, -1.0, -1.0])        # MuJoCo camera (x right, y up, z back) -> CV camera (x right, y down, z forward)

    def __init__(self, mesh_path, port=FPOSE_PORT, min_px=(150, 60), track=0, conn=None, lost_after=5, axial=False):
        self.conn = conn or Client(("127.0.0.1", port), authkey=b"fpose")
        self.mesh, self.track, self.axial = str(mesh_path), int(track), bool(axial)
        self.min_px = {"wrist": min_px[0], "global": min_px[1], "global2": min_px[1]}
        self.lost_after = lost_after
        self.T_world_part, self.cam, self.stale, self.lost = None, None, 0, 0

    @staticmethod
    def extent(mask):
        """Longest visible stretch of the part in pixels (bounding-box diagonal); 0 if unseen."""
        ys, xs = np.nonzero(mask)
        return float(np.hypot(ys.max() - ys.min(), xs.max() - xs.min())) if len(ys) else 0.0

    def _call(self, **req):
        self.conn.send(dict(mesh=self.mesh, track=self.track, **req))
        r = self.conn.recv()
        if "error" in r:
            raise RuntimeError(r["error"])
        return r["pose"]

    @classmethod
    def cv_pose(cls, view):
        """T_world<-cv_cam and its inverse for a captured view."""
        Rw = view["cam_R"] @ cls.F
        T = np.eye(4); T[:3, :3] = Rw; T[:3, 3] = view["cam_pos"]
        Ti = np.eye(4); Ti[:3, :3] = Rw.T; Ti[:3, 3] = -Rw.T @ view["cam_pos"]
        return T, Ti

    def estimate(self, views, approach=None):
        """approach: the hand's approach direction (tcp +z, palm -> fingertips) in the world; for a rod / peg held
        in the fingers it fixes which end is the inserting end (the two ends look alike)."""
        px = {n: int(v["mask"].sum()) for n, v in views.items()}
        ext = {n: (self.extent(v["mask"]) if px[n] >= self.min_px[n] else 0.0) for n, v in views.items()}
        ext["wrist"] *= 0.5                                           # the wrist camera mostly sees the tip end-on
        best = max(ext, key=ext.get)
        if self.cam is not None and ext[self.cam] > 0 and ext[self.cam] >= 0.7 * ext[best]:
            cam, self.lost = self.cam, 0                              # keep the current camera unless another sees much more
        elif ext[best] > 0:
            cam, self.lost = best, 0
        else:
            self.lost += 1
            if self.cam is not None and self.lost <= self.lost_after and self.T_world_part is not None:
                self.stale += 1                                       # brief occlusion: hold the last estimate
                return self.T_world_part, dict(n_points=px[self.cam], rms_mm=None, stale=self.stale, views=px, cam=self.cam)
            cam = None
        if cam is None:
            self.stale += 1
            return self.T_world_part, dict(n_points=max(px.values()), rms_mm=None, stale=self.stale, views=px, cam=None)
        v = views[cam]
        depth = np.where(v["depth"] < 2.5, v["depth"], 0.0).astype(np.float32)
        T_wc, T_cw = self.cv_pose(v)
        if self.T_world_part is None or self.cam != cam:                        # new camera: register afresh
            T_cp = self._call(cmd="register", K=v["K"], rgb=v["rgb"], depth=depth, mask=v["mask"])
        else:
            T_cp = self._call(cmd="track", K=v["K"], rgb=v["rgb"], depth=depth)
        T = T_wc @ np.asarray(T_cp)
        if self.axial:
            # A rod / peg held in the fingers shows the estimator only a stretch of its length, which leaves the pose
            # free to slide along the axis, and its two ends look alike. The inserting end (part-frame origin, the
            # base of the +z axis) points away from the palm; it is the extreme visible point along the axis, taken
            # over every camera that sees the part (the hand hides different stretches from different cameras).
            R = T[:3, :3].copy()
            if approach is not None and float(R[:, 2] @ np.asarray(approach)) > 0:
                R = R @ np.diag([1.0, -1.0, -1.0])                 # 180 deg about x: flips z, keeps a right-handed frame
            # every camera that sees anything of it, the wrist camera included: it looks at the tip face end-on
            pts = [PartPerceiver.points(views[n]) for n in views if px[n] >= 20]
            pts = np.concatenate(pts) if pts else np.zeros((0, 3))
            if len(pts) >= 20:
                axis = R[:, 2]
                proj = (pts - T[:3, 3]) @ axis
                T[:3, 3] = T[:3, 3] + axis * float(np.percentile(proj, 0.5))
            T[:3, :3] = R
        if self.T_world_part is not None and self.cam == cam and np.linalg.norm(T[:3, 3] - self.T_world_part[:3, 3]) > 0.03:
            self.stale += 1                                           # a 3 cm jump in one frame is a glitch: hold
            return self.T_world_part, dict(n_points=px[cam], rms_mm=None, stale=self.stale, views=px, cam=cam)
        self.T_world_part, self.cam, self.stale, self.lost = T, cam, 0, 0
        return self.T_world_part, dict(n_points=px[cam], rms_mm=None, stale=0, views=px, cam=cam)


class PartPerceiver:
    def __init__(self, xml: str, part_body: str = "part", part_mesh: str = "", w=W, h=H, seed=0):
        key = hashlib.sha1(xml.encode()).hexdigest()

        def build():
            if key not in _RENDERERS:
                m = mujoco.MjModel.from_xml_string(add_cameras(xml))
                _RENDERERS[key] = (m, mujoco.MjData(m), {"global": mujoco.Renderer(m, h, w), "global2": mujoco.Renderer(m, h, w),
                                                         "wrist": mujoco.Renderer(m, WH, WW)})
        on_render_thread(build)
        self.m, self.d, self.rs = _RENDERERS[key]
        self.cams = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in ("global", "global2", "wrist")}
        self.body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, part_body)
        self.part_geoms = np.nonzero(self.m.geom_bodyid == self.body)[0]
        self.fpose = FPoseTracker(MESH_DIR / f"{part_mesh}.obj", axial=part_mesh.startswith(("round_peg", "rectangular_peg")))

    # ------------------------------------------------------------- cameras
    def _view(self, name, cam_id):
        r = self.rs[name]
        r.update_scene(self.d, camera=cam_id)
        rgb = r.render().copy()
        r.enable_depth_rendering(); depth = r.render().copy(); r.disable_depth_rendering()
        r.enable_segmentation_rendering(); seg = r.render()[:, :, 0].copy(); r.disable_segmentation_rendering()
        mask = np.isin(seg, self.part_geoms) & np.isfinite(depth) & (depth > 0.01) & (depth < 3.0)
        h, w = r.height, r.width
        fy = (h / 2) / math.tan(math.radians(float(self.m.cam_fovy[cam_id])) / 2)
        K = np.array([[fy, 0.0, w / 2], [0.0, fy, h / 2], [0.0, 0.0, 1.0]])
        return dict(rgb=rgb, depth=depth.astype(np.float32), mask=mask, K=K,
                    cam_pos=self.d.cam_xpos[cam_id].copy(), cam_R=self.d.cam_xmat[cam_id].reshape(3, 3).copy())

    def capture(self, state):
        """The three cameras at this physics state. depth in metres (z along the optical axis); the camera frame has
        x right, y up, z backwards (MuJoCo): world = cam_pos + cam_R @ [(u+.5-cx)/f*d, -(v+.5-cy)/f*d, -d]."""
        def cap():
            mujoco.mj_setState(self.m, self.d, state, STATE_SPEC)
            mujoco.mj_forward(self.m, self.d)
            return {n: self._view(n, c) for n, c in self.cams.items()}
        return on_render_thread(cap)

    @staticmethod
    def points(view, mask=None):
        """World points of the masked pixels of one view (default: the part's instance mask)."""
        mask = view["mask"] if mask is None else (mask & np.isfinite(view["depth"]) & (view["depth"] > 0.01))
        v, u = np.nonzero(mask)
        if len(u) == 0:
            return np.zeros((0, 3))
        K, dd = view["K"], view["depth"][v, u]
        pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * dd, -(v + 0.5 - K[1, 2]) / K[1, 1] * dd, -dd], 1)
        return view["cam_pos"] + pc @ view["cam_R"].T

    def estimate(self, state, views=None, approach=None):
        """(pos, quat, conf) of the part from the cameras; pos/quat None until the part has been seen once."""
        views = views if views is not None else self.capture(state)
        T, conf = self.fpose.estimate(views, approach=approach)
        if T is None:
            return None, None, conf
        return T[:3, 3].copy(), mat_to_quat(T[:3, :3][None])[0], conf

    def reset(self):
        self.fpose.T_world_part, self.fpose.cam, self.fpose.stale, self.fpose.lost = None, None, 0, 0
