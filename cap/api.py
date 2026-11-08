"""Primitive API for driving one AssembleNIST episode from a policy we author elsewhere.

The simulator only accepts 8-D relative joint increments at 25 Hz (q_target = q + scale*a,
a[7] latches the gripper), so any LLM-authored policy -- a program written once, or an agent
deciding in the loop -- needs a layer above that. This file is that layer, and it is also the
experiment's prior-knowledge boundary, so the boundary is made explicit and enforced.

Three nested levels; a policy declares one and calls nothing above it:

  raw    observe() / step() / constants()        nothing but the simulator
  kin    + ik() move_to() move_to_joints()       task-agnostic kinematics: identical for all
           set_gripper() wait()                  20 assemblies and any 7-DoF arm
  grasp  + grasp_candidates()                    unlabelled antipodal candidates for the part

Withheld at every level: insert() / align() / reorient() / screw(), the grasp sampler's
`kind` labels ("axial" / "side" / "side_top" -- side_top is the paper's "picks it up near one
end" finding), and any worded hint about how to assemble. Which candidate to take, how to
bring a part upright, when to press down: all of that is the policy's job.

`spec(level)` renders the prompt-facing documentation from this same table, so what a policy
is told it may call cannot drift from what it is actually allowed to call.
"""
from __future__ import annotations

import os

import numpy as np
import mujoco

from assembly_policy.sim.env import STATE_SPEC, BatchedAssemblyEnv, EnvConfig
from assembly_policy.sim.resets import ResetGenerator, any_grasp
from assembly_policy.sim.robot import mat_to_quat, quat_mul, quat_to_mat, solve_ik
from assembly_policy.sim.scene import build_xml
from assembly_policy.sim.tasks import TASKS, scene_pose

# "camera": part pose from the two-camera tracker (cap/perception.py); "oracle": the simulator's pose (ablation)
PERCEPTION = os.environ.get("CAP_PERCEPTION", "camera")

LEVELS = ("raw", "kin", "grasp")

# method -> lowest level at which it may be called
_LEVEL_OF = {
    "observe": "raw", "step": "raw", "constants": "raw", "camera": "raw", "locate_part": "grasp",
    "ik": "kin", "move_to": "kin", "move_to_joints": "kin", "set_gripper": "kin", "wait": "kin",
    "grasp_candidates": "grasp",
}


class EpisodeOver(Exception):
    """The episode terminated (success, safety violation, or the 15 s horizon)."""


class LevelError(Exception):
    """The policy called a primitive above its declared level."""


class AssemblyAPI:
    """One episode of one task. A policy drives it through these methods only."""

    def __init__(self, env: BatchedAssemblyEnv, rng: np.random.Generator, level: str = "grasp",
                 record: bool = False):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        self.env, self.rng, self.level = env, rng, level
        self.tm = env.models[0]
        self.m = self.tm.model
        self._ik_d = mujoco.MjData(self.m)
        self._obs_d = mujoco.MjData(self.m)     # scratch for the contact solver in observe()
        self.task = TASKS[env.task_names[0]]
        self._goal_pos, self._goal_quat = scene_pose(self.task.part_scene)
        ecfg = env.ecfg
        self.scale = (np.asarray(ecfg.action_scale_joints, float) if len(ecfg.action_scale_joints)
                      else np.full(7, ecfg.action_scale))
        self.trace: list[dict] = []
        self.frames: list | None = [] if record else None   # per-step states, for offline rendering
        self.key_states: list = []                           # state at every recorded primitive call (keyframes)
        self.done = False
        self.outcome: dict = {}
        # start states may hand over an already-closed gripper (pickup/inhand/partial/near);
        # env.set_states latches grip_cmd from the finger opening, so mirror it or the first
        # servo step would re-open the fingers and drop the part
        self.grip_open = bool(env.grip_cmd[0] > 0)
        self._a7 = 1.0 if self.grip_open else -1.0      # what the kin primitives send on action[7]
        self.perception = PERCEPTION
        self._perc, self._perc_cache, self._views_cache, self._perc_err = None, None, None, []
        if self.perception == "camera":
            from perception import PartPerceiver
            if hasattr(env, "tasks"):                                   # multi-part board (cap/multi_env.py)
                from multi_scene import build_multi_xml
                xml, body = build_multi_xml(env.tasks, visual=True), f"part_{self.task.name}"
            else:
                xml, body = build_xml(self.task.name, visual=True), "part"
            self._perc = PartPerceiver(xml, body, self.task.part_asset)

    def _part_pose(self):
        """(pos, quat) of the part as the policy may know it: the camera estimate, or the simulator's in oracle mode."""
        s = self.env.state[0]
        true_p, true_q = s[self.env.sp[:3]].copy(), s[self.env.sp[3:7]].copy()
        if self._perc is None:
            return true_p, true_q, None
        if self._perc_cache is None or self._perc_cache[0] != int(self.env.t[0]):
            approach = self.tm.fk.forward(self.env.state[:1, self.env.sq])["tcp_R"][0][:, 2]
            pos, quat, conf = self._perc.estimate(s, self._views(), approach=approach)
            if pos is None:                                             # never seen: no pose
                pos, quat = np.full(3, np.nan), np.full(4, np.nan)
            else:
                self._perc_err.append(float(np.linalg.norm(pos - true_p)))
            self._perc_cache = (int(self.env.t[0]), pos, quat, conf)
        return self._perc_cache[1].copy(), self._perc_cache[2].copy(), dict(self._perc_cache[3])

    def _views(self):
        """The two camera images at this control step (rendered once per step)."""
        if self._views_cache is None or self._views_cache[0] != int(self.env.t[0]):
            self._views_cache = (int(self.env.t[0]), self._perc.capture(self.env.state[0]))
        return self._views_cache[1]

    # ------------------------------------------------------------------ enforcement
    def _require(self, name):
        need = _LEVEL_OF[name]
        if LEVELS.index(need) > LEVELS.index(self.level):
            raise LevelError(f"{name}() needs level {need!r}; this policy declared {self.level!r}")

    def _rec(self, call, **kw):
        """Append a trace entry: the primitive, its result, and what the robot could see afterwards."""
        s = self.env.state[0]
        fk = self.tm.fk.forward(self.env.state[:, self.env.sq])
        _, wrench = self._external_load(self._jacobian())
        pp, _, conf = self._part_pose()
        e = dict(t=round(float(self.env.t[0]) * self.env.ecfg.control_dt, 2), call=call, **kw,
                 tcp=np.round(fk["tcp_p"][0], 4).tolist(), part=np.round(pp, 4).tolist(),
                 **({"part_seen": [conf["n_points"], conf["rms_mm"], conf["stale"]]} if conf else {}),
                 grip=round(float(s[self.env.sf].sum()), 4), wrench=np.round(wrench, 1).tolist())
        if self.frames is not None:
            self.key_states.append(s.copy())
            e["key"] = len(self.key_states) - 1
        self.trace.append(e)

    # ------------------------------------------------------------------ raw
    def constants(self):
        """Fixed numbers of the MDP (same every episode)."""
        self._require("constants")
        return dict(control_hz=round(1.0 / self.env.ecfg.control_dt), horizon_s=self.env.ecfg.horizon_s,
                    action_dim=8, action_scale=self.scale.round(4).tolist(),
                    success_threshold_m=self.env.ecfg.success_threshold,
                    joint_limits=self.env.idx.jnt_range.round(3).tolist(),
                    velocity_limit=self.env.vlim.round(3).tolist(),
                    task=self.task.name, family=self.task.family)

    def observe(self):
        """Robot state, limited to what a real robot cell reports. The part is NOT in here: it is seen
        only through the cameras (camera()) or the tracker built on them (locate_part()).

        pos are metres in the world frame, quat are (w,x,y,z). `goal_*` is the pose the part
        must reach (success_threshold_m on both of its keypoints). `tcp` is the fingertip
        centre, its +z is the approach direction (palm -> fingertips) and +y the finger closing
        axis. `joint_torque_ext` [7] and `tcp_wrench_ext` [6: force xyz, torque xyz, world
        frame] are the external load the environment exerts on the arm, as libfranka's
        tau_ext_hat / O_F_ext_hat_K. The simulator's assembly distance is NOT exposed: no
        robot can measure it.
        """
        self._require("observe")
        s = self.env.state[0]
        fk = self.tm.fk.forward(self.env.state[:, self.env.sq])
        J = self._jacobian()
        tau_ext, wrench = self._external_load(J)
        oracle = {} if self._perc is not None else dict(part_pos=s[self.env.sp[:3]].copy(), part_quat=s[self.env.sp[3:7]].copy())
        return dict(
            t=round(float(self.env.t[0]) * self.env.ecfg.control_dt, 2),
            time_left=round(float(self.env.horizon - self.env.t[0]) * self.env.ecfg.control_dt, 2),
            tcp_pos=fk["tcp_p"][0].copy(), tcp_quat=mat_to_quat(fk["tcp_R"][:1])[0],
            **oracle,
            goal_pos=self._goal_pos.copy(), goal_quat=self._goal_quat.copy(),
            joints=s[self.env.sq].copy(), joint_vel=s[self.env.sqd].copy(),
            gripper_width=float(s[self.env.sf].sum()),
            gripper_open_cmd=self.grip_open,
            gripper_target_width=round(float(self.env.grip_cmd[0]) * 2, 4),
            tcp_jacobian=J,
            joint_torque_ext=tau_ext, tcp_wrench_ext=wrench,
        )

    def camera(self):
        """The two cameras at this control step: dict(global=..., wrist=...), each dict(rgb [H,W,3] uint8,
        depth [H,W] metres along the optical axis, mask [H,W] bool = the part's instance mask (a segmentation
        model's output), K 3x3 intrinsics, cam_pos [3], cam_R [3,3] camera-to-world). `global` is fixed over the
        board, `wrist` rides on the hand looking along the fingers. A pixel (u,v) with depth d is at
        cam_pos + cam_R @ [(u+.5-cx)/fx*d, -(v+.5-cy)/fy*d, -d] in the world. 320x240, no noise."""
        self._require("camera")
        if self._perc is None:
            raise RuntimeError("camera() needs CAP_PERCEPTION=camera")
        return {n: dict(v) for n, v in self._views().items()}

    def locate_part(self):
        """Pose of the part from the two cameras (depth + mask -> points, ICP on the part's CAD model, tracked
        from step to step): (pos [3], quat [4], conf) with conf = dict(n_points seen, rms_mm fit residual,
        stale = steps since the part was last seen, views = points per camera). pos/quat are None until the part
        has been seen once; while it is hidden the last estimate is returned (stale > 0). A few mm / degrees
        off when the part is far from the wrist camera or mostly hidden; symmetric parts have an arbitrary roll."""
        self._require("locate_part")
        pp, pq, conf = self._part_pose()
        if not np.isfinite(pp).all():
            return None, None, conf
        return pp, pq, conf

    def _external_load(self, J):
        """External joint torques from the constraint solver and the TCP wrench they imply.

        mj_forward on the current full state recomputes contacts without stepping;
        qfrc_constraint is the generalized force the environment (contacts, joint limits)
        applies to the arm. A real Franka estimates the same quantities from its joint torque
        sensors every cycle, so exposing them models a robot that can feel a jam.
        """
        d = self._obs_d
        mujoco.mj_setState(self.m, d, self.env.state[0], STATE_SPEC)
        mujoco.mj_forward(self.m, d)
        tau_ext = d.qfrc_constraint[self.env.idx.arm_dof].copy()
        wrench = np.linalg.pinv(J.T) @ tau_ext          # tau = J^T f  ->  f (6,)
        return tau_ext, wrench

    def _jacobian(self):
        """6x7 TCP Jacobian at the current joints (rows: linear xyz, then angular xyz).

        A real Franka controller has this -- libfranka publishes it every cycle -- so
        withholding it would model a robot that does not know its own kinematics.
        """
        d = self._ik_d
        d.qpos[self.env.idx.arm_qpos] = self.env.state[0, self.env.sq]
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, d, jp, jr, self.env.idx.tcp_site)
        return np.vstack([jp[:, self.env.idx.arm_dof], jr[:, self.env.idx.arm_dof]])

    def step(self, action):
        """One 25 Hz control step. action[0:7] are relative joint targets in units of
        action_scale (clipped to [-1,1]); action[7] > 0.5 commands fully open (0.08 m), < -0.5
        fully closed, in between keeps the current gripper target (set_gripper(width=) sets an
        intermediate one). Returns observe(). Raises EpisodeOver when the episode ends."""
        self._require("step")
        return self._step(np.asarray(action, float).reshape(8))

    # ------------------------------------------------------------------ kin
    def ik(self, pos, quat, q_init=None):
        """Damped-least-squares IK for the TCP pose. Returns (joints[7], reachable)."""
        self._require("ik")
        q, ok = solve_ik(self.m, self._ik_d, self.env.idx, np.asarray(pos, float),
                         np.asarray(quat, float),
                         q_init=self.env.state[0, self.env.sq] if q_init is None else q_init,
                         rng=self.rng, restarts=2)
        return q, bool(ok)

    def move_to(self, pos, quat, timeout_s=3.0, tol=0.01, gain=1.0):
        """Servo the TCP to a world pose. Returns True if the joint target was reached within
        `tol` rad on every joint. Consumes episode time; raises EpisodeOver if it runs out."""
        self._require("move_to")
        q_target, reachable = self.ik(pos, quat)
        if not reachable:
            self._rec("move_to", ok=False, why="unreachable", pos=np.round(pos, 4).tolist())
            return False
        ok = self.move_to_joints(q_target, timeout_s=timeout_s, tol=tol, gain=gain, _rec=False)
        self._rec("move_to", ok=ok, why="" if ok else "servo_timeout",
                  pos=np.round(pos, 4).tolist(), err=self._joint_err(q_target))
        return ok

    def move_to_joints(self, q_target, timeout_s=3.0, tol=0.01, gain=1.0, _rec=True):
        """Servo the arm to a joint configuration, holding the current gripper command."""
        self._require("move_to_joints")
        q_target = np.asarray(q_target, float).reshape(7)
        a = np.zeros(8)
        for _ in range(max(1, int(timeout_s / self.env.ecfg.control_dt))):
            err = q_target - self.env.state[0, self.env.sq]
            if np.abs(err).max() < tol:
                break
            a[:7] = np.clip(gain * err / self.scale, -1.0, 1.0)
            a[7] = self._a7
            self._step(a)
        ok = bool(np.abs(q_target - self.env.state[0, self.env.sq]).max() < tol)
        if _rec:
            self._rec("move_to_joints", ok=ok, err=self._joint_err(q_target))
        return ok

    def set_gripper(self, open_: bool | None = None, hold_s=0.5, width: float | None = None):
        """Command the gripper and hold the arm still while the fingers act. Either open_=True/False
        (fully open 0.08 m / fully closed, force-limited) or width= a target opening in metres
        (0 .. 0.08), as a Franka Hand's move(width). The target persists until the next command."""
        self._require("set_gripper")
        if width is not None:
            per_finger = float(np.clip(width / 2.0, self.env.ecfg.grip_close, 0.04))
            self.env.grip_cmd[0] = per_finger
            self.grip_open = per_finger > 0.03
            self._a7 = 0.0                              # hold: action[7] in the deadband keeps grip_cmd
        else:
            self.grip_open = bool(open_)
            self._a7 = 1.0 if self.grip_open else -1.0
        a = np.zeros(8)
        a[7] = self._a7
        for _ in range(max(1, int(hold_s / self.env.ecfg.control_dt))):
            self._step(a)
        self._rec("set_gripper", open=self.grip_open, target=round(float(self.env.grip_cmd[0]) * 2, 4),
                  width=round(float(self.env.state[0, self.env.sf].sum()), 4))

    def wait(self, seconds=0.2):
        """Hold the current joint and gripper targets."""
        self._require("wait")
        a = np.zeros(8)
        a[7] = self._a7
        for _ in range(max(1, int(seconds / self.env.ecfg.control_dt))):
            self._step(a)

    # ------------------------------------------------------------------ grasp
    def grasp_candidates(self, part_pos, part_quat, n=32):
        """n antipodal grasp candidates for the part at the pose YOU pass (e.g. from locate_part()).

        Each is dict(pos, quat, half_width): the world TCP pose at which closing the gripper
        grasps the part, and the part's half-width along the closing axis. Candidates are
        unfiltered -- some are unreachable, collide with the board, or are poor for what
        follows. Sampling is i.i.d.; call again for more.
        """
        self._require("grasp_candidates")
        pp, pq = np.asarray(part_pos, float).reshape(3), np.asarray(part_quat, float).reshape(4)
        Rp = quat_to_mat(pq)
        out = []
        for _ in range(n):
            gp, gq, half = any_grasp(self.task.part_asset, self.rng)
            out.append(dict(pos=pp + Rp @ gp, quat=quat_mul(pq, gq), half_width=float(half)))
        return out

    # ------------------------------------------------------------------ internals
    def _joint_err(self, q_target):
        return round(float(np.abs(q_target - self.env.state[0, self.env.sq]).max()), 4)

    def _step(self, a):
        if self.done:
            raise EpisodeOver()
        if self.frames is not None:
            self.frames.append(self.env.state[0].copy())
        _, done, info = self.env.step(a[None, :])
        if bool(done[0]):
            self.done = True
            self.outcome = dict(success=bool(info["success"][0]), safety=bool(info["safety"][0]),
                                timeout=bool(info["timeout"][0]), d_asm=float(info["d_asm"][0]),
                                steps=int(self.env.t[0]), perception=self.perception,
                                perception_err_mm=(dict(mean=round(1000 * float(np.mean(self._perc_err)), 2),
                                                        p90=round(1000 * float(np.percentile(self._perc_err, 90)), 2),
                                                        max=round(1000 * float(np.max(self._perc_err)), 2), n=len(self._perc_err))
                                                   if self._perc_err else None))
            raise EpisodeOver()
        return self.observe()


# ---------------------------------------------------------------------- prompt spec
def spec(level="grasp"):
    """The API documentation a policy is given, rendered from _LEVEL_OF so it cannot drift."""
    allowed = [n for n, lv in _LEVEL_OF.items() if LEVELS.index(lv) <= LEVELS.index(level)]
    if PERCEPTION == "oracle":
        allowed = [n for n in allowed if n != "camera"]
    lines = [f"API level {level!r}. You may call ONLY these methods on `api`:", ""]
    for name in allowed:
        fn = getattr(AssemblyAPI, name)
        doc = " ".join((fn.__doc__ or "").split())
        if PERCEPTION == "oracle":
            doc = {"observe": doc.replace("The part is NOT in here: it is seen only through the cameras (camera()) or the "
                                          "tracker built on them (locate_part()).",
                                          "ORACLE mode: part_pos [3] and part_quat [4] are the simulator's exact part pose."),
                   "locate_part": "ORACLE mode: (pos [3], quat [4], None), the simulator's exact part pose (the same as "
                                  "observe()['part_pos'/'part_quat'])."}.get(name, doc)
        import inspect
        sig = str(inspect.signature(fn)).replace("(self, ", "(").replace("(self)", "()")
        sig = sig.split(" -> ")[0]
        lines.append(f"  api.{name}{sig}\n      {doc}")
    pose = ("The part's pose is never given: see it through camera() (write your own vision on the images) or "
            "locate_part() (the provided tracker)." if PERCEPTION != "oracle" else
            "ORACLE MODE (ablation): observe() also returns part_pos [3] and part_quat [4], the simulator's exact pose of "
            "the part frame (the frame the CAD model and the keypoints are defined in), and locate_part() returns that "
            "same exact pose with conf=None. There is no camera().")
    lines += ["", "Calling anything else raises LevelError. There is no insert/align/reorient/screw "
                  "primitive; the assembly strategy is yours to compute. " + pose,
              "`api.EpisodeOver` is raised as soon as the episode ends -- let it propagate."]
    return "\n".join(lines)


# ---------------------------------------------------------------------- episode plumbing
def make_env(task_name, nthread=1, horizon_s=None):
    """One-env simulator for `task_name`. horizon_s overrides the benchmark's episode limit (default 15 s)."""
    kw = dict(nthread=nthread)
    if horizon_s is not None:
        kw["horizon_s"] = float(horizon_s)
    return BatchedAssemblyEnv([task_name], [0], EnvConfig(**kw))


def sample_starts(task_name, family, n, seed=0):
    """n start states of one reset family ('loose'|'pickup'|'inhand'|'partial'|'near'|'board')."""
    states, _ = ResetGenerator(task_name, seed=seed).generate(family, n)
    return states


def run_episode(env, state, policy, level="grasp", seed=0, record=False):
    """Run `policy(api)` on one start state. Returns the outcome and the primitive trace."""
    env.set_states([0], np.asarray(state)[None, :], task_ids=[0])
    api = AssemblyAPI(env, np.random.default_rng(seed), level=level, record=record)
    err = None
    try:
        policy(api)
    except EpisodeOver:
        pass
    except LevelError:
        raise
    except Exception as e:                       # a buggy policy is a result, not a crash
        err = f"{type(e).__name__}: {e}"
    if not api.done:                             # policy returned early; let the clock run out
        try:
            while True:
                api.wait(0.2)
        except EpisodeOver:
            pass
    out = dict(**api.outcome, error=err, trace=api.trace)
    if api.frames is not None:
        out["frames"] = np.asarray(api.frames)
        out["keyframes"] = np.asarray(api.key_states) if api.key_states else np.zeros((0, api.frames and len(api.frames[0]) or 0))
    return out
