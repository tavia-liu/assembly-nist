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

import numpy as np
import mujoco

from assembly_policy.sim.env import BatchedAssemblyEnv, EnvConfig
from assembly_policy.sim.resets import ResetGenerator, any_grasp
from assembly_policy.sim.robot import mat_to_quat, quat_mul, quat_to_mat, solve_ik
from assembly_policy.sim.tasks import TASKS, scene_pose

LEVELS = ("raw", "kin", "grasp")

# method -> lowest level at which it may be called
_LEVEL_OF = {
    "observe": "raw", "step": "raw", "constants": "raw",
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
        self.task = TASKS[env.task_names[0]]
        self._goal_pos, self._goal_quat = scene_pose(self.task.part_scene)
        ecfg = env.ecfg
        self.scale = (np.asarray(ecfg.action_scale_joints, float) if len(ecfg.action_scale_joints)
                      else np.full(7, ecfg.action_scale))
        self.trace: list[dict] = []
        self.frames: list | None = [] if record else None   # per-step states, for offline rendering
        self.done = False
        self.outcome: dict = {}
        # start states may hand over an already-closed gripper (pickup/inhand/partial/near);
        # env.set_states latches grip_cmd from the finger opening, so mirror it or the first
        # servo step would re-open the fingers and drop the part
        self.grip_open = bool(env.grip_cmd[0] > 0)

    # ------------------------------------------------------------------ enforcement
    def _require(self, name):
        need = _LEVEL_OF[name]
        if LEVELS.index(need) > LEVELS.index(self.level):
            raise LevelError(f"{name}() needs level {need!r}; this policy declared {self.level!r}")

    def _rec(self, call, **kw):
        self.trace.append(dict(t=round(float(self.env.t[0]) * self.env.ecfg.control_dt, 2), call=call, **kw))

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
        """Full privileged state of the episode.

        pos are metres in the world frame, quat are (w,x,y,z). `goal_*` is the pose the part
        must reach; success is d_asm < success_threshold_m. `tcp` is the fingertip centre,
        its +z is the approach direction (palm -> fingertips) and +y the finger closing axis.
        """
        self._require("observe")
        s = self.env.state[0]
        fk = self.tm.fk.forward(self.env.state[:, self.env.sq])
        return dict(
            t=round(float(self.env.t[0]) * self.env.ecfg.control_dt, 2),
            time_left=round(float(self.env.horizon - self.env.t[0]) * self.env.ecfg.control_dt, 2),
            tcp_pos=fk["tcp_p"][0].copy(), tcp_quat=mat_to_quat(fk["tcp_R"][:1])[0],
            part_pos=s[self.env.sp[:3]].copy(), part_quat=s[self.env.sp[3:7]].copy(),
            goal_pos=self._goal_pos.copy(), goal_quat=self._goal_quat.copy(),
            joints=s[self.env.sq].copy(), gripper_width=float(s[self.env.sf].sum()),
            gripper_open_cmd=self.grip_open,
            d_asm=float(self.env.assembly_distance()[0]),
        )

    def step(self, action):
        """One 25 Hz control step. action[0:7] are relative joint targets in units of
        action_scale (clipped to [-1,1]); action[7] > 0.5 opens the gripper, < -0.5 closes it,
        in between holds. Returns observe(). Raises EpisodeOver when the episode ends."""
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
            a[7] = 1.0 if self.grip_open else -1.0
            self._step(a)
        ok = bool(np.abs(q_target - self.env.state[0, self.env.sq]).max() < tol)
        if _rec:
            self._rec("move_to_joints", ok=ok, err=self._joint_err(q_target))
        return ok

    def set_gripper(self, open_: bool, hold_s=0.5):
        """Open or close the gripper, holding the arm still while the fingers act."""
        self._require("set_gripper")
        self.grip_open = bool(open_)
        a = np.zeros(8)
        a[7] = 1.0 if self.grip_open else -1.0
        for _ in range(max(1, int(hold_s / self.env.ecfg.control_dt))):
            self._step(a)
        self._rec("set_gripper", open=self.grip_open, width=round(float(self.env.state[0, self.env.sf].sum()), 4))

    def wait(self, seconds=0.2):
        """Hold the current joint and gripper targets."""
        self._require("wait")
        a = np.zeros(8)
        a[7] = 1.0 if self.grip_open else -1.0
        for _ in range(max(1, int(seconds / self.env.ecfg.control_dt))):
            self._step(a)

    # ------------------------------------------------------------------ grasp
    def grasp_candidates(self, n=32):
        """n antipodal grasp candidates for the part at its CURRENT pose.

        Each is dict(pos, quat, half_width): the world TCP pose at which closing the gripper
        grasps the part, and the part's half-width along the closing axis. Candidates are
        unfiltered -- some are unreachable, collide with the board, or are poor for what
        follows. Sampling is i.i.d.; call again for more.
        """
        self._require("grasp_candidates")
        s = self.env.state[0]
        pp, pq = s[self.env.sp[:3]].copy(), s[self.env.sp[3:7]].copy()
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
                                steps=int(self.env.t[0]))
            raise EpisodeOver()
        return self.observe()


# ---------------------------------------------------------------------- prompt spec
def spec(level="grasp"):
    """The API documentation a policy is given, rendered from _LEVEL_OF so it cannot drift."""
    allowed = [n for n, lv in _LEVEL_OF.items() if LEVELS.index(lv) <= LEVELS.index(level)]
    lines = [f"API level {level!r}. You may call ONLY these methods on `api`:", ""]
    for name in allowed:
        fn = getattr(AssemblyAPI, name)
        doc = " ".join((fn.__doc__ or "").split())
        import inspect
        sig = str(inspect.signature(fn)).replace("(self, ", "(").replace("(self)", "()")
        sig = sig.split(" -> ")[0]
        lines.append(f"  api.{name}{sig}\n      {doc}")
    lines += ["", "Calling anything else raises LevelError. There is no insert/align/reorient/screw "
                  "primitive; the assembly strategy is yours to compute.",
              "`api.EpisodeOver` is raised as soon as the episode ends -- let it propagate."]
    return "\n".join(lines)


# ---------------------------------------------------------------------- episode plumbing
def make_env(task_name, nthread=1):
    return BatchedAssemblyEnv([task_name], [0], EnvConfig(nthread=nthread))


def sample_starts(task_name, family, n, seed=0):
    """n start states of one reset family ('loose'|'pickup'|'inhand'|'partial'|'near')."""
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
    return out
