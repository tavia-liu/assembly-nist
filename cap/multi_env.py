"""Simulator for several parts on the board at once, with the single-env BatchedAssemblyEnv interface.

All parts are physical all the time. One part is *active*: the observation (part_p / part_q), the assembly
distance and the success flag refer to it, so the cap API, the RL skill and the policies work unchanged; code
switches the active part with set_active(). The clock runs over the whole board (horizon_s = the full budget).
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

from assembly_policy.sim.tasks import equivalent_keypoints

from assembly_policy.sim.env import HISTORY, PROPRIO_DIM, STATE_SPEC, BatchedAssemblyEnv, EnvConfig, FrankaFK
from assembly_policy.sim.robot import FINGER_OPEN, HOME_Q, quat_to_mat
from assembly_policy.sim.scene import ARM_KD, ARM_KP, TORQUE_LIMIT, VELOCITY_LIMIT, PhysicsConfig
from assembly_policy.sim.tasks import TASKS, scene_pose
from multi_scene import build_multi_xml


class MultiIndices:
    """assembly_policy.sim.robot.Indices for the multi-part model; the part fields follow the active task."""

    def __init__(self, m: mujoco.MjModel, tasks):
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
        self.arm_qpos = np.array([m.jnt_qposadr[jid(f"joint{i + 1}")] for i in range(7)])
        self.arm_dof = np.array([m.jnt_dofadr[jid(f"joint{i + 1}")] for i in range(7)])
        self.finger_qpos = np.array([m.jnt_qposadr[jid(f"finger_joint{i + 1}")] for i in range(2)])
        self.finger_dof = np.array([m.jnt_dofadr[jid(f"finger_joint{i + 1}")] for i in range(2)])
        self.tcp_site = sid("tcp")
        self.hand_body = bid("hand")
        self.finger_bodies = [bid(n) for n in ("left_finger", "right_finger")]
        self.jnt_range = np.array([m.jnt_range[jid(f"joint{i + 1}")] for i in range(7)])
        self.part_qpos_of = {t: int(m.jnt_qposadr[jid(f"part_free_{t}")]) for t in tasks}
        self.part_dof_of = {t: int(m.jnt_dofadr[jid(f"part_free_{t}")]) for t in tasks}
        self.part_body_of = {t: bid(f"part_{t}") for t in tasks}
        self.part_site_of = {t: sid(f"part_frame_{t}") for t in tasks}
        self.fixture_site_of = {t: sid(f"fixture_frame_{t}") for t in tasks}
        self.set_active(tasks[0])

    def set_active(self, t):
        self.part_qpos, self.part_dof = self.part_qpos_of[t], self.part_dof_of[t]
        self.part_body, self.part_site, self.fixture_site = self.part_body_of[t], self.part_site_of[t], self.fixture_site_of[t]


class MultiModel:
    """assembly_policy.sim.env.TaskModel for the multi-part scene."""

    def __init__(self, tasks, pcfg: PhysicsConfig, nthread, physics_of=None):
        self.tasks, self.name = list(tasks), "+".join(tasks)
        self.model = mujoco.MjModel.from_xml_string(build_multi_xml(tasks, pcfg, physics_of=physics_of))
        self.datas = [mujoco.MjData(self.model) for _ in range(nthread)]
        self.idx = MultiIndices(self.model, tasks)
        self.fk = FrankaFK(self.model)
        self.goal = {}
        for t in tasks:
            task = TASKS[t]
            kp = np.asarray(task.keypoints)
            ppos, pquat = scene_pose(task.part_scene)
            goal_kp = ppos + kp @ quat_to_mat(pquat).T
            g = goal_kp[1] - goal_kp[0]
            fpos, fquat = scene_pose(task.fixture_scene)
            self.goal[t] = dict(keypoints=kp, kp_eq=equivalent_keypoints(task), goal_kp=goal_kp, goal_axis=g / np.linalg.norm(g),
                                fixture_pos=fpos, fixture_R=quat_to_mat(fquat))
        self.set_active(tasks[0])

    def set_active(self, t):
        self.idx.set_active(t)
        self.active = t
        for k, v in self.goal[t].items():
            setattr(self, k, v)


class MultiPartEnv(BatchedAssemblyEnv):
    def __init__(self, tasks, ecfg: EnvConfig | None = None, pcfg: PhysicsConfig | None = None, n_envs: int = 1,
                 active: str | None = None, physics_of: str | None = None):
        """n_envs copies of the same scene (training); `active` = the part every env works on (default tasks[0]);
        physics_of = the task whose physics the whole scene uses (default: the finest needed)."""
        self.ecfg = ecfg or EnvConfig()
        self.pcfg = pcfg or PhysicsConfig(fine_families=self.ecfg.fine_families,
                                          fine_sdf_initpoints=self.ecfg.fine_sdf_initpoints,
                                          fine_sdf_iterations=self.ecfg.fine_sdf_iterations)
        self.tasks = list(tasks)
        self.models = [MultiModel(self.tasks, self.pcfg, self.ecfg.nthread, physics_of=physics_of)]
        m0 = self.models[0].model
        self.nstate = mujoco.mj_stateSize(m0, STATE_SPEC)
        self.idx = self.models[0].idx
        self.substeps_k = [int(round(self.ecfg.control_dt / m0.opt.timestep))]
        self.dt_k = np.array([self.substeps_k[0] * m0.opt.timestep])
        self.substeps = self.substeps_k[0]
        self.horizon = int(round(self.ecfg.horizon_s / self.ecfg.control_dt))
        self.N = int(n_envs)
        self.task = np.zeros(self.N, dtype=np.int64)
        self.state = np.zeros((self.N, self.nstate))
        self.prev_action = np.zeros((self.N, 8))
        self.grip_cmd = np.zeros(self.N)
        self.hist = np.zeros((self.N, HISTORY, PROPRIO_DIM), dtype=np.float32)
        self.t = np.zeros(self.N, dtype=np.int64)
        self.pool = ThreadPoolExecutor(self.ecfg.nthread) if self.ecfg.nthread > 1 else None
        self._tls = threading.local()
        self.vlim = np.asarray(VELOCITY_LIMIT) * self.ecfg.velocity_limit_scale
        self.kp, self.kd, self.tlim = np.asarray(ARM_KP), np.asarray(ARM_KD), np.asarray(TORQUE_LIMIT)
        self.jlo, self.jhi = self.idx.jnt_range[:, 0], self.idx.jnt_range[:, 1]
        self.nq = m0.nq
        self.sq = 1 + self.idx.arm_qpos
        self.sf = 1 + self.idx.finger_qpos
        self.sqd = 1 + self.nq + self.idx.arm_dof
        self.scoring = True          # False: step() never reports success (code finishing / recovery moves)
        self.set_active(active or self.tasks[0])

    # ------------------------------------------------------------- parts
    def part_slices(self, t):
        qp, dp = self.idx.part_qpos_of[t], self.idx.part_dof_of[t]
        return np.arange(1 + qp, 8 + qp), np.arange(1 + self.nq + dp, 7 + self.nq + dp)

    def set_active(self, t):
        self.models[0].set_active(t)
        self.active, self.task_names = t, [t]
        self.sp, self.spd = self.part_slices(t)

    def assembly_distance(self, t=None):
        if t is None and not self.scoring:
            return np.full(self.N, np.inf)
        t = t or self.active
        g, sp = self.models[0].goal[t], self.part_slices(t)[0]
        pp, pq = self.state[:, sp[:3]], self.state[:, sp[3:7]]
        kp = pp[:, None, None, :] + np.einsum("nij,skj->nski", quat_to_mat(pq), g["kp_eq"])
        return np.linalg.norm(kp - g["goal_kp"][None, None], axis=3).max(2).min(1)        # min over mesh symmetries

    def assembled(self):
        """Final scoring: every part's assembly distance (m) and whether it is seated."""
        return {t: dict(d_asm=float(self.assembly_distance(t)[0]),
                        ok=bool(self.assembly_distance(t)[0] < self.ecfg.success_threshold)) for t in self.tasks}

    def part_pos(self, t):
        return self.state[0, self.part_slices(t)[0][:3]].copy()

    def tcp_pos(self):
        return self.models[0].fk.forward(self.state[:, self.sq])["tcp_p"][0].copy()

    # ------------------------------------------------------------- reset
    def set_states(self, env_ids, states, task_ids=None):
        super().set_states(env_ids, states, None)

    def compose(self, part_states, sp_single, robot=None):
        """A full state: robot at home with the gripper open (or `robot` = a single-task state whose arm, fingers
        and part pose are copied for the active part) and each part of part_states at its single-task pose."""
        s = np.zeros(self.nstate)
        s[self.sq] = HOME_Q
        s[self.sf] = FINGER_OPEN
        for t, st in part_states.items():
            s[self.part_slices(t)[0]] = np.asarray(st)[sp_single]
        if robot is not None:
            robot = np.asarray(robot)
            s[self.sq], s[self.sf] = robot[self.sq], robot[self.sf]
            s[self.sp] = robot[sp_single]
        return s

    def reset_board(self, part_states, sp_single, settle_s=1.0):
        """Robot at home with the gripper open, each part at the pose of its single-task start state
        (part_states[t], read at the single-task part slice sp_single), settled for settle_s seconds."""
        s = np.zeros(self.nstate)
        s[self.sq] = HOME_Q
        s[self.sf] = FINGER_OPEN
        for t, st in part_states.items():
            s[self.part_slices(t)[0]] = np.asarray(st)[sp_single]
        self.set_states([0], s[None], [0])
        hold = np.zeros((1, 8)); hold[0, 7] = 1.0
        for _ in range(int(settle_s / self.ecfg.control_dt)):
            self.step(hold)
        self.set_states([0], self.state.copy(), [0])      # clock, history and action memory back to zero
