"""Hand-written reference policy for R16, used only to answer one question:

    is the `api` layer expressive enough to solve the task at all?

It is the ceiling probe for the CaP route, not the measured artifact. If this cannot clear
R16 the primitive boundary is drawn wrong and no LLM-authored program will do better; if it
can, whatever an LLM-authored program fails to reach is a fact about the authoring loop
rather than about the API.

Strategy (nothing here is handed to the policies under test):
  while grasped, the TCP pose relative to the part is fixed, so the TCP pose that seats the
  part is  goal_pose * (part_pose^-1 * tcp_pose).  Choosing a grasp therefore means choosing
  an insertion pose, and a grasp is only usable if BOTH its own and its implied insertion
  pose are IK-reachable -- which is what ends up standing a rod that lies flat upright.
"""
from __future__ import annotations

import numpy as np

from assembly_policy.sim.robot import quat_conj, quat_mul, quat_to_mat

UP = np.array([0.0, 0.0, 1.0])


def tcp_in_part(part_pos, part_quat, tcp_pos, tcp_quat):
    R = quat_to_mat(part_quat)
    return R.T @ (tcp_pos - part_pos), quat_mul(quat_conj(part_quat), tcp_quat)


def tcp_for_part_at(goal_pos, goal_quat, rel_pos, rel_quat):
    """TCP pose that puts the part at (goal_pos, goal_quat), given the fixed grasp offset."""
    return goal_pos + quat_to_mat(goal_quat) @ rel_pos, quat_mul(goal_quat, rel_quat)


def holding(obs):
    return (not obs["gripper_open_cmd"]) and obs["gripper_width"] < 0.075


def insert(api, obs, approach_h=0.07, press=0.004):
    """Carry the grasped part to its assembled pose and press it home."""
    rel_p, rel_q = tcp_in_part(obs["part_pos"], obs["part_quat"], obs["tcp_pos"], obs["tcp_quat"])
    tgt_p, tgt_q = tcp_for_part_at(obs["goal_pos"], obs["goal_quat"], rel_p, rel_q)
    api.move_to(tgt_p + UP * approach_h, tgt_q, timeout_s=3.5)
    api.move_to(tgt_p + UP * 0.012, tgt_q, timeout_s=2.0, tol=0.006)
    api.move_to(tgt_p - UP * press, tgt_q, timeout_s=3.0, tol=0.004)
    api.wait(0.4)


def pick_grasp(api, obs, n=64):
    """Score candidates by whether the grasp AND the insertion pose it implies are reachable."""
    best = None
    for c in api.grasp_candidates(n):
        rel_p, rel_q = tcp_in_part(obs["part_pos"], obs["part_quat"], c["pos"], c["quat"])
        ins_p, ins_q = tcp_for_part_at(obs["goal_pos"], obs["goal_quat"], rel_p, rel_q)
        if ins_p[2] < obs["goal_pos"][2] + 0.005:        # grasp would be inside the board
            continue
        q_ins, ok_ins = api.ik(ins_p, ins_q)
        if not ok_ins:
            continue
        q_g, ok_g = api.ik(c["pos"], c["quat"])
        if not ok_g:
            continue
        # prefer an approach that is not far from straight down and an insertion that keeps
        # the hand clear of the board
        down = -quat_to_mat(c["quat"])[:, 2] @ UP
        score = down + 2.0 * min(ins_p[2] - obs["goal_pos"][2], 0.15)
        if best is None or score > best[0]:
            best = (score, c, q_g)
    return best


def grasp(api, obs, lift=0.10):
    picked = pick_grasp(api, obs)
    if picked is None:
        return False
    _, c, _ = picked
    approach = quat_to_mat(c["quat"])[:, 2]              # palm -> fingertips
    api.set_gripper(True, hold_s=0.2)
    if not api.move_to(c["pos"] - approach * 0.06, c["quat"], timeout_s=3.0):
        return False
    api.move_to(c["pos"], c["quat"], timeout_s=1.5, tol=0.006)
    api.set_gripper(False, hold_s=0.6)
    o = api.observe()
    api.move_to(o["tcp_pos"] + UP * lift, o["tcp_quat"], timeout_s=1.5)
    return True


def policy(api):
    obs = api.observe()
    if not holding(obs):
        if not grasp(api, obs):
            return
        obs = api.observe()
        if not holding(obs):
            return
    insert(api, obs)
