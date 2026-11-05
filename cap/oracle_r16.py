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


def seating_pose(api):
    """TCP pose that seats the part, recomputed from the CURRENT grasp.

    The part can shift in the fingers during transport, so the offset is re-read before every
    leg instead of being captured once -- an open-loop target goes stale the moment it slips.
    """
    o = api.observe()
    rel_p, rel_q = tcp_in_part(o["part_pos"], o["part_quat"], o["tcp_pos"], o["tcp_quat"])
    return (*tcp_for_part_at(o["goal_pos"], o["goal_quat"], rel_p, rel_q), o)


def insert(api, approach_h=0.055, press=0.005, tries=3):
    """Carry the grasped part to its assembled pose and press it home, retrying on a jam."""
    for _ in range(tries):
        tgt_p, tgt_q, _ = seating_pose(api)
        api.move_to(tgt_p + UP * approach_h, tgt_q, timeout_s=2.0, gain=3.0)
        tgt_p, tgt_q, _ = seating_pose(api)
        api.move_to(tgt_p + UP * 0.010, tgt_q, timeout_s=1.2, tol=0.005, gain=3.0)
        tgt_p, tgt_q, _ = seating_pose(api)
        api.move_to(tgt_p - UP * press, tgt_q, timeout_s=1.5, tol=0.003, gain=3.0)
        api.wait(0.2)
        # still here => not seated (success ends the episode): back off and line up again
        o = api.observe()
        api.move_to(o["tcp_pos"] + UP * 0.025, o["tcp_quat"], timeout_s=1.0, gain=3.0)


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


def grasp(api, obs, lift=0.08):
    picked = pick_grasp(api, obs)
    if picked is None:
        return False
    _, c, _ = picked
    approach = quat_to_mat(c["quat"])[:, 2]              # palm -> fingertips
    api.set_gripper(True, hold_s=0.2)
    if not api.move_to(c["pos"] - approach * 0.05, c["quat"], timeout_s=2.0, gain=3.0):
        return False
    api.move_to(c["pos"], c["quat"], timeout_s=1.0, tol=0.006, gain=3.0)
    api.set_gripper(False, hold_s=0.5)
    o = api.observe()
    api.move_to(o["tcp_pos"] + UP * lift, o["tcp_quat"], timeout_s=1.2, gain=3.0)
    return True


def policy(api):
    obs = api.observe()
    if not holding(obs):
        if not grasp(api, obs):
            return
        if not holding(api.observe()):               # grasp missed or the part slipped out
            return
    insert(api)
