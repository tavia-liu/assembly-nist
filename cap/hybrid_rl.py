"""Code + RL hybrid for one part (any task): code does the geometric part, the RL checkpoint the contact part.

policy:          code opens the hand and parks it at a pre-grasp standoff above an antipodal grasp of the part
                 (approach within 60 deg of straight down), then hands over to RL; if RL has not finished within
                 its budget, code recovers (open, lift clear, fresh pre-grasp) and hands over again.
policy_rl_first: RL from the start state; code only recovers between attempts.
"""
from __future__ import annotations

import numpy as np

from assembly_policy.sim.robot import quat_to_mat
from rl_skill import RLSkill

UP = np.array([0.0, 0.0, 1.0])
STANDOFF = 0.03      # pre-grasp: this far back along the approach axis from the grasp pose
SAFE_Z = 0.16      # travel height: above every assembled part on the board
PRESS_MM = 4.0     # finish(): push the seated part this much further along the approach before letting go
RL_BUDGET_S = 12.0
ATTEMPTS = 4


def _ok_q(api, q):
    lim = np.asarray(api.constants()["joint_limits"])
    return bool(np.all(q > lim[:, 0] + 0.05) and np.all(q < lim[:, 1] - 0.05))


def pregrasp(api, n=64):
    """Reachable pre-grasp (pos, quat, joints) above the part, approach as close to straight down as possible."""
    o = api.observe()
    pp, pq, conf = api.locate_part()
    if pp is None:
        print("[hybrid] part not seen by either camera", flush=True)
        return None
    best = None
    for c in api.grasp_candidates(pp, pq, n):
        a = quat_to_mat(c["quat"])[:, 2]          # approach direction (palm -> fingertips)
        down = -a[2]
        if down < 0.5:                             # more than 60 deg from vertical
            continue
        pre = c["pos"] - a * STANDOFF
        if pre[2] < 0.015:
            continue
        q, ok = api.ik(pre, c["quat"])
        if not ok or not _ok_q(api, q):
            continue
        cost = (1.0 - down) + 0.15 * np.abs(q - o["joints"]).max()
        if best is None or cost < best[0]:
            best = (cost, pre, c["quat"], q)
    return None if best is None else best[1:]


def recover(api):
    """Open the hand and lift clear so the next pre-grasp is approached from above."""
    api.set_gripper(True, hold_s=0.3)
    o = api.observe()
    if o["tcp_pos"][2] < SAFE_Z - 0.02:
        api.move_to(np.r_[o["tcp_pos"][:2], SAFE_Z], o["tcp_quat"], timeout_s=1.0, tol=0.03, gain=3.0)


def finish(api):
    """After the RL reports the part seated (still in the fingers): press it home along the approach axis,
    open, and lift straight back out. Counters the spring-back of friction fits (USB) on release."""
    o = api.observe()
    a = quat_to_mat(o["tcp_quat"])[:, 2]
    api.move_to(o["tcp_pos"] + a * PRESS_MM / 1000, o["tcp_quat"], timeout_s=0.5, tol=0.002, gain=2.0)
    api.wait(0.2)
    api.set_gripper(True, hold_s=0.4)
    o = api.observe()
    api.move_to(o["tcp_pos"] - a * 0.03, o["tcp_quat"], timeout_s=0.6, tol=0.01, gain=2.0)
    o = api.observe()
    if o["tcp_pos"][2] < SAFE_Z - 0.02:
        api.move_to(np.r_[o["tcp_pos"][:2], SAFE_Z], o["tcp_quat"], timeout_s=1.0, tol=0.03, gain=3.0)


def go_pregrasp(api):
    b = pregrasp(api)
    if b is None:
        print("[hybrid] no reachable pre-grasp", flush=True)
        return False
    pre, quat, q = b
    q_hi, ok = api.ik(pre + UP * 0.08, quat, q_init=q)
    if ok and _ok_q(api, q_hi):
        api.move_to_joints(q_hi, timeout_s=2.0, tol=0.03, gain=3.0)
    ok = api.move_to_joints(q, timeout_s=1.5, tol=0.02, gain=3.0)
    o = api.observe()
    print(f"[hybrid] pre-grasp t={o['t']} reached={ok} tcp={np.round(o['tcp_pos'], 3).tolist()}", flush=True)
    return True


def _loop(api, code_first):
    rl = RLSkill(api)
    for k in range(ATTEMPTS):
        if k > 0:
            recover(api)
        if code_first or k > 0:
            go_pregrasp(api)
        o = api.observe()
        print(f"[hybrid] RL attempt {k} t={o['t']} budget={RL_BUDGET_S}s", flush=True)
        rl.run(max_s=RL_BUDGET_S)


def policy(api):
    _loop(api, code_first=True)


def policy_rl_first(api):
    _loop(api, code_first=False)
