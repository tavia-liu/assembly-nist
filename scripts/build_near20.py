"""Reset bank for training on the full board: the active part already in the hand near its fixture (the single-task
`near` family) while the other 19 parts lie loose on the board (the `board` family of their own banks).

For every single-task near state of <task> (train and eval split kept), a board layout for the other parts is drawn
such that no part overlaps a fixture or another part; the composed state is settled for 0.5 s with the robot holding
position; it is kept if the active part is still in the fingers, nothing else touches the robot or the held part,
and no part flew off the board. Output: <out>/<task>_{train,eval}.npz with the multi-part state vectors (family =
near, level copied), readable by assembly_policy.train through cap/train_board.py.

  PYTHONPATH=.:cap python scripts/build_near20.py --task P4 --out data/banks_near20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cap"))
import api                                                     # noqa: E402
from assembly_policy.sim.env import EnvConfig                  # noqa: E402
from assembly_policy.sim.resets import FAMILIES                # noqa: E402
from assembly_policy.sim.tasks import BANK_DIR, resolve_tasks  # noqa: E402
from eval_protocol import CAP_BANK_DIR                         # noqa: E402
from multi_env import MultiPartEnv                             # noqa: E402
from multi_part import choose_layout                           # noqa: E402


ROBOT_BODIES = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger")


def robot_body_ids(m):
    return {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ROBOT_BODIES} - {-1}


def contacts_bad(m, d, robot, active_body, other_parts):
    """True if the arm, the hand or the held part touches one of the other parts. Loose parts resting against each
    other, the board or a fixture are fine."""
    mine = robot | {active_body}
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in mine and b2 in other_parts) or (b2 in mine and b1 in other_parts):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--scene", default="all20")
    ap.add_argument("--near-bank", default=str(BANK_DIR), help="single-task banks with the near family")
    ap.add_argument("--board-bank", default=str(CAP_BANK_DIR), help="board-family banks for the other parts")
    ap.add_argument("--out", default="data/banks_near20")
    ap.add_argument("--n-train", type=int, default=640)
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--nthread", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tasks = resolve_tasks(a.scene)
    T = a.task
    others = [t for t in tasks if t != T]
    env = MultiPartEnv(tasks, EnvConfig(nthread=a.nthread, horizon_s=15.0), n_envs=a.batch, active=T, physics_of=T)
    m = env.models[0].model
    d = mujoco.MjData(m)
    e1 = api.make_env(T); sp1 = e1.sp.copy(); e1.close()
    part_bodies = {t: env.idx.part_body_of[t] for t in tasks}
    robot = robot_body_ids(m)
    other_parts = {part_bodies[t] for t in others}
    rejected = dict(held=0, moved=0, fell=0, contact=0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for split, n_want in (("train", a.n_train), ("eval", a.n_eval)):
        z = np.load(Path(a.near_bank) / f"{T}_{split}.npz")
        near = np.nonzero(z["family"] == FAMILIES.index("near"))[0]
        rng = np.random.default_rng(a.seed + (0 if split == "train" else 1))
        states, levels, src, tries, t0 = [], [], [], 0, time.time()
        k = 0
        while len(states) < n_want and tries < 20 * n_want:
            batch_idx = [near[(k + j) % len(near)] for j in range(a.batch)]
            k += a.batch
            cand = []
            for j in batch_idx:
                layout = choose_layout(others, a.board_bank, sp1, rng)
                cand.append(env.compose(layout, sp1, robot=z["states"][j]))
            cand = np.stack(cand)
            env.set_states(np.arange(a.batch), cand)
            hold = np.zeros((a.batch, 8)); hold[:, 7] = -1.0            # keep the fingers closed on the part
            for _ in range(int(0.5 / env.ecfg.control_dt)):
                env.step(hold)
            for i, j in enumerate(batch_idx):
                tries += 1
                s = env.state[i]
                mujoco.mj_setState(m, d, s, mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(m, d)
                held = -0.004 < s[env.sf].sum() < 0.075      # fingers at the part (P4: ~0.0006), not slammed shut (-0.02)
                d_act = np.linalg.norm(s[env.sp[:3]] - z["states"][j][sp1[:3]])
                z_ok = all(s[env.part_slices(t)[0][2]] > -0.02 for t in tasks)
                bad = contacts_bad(m, d, robot, part_bodies[T], other_parts)
                if not held:
                    rejected["held"] += 1
                elif d_act >= 0.01:
                    rejected["moved"] += 1
                elif not z_ok:
                    rejected["fell"] += 1
                elif bad:
                    rejected["contact"] += 1
                else:
                    states.append(s.copy()); levels.append(float(z["level"][j])); src.append(int(j))
                if len(states) >= n_want:
                    break
            print(f"{T} {split}: {len(states)}/{n_want} kept of {tries} tried  rejected {rejected}  [{time.time() - t0:.0f} s]", flush=True)
        S = np.stack(states[:n_want])
        np.savez_compressed(out / f"{T}_{split}.npz", states=S, family=np.full(len(S), FAMILIES.index("near")),
                            level=np.asarray(levels[:n_want]), source=np.asarray(src[:n_want]))
        (out / f"{T}_{split}.json").write_text(json.dumps(dict(task=T, scene=tasks, split=split, n=len(S), tried=tries,
                                                               nstate=int(S.shape[1]), wall_s=round(time.time() - t0, 1)), indent=1))
        print(f"wrote {out / f'{T}_{split}.npz'}: {len(S)} states, nstate {S.shape[1]}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
