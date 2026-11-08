# NIST Task Board benchmark (camera-only)

A MuJoCo benchmark for the 20 assemblies of the **NIST Assembly Task Board 1** with a Franka arm, in which a policy
sees the part **only through cameras**. The simulator's part pose is never exposed; everything a policy observes
exists on a real cell with two fixed RGB-D cameras and a wrist camera.

Built on [AssembleNIST](docs/ASSEMBLENIST_README.md) (the simulator, the 20 tasks and its PPO baseline, kept
unchanged under `assembly_policy/`). The benchmark layer is `cap/`.

## Task

One episode = one part. The part lies loose on the board at a random settled pose, the robot is at its home
configuration with the gripper open, and the policy has 40 s to assemble it. Success: the part's two keypoints
are within 1 mm of their goal, for whichever of the part's mesh symmetries fits best (a rod seated either way up
counts; gears and connectors have one orientation).

| group | tasks |
|---|---|
| round rods | R4 R8 R12 R16 |
| rectangular bars | P4 P8 P12 P16 |
| gears | Gs Gm Gl |
| nuts | N4 N8 N12 N16 |
| connectors | BNC DSUB RJ45 USB WP |

The task specification a policy gets: the part's ID, its CAD mesh (`api.part_model()`), the board/fixture (in the
scene) and the goal pose (`goal_pos`, `goal_quat` in `api.observe()`).

## What a policy observes (API level `final`)

```
PYTHONPATH=.:cap python -c "import api; print(api.spec('final'))"
```

| call | gives | real-cell counterpart |
|---|---|---|
| `observe()` | joints, joint velocities, TCP pose, gripper width, external joint torques / TCP wrench, goal pose, time left | robot state interface |
| `camera()` | three views (`global`, `global2` 1280x960; `wrist` 640x480): RGB, metric depth, the part's instance mask, intrinsics K, camera pose | RGB-D cameras + a segmentation model |
| `locate_part()` | 6-DoF pose estimate of the part from FoundationPose (RGB-D + mask + CAD), tracked frame to frame | the same estimator on real images |
| `part_model()` | the part's CAD mesh | the CAD |
| `ik / move_to / move_to_joints / set_gripper / wait` | task-agnostic kinematics | the arm controller |
| `step(a)` | raw 25 Hz joint increments | the arm controller |

Idealisations to keep in mind: the instance mask comes from the renderer (a real cell runs a segmentation
model), depth has no sensor noise, and the pose estimator's compute time does not consume episode time.

Withheld at every level: the simulator's part pose and assembly distance; any insert/align/screw primitive.

## Running an evaluation

A policy is a Python file defining `policy(api)` (see `examples/noop_policy.py`). The held-out protocol runs 100
`board` starts from seed 1000, saves `eval.json` and one video per episode:

```
sbatch --export=ALL,TASK=R8,POLICY=path/to/policy.py cap/eval.sbatch     # starts the FoundationPose service, then:
MUJOCO_GL=egl PYTHONPATH=.:cap python cap/eval_protocol.py --task R8 --policy path/to/policy.py
```

Develop on other seeds (`--seed 0`); seed 1000 is for reporting. Stage verifiers inside a policy do not stop an
episode during evaluation (`api.strict_stages` is False there).

Whole-board protocol (several parts physically on the board, one continuous episode, 40 s per part, scored at
the end):

```
MUJOCO_GL=egl PYTHONPATH=.:cap python cap/multi_board.py --tasks board20 --policy path/to/policy.py --n-seeds 10
```

## Setup

```
python -m venv .venv && .venv/bin/pip install -e . && .venv/bin/pip install imageio[ffmpeg]
```

`locate_part()` needs the FoundationPose service, `cap/fpose_server.py`, running in a conda env with
[FoundationPose](https://github.com/NVlabs/FoundationPose) installed (`FOUNDATIONPOSE_DIR`), on a GPU; it listens
on `$FPOSE_PORT` (default 8130). `cap/eval.sbatch` starts it for the job. Rendering uses EGL (`MUJOCO_GL=egl`).

## Layout

```
assembly_policy/   simulator, 20 tasks, reset generator, symmetric success metric, PPO baseline (AssembleNIST)
assets/            board, fixtures and part meshes (NIST ATB1 CAD)
cap/api.py         the policy-facing API and its levels (raw, kin, final); api.spec(level) renders the docs
cap/perception.py  cameras, masks, depth; FoundationPose client
cap/fpose_server.py FoundationPose as a local service
cap/eval_protocol.py  single-part held-out protocol (seeded starts, videos)
cap/multi_board.py, multi_env.py, multi_scene.py   whole-board protocol
examples/          a minimal policy
```
