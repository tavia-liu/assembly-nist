# NIST Task Board benchmark (camera-only)

A MuJoCo benchmark for the 20 assemblies of the **NIST Assembly Task Board 1** with a Franka arm, in which a policy
sees the part **only through cameras**. The simulator's part pose is never exposed; everything a policy observes
exists on a real cell with two fixed RGB-D cameras and a wrist camera.

The simulator (`assembly_policy/sim`, `assets/`, `plugin/`) comes from
[AssembleNIST](https://github.com/suniique/assembly-nist) by Tao Sun; the benchmark layer is `benchmark/`.

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

A policy is given the part's ID, its CAD mesh (`api.part_model()`), the board with its fixtures (in the scene)
and the goal pose (`goal_pos`, `goal_quat` in `api.observe()`).

## What a policy observes

```
PYTHONPATH=.:benchmark python -c "import api; print(api.spec())"
```

| call | gives | real-cell counterpart |
|---|---|---|
| `observe()` | joints, joint velocities, TCP pose, gripper width, external joint torques / TCP wrench, goal pose, time left | robot state interface |
| `camera()` | three views (`global`, `global2` 1280x960; `wrist` 640x480): RGB, metric depth, the part's instance mask, intrinsics K, camera pose | RGB-D cameras + a segmentation model |
| `locate_part()` | 6-DoF pose estimate of the part from FoundationPose (RGB-D + mask + CAD), tracked frame to frame | the same estimator on real images |
| `part_model()` | the part's CAD mesh | the CAD |
| `ik / move_to / move_to_joints / set_gripper / wait` | task-agnostic kinematics | the arm controller |
| `step(a)` | raw 25 Hz joint increments | the arm controller |

Idealisations: the instance mask comes from the renderer (a real cell runs a segmentation model), depth has no
sensor noise, and the pose estimator's compute time does not consume episode time.

Never exposed: the simulator's part pose and assembly distance. There is no insert/align/screw primitive.

## Running an evaluation

A policy is a Python file defining `policy(api)` (see `examples/noop_policy.py`). It may use the public methods of
`api` and nothing else: the evaluation refuses a policy file that touches `api.env`, private attributes or the
simulator's modules. How the policy was obtained is unconstrained (hand-written code, learned components trained
in this simulator, anything else). The held-out protocol runs 100 `board` starts from seed 1000 and writes
`eval.json` plus one video per episode:

```
sbatch --export=ALL,TASK=R8,POLICY=path/to/policy.py benchmark/eval.sbatch     # starts the FoundationPose service, then:
MUJOCO_GL=egl PYTHONPATH=.:benchmark python benchmark/eval_protocol.py --task R8 --policy path/to/policy.py
```

Anything that renders (cameras, videos) needs a GPU: `gpu.sbatch` is the template for running any command on one
(`sbatch --export=ALL,CMD="..." gpu.sbatch`); it also starts the FoundationPose service for the job.

Develop on other seeds (`--seed 0`); seed 1000 is for reporting.

Whole board (several parts physically on the board, one continuous episode, 40 s per part, scored at the end):

```
MUJOCO_GL=egl PYTHONPATH=.:benchmark python benchmark/multi_board.py --tasks board20 --policy path/to/policy.py --n-seeds 10
```

## Setup

```
python -m venv .venv && .venv/bin/pip install -e .
```

`locate_part()` needs the FoundationPose service, `benchmark/fpose_server.py`, running on a GPU in a conda env with
[FoundationPose](https://github.com/NVlabs/FoundationPose) installed (`FOUNDATIONPOSE_DIR`); it listens on
`$FPOSE_PORT` (default 8130). `benchmark/eval.sbatch` starts it for the job. Rendering uses EGL (`MUJOCO_GL=egl`).

## Layout

```
assembly_policy/sim/   simulator: scene, robot, 20 tasks, reset generator, symmetric success metric
assets/                board, fixtures and part meshes (NIST ATB1 CAD), SDF grids, collision proxies
plugin/                MuJoCo SDF collision plugin
benchmark/api.py             the policy-facing API; api.spec() renders its docs
benchmark/perception.py      cameras, masks, depth; FoundationPose client
benchmark/fpose_server.py    FoundationPose as a local service
benchmark/eval_protocol.py   single-part held-out protocol (seeded starts, videos); benchmark/eval.sbatch runs it on a GPU
gpu.sbatch             GPU job template for any command (starts the FoundationPose service)
benchmark/multi_board.py, multi_env.py, multi_scene.py   whole-board protocol
examples/              a minimal policy
```
