# NIST Task Board benchmark

MuJoCo simulation of the NIST Assembly Task Board 1 (20 parts, Franka arm), built on
[AssembleNIST](https://github.com/suniique/assembly-nist). A policy is a Python file defining `policy(api)`;
`api` (benchmark/api.py) is its only interface to the robot and the cameras: `python -c "import api; print(api.spec())"`.

```
PYTHONPATH=.:benchmark MUJOCO_GL=egl
python benchmark/eval_protocol.py --task R8 --policy examples/noop_policy.py      # one part, 100 held-out starts, videos
python benchmark/multi_board.py --tasks board20 --policy examples/noop_policy.py  # all 20 parts on the board
sbatch --export=ALL,CMD="..." gpu.sbatch                                          # any command on a GPU
```

`locate_part()` needs the FoundationPose service (benchmark/fpose_server.py); gpu.sbatch starts it.
