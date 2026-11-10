# cap/clean -- runs that never saw ground truth

Criterion: the policy (and the search loop around it) only used what a real cell has: proprioception, RGB-D cameras
with the part instance mask, the task spec (part ID, CAD, goal pose) and the FoundationPose tracker built on them.
Nothing here read the simulator part pose (observe() part_pos, CAP_PERCEPTION=oracle, grasp_candidates from truth)
or an RL checkpoint trained on simulator point clouds.

Excluded on purpose (still under cap/runs): agent_loop/ (fixloop; observe() gave the true part pose, loose starts),
autoloop/rod_8mm_oracle, autoloop/rod_8mm_kin (kin level) and autoloop/rod_8mm_auto, rod_16mm_auto (11-07 grasp-level API, before the final level existed), board_loop/ and rl/ (RL skill = sim point-cloud checkpoint), agent_policy/ (true pose), archive/.

## autoloop/ (cap/auto/loop.py, AutoRobo-style staged search; board starts, 40 s horizon, held-out = first 100 eval-bank starts)

| run | task | API level / inputs | held-out |
|---|---|---|---|
| rod_8mm_final | R8 | final: camera(mask,depth) + part_model (CAD) + locate_part (FoundationPose) | 28/100 (eval_r25.json) |

The old path cap/runs/autoloop/rod_8mm_final is a forwarding symlink.
Camera-only RL (symmetric pixel / ResNet PPO, actor and critic on images) lives in runs/peg_4mm_{pixel,resnet}_near, not here.
