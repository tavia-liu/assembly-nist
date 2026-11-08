#!/bin/bash
#SBATCH --job-name=feat-P4
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=r28-1801
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
# Frozen-encoder PPO (actor on ResNet-18 features, critic on point clouds, binary sparse reward,
# uniform near starts). Resumes automatically from $RUN_DIR/checkpoint.pt; resubmit to continue.
#   sbatch --export=ALL,TASK=P4,ITERS=12000 cap/train_feat.sbatch
set -uo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export PYTHONPATH=.:cap MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
TASK=${TASK:-P4}
LONG=$(.venv/bin/python -c "from assembly_policy.sim.tasks import task_name; print(task_name('$TASK'))")
# run ids stay feat_<TASK>_near_ppo so resubmits resume the same wandb run; display name uses the long task name
export WANDB_ENTITY=WASHU-MARL WANDB_PROJECT=${WANDB_PROJECT:-assemble} WANDB_RUN_NAME=${WANDB_RUN_NAME:-${LONG}_encoder_near_ppo} WANDB_RUN_ID=${WANDB_RUN_ID:-feat_${TASK}_near_ppo}
.venv/bin/python -u cap/train_feat.py --task "$TASK" --run-dir "${RUN_DIR:-runs/${LONG}_encoder_near}" --envs "${ENVS:-128}" \
    --steps 32 --iters "${ITERS:-12000}" --save-every 25 --keep-every 100 --max-hours 23.5 ${TRAIN_ARGS:-}
