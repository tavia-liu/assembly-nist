#!/bin/bash
#SBATCH --job-name=pixel-ppo
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=r28-1801
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
# Symmetric pixel PPO (learned conv encoder shared by actor and critic, two cameras + proprioception, no oracle
# information). Resumes automatically from $RUN_DIR/checkpoint.pt; resubmit (or chain with --dependency) to continue.
#   sbatch --export=ALL,TASK=P4,ITERS=40000 cap/train_pixel.sbatch
set -uo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export PYTHONPATH=.:cap MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
TASK=${TASK:-P4}
LONG=$(.venv/bin/python -c "from assembly_policy.sim.tasks import task_name; print(task_name('$TASK'))")
RUN_DIR=${RUN_DIR:-runs/${LONG}_pixel_near}
NAME=$(basename "$RUN_DIR")
export CAP_PIX_RES=${CAP_PIX_RES:-96}
export WANDB_ENTITY=WASHU-MARL WANDB_PROJECT=${WANDB_PROJECT:-assemble} WANDB_RUN_NAME=${WANDB_RUN_NAME:-${NAME}_ppo} WANDB_RUN_ID=${WANDB_RUN_ID:-${NAME}_ppo}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-$LONG} WANDB_TAGS=${WANDB_TAGS:-pixel,image,symmetric,near,ppo}
.venv/bin/python -u cap/train_pixel.py --task "$TASK" --run-dir "$RUN_DIR" --envs "${ENVS:-128}" \
    --steps 32 --iters "${ITERS:-40000}" --save-every 25 --keep-every 100 --max-hours 23.5 --nthread 16 ${TRAIN_ARGS:-}
