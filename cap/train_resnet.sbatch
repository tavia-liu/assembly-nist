#!/bin/bash
#SBATCH --job-name=resnet-ppo
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=r28-1801
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
# Symmetric pixel PPO with a fine-tuned ImageNet ResNet-18 trunk (cap/train_resnet.py): actor and critic share one
# image latent, no oracle pose / point clouds anywhere. Resumes from $RUN_DIR/checkpoint.pt; resubmit to continue.
#   sbatch --export=ALL,TASK=P4,ITERS=30000 cap/train_resnet.sbatch
set -uo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export PYTHONPATH=.:cap MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CAP_PIX_RES=${CAP_PIX_RES:-128}
TASK=${TASK:-P4}
LONG=$(.venv/bin/python -c "from assembly_policy.sim.tasks import task_name; print(task_name('$TASK'))")
RUN_DIR=${RUN_DIR:-runs/${LONG}_resnet_near}
NAME=$(basename "$RUN_DIR")
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-$LONG} WANDB_TAGS=${WANDB_TAGS:-resnet,image,symmetric,near,ppo}
export WANDB_ENTITY=WASHU-MARL WANDB_PROJECT=${WANDB_PROJECT:-assemble} WANDB_RUN_NAME=${WANDB_RUN_NAME:-${NAME}_ppo} WANDB_RUN_ID=${WANDB_RUN_ID:-${NAME}_ppo}
.venv/bin/python -u cap/train_resnet.py --task "$TASK" --run-dir "$RUN_DIR" --envs "${ENVS:-128}" --steps 32 \
    --iters "${ITERS:-30000}" --save-every 25 --keep-every 100 --max-hours 23.5 --nthread 16 ${TRAIN_ARGS:-}
