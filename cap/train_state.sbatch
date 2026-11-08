#!/bin/bash
#SBATCH --job-name=ppo-state
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=r28-1801
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
# State-based (oracle part pose) PPO from scratch on one task, uniform starts from one bank. Resubmit to resume.
#   sbatch --export=ALL,TASK=P4,BANK=data/banks_board,RUN_DIR=runs/state_P4_board cap/train_state.sbatch
set -uo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export PYTHONPATH=.:cap MUJOCO_GL=egl
TASK=${TASK:-P4}; BANK=${BANK:-data/banks_near}; RUN_DIR=${RUN_DIR:-runs/state_${TASK}}
NAME=$(basename "$RUN_DIR")
export WANDB_ENTITY=WASHU-MARL WANDB_PROJECT=${WANDB_PROJECT:-assemble} WANDB_RUN_NAME=${NAME}_ppo WANDB_RUN_ID=${NAME}_ppo
.venv/bin/python -u -m assembly_policy.train --tasks "$TASK" --bank-dir "$BANK" --sampler uniform --no-eval \
    --run-dir "$RUN_DIR" --envs "${ENVS:-2048}" --iters "${ITERS:-8000}" --steps 32 --nthread 16 \
    --save-every 50 --keep-every 250 --max-hours 23.5 ${TRAIN_ARGS:-}
