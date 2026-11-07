#!/bin/bash
#SBATCH --job-name=perc-learner
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=r28-1801
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
# PPO learner; the environments live on NCONN perceived workers (cap/perceived_worker.sbatch). Resubmit to resume.
#   sbatch --export=ALL,RUN_DIR=runs/perc_P4,NCONN=8,ENVS=192,TASK=P4 cap/perceived_learner.sbatch
set -uo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export PYTHONPATH=.:cap
.venv/bin/python -u -m assembly_policy.train --run-dir "${RUN_DIR:?}" --remote-workers "${NCONN:?}" --envs "${ENVS:-192}" \
    --tasks "${TASK:-P4}" --bank-dir data/banks_near --sampler uniform --no-eval --steps "${STEPS:-32}" \
    --iters "${ITERS:-3000}" --save-every 25 --keep-every 100 --accept-timeout 900 --min-conn-frac 0.5 \
    --max-hours 11.5 ${TRAIN_ARGS:-}
