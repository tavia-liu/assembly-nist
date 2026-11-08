#!/bin/bash
#SBATCH --job-name=eval-sweep
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
# Held-out eval of every kept checkpoint of a run, logged to its wandb run as eval/*.
#   sbatch --export=ALL,RUN_DIR=runs/feat_P4,KIND=feat,WID=feat_P4_near_ppo_full cap/eval_sweep.sbatch
set -uo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export PYTHONPATH=.:cap MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
.venv/bin/python -u cap/eval_sweep.py --run-dir "${RUN_DIR:?}" --kind "${KIND:?}" --task "${TASK:-P4}" --wandb-id "${WID:?}" --every "${EVERY:-1}"
