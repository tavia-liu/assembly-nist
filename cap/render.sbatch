#!/bin/bash
#SBATCH --job-name=cap-render
#SBATCH --account=engr-class-any
#SBATCH --partition=gpu-linuxlab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cap/runs/render-%j.log
set -euo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export MUJOCO_GL=egl PYTHONPATH=.:cap
.venv/bin/python cap/record.py --task "${TASK:-R16}" --family "${FAMILY:-near}" \
    --n "${N:-6}" --out "${OUT:-cap/runs/R16_near.mp4}"
