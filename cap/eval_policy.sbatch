#!/bin/bash
#SBATCH --job-name=cap-eval
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
set -euo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export PYTHONPATH=.:cap
.venv/bin/python cap/eval_policy.py --task "${TASK:-R16}" --family "${FAMILY:-near}" \
    --n "${N:-6}" --policy "${POLICY:-oracle_r16:policy}" --level "${LEVEL:-grasp}" --trace
