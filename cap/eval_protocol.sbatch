#!/bin/bash
#SBATCH --job-name=cap-eval160
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
set -euo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export PYTHONPATH=.:cap
.venv/bin/python -u cap/eval_protocol.py --task "${TASK:-R16}" --run-dir "${RUN_DIR:-cap/runs/fixloop/R16_near}" \
    --level "${LEVEL:-grasp}" ${POLICY:+--policy "$POLICY"}
