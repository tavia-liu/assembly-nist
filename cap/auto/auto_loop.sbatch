#!/bin/bash
#SBATCH --job-name=cap-autoloop
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
# sbatch --export=ALL,TASK=R8 cap/auto/auto_loop.sbatch        (RESUME=1 continues an existing run)
set -euo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export PYTHONPATH=.:cap:cap/auto MUJOCO_GL=egl CAP_CLAUDE_BIN="$HOME/.local/bin/claude"
.venv/bin/python -u cap/auto/loop.py --task "${TASK:-R8}" --schedule "${SCHEDULE:-2,5,10,15,20}" \
    --rounds "${ROUNDS:-25}" --horizon "${HORIZON:-40}" --level "${LEVEL:-grasp}" \
    --model "${MODEL:-claude-opus-5-5[1m]}" --audit-model "${AUDIT_MODEL:-claude-opus-5-5[1m]}" \
    --max-experiments "${MAX_EXPERIMENTS:-3}" ${OUT:+--out "$OUT"} ${RESUME:+--resume}
