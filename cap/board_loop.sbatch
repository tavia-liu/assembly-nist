#!/bin/bash
#SBATCH --job-name=cap-boardloop
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
set -euo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export PYTHONPATH=.:cap MUJOCO_GL=egl CAP_CLAUDE_BIN="$HOME/.local/bin/claude"
.venv/bin/python -u cap/board_loop.py --tasks "${TASKS:-board10}" --schedule "${SCHEDULE:-2,4}" \
    --rounds "${ROUNDS:-10}" --per-part-s "${PER_PART_S:-60}" --model "${MODEL:-claude-opus-5-5[1m]}" ${OUT:+--out "$OUT"}
