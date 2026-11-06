#!/bin/bash
#SBATCH --job-name=cap-agentloop
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
set -euo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export PYTHONPATH=.:cap MUJOCO_GL=egl CAP_CLAUDE_BIN="$HOME/.local/bin/claude"
.venv/bin/python -u cap/agent_loop.py --task "${TASK:-R16}" --per-family "${PER_FAMILY:-5}" --grow "${GROW:-5}" \
    --rounds "${ROUNDS:-10}" --level "${LEVEL:-grasp}" --model "${MODEL:-claude-opus-5-5[1m]}" ${OUT:+--out "$OUT"}
