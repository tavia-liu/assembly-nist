#!/bin/bash
#SBATCH --job-name=cap-fixloop
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
set -euo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export PYTHONPATH=.:cap CAP_CLAUDE_BIN="$HOME/.local/bin/claude"
.venv/bin/python -u cap/fix_loop.py --task "${TASK:-R16}" --per-family "${PER_FAMILY:-5}" --grow "${GROW:-5}" \
    --turns "${TURNS:-10}" --level "${LEVEL:-grasp}" --start "${START:-cap/naive_r16.py}" \
    --model "${MODEL:-claude-opus-5-5[1m]}" ${OUT:+--out "$OUT"}
