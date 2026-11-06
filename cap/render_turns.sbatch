#!/bin/bash
#SBATCH --job-name=cap-render-turns
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
# Render every turn of a fix-loop run that has no vNN.mp4 yet (safe to resubmit).
# Bank-based runs (log.jsonl has bank_idx) replay the exact debug starts; older runs use generator starts.
set -euo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
DIR="${DIR:-cap/runs/fixloop/R16_bank}"
export MUJOCO_GL=egl PYTHONPATH=.:cap:$DIR
for f in "$DIR"/v*.py; do
  v=$(basename "$f" .py); t=${v#v}
  [ -f "$DIR/$v.mp4" ] && continue
  echo "== $v"
  if grep -q '"bank_idx"' "$DIR/log.jsonl"; then
    .venv/bin/python cap/record.py --task "${TASK:-R16}" --level "${LEVEL:-grasp}" \
        --from-log "$DIR/log.jsonl" --turn "$((10#$t))" --out "$DIR/$v.mp4" || echo "!! $v failed"
  else
    .venv/bin/python cap/record.py --task "${TASK:-R16}" --family "${FAMILY:-near}" --n "${N:-6}" \
        --level "${LEVEL:-grasp}" --policy "$v:policy" --out "$DIR/$v.mp4" || echo "!! $v failed"
  fi
done
