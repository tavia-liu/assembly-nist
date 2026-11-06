#!/bin/bash
#SBATCH --job-name=cap-render-turns
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=cap/runs/render-turns-%j.log
# Render every vNN.py in a fix-loop run dir that has no vNN.mp4 yet (safe to resubmit).
set -euo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
DIR="${DIR:-cap/runs/fixloop_R16_near}"
export MUJOCO_GL=egl PYTHONPATH=.:cap:$DIR
for f in "$DIR"/v*.py; do
  v=$(basename "$f" .py)
  [ -f "$DIR/$v.mp4" ] && continue
  echo "== $v"
  .venv/bin/python cap/record.py --task "${TASK:-R16}" --family "${FAMILY:-near}" --n "${N:-6}" \
      --level "${LEVEL:-grasp}" --policy "$v:policy" --out "$DIR/$v.mp4" || echo "!! $v failed"
done
