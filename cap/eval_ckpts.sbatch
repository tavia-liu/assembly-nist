#!/bin/bash
#SBATCH --job-name=eval-ckpts
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
# Held-out near eval (128 starts, sampled + deterministic, videos) of a state ckpt and/or a feat ckpt.
#   sbatch --export=ALL,STATE=runs/state_P4/model_004000.pt,FEAT=runs/feat_P4/model_005200.pt cap/eval_ckpts.sbatch
set -uo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export PYTHONPATH=.:cap MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
TASK=${TASK:-P4}
if [ -n "${STATE:-}" ]; then
  .venv/bin/python -u cap/eval_state.py --ckpt "$STATE" --task "$TASK" --nthread 8
  .venv/bin/python -u cap/eval_state.py --ckpt "$STATE" --task "$TASK" --nthread 8 --deterministic
fi
if [ -n "${FEAT:-}" ]; then
  .venv/bin/python -u cap/eval_feat.py --ckpt "$FEAT" --task "$TASK" --nthread 8
  .venv/bin/python -u cap/eval_feat.py --ckpt "$FEAT" --task "$TASK" --nthread 8 --deterministic
  .venv/bin/python -u cap/render_feat.py --ckpt "$FEAT" --task "$TASK" --episodes 8
fi
