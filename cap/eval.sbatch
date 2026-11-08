#!/bin/bash
#SBATCH --job-name=nist-eval
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=eval-%j.log
# Submit from the repository root:
#   sbatch --export=ALL,TASK=R8,POLICY=examples/noop_policy.py cap/eval.sbatch        (N, SEED, HORIZON, OUT optional)
# Starts the FoundationPose service on the job's GPU (conda env $FPOSE_CONDA_ENV, FoundationPose checkout
# $FOUNDATIONPOSE_DIR), then runs cap/eval_protocol.py.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=.:cap MUJOCO_GL=egl
export FPOSE_PORT=$((8200 + SLURM_JOB_ID % 500))
FPLOG=fpose-$SLURM_JOB_ID.log
( source "${CONDA_SH:-/engrfs/project/chongjie/buzhao/miniconda3/etc/profile.d/conda.sh}" && conda activate "${FPOSE_CONDA_ENV:-fpose}" &&
  export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-} &&
  exec python -u cap/fpose_server.py --port $FPOSE_PORT ) > $FPLOG 2>&1 &
FPSRV=$!
trap 'kill $FPSRV 2>/dev/null' EXIT
for i in $(seq 1 150); do grep -q listening $FPLOG 2>/dev/null && break; kill -0 $FPSRV 2>/dev/null || { echo "fpose server died"; tail $FPLOG; exit 1; }; sleep 2; done
grep -q listening $FPLOG || { echo "fpose server not up"; tail $FPLOG; exit 1; }
"${PYTHON:-.venv/bin/python}" -u cap/eval_protocol.py --task "${TASK:-R8}" --policy "$POLICY" --level "${LEVEL:-final}" \
    --n "${N:-100}" --seed "${SEED:-1000}" --horizon "${HORIZON:-40}" ${OUT:+--out "$OUT"}
