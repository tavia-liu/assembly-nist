#!/bin/bash
#SBATCH --job-name=cell-server
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=52:00:00
#SBATCH --output=runs/server/slurm-%j.log
# The robot cell as a service for a sandboxed agent. Submit from the repository root:
#   mkdir -p runs/server; SERVER_TOKEN=<secret> sbatch --export=ALL benchmark/server.sbatch
# Writes runs/server/<jobid>/server.addr (host:port) for the sandbox's ROBOT_ADDR. Starts FoundationPose first.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=.:benchmark MUJOCO_GL=egl
export FPOSE_PORT=$((8200 + SLURM_JOB_ID % 500)) PORT=$((8700 + SLURM_JOB_ID % 300))
OUT=runs/server/$SLURM_JOB_ID; mkdir -p $OUT
FPLOG=$OUT/fpose.log
( source "${CONDA_SH:-/engrfs/project/chongjie/buzhao/miniconda3/etc/profile.d/conda.sh}" && conda activate "${FPOSE_CONDA_ENV:-fpose}" &&
  export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-} &&
  exec python -u benchmark/fpose_server.py --port $FPOSE_PORT ) > $FPLOG 2>&1 &
FPSRV=$!
trap 'kill $FPSRV 2>/dev/null' EXIT
for i in $(seq 1 150); do grep -q listening $FPLOG 2>/dev/null && break; kill -0 $FPSRV 2>/dev/null || { echo "fpose server died"; tail $FPLOG; exit 1; }; sleep 2; done
grep -q listening $FPLOG || { echo "fpose server not up"; tail $FPLOG; exit 1; }
exec "${PYTHON:-.venv/bin/python}" -u benchmark/server.py --port $PORT --out $OUT
