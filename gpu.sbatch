#!/bin/bash
#SBATCH --job-name=nist-gpu
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
# GPU job template. Submit from the repository root with the command to run in $CMD:
#
#   sbatch --export=ALL,CMD="cap/eval_protocol.py --task R8 --policy examples/noop_policy.py --n 10 --seed 0" gpu.sbatch
#   sbatch --export=ALL,CMD="cap/multi_board.py --tasks board5 --policy my_policy.py --n-seeds 2" gpu.sbatch
#
# Rendering (cameras, videos) needs a GPU with EGL; never run these on the login node. Interactive alternative:
#   srun -p general-gpu -A engr-lab-chongjie --gres=gpu:a40:1 --cpus-per-task=4 --mem=24G --time=2:00:00 --pty bash
#
# The job starts the FoundationPose service (locate_part's estimator) on its GPU, waits until it listens, runs $CMD
# with the repository's Python, and stops the service at exit. Set FPOSE=0 to skip the service (policies that never
# call locate_part(); the API still constructs the client, so the service is normally needed).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=.:cap MUJOCO_GL=egl
export FPOSE_PORT=$((8200 + SLURM_JOB_ID % 500))
if [ "${FPOSE:-1}" != "0" ]; then
  FPLOG=fpose-$SLURM_JOB_ID.log
  ( source "${CONDA_SH:-/engrfs/project/chongjie/buzhao/miniconda3/etc/profile.d/conda.sh}" && conda activate "${FPOSE_CONDA_ENV:-fpose}" &&
    export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-} &&
    exec python -u cap/fpose_server.py --port $FPOSE_PORT ) > $FPLOG 2>&1 &
  FPSRV=$!
  trap 'kill $FPSRV 2>/dev/null' EXIT
  for i in $(seq 1 150); do grep -q listening $FPLOG 2>/dev/null && break; kill -0 $FPSRV 2>/dev/null || { echo "fpose server died"; tail $FPLOG; exit 1; }; sleep 2; done
  grep -q listening $FPLOG || { echo "fpose server not up"; tail $FPLOG; exit 1; }
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${PYTHON:-.venv/bin/python}" -u $CMD
