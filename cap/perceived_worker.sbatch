#!/bin/bash
#SBATCH --job-name=perc-worker
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=r28-1801
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
# One GPU: a FoundationPose server + a simulation worker whose envs observe the part through the cameras.
#   sbatch --export=ALL,RUN_DIR=runs/perc_P4 cap/perceived_worker.sbatch
set -uo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
source /engrfs/project/chongjie/buzhao/miniconda3/etc/profile.d/conda.sh
conda activate fpose
export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export FPOSE_PORT=$((8200 + SLURM_JOB_ID % 500))
python -u cap/fpose_server.py --port $FPOSE_PORT > cap/runs/slurm/fpose_server_$SLURM_JOB_ID.log 2>&1 &
SRV=$!
for i in $(seq 1 150); do grep -q listening cap/runs/slurm/fpose_server_$SLURM_JOB_ID.log 2>/dev/null && break; sleep 2; done
conda deactivate
export PYTHONPATH=.:cap MUJOCO_GL=egl CAP_TRACKER=fpose CAP_CAM_RES=640x480 CAP_TRACK_EVERY=${TRACK_EVERY:-2}
.venv/bin/python -u cap/perceived_worker.py --run-dir "${RUN_DIR:?}" --nthread "$SLURM_CPUS_PER_TASK"
kill $SRV
