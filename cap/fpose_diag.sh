#!/bin/bash
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
source /engrfs/project/chongjie/buzhao/miniconda3/etc/profile.d/conda.sh
conda activate fpose
export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export FPOSE_PORT=$((8200 + SLURM_JOB_ID % 500))
python -u cap/fpose_server.py --port $FPOSE_PORT > cap/runs/slurm/fpose_server_$SLURM_JOB_ID.log 2>&1 &
SRV=$!
for i in $(seq 1 150); do grep -q listening cap/runs/slurm/fpose_server_$SLURM_JOB_ID.log 2>/dev/null && break; sleep 2; done
conda deactivate
export PYTHONPATH=.:cap MUJOCO_GL=egl CAP_TRACKER=fpose CAP_CAM_RES=1280x960 CAP_WRIST_RES=320x240
.venv/bin/python -u cap/fpose_diag.py P4 6 2>&1 | grep -v "Warning\|EGLError\|__del__\|glCheckError\|^  File\|^    "
kill $SRV
