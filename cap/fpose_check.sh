#!/bin/bash
# FoundationPose server + the perception check (tracker = fpose) on one GPU node
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
source /engrfs/project/chongjie/buzhao/miniconda3/etc/profile.d/conda.sh
conda activate fpose
export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
python -u cap/fpose_server.py --port 8130 > cap/runs/slurm/fpose_server_$SLURM_JOB_ID.log 2>&1 &
SRV=$!
for i in $(seq 1 120); do grep -q listening cap/runs/slurm/fpose_server_$SLURM_JOB_ID.log 2>/dev/null && break; sleep 2; done
conda deactivate
export PYTHONPATH=.:cap MUJOCO_GL=egl CAP_TRACKER=fpose CAP_CAM_RES=640x480
for T in R8 P4; do .venv/bin/python -u cap/perception_check.py $T 3; done
kill $SRV
tail -n 5 cap/runs/slurm/fpose_server_$SLURM_JOB_ID.log
