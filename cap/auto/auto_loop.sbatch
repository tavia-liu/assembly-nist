#!/bin/bash
#SBATCH --job-name=cap-autoloop
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=cap/runs/slurm/%x-%j.log
# sbatch --export=ALL,TASK=R8 cap/auto/auto_loop.sbatch        (RESUME=1 continues an existing run)
set -euo pipefail
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
export PYTHONPATH=.:cap:cap/auto MUJOCO_GL=egl CAP_CLAUDE_BIN="$HOME/.local/bin/claude"
export CLAUDE_CONFIG_DIR="${CAP_LOOP_CONFIG:-/engrfs/project/chongjie/buzhao/claude/config-loop}"   # own creds/ slots; see save_creds.sh
# FoundationPose (locate_part's estimator) as a local service on this node's GPU
export FPOSE_PORT=$((8200 + SLURM_JOB_ID % 500)) CAP_TRACKER="${CAP_TRACKER:-fpose}" CAP_WRIST_RES="${CAP_WRIST_RES:-640x480}"
FPLOG=cap/runs/slurm/fpose_server_$SLURM_JOB_ID.log
( source /engrfs/project/chongjie/buzhao/miniconda3/etc/profile.d/conda.sh && conda activate fpose &&
  export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-} &&
  exec python -u cap/fpose_server.py --port $FPOSE_PORT ) > $FPLOG 2>&1 &
FPSRV=$!
trap 'kill $FPSRV 2>/dev/null' EXIT
for i in $(seq 1 150); do grep -q listening $FPLOG 2>/dev/null && break; kill -0 $FPSRV 2>/dev/null || { echo "fpose server died"; tail $FPLOG; exit 1; }; sleep 2; done
grep -q listening $FPLOG || { echo "fpose server not up"; tail $FPLOG; exit 1; }
.venv/bin/python -u cap/auto/loop.py --task "${TASK:-R8}" --schedule "${SCHEDULE:-2,5,10,15,20}" \
    --rounds "${ROUNDS:-25}" --horizon "${HORIZON:-40}" --level "${LEVEL:-final}" \
    --model "${MODEL:-claude-opus-5-5[1m]}" --audit-model "${AUDIT_MODEL:-claude-opus-5-5[1m]}" \
    --max-experiments "${MAX_EXPERIMENTS:-3}" ${OUT:+--out "$OUT"} ${RESUME:+--resume}
