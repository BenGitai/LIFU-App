#!/bin/bash
# Run the LIFU pipeline on a Brains GPU box (no scheduler -- just ssh in and run this).
#   chmod +x run_brains.sh   &&   ./run_brains.sh
# Long GPU run: launch it under tmux (or nohup) so it survives an SSH drop, e.g.
#   tmux new -s lifu './run_brains.sh 2>&1 | tee lifu.log'   (detach: Ctrl-b d)
set -e
cd "$(dirname "$0")"

# --- environment (edit to match the box) -------------------------------------
# conda activate <ENV>          # env with: numpy scipy nibabel trimesh matplotlib kwave cupy
# export CUDA_VISIBLE_DEVICES=0 # pin to one GPU if the box is shared

# --- sanity checks -----------------------------------------------------------
python -c "import numpy,scipy,nibabel,trimesh,matplotlib; print('geometry deps OK')"
python -c "import kwave, cupy; print('kwave+cupy OK')" || echo "WARN: no kwave/cupy -> geometry only"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- run ---------------------------------------------------------------------
python lifu_pipeline.py --data . --out ./pipeline_out --dx-sim 0.3 --run-sim
