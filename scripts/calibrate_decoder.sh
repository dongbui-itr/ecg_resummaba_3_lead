#!/usr/bin/env bash
# Decoder calibration ON PORTAL-EVAL ONLY (README 8i / config.DECODE_MIN_PEAK_PROB): sweep
# s_boost x min_peak_prob for one checkpoint, scoring the 5,000-record portal-eval sample
# each time. Never calibrate on mitdb or the beat-eval set - those are reported on.
#   scripts/calibrate_decoder.sh <model> <checkpoint.keras> <gpu>
set -u
cd "$(dirname "$0")/.."
MODEL="${1:?model}"; CKPT="${2:?checkpoint}"; GPU="${3:-0}"
export ECGR_RUN_TAG="${ECGR_RUN_TAG:-260923_60s}" CUDA_VISIBLE_DEVICES="$GPU" TF_CPP_MIN_LOG_LEVEL=2
PY=/home/ai-server/miniconda3/envs/beat/bin/python
for sb in 1.0 0.8 0.65 0.5; do
  for pp in 0.0 0.6 0.8; do
    tag="calib_${MODEL}_sb${sb}_pp${pp}"
    if [[ -f "$($PY -c "from ecgr import config; import os; print(os.path.join(config.EC57_DIR, '$tag', 'portal-eval', 'portal-eval_QRS_report_line.out'))")" ]]; then
      echo "[$(date +%H:%M:%S)] $tag already scored"; continue
    fi
    echo "[$(date +%H:%M:%S)] $tag"
    ECGR_DECODE_MIN_PEAK_PROB="$pp" "$PY" -m ecgr ec57 --model "$MODEL" --checkpoint "$CKPT" --tag "$tag" \
      --skip-physionet --skip-portal --splits eval --split-records 5000 --s-boost "$sb" \
      > "logs/${tag}.log" 2>&1 || echo "  FAILED rc=$?"
    grep -E "^  portal-eval" "logs/${tag}.log"
  done
done
echo "[$(date +%H:%M:%S)] calibration sweep done"
