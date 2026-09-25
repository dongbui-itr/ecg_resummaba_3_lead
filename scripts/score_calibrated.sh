#!/usr/bin/env bash
# Full EC57 + beat-eval + splits for one checkpoint at a calibrated decoder operating point
# (chosen on portal-eval, scripts/calibrate_decoder.sh), then the no-regression diff.
#   scripts/score_calibrated.sh <model> <checkpoint> <s_boost> <min_peak_prob> <gpu> [tag]
set -u
cd "$(dirname "$0")/.."
MODEL="${1:?}"; CKPT="${2:?}"; SB="${3:?}"; PP="${4:?}"; GPU="${5:-0}"; TAG="${6:-${MODEL}_calib}"
export ECGR_RUN_TAG="${ECGR_RUN_TAG:-260923_60s}" CUDA_VISIBLE_DEVICES="$GPU" TF_CPP_MIN_LOG_LEVEL=2
PY=/home/ai-server/miniconda3/envs/beat/bin/python
echo "[$(date +%H:%M:%S)] ec57 $TAG (s_boost $SB, min_peak_prob $PP)"
ECGR_DECODE_MIN_PEAK_PROB="$PP" "$PY" -m ecgr ec57 --model "$MODEL" --checkpoint "$CKPT" --tag "$TAG" --s-boost "$SB" \
  > "logs/${TAG}_ec57.log" 2>&1 || echo "ec57 FAILED rc=$?"
grep -aE "^  [a-z0-9-]+  Q_Se=" "logs/${TAG}_ec57.log"
echo "[$(date +%H:%M:%S)] regress $TAG"
"$PY" -m ecgr regress --model "$MODEL" --tag "$TAG" > "logs/${TAG}_regress.log" 2>&1; echo "regress rc=$?"
sed 's/\x1b\[[0-9;]*m//g' "logs/${TAG}_regress.log" | tail -20
echo "[$(date +%H:%M:%S)] done $TAG"
