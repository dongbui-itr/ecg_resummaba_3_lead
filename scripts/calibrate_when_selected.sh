#!/usr/bin/env bash
# Wait for `select` to write epochs/selected_by_bxb.keras for one model, then run the
# portal-eval-only decoder calibration sweep (scripts/calibrate_decoder.sh) on the given GPU.
# Unattended follow-up so the calibration does not wait for a human; the operating point is
# still chosen by hand from logs/calib_<model>_*.log and scored with scripts/score_calibrated.sh.
#   scripts/calibrate_when_selected.sh <model> <gpu>
set -u
cd "$(dirname "$0")/.."
export ECGR_RUN_TAG="${ECGR_RUN_TAG:-260923_60s}"
MODEL="${1:?model}"; GPU="${2:-0}"
PY=/home/ai-server/miniconda3/envs/beat/bin/python
CKPT="$("$PY" -c "from ecgr import config, models; import os; print(os.path.join(config.CHECKPOINT_DIR, models.keras_name('$MODEL'), 'epochs', 'selected_by_bxb.keras'))")"
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for $CKPT"
while [[ ! -f "$CKPT" ]]; do sleep 120; done
sleep 60   # let the writer finish
log "found; calibrating $MODEL on GPU $GPU"
scripts/calibrate_decoder.sh "$MODEL" "$CKPT" "$GPU"
log "calibration finished for $MODEL"
touch "logs/calib_${MODEL}.done"
