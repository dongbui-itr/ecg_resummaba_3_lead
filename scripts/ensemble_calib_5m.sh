#!/usr/bin/env bash
# Portal-eval-only check of checkpoint ensembles for resumamba_5m (README 8i: never calibrate
# on mitdb / beat-eval). Waits for the single-checkpoint calibration sweep to finish so the
# GPU is not shared three ways, then scores each ensemble x decoder point on the 5,000-record
# portal-eval sample. Epoch 29 (bxb winner) calls noise V far more often than epoch 17 on
# nstdb; averaging softmax outputs is the cheapest candidate fix and must first earn its place
# on portal-eval like any other checkpoint.
#   scripts/ensemble_calib_5m.sh <gpu>
set -u
cd "$(dirname "$0")/.."
export ECGR_RUN_TAG="${ECGR_RUN_TAG:-260923_60s}" CUDA_VISIBLE_DEVICES="${1:-0}" TF_CPP_MIN_LOG_LEVEL=2
PY=/home/ai-server/miniconda3/envs/beat/bin/python
E=/mnt/md0/Dong_data/portal_data/train/260923_60s/checkpoints/resumamba_seq2seq_5m/epochs
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for logs/calib_resumamba_5m.done"
while [[ ! -f logs/calib_resumamba_5m.done ]]; do sleep 60; done
run() {  # run <name> <sb> <pp> <ckpt...>
  local name="$1" sb="$2" pp="$3"; shift 3
  local tag="calib_resumamba_5m_${name}_sb${sb}_pp${pp}"
  log "$tag"
  ECGR_DECODE_MIN_PEAK_PROB="$pp" "$PY" -m ecgr ec57 --model resumamba_5m --checkpoint "$@" --tag "$tag" \
    --skip-physionet --skip-portal --splits eval --split-records 5000 --s-boost "$sb" \
    > "logs/${tag}.log" 2>&1 || echo "  FAILED rc=$?"
  grep -E "^  portal-eval" "logs/${tag}.log"
}
run ens17-29    1.0 0.0 "$E/epoch_17.keras" "$E/epoch_29.keras"
run ens17-29    1.0 0.8 "$E/epoch_17.keras" "$E/epoch_29.keras"
run ens17-29    0.8 0.8 "$E/epoch_17.keras" "$E/epoch_29.keras"
run ens17-25-29 1.0 0.8 "$E/epoch_17.keras" "$E/epoch_25.keras" "$E/epoch_29.keras"
log "ensemble calibration done"
touch logs/ensemble_calib_5m.done
