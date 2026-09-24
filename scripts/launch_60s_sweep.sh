#!/usr/bin/env bash
# Wait for the 60 s data build (and any running test suite) to finish, then start the two
# GPU queues of the 260923_60s sweep, detached. Idempotent: rerunning skips finished stages.
set -u
cd "$(dirname "$0")/.."
MANIFEST=/mnt/md0/Dong_data/portal_data/train/tfrecord_60s_3lead/dataset_manifest.json
export ECGR_RUN_TAG="${ECGR_RUN_TAG:-260923_60s}"
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for the data build: ${MANIFEST}"
# The bracket trick keeps pgrep from matching a shell whose own command line quotes this
# pattern (a monitor or an interactive check would otherwise hold the launcher forever).
while [[ ! -f "${MANIFEST}" ]] || pgrep -f "[e]cgr data --step" >/dev/null; do sleep 60; done
log "data build done"
while pgrep -f "[p]ytest tests" >/dev/null; do log "test suite still on GPU 1, waiting"; sleep 30; done
mkdir -p logs
ECGR_GPU=0 setsid nohup ./run_pipeline.sh sweep resumamba_5m resumamba_100k </dev/null >logs/q0_60s.log 2>&1 &
log "queue 0 (GPU 0): resumamba_5m resumamba_100k -> logs/q0_60s.log (pid $!)"
sleep 5
ECGR_GPU=1 setsid nohup ./run_pipeline.sh sweep resumamba_3m resumamba_1m </dev/null >logs/q1_60s.log 2>&1 &
log "queue 1 (GPU 1): resumamba_3m resumamba_1m -> logs/q1_60s.log (pid $!)"
