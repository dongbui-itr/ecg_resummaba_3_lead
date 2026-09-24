#!/usr/bin/env bash
# Queue 1 lost its 3m training to an external SIGTERM (2026-09-24 10:12) and moved on to 1m.
# Wait for that queue to finish, then rerun the 3m size on GPU 1: ssl/cpc weights exist and
# are skipped, so only train -> select -> stepeval -> ec57 -> regress run.
set -u
cd "$(dirname "$0")/.."
export ECGR_RUN_TAG=260923_60s
log() { echo "[$(date '+%F %T')] $*"; }
Q1_PID="${1:?pid of the queue-1 run_pipeline.sh}"
log "waiting for queue 1 (pid ${Q1_PID}) to finish"
while kill -0 "${Q1_PID}" 2>/dev/null; do sleep 120; done
log "queue 1 finished; relaunching resumamba_3m on GPU 1"
ECGR_GPU=1 setsid nohup ./run_pipeline.sh sweep resumamba_3m </dev/null >logs/q1b_3m_60s.log 2>&1 &
log "queue 1b (GPU 1): resumamba_3m -> logs/q1b_3m_60s.log (pid $!)"
