#!/usr/bin/env bash
# One CfC size end to end on one card: distill -> stepeval -> ec57.
#
#   ECGR_GPU=1 ECGR_RUN_TAG=261012_cfc \
#       setsid nohup scripts/run_cfc.sh resucfc_2m </dev/null >logs/cfc_2m.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export ECGR_DATA_DIR="${ECGR_DATA_DIR:-/mnt/md0/Dong_data/portal_data/}"
export ECGR_PHYSIONET_DIR="${ECGR_PHYSIONET_DIR:-/mnt/md0/Dong_data/physionet/}"
export ECGR_RUN_TAG="${ECGR_RUN_TAG:-$(date +%y%m%d)_cfc}"
export ECGR_SAVE_EPOCH_INTERVAL="${ECGR_SAVE_EPOCH_INTERVAL:-3}"
exec ./run_pipeline.sh cfc "$@"
