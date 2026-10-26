#!/bin/bash
# long-Holter validation inference (eval side of ltafdb / nsrdb / incartdb / svdb) for one checkpoint
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
export ECGR_RHYTHM_RUN_TAG=summary_111026 ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/ ECGR_RHYTHM_PREDICT_HOP=5 ECGR_RHYTHM_PREDICT_TTA=id,flip,swap ECGR_RHYTHM_PREDICT_TAPER=0
key=$1; ckpt=$2
echo "[$(date +%T)] $key"
python3 -u -c "
from ecgr.rhythm import ec57
ec57.run('$ckpt', 'holterval_$key', dbs=['ltafdb','nsrdb','incartdb','svdb'], skip_rhythm_eval=True, decode={})
" 2>&1 | grep -E "=====|records predicted|error on|E_Se=|Traceback"
echo "[$(date +%T)] $key HOLTER_DONE"
