#!/bin/bash
# EC57 (5 DBs, with bxb) + test-split data eval for one tuned model
set -u
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
key=$1; ckpt=$2; dec=$3
export ECGR_RHYTHM_RUN_TAG=summary_111026 ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/ ECGR_RHYTHM_PREDICT_HOP=5 ECGR_RHYTHM_PREDICT_TTA=id,flip,swap ECGR_RHYTHM_PREDICT_TAPER=0
echo "[$(date +%T)] $key: EC57"
python3 -u -c "
import json
from ecgr.rhythm import ec57
from ecgr.evaluation.report import summarize
d = json.load(open('$dec'))['decode']
ec57.run('$ckpt', '$key', dbs=['mitdb','nstdb','escdb','ahadb','afdb'], skip_rhythm_eval=True, decode=d)
summarize('/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/ec57/$key/beats')
" 2>&1 | grep -E "=====|E_Se=|beats .*Q_Se|Traceback|Error"
echo "[$(date +%T)] $key: test split"
python3 -u -c "
from ecgr.rhythm.evaluate import evaluate
evaluate('$ckpt', split='test', out_dir='/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/eval/$key')
" 2>&1 | grep -E "macro F1|Traceback|report ->"
echo "[$(date +%T)] $key: EC57_DONE"
