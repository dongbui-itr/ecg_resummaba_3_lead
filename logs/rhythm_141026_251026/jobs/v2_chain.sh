#!/bin/bash
# validation v2 (SVT/VT filled with portal runs): extra inference, decode re-tuning, final rhythm_eval score
set -u
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
W=/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval
key=$1; ckpt=$2; workers=$3
export STRIP_VAL_EVENTS=$W/strip_val2_events.json
echo "[$(date +%T)] $key: v2 inference"
python3 -u $W/jobs/val_predict.py $key $ckpt 2>&1 | grep -E "predicted|PREDICT_DONE|Traceback"
echo "[$(date +%T)] $key: v2 decode tuning"
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python3 -u -m ecgr.rhythm.strip_tune $W/val_npz/$key $W/tunedv2_$key.json --events $STRIP_VAL_EVENTS --workers $workers --rounds 3 --random 300 --start $W/tuned_$key.json 2>&1 | grep -E "start:|target cells|below target|^->|Traceback"
echo "[$(date +%T)] $key: final"
python3 -u $W/jobs/final_strip.py ${key}_v2 $ckpt $W/test_npz/$key $W/val_npz/$key $W/tunedv2_$key.json 2>&1 | grep -E "^## |target cells|below target|FINAL_DONE|Traceback"
echo "[$(date +%T)] $key: V2_DONE"
