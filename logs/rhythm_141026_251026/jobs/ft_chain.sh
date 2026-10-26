#!/bin/bash
# strip_val inference of a fine-tuned checkpoint, then decode tuning starting from the base model's tuned params
set -u
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
W=/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval
key=$1; ckpt=$2; start=$3; workers=$4
echo "[$(date +%T)] $key: validation inference"
python3 -u $W/jobs/val_predict.py $key $ckpt
echo "[$(date +%T)] $key: decode tuning"
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python3 -u -m ecgr.rhythm.strip_tune $W/val_npz/$key $W/tuned_$key.json --events $W/strip_val_events.json --workers $workers --rounds 3 --random 300 --start $start
echo "[$(date +%T)] $key: CHAIN_DONE"
