#!/bin/bash
# continue the source-focused fine-tune, then strip_val inference + decode tuning for the best and the last epoch
set -u
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
R=/media/Project/ECG/Model_Dong/ecgr_rhythm; W=$R/summary_111026/strip_eval
size=$1; init=$2; lr=$3; start=$4; workers=$5
tag=121026_rhythm_dual_${size}_ftstrip2
export ECGR_RHYTHM_RUN_TAG=$tag ECGR_RHYTHM_CKPT_START_EPOCH=1 ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
export ECGR_RHYTHM_SOURCES=afib-2,rhythm-2,rhythm-3,dataset-4,dataset-5,sinus-2,sinus-3,sinus-4,sinus-4p
echo "[$(date +%T)] $size: fine-tune from $init lr $lr"
python3 -u -m ecgr.rhythm train --model rhythm_dual_$size --init-from $init --lr $lr --schedule cosine --sampler stratified --epochs 6 --patience 3 2>&1 | tr '\r' '\n' | grep -aE "Epoch [0-9]+|val_rhythm_f1|early stopping|Restoring|Traceback|Error|finished"
unset ECGR_RHYTHM_SOURCES
D=$R/$tag/checkpoints/dualunet_rhythm_$size
last=$(ls $D/epochs/epoch_*.keras | sort | tail -1)
for pair in "best:$D/best_model.keras" "last:$last"; do
  k=${pair%%:*}; ck=${pair#*:}
  key=dual${size}_ft2_$k
  echo "[$(date +%T)] $key: validation inference ($ck)"
  python3 -u $W/jobs/val_predict.py $key $ck 2>&1 | grep -E "predicted|PREDICT_DONE|Traceback"
  echo "[$(date +%T)] $key: decode tuning"
  CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python3 -u -m ecgr.rhythm.strip_tune $W/val_npz/$key $W/tuned_$key.json --events $W/strip_val_events.json --workers $workers --rounds 3 --random 300 --start $start 2>&1 | grep -E "start:|target cells|below target|^->|Traceback"
done
echo "[$(date +%T)] $size: CHAIN2_DONE"
