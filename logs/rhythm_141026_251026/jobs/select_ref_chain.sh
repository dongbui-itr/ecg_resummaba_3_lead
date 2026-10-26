#!/bin/bash
# select_ref_chain.sh <unit> <run_tag> <keyprefix>: after <unit> ends, overlap-1 validation inference for epochs 1-4 + reference-guarded decode tuning
set -u
unit=$1; tag=$2; kp=$3
R=/media/Project/ECG/Model_Dong/ecgr_rhythm; J=$R/summary_111026/strip_eval/jobs; S=$R/summary_111026/strip_eval
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
while systemctl --user is-active -q $unit; do sleep 60; done
E=$R/$tag/checkpoints/dualunet_rhythm_1m/epochs
for e in 1 2 3 4; do
  ck=$E/epoch_0$e.keras; key=${kp}_e$e
  [ -f $ck ] || { echo "no $ck"; continue; }
  echo "[$(date +%T)] $key: validation inference (overlap 1)"
  ECGR_RHYTHM_PREDICT_OVERLAP=1 $J/holter_val.sh $key $ck > $J/hv_$key.log 2>&1
  STRIP_VAL_EVENTS=$S/strip_val2_events.json ECGR_RHYTHM_PREDICT_OVERLAP=1 ECGR_RHYTHM_PREDICT_TAPER=0 python3 -u $J/val_predict.py $key $ck > $J/vp_$key.log 2>&1
  echo "[$(date +%T)] $key: reference-guarded tuning (background)"
  (cd /tmp && python3 -u $J/tune_pplus.py $key --ref ${REF:-v4ov1_tight} --sweeps 3 --random 60 --workers 10 --out $S/tune_pplus_${key}ref > $J/tune_pplus_${key}ref.log 2>&1) &
done
wait
echo "[$(date +%T)] SELECT_DONE $kp"
cd /tmp && python3 $J/compare_candidates.py ${REF:-v4ov1_tight}
