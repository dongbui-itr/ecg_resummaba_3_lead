#!/bin/bash
# distill2_chain.sh <size> : KD of the beatft2_e2 teacher into rhythm_dual_<size> (SSL encoder reused from the
# 151026 run: same teacher beat positions), validation-only decoder tune (own guard), then EC57 + dataset-eval v4
# ONCE at overlap 1 with the AF gate.
set -uo pipefail
size=$1
R=/media/Project/ECG/Model_Dong/ecgr_rhythm; J=$R/summary_111026/strip_eval/jobs; S=$R/summary_111026/strip_eval
TAG=191026_rhythm_dual_small_distill2; V=$R/$TAG
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
export ECGR_RHYTHM_NPY_DIR=$R/161026_rhythm_dual_1m_v5ft/npy_ft ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
export ECGR_RHYTHM_RUN_TAG=$TAG ECGR_RHYTHM_TEACHER=$R/release_dualunet_1m_e26/checkpoint/candidates/beatft2_e2.keras
export ECGR_RHYTHM_CKPT_START_EPOCH=3 ECGR_RHYTHM_STRAT_LONG=VT:5 ECGR_RHYTHM_STRAT_LONG_BOOST=3
SSL=$R/151026_rhythm_dual_small_v4/checkpoints/dualunet_rhythm_$size/ssl_encoder.keras
if [ ! -f $SSL ]; then
  echo "[$(date +%T)] $size: SSL encoder (teacher beat positions)"
  python3 -u -m ecgr.rhythm ssl --model rhythm_dual_$size 2>&1 | tr '\r' '\n' | grep --line-buffered -aE "SSL|Traceback|Error|saved" || true
  SSL=$V/checkpoints/dualunet_rhythm_$size/ssl_encoder.keras
fi
echo "[$(date +%T)] $size: distill from beatft2_e2 teacher (KD a=0.5 T=2), ssl $SSL"
python3 -u -m ecgr.rhythm distill --model rhythm_dual_$size --ssl-from $SSL --epochs 40 --patience 8 2>&1 | tr '\r' '\n' | grep --line-buffered -aE "teacher|EPOCH .*(val_rhythm_f1|beat:)|Traceback|Error|finished"
mkdir -p $V/release; cp $V/checkpoints/dualunet_rhythm_$size/best_model.keras $V/release/distill2_$size.keras
key=distill2_$size; ck=$V/release/distill2_$size.keras
echo "[$(date +%T)] $size: validation inference (overlap 1)"
ECGR_RHYTHM_PREDICT_OVERLAP=1 $J/holter_val.sh $key $ck > $J/hv_$key.log 2>&1
STRIP_VAL_EVENTS=$S/strip_val2_events.json ECGR_RHYTHM_PREDICT_OVERLAP=1 ECGR_RHYTHM_PREDICT_TAPER=0 python3 -u $J/val_predict.py $key $ck > $J/vp_$key.log 2>&1
echo "[$(date +%T)] $size: decoder tune on validation (own guard, SE_TOL=3)"
(cd /tmp && SE_TOL=3 python3 -u $J/tune_pplus.py $key --sweeps 3 --random 80 --workers 8 --out $S/tune_pplus_${key} > $J/tune_pplus_${key}.log 2>&1)
echo "[$(date +%T)] $size: EC57 + dataset-eval v4 (once)"
$J/ec57_final_ov1.sh $key $ck $S/tune_pplus_$key/tuned_decode.json > $J/ec57_$key.log 2>&1
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
TUNED_DECODE=$S/tune_pplus_$key/tuned_decode.json ECGR_RHYTHM_PREDICT_OVERLAP=1 ECGR_RHYTHM_PREDICT_TTA=id,flip,swap ECGR_RHYTHM_PREDICT_TAPER=0 python3 -u $J/eval_dataset_v4.py $key $ck > $J/dsv4_$key.log 2>&1
echo "[$(date +%T)] DISTILL2_DONE $size"
