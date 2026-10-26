#!/bin/bash
# retrain30k_chain.sh <variant> <kd_alpha> : retrain rhythm_dual_30k with the SSL encoder + KD from beatft2_e2 (npy_ft, lr 1e-3
# cosine to 0, 36 epochs, NO early stop; the student is saved after every epoch), then validation-only candidate screening:
# best_model + epochs 30/33/36 -> overlap-1 validation inference -> decoder tune (own guard, SE_TOL=3). Final EC57/v4 is run
# once afterwards on the candidate chosen from these validation numbers.
set -uo pipefail
v=$1; alpha=$2
R=/media/Project/ECG/Model_Dong/ecgr_rhythm; J=$R/summary_111026/strip_eval/jobs; S=$R/summary_111026/strip_eval; REL=$R/release_dualunet_1m_e26/checkpoint/candidates
TAG=201026_rhythm_dual_30k_$v; V=$R/$TAG
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt 7000 ]; do sleep 60; done
export ECGR_RHYTHM_NPY_DIR=$R/161026_rhythm_dual_1m_v5ft/npy_ft ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
export ECGR_RHYTHM_RUN_TAG=$TAG ECGR_RHYTHM_TEACHER=$REL/beatft2_e2.keras ECGR_RHYTHM_CKPT_START_EPOCH=3 ECGR_RHYTHM_STRAT_LONG=VT:5 ECGR_RHYTHM_STRAT_LONG_BOOST=3
SSL=$R/151026_rhythm_dual_small_v4/checkpoints/dualunet_rhythm_30k/ssl_encoder.keras
echo "[$(date +%T)] $v: KD from beatft2_e2 (alpha $alpha, T 2) ssl $SSL, 36 epochs, no early stop"
python3 -u -m ecgr.rhythm distill --model rhythm_dual_30k --ssl-from $SSL --epochs 36 --patience 999 --alpha $alpha 2>&1 | tr '\r' '\n' | grep --line-buffered -aE "student|teacher|KD |EPOCH .*(val_rhythm_f1|beat:)|Traceback|Error|finished"
E=$V/checkpoints/dualunet_rhythm_30k
mkdir -p $V/release; cp $E/best_model.keras $V/release/best.keras
for e in 30 33 36; do cp $E/epochs/epoch_$e.keras $V/release/e$e.keras; done
for c in best e30 e33 e36; do
  key=r${v}_$c; ck=$V/release/$c.keras
  echo "[$(date +%T)] $key: validation inference (overlap 1)"
  ECGR_RHYTHM_PREDICT_OVERLAP=1 $J/holter_val.sh $key $ck > $J/hv_$key.log 2>&1
  STRIP_VAL_EVENTS=$S/strip_val2_events.json ECGR_RHYTHM_PREDICT_OVERLAP=1 ECGR_RHYTHM_PREDICT_TAPER=0 python3 -u $J/val_predict.py $key $ck > $J/vp_$key.log 2>&1
  echo "[$(date +%T)] $key: decoder tune (background)"
  (cd /tmp && SE_TOL=3 python3 -u $J/tune_pplus.py $key --sweeps 3 --random 80 --workers 6 --out $S/tune_pplus_$key > $J/tune_pplus_$key.log 2>&1) &
done
wait
echo "[$(date +%T)] RETRAIN30K_SCREEN_DONE $v"
