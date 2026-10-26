#!/bin/bash
# rhythm-decoder-only fine-tune of r30k_A_e36 under beatft2_e2 (encoder + beat head + channel head frozen, so beat output is
# bit-identical): R1 = AFIB CE x1.5, R2 = AFIB CE x2.0; KD alpha 0.5, lr 5e-4 cosine, 10 epochs, every epoch saved.
set -uo pipefail
R=/media/Project/ECG/Model_Dong/ecgr_rhythm
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
export ECGR_RHYTHM_NPY_DIR=$R/161026_rhythm_dual_1m_v5ft/npy_ft ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
export ECGR_RHYTHM_TEACHER=$R/191026_rhythm_dual_1m_v4_beatft2/checkpoints/dualunet_rhythm_1m/epochs/epoch_02.keras
export ECGR_RHYTHM_CKPT_START_EPOCH=1 ECGR_RHYTHM_STRAT_LONG=VT:5 ECGR_RHYTHM_STRAT_LONG_BOOST=3
for spec in "R1 0.7916" "R2 1.0554"; do
  set -- $spec; v=$1; af=$2
  while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt 7000 ]; do sleep 60; done
  export ECGR_RHYTHM_RUN_TAG=241026_rhythm_dual_30k_e36rhy_$v ECGR_RHYTHM_CLASS_WEIGHTS=0.2407,$af,1.0,2.1314,1.8037
  echo "[$(date +%T)] $v: rhythm-decoder-only fine-tune, AFIB weight $af"
  python3 -u -m ecgr.rhythm distill --model rhythm_dual_30k --init-from $R/201026_rhythm_dual_30k_A/release/e36.keras --train-only '^(rhythm|rr_)' --epochs 10 --patience 999 --lr 5e-4 --alpha 0.5 2>&1 | tr '\r' '\n' | grep --line-buffered -aE "student|train-only|KD |EPOCH .*(val_rhythm_f1|beat:)|Traceback|Error|finished"
  echo "[$(date +%T)] FT_DONE $v"
done
echo "[$(date +%T)] RHY_ALL_DONE"
