#!/bin/bash
# ft_e36_chain.sh : fine-tune ALL weights (encoder + decoders) of r30k_A_e36 under the beatft2_e2 teacher, two variants run
# one after the other (GPU is shared): F1 = KD alpha 0.9 (lean on the teacher, which beats the student on every class),
# F2 = KD alpha 0.7 + AFIB rhythm-CE weight x1.5. lr 2e-4 cosine, 12 epochs, no early stop, every epoch saved.
# Post-processing is NOT tuned: candidates are later scored with r30k_A_e36's own decode json unchanged.
set -uo pipefail
R=/media/Project/ECG/Model_Dong/ecgr_rhythm; REL=$R/release_dualunet_1m_e26/checkpoint/candidates
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
export ECGR_RHYTHM_NPY_DIR=$R/161026_rhythm_dual_1m_v5ft/npy_ft ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
export ECGR_RHYTHM_TEACHER=$REL/beatft2_e2.keras ECGR_RHYTHM_CKPT_START_EPOCH=1 ECGR_RHYTHM_STRAT_LONG=VT:5 ECGR_RHYTHM_STRAT_LONG_BOOST=3
for spec in "F1 0.9 -" "F2 0.7 0.2407,0.7916,1.0,2.1314,1.8037"; do
  set -- $spec; v=$1; alpha=$2; cw=$3
  while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt 9000 ]; do sleep 60; done
  export ECGR_RHYTHM_RUN_TAG=231026_rhythm_dual_30k_e36ft_$v
  if [ "$cw" = "-" ]; then unset ECGR_RHYTHM_CLASS_WEIGHTS; else export ECGR_RHYTHM_CLASS_WEIGHTS=$cw; fi
  echo "[$(date +%T)] $v: fine-tune r30k_A_e36 (alpha $alpha, class weights $cw), lr 2e-4, 12 epochs"
  python3 -u -m ecgr.rhythm distill --model rhythm_dual_30k --init-from $REL/r30k_A_e36.keras --epochs 12 --patience 999 --lr 2e-4 --alpha $alpha 2>&1 | tr '\r' '\n' | grep --line-buffered -aE "student|KD |EPOCH .*(val_rhythm_f1|beat:)|Traceback|Error|finished"
  echo "[$(date +%T)] FT_DONE $v"
done
echo "[$(date +%T)] FT_E36_ALL_DONE"
