#!/bin/bash
# resume F2 (interrupted after epoch 7/12): epochs 8-12 = 5 epochs from F2 epoch_07, lr 7.4e-5 (F2's cosine value at epoch 8) cosine to 0
set -uo pipefail
R=/media/Project/ECG/Model_Dong/ecgr_rhythm; REL=$R/release_dualunet_1m_e26/checkpoint/candidates
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt 7000 ]; do sleep 60; done
export ECGR_RHYTHM_NPY_DIR=$R/161026_rhythm_dual_1m_v5ft/npy_ft ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
export ECGR_RHYTHM_TEACHER=$REL/beatft2_e2.keras ECGR_RHYTHM_CKPT_START_EPOCH=1 ECGR_RHYTHM_STRAT_LONG=VT:5 ECGR_RHYTHM_STRAT_LONG_BOOST=3
export ECGR_RHYTHM_RUN_TAG=231026_rhythm_dual_30k_e36ft_F2b ECGR_RHYTHM_CLASS_WEIGHTS=0.2407,0.7916,1.0,2.1314,1.8037
python3 -u -m ecgr.rhythm distill --model rhythm_dual_30k --init-from $R/231026_rhythm_dual_30k_e36ft_F2/checkpoints/dualunet_rhythm_30k/epochs/epoch_07.keras --epochs 5 --patience 999 --lr 7.4e-5 --alpha 0.7 2>&1 | tr '\r' '\n' | grep --line-buffered -aE "student|EPOCH .*(val_rhythm_f1|beat:)|Traceback|Error|finished"
echo "[$(date +%T)] F2B_DONE"
