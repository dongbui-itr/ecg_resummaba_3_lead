#!/bin/bash
# P+-oriented fine-tune of v4_scratch_best on the v4 TRAIN data (validation = original eval split; no eval sets)
set -euo pipefail
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt 14000 ]; do sleep 60; done
export ECGR_RHYTHM_NPY_DIR=/media/Project/ECG/Model_Dong/ecgr_rhythm/npy_curated_v4_131026 ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
export ECGR_RHYTHM_RUN_TAG=161026_rhythm_dual_1m_v4_pplus2 ECGR_RHYTHM_CKPT_START_EPOCH=1
export ECGR_RHYTHM_CLASS_WEIGHTS=0.33,0.5,0.75,1.5,1.3 ECGR_RHYTHM_BEAT_CLASS_WEIGHTS=0.15,1.0,3.0,2.5
echo "[$(date +%T)] fine-tune from v4_scratch_best: milder class weights (0.33,0.5,0.75,1.5,1.3), beat S 3 V 2.5, lr 1e-4 cosine, 8 epochs"
python3 -u -m ecgr.rhythm train --model rhythm_dual_1m --init-from /media/Project/ECG/Model_Dong/ecgr_rhythm/141026_rhythm_dual_1m_v4_scratch/release/v4_scratch_best.keras --lr 1e-4 --schedule cosine --sampler stratified --epochs 4 --patience 4 2>&1 | tr '\r' '\n' | grep --line-buffered -aE "windows|class weights|EPOCH .*val_rhythm_f1|Traceback|Error|finished"
echo "[$(date +%T)] FT_PPLUS2_DONE"
