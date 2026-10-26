#!/bin/bash
# Beat-decoder-only fine-tune of v4_scratch_best (encoder + rhythm decoder frozen): data = npy_ft
# (curated v4 + nsrdb/ltdb/svdb train sides, no EC57 / dataset-eval), beat weights S 4 (as v4), V 2, 4 epochs.
set -euo pipefail
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt 14000 ]; do sleep 60; done
export ECGR_RHYTHM_NPY_DIR=/media/Project/ECG/Model_Dong/ecgr_rhythm/161026_rhythm_dual_1m_v5ft/npy_ft ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
export ECGR_RHYTHM_RUN_TAG=191026_rhythm_dual_1m_v4_beatft2 ECGR_RHYTHM_CKPT_START_EPOCH=1
export ECGR_RHYTHM_BEAT_CLASS_WEIGHTS=0.15,1.0,4.0,2.0
echo "[$(date +%T)] beat-decoder-only fine-tune from v4_scratch_best: --train-only '^(pos[123]|beat_pos|type)', lr 1e-4 cosine, 4 epochs"
python3 -u -m ecgr.rhythm train --model rhythm_dual_1m --init-from /media/Project/ECG/Model_Dong/ecgr_rhythm/141026_rhythm_dual_1m_v4_scratch/release/v4_scratch_best.keras --lr 1e-4 --schedule cosine --sampler stratified --epochs 4 --patience 4 --train-only '^(pos[123]|beat_pos|type)' 2>&1 | tr '\r' '\n' | grep --line-buffered -aE "train-only|windows|class weights|EPOCH .*(val_rhythm_f1|beat)|Traceback|Error|finished"
echo "[$(date +%T)] FT_BEATFT2_DONE"
