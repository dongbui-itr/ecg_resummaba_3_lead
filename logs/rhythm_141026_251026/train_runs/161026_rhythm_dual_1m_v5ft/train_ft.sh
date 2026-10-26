#!/bin/bash
# v5 fine-tune of v4_scratch_best: + nsrdb (sinus), ltdb and svdb (MIT-style beats), x3; no EC57 / v4 data
set -euo pipefail
FT=/media/Project/ECG/Model_Dong/ecgr_rhythm/161026_rhythm_dual_1m_v5ft
cd $FT/code
export ECGR_RHYTHM_NPY_DIR=$FT/npy_ft ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
export ECGR_RHYTHM_RUN_TAG=161026_rhythm_dual_1m_v5ft ECGR_RHYTHM_CKPT_START_EPOCH=1
export ECGR_RHYTHM_STRAT_LONG=VT:5 ECGR_RHYTHM_STRAT_LONG_BOOST=3
export ECGR_RHYTHM_BEAT_CLASS_WEIGHTS=0.15,1.0,3.0,2.5
python3 -u -m ecgr.rhythm train --model rhythm_dual_1m \
  --init-from /media/Project/ECG/Model_Dong/ecgr_rhythm/141026_rhythm_dual_1m_v4_scratch/release/v4_scratch_best.keras \
  --lr 1e-4 --schedule cosine --sampler stratified --epochs 8 --patience 8
