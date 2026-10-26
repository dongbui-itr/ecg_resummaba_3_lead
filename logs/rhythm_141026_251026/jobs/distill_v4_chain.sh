#!/bin/bash
set -euo pipefail
size=$1
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
export ECGR_RHYTHM_NPY_DIR=/media/Project/ECG/Model_Dong/ecgr_rhythm/npy_curated_v4_131026 ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
export ECGR_RHYTHM_RUN_TAG=151026_rhythm_dual_small_v4 ECGR_RHYTHM_TEACHER=/media/Project/ECG/Model_Dong/ecgr_rhythm/141026_rhythm_dual_1m_v4_scratch/release/v4_scratch_best.keras
export ECGR_RHYTHM_CKPT_START_EPOCH=3 ECGR_RHYTHM_STRAT_LONG=VT:5 ECGR_RHYTHM_STRAT_LONG_BOOST=3
echo "[$(date +%T)] $size: SSL encoder (teacher v4 beat positions)"
python3 -u -m ecgr.rhythm ssl --model rhythm_dual_$size 2>&1 | tr '\r' '\n' | grep --line-buffered -aE "SSL|masking|Epoch [0-9]+/|loss:.*val|Traceback|Error|saved" | grep -av "ETA" || true
test -f /media/Project/ECG/Model_Dong/ecgr_rhythm/151026_rhythm_dual_small_v4/checkpoints/dualunet_rhythm_$size/ssl_encoder.keras
echo "[$(date +%T)] $size: distill from v4 teacher (KD a=0.5 T=2)"
python3 -u -m ecgr.rhythm distill --model rhythm_dual_$size --epochs 40 --patience 8 2>&1 | tr '\r' '\n' | grep --line-buffered -aE "teacher|EPOCH .*val_rhythm_f1|improved|Traceback|Error|finished"
V=/media/Project/ECG/Model_Dong/ecgr_rhythm/151026_rhythm_dual_small_v4; mkdir -p $V/release; cp $V/checkpoints/dualunet_rhythm_$size/best_model.keras $V/release/v4_distill_$size.keras
echo "[$(date +%T)] $size: EC57 + dataset-eval/v4"
/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs/ec57_official.sh v4_distill_$size $V/release/v4_distill_$size.keras > /media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs/ec57_v4_distill_$size.log 2>&1
ECGR_RHYTHM_PREDICT_HOP=5 ECGR_RHYTHM_PREDICT_TTA=id,flip,swap ECGR_RHYTHM_PREDICT_TAPER=0 python3 -u /media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs/eval_dataset_v4.py v4_distill_$size $V/release/v4_distill_$size.keras > /media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs/eval_dataset_v4_distill_$size.log 2>&1
echo "[$(date +%T)] $size: DISTILL_CHAIN_DONE"
