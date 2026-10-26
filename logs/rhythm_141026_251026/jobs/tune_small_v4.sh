#!/bin/bash
# tune_small_v4.sh <size> : fine-tune the v4-distilled student on the v4 train data (lower LR, more label weight)
set -euo pipefail
size=$1
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
while systemctl --user is-active -q ecgr-distill-$size; do sleep 60; done
export ECGR_RHYTHM_NPY_DIR=/media/Project/ECG/Model_Dong/ecgr_rhythm/npy_curated_v4_131026 ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
export ECGR_RHYTHM_RUN_TAG=151026_rhythm_dual_small_v4_tune ECGR_RHYTHM_TEACHER=/media/Project/ECG/Model_Dong/ecgr_rhythm/141026_rhythm_dual_1m_v4_scratch/release/v4_scratch_best.keras
export ECGR_RHYTHM_CKPT_START_EPOCH=3 ECGR_RHYTHM_STRAT_LONG=VT:5 ECGR_RHYTHM_STRAT_LONG_BOOST=3
echo "[$(date +%T)] $size: tune from v4_distill_$size (lr 2e-4, KD a=0.3, 15 epochs)"
python3 -u -m ecgr.rhythm distill --model rhythm_dual_$size --ssl-from /media/Project/ECG/Model_Dong/ecgr_rhythm/151026_rhythm_dual_small_v4/release/v4_distill_$size.keras --lr 2e-4 --alpha 0.3 --epochs 15 --patience 6 2>&1 | tr '\r' '\n' | grep --line-buffered -aE "teacher|encoder from|EPOCH .*val_rhythm_f1|Traceback|Error|finished"
V=/media/Project/ECG/Model_Dong/ecgr_rhythm/151026_rhythm_dual_small_v4_tune; mkdir -p $V/release; cp $V/checkpoints/dualunet_rhythm_$size/best_model.keras $V/release/v4_tune_$size.keras
echo "[$(date +%T)] $size: EC57 + dataset-eval/v4"
/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs/ec57_official.sh v4_tune_$size $V/release/v4_tune_$size.keras > /media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs/ec57_v4_tune_$size.log 2>&1
ECGR_RHYTHM_PREDICT_HOP=5 ECGR_RHYTHM_PREDICT_TTA=id,flip,swap ECGR_RHYTHM_PREDICT_TAPER=0 python3 -u /media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs/eval_dataset_v4.py v4_tune_$size $V/release/v4_tune_$size.keras > /media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs/eval_dataset_v4_tune_$size.log 2>&1
echo "[$(date +%T)] $size: TUNE_CHAIN_DONE"
