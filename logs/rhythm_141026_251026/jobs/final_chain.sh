#!/bin/bash
# final scoring (once): EC57 ov1 + dataset-eval v4 for v4 official (ref), v4 + tuned decode, pplus3_e1 + tuned decode
set -u
J=/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs; S=/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval
V4=/media/Project/ECG/Model_Dong/ecgr_rhythm/141026_rhythm_dual_1m_v4_scratch/release/v4_scratch_best.keras
P3=/media/Project/ECG/Model_Dong/ecgr_rhythm/161026_rhythm_dual_1m_v4_pplus3/checkpoints/dualunet_rhythm_1m/epochs/epoch_01.keras
while systemctl --user is-active -q ecgr-ec57-v4ov1ref; do sleep 60; done
$J/ec57_final_ov1.sh v4ov1_tuned $V4 $S/tune_pplus_v4ov1_tight/tuned_decode.json > $J/ec57_v4ov1_tuned.log 2>&1
$J/ec57_final_ov1.sh pplus3_e1 $P3 $S/tune_pplus_pplus3_e1ref/tuned_decode.json > $J/ec57_pplus3_e1.log 2>&1
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
export ECGR_RHYTHM_PREDICT_OVERLAP=1 ECGR_RHYTHM_PREDICT_TTA=id,flip,swap ECGR_RHYTHM_PREDICT_TAPER=0 ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
python3 -u $J/eval_dataset_v4.py v4ov1_official $V4 > $J/dsv4_v4ov1_official.log 2>&1
TUNED_DECODE=$S/tune_pplus_v4ov1_tight/tuned_decode.json python3 -u $J/eval_dataset_v4.py v4ov1_tuned $V4 > $J/dsv4_v4ov1_tuned.log 2>&1
TUNED_DECODE=$S/tune_pplus_pplus3_e1ref/tuned_decode.json python3 -u $J/eval_dataset_v4.py pplus3_e1 $P3 > $J/dsv4_pplus3_e1.log 2>&1
echo "[$(date +%T)] FINAL_DONE"
