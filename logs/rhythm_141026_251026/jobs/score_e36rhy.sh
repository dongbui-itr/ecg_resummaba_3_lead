#!/bin/bash
# score_e36rhy.sh : EC57 (5 DBs + bxb + test split) + dataset-eval v4 ONCE, overlap 1, for two rhythm-decoder fine-tunes of e36,
# each with r30k_A_e36's own decode json unchanged (no post-process tuning); Excel vs e36 and the teacher.
set -u
R=/media/Project/ECG/Model_Dong/ecgr_rhythm; J=$R/summary_111026/strip_eval/jobs; S=$R/summary_111026
DEC=$S/strip_eval/tune_pplus_rA_e36/tuned_decode.json
for spec in "e36rhy_R2_e05 $R/241026_rhythm_dual_30k_e36rhy_R2/checkpoints/dualunet_rhythm_30k/epochs/epoch_05.keras" \
            "e36rhy_R1_e03 $R/241026_rhythm_dual_30k_e36rhy_R1/checkpoints/dualunet_rhythm_30k/epochs/epoch_03.keras"; do
  set -- $spec; key=$1; ck=$2
  while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt 6000 ]; do sleep 60; done
  $J/ec57_final_ov1.sh $key $ck $DEC > $J/ec57_$key.log 2>&1
  cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
  TUNED_DECODE=$DEC ECGR_RHYTHM_PREDICT_OVERLAP=1 ECGR_RHYTHM_PREDICT_TTA=id,flip,swap ECGR_RHYTHM_PREDICT_TAPER=0 ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/ python3 -u $J/eval_dataset_v4.py $key $ck > $J/dsv4_$key.log 2>&1
  echo "[$(date +%T)] SCORED $key"
done
python3 $S/build_summary_xlsx.py $S/ec57_v4_e36rhy_summary.xlsx e36rhy_R2_e05 e36rhy_R1_e03 r30k_A_e36 beatft2_e2 > $J/xlsx_e36rhy.log 2>&1
echo "[$(date +%T)] SCORE_E36RHY_DONE"
