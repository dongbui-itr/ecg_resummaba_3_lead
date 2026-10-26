#!/bin/bash
# final_se3.sh <key> <ckpt> <decode_json> : score the chosen candidate ONCE (EC57 5 DBs + bxb + test split, dataset-eval v4)
# at overlap 1 with its tuned decode, then file checkpoint + decode under the release project.
set -u
key=$1; ckpt=$2; dec=$3
J=/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs; REL=/media/Project/ECG/Model_Dong/ecgr_rhythm/release_dualunet_1m_e26
$J/ec57_final_ov1.sh $key $ckpt $dec > $J/ec57_$key.log 2>&1
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
TUNED_DECODE=$dec ECGR_RHYTHM_PREDICT_OVERLAP=1 ECGR_RHYTHM_PREDICT_TTA=id,flip,swap ECGR_RHYTHM_PREDICT_TAPER=0 ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/ python3 -u $J/eval_dataset_v4.py $key $ckpt > $J/dsv4_$key.log 2>&1
mkdir -p $REL/checkpoint/candidates; cp $ckpt $REL/checkpoint/candidates/$key.keras; cp $dec $REL/checkpoint/candidates/${key}_decode.json
sha256sum $REL/checkpoint/candidates/$key.keras > $REL/checkpoint/candidates/$key.SHA256
echo "[$(date +%T)] FINAL_SE3_DONE $key"
