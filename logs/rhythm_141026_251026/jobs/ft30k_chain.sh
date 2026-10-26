#!/bin/bash
# ft30k_chain.sh : fine-tune ALL weights (encoder + decoders) of distill2_30k under the beatft2_e2 teacher (KD a=0.5 T=2,
# lr 2e-4 cosine), validation-only decoder tune (own guard), then EC57 + dataset-eval v4 ONCE at overlap 1 (+AF gate),
# checkpoint + decode filed under the release project, Excel summary vs distill2_30k and the teacher.
set -uo pipefail
R=/media/Project/ECG/Model_Dong/ecgr_rhythm; J=$R/summary_111026/strip_eval/jobs; S=$R/summary_111026/strip_eval; REL=$R/release_dualunet_1m_e26/checkpoint/candidates
TAG=201026_rhythm_dual_30k_ftd; V=$R/$TAG; key=distill3_30k
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt 6000 ]; do sleep 60; done
export ECGR_RHYTHM_NPY_DIR=$R/161026_rhythm_dual_1m_v5ft/npy_ft ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
export ECGR_RHYTHM_RUN_TAG=$TAG ECGR_RHYTHM_TEACHER=$REL/beatft2_e2.keras ECGR_RHYTHM_CKPT_START_EPOCH=1 ECGR_RHYTHM_STRAT_LONG=VT:5 ECGR_RHYTHM_STRAT_LONG_BOOST=3
echo "[$(date +%T)] 30k: fine-tune all weights of distill2_30k under beatft2_e2 (KD a=0.5 T=2, lr 2e-4 cosine, 20 ep)"
python3 -u -m ecgr.rhythm distill --model rhythm_dual_30k --init-from $REL/distill2_30k.keras --epochs 20 --patience 6 --lr 2e-4 2>&1 | tr '\r' '\n' | grep --line-buffered -aE "student|teacher|EPOCH .*(val_rhythm_f1|beat:)|improved|Traceback|Error|finished"
mkdir -p $V/release; cp $V/checkpoints/dualunet_rhythm_30k/best_model.keras $V/release/$key.keras; ck=$V/release/$key.keras
echo "[$(date +%T)] validation inference (overlap 1)"
ECGR_RHYTHM_PREDICT_OVERLAP=1 $J/holter_val.sh $key $ck > $J/hv_$key.log 2>&1
STRIP_VAL_EVENTS=$S/strip_val2_events.json ECGR_RHYTHM_PREDICT_OVERLAP=1 ECGR_RHYTHM_PREDICT_TAPER=0 python3 -u $J/val_predict.py $key $ck > $J/vp_$key.log 2>&1
echo "[$(date +%T)] decoder tune on validation (own guard)"
(cd /tmp && python3 -u $J/tune_pplus.py $key --sweeps 3 --random 80 --workers 8 --out $S/tune_pplus_$key > $J/tune_pplus_$key.log 2>&1)
echo "[$(date +%T)] EC57 + dataset-eval v4 (once)"
$J/ec57_final_ov1.sh $key $ck $S/tune_pplus_$key/tuned_decode.json > $J/ec57_$key.log 2>&1
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
TUNED_DECODE=$S/tune_pplus_$key/tuned_decode.json ECGR_RHYTHM_PREDICT_OVERLAP=1 ECGR_RHYTHM_PREDICT_TTA=id,flip,swap ECGR_RHYTHM_PREDICT_TAPER=0 python3 -u $J/eval_dataset_v4.py $key $ck > $J/dsv4_$key.log 2>&1
cp $ck $REL/$key.keras; cp $S/tune_pplus_$key/tuned_decode.json $REL/${key}_decode.json; (cd $REL && sha256sum $key.keras > $key.SHA256)
python3 $R/summary_111026/build_summary_xlsx.py $R/summary_111026/ec57_v4_distill3_30k_summary.xlsx $key distill2_30k beatft2_e2 > $J/xlsx_$key.log 2>&1
echo "[$(date +%T)] FT30K_DONE"
