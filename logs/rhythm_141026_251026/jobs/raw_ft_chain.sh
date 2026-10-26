#!/bin/bash
set -u
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
R=/media/Project/ECG/Model_Dong/ecgr_rhythm; J=$R/summary_111026/strip_eval/jobs
size=$1; init=$2
N=$R/npy_portal_mark10
until [ -f $N/manifest.json ] && python3 -c "import json,sys; m=json.load(open('$N/manifest.json')); sys.exit(0 if {'train','eval'} <= set(m['splits']) else 1)" 2>/dev/null && ! systemctl --user is-active -q ecgr-data-mark10; do sleep 60; done
tag=131026_rhythm_dual_${size}_raw
echo "[$(date +%T)] $size: fine-tune from $init"
ECGR_RHYTHM_NPY_DIR=$N ECGR_RHYTHM_CLASS_WEIGHTS=1,1,1,1,1 ECGR_RHYTHM_RUN_TAG=$tag ECGR_RHYTHM_CKPT_START_EPOCH=1 ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/ \
  python3 -u -m ecgr.rhythm train --model rhythm_dual_$size --init-from $init --lr 1e-4 --schedule cosine --sampler stratified --epochs 8 --patience 8 2>&1 | tr '\r' '\n' | grep --line-buffered -aE "windows|val_rhythm_f1 improved|did not improve|Traceback|Error|finished"
echo "[$(date +%T)] $size: epoch selection + bias calibration + rhythm_eval"
python3 -u $J/finalize_raw.py $size $R/$tag/checkpoints/dualunet_rhythm_$size 2>&1 | grep --line-buffered -E "epoch_|picked|bias|VAL|TEST|FINAL_RAW_DONE|Traceback"
echo "[$(date +%T)] $size: RAW_CHAIN_DONE"
