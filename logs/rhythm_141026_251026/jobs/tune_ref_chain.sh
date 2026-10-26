#!/bin/bash
# reference-guarded decode tuning (guards vs v4ov1_tight base) for the fine-tuned epochs, as their inference lands
R=/media/Project/ECG/Model_Dong/ecgr_rhythm; J=$R/summary_111026/strip_eval/jobs; S=$R/summary_111026/strip_eval
for e in 2 4 6 8; do
  key=pplus_e$e
  until [ "$(ls $S/val_npz/$key 2>/dev/null | wc -l)" -ge 4719 ] && grep -qE "VAL_INFER_DONE|PREDICT_DONE" $J/vp_$key.log 2>/dev/null; do sleep 60; done
  echo "[$(date +%T)] $key: reference-guarded tuning"
  (cd /tmp && python3 -u $J/tune_pplus.py $key --ref v4ov1_tight --sweeps 3 --random 60 --workers 10 --out $S/tune_pplus_${key}ref > $J/tune_pplus_${key}ref.log 2>&1)
done
echo "[$(date +%T)] REF_TUNE_DONE"
