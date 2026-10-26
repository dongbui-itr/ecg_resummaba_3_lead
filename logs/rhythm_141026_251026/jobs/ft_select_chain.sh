#!/bin/bash
# after the P+ fine-tune: overlap-1 validation inference for epochs 2/4/6/8, decode tuning on each, pick by validation score
set -u
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
while systemctl --user is-active -q ecgr-ft-pplus; do sleep 60; done
E=/media/Project/ECG/Model_Dong/ecgr_rhythm/161026_rhythm_dual_1m_v4_pplus/checkpoints/dualunet_rhythm_1m/epochs
for e in 2 4 6 8; do
  ck=$E/epoch_0$e.keras; key=pplus_e$e
  [ -f $ck ] || { echo "no $ck"; continue; }
  echo "[$(date +%T)] $key: validation inference (overlap 1)"
  ECGR_RHYTHM_PREDICT_OVERLAP=1 /media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs/holter_val.sh $key $ck > /media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs/hv_$key.log 2>&1
  STRIP_VAL_EVENTS=/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/strip_val2_events.json ECGR_RHYTHM_PREDICT_OVERLAP=1 ECGR_RHYTHM_PREDICT_TAPER=0 python3 -u /media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs/val_predict.py $key $ck > /media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs/vp_$key.log 2>&1
  echo "[$(date +%T)] $key: decode tuning (background)"
  (cd /tmp && python3 -u /media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs/tune_pplus.py $key --sweeps 2 --random 40 --workers 12 > /media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs/tune_pplus_$key.log 2>&1) &
done
wait
echo "[$(date +%T)] SELECT_DONE"
python3 - <<'PY'
import json, glob, os
S='/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval'
for d in sorted(glob.glob(f'{S}/tune_pplus_*')):
    try:
        b=json.load(open(f'{d}/best.json')); base=json.load(open(f'{d}/base_metrics.json'))
        print(f"{os.path.basename(d):22s} tuned score {b['score']:8.2f} short {b['short']:6.1f} guard {b['guard']:5.1f} | base S+P {base['B|all|S|PP']:.1f} V+P {base['B|all|V|PP']:.1f} AF E+P {base['H|ltafdb|AFIB|E_PP']:.1f} AVB fph {base['FPH|AVB']:.3f}")
    except Exception as ex: print(d, ex)
PY
