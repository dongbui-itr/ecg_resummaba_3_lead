#!/bin/bash
# Cách 2: one decode per model that meets the strip targets AND does not fall below the base model's
# long-Holter validation bar. Then choose base vs fine-tuned on validation, score rhythm_eval once,
# re-decode EC57 with the chosen decode.
set -u
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
R=/media/Project/ECG/Model_Dong/ecgr_rhythm; S=$R/summary_111026; T=$S/strip_eval; J=$T/jobs
export TF_CPP_MIN_LOG_LEVEL=3 STRIP_VAL_EVENTS=$T/strip_val2_events.json
size=$1; base_key=$2; base_ckpt=$3; base_hann=$4; base_start=$5; ft_key=$6; ft_ckpt=$7; ft_start=$8; workers=$9
ft_hann=$S/ec57/holterval_$ft_key/_ann
log() { echo "[$(date +%T)] $size: $*"; }
for d in $base_hann $ft_hann; do
  until [ $(ls $d/ltafdb/*.npz $d/nsrdb/*.npz $d/incartdb/*.npz $d/svdb/*.npz 2>/dev/null | wc -l) -ge 42 ]; do sleep 60; done
done
log "Holter validation ready; bar from the base model (official decode)"
CUDA_VISIBLE_DEVICES="" python3 -m ecgr.rhythm.strip_tune make-bar $base_hann $T/holter_bar_$size.json $workers | tail -0
log "v2 strip inference for the base model"
python3 -u $J/val_predict.py $base_key $base_ckpt 2>&1 | grep -E "PREDICT_DONE|Traceback"
for pair in "$base_key:$base_hann:$base_start" "$ft_key:$ft_hann:$ft_start"; do
  IFS=: read k h st <<< "$pair"
  log "tune $k (strip_val2 + Holter bar)"
  CUDA_VISIBLE_DEVICES="" python3 -u -m ecgr.rhythm.strip_tune $T/val_npz/$k $T/tunedH_$k.json --events $STRIP_VAL_EVENTS --workers $workers --rounds 3 --random 200 --start $st --holter-ann $h --holter-bar $T/holter_bar_$size.json 2>&1 | grep -E "start:|target cells|below target|Holter validation below|^->|Traceback"
done
pick=$(python3 -c "
import json
def k(p):
    d=json.load(open(p)); return d['validation']['shortfall_safety']
a=k('$T/tunedH_$base_key.json'); b=k('$T/tunedH_$ft_key.json')
print('ft' if b < a else 'base')")
if [ $pick = ft ]; then key=$ft_key; ckpt=$ft_ckpt; tnpz=$T/test_npz/$ft_key; else key=$base_key; ckpt=$base_ckpt; tnpz=$S/ec57/$base_key/_ann/rhythm_eval; fi
log "picked $pick: $key"
echo "$pick $key $ckpt" > $T/pickH_$size.txt
log "final rhythm_eval (strip protocol, real epicmp)"
python3 -u $J/final_strip.py ${key}_H $ckpt $tnpz $T/val_npz/$key $T/tunedH_$key.json 2>&1 | grep -E "^## |target cells|below target|FINAL_DONE|Traceback"
echo "$key" > $T/pickH_${size}_key.txt
log "HOLTER_TUNE_DONE"
