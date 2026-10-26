#!/bin/bash
# ref-guarded tunes with the 3-point V/S Se allowance (SE_TOL=3) for v4 and the beat fine-tune epochs
J=/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs; S=/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval
export SE_TOL=3
cd /tmp
for key in v4ov1 beatft_e2 beatft_e1; do python3 -u $J/tune_pplus.py $key --ref v4ov1_gate --sweeps 3 --random 80 --workers 8 --out $S/tune_pplus_${key}se3 > $J/tune_pplus_${key}se3.log 2>&1 & done; wait
for key in beatft_e4 beatft2_e1 beatft2_e2; do python3 -u $J/tune_pplus.py $key --ref v4ov1_gate --sweeps 3 --random 80 --workers 8 --out $S/tune_pplus_${key}se3 > $J/tune_pplus_${key}se3.log 2>&1 & done; wait
echo "[$(date +%T)] SE3_DONE"; SE_TOL=3 python3 $J/compare_candidates.py v4ov1_gate
