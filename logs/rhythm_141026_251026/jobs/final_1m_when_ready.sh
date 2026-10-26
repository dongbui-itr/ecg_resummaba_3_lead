#!/bin/bash
# wait for the 1m chain, pick ft2_best vs ft2_last by validation shortfall, then final scoring
J=/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval/jobs; R=/media/Project/ECG/Model_Dong/ecgr_rhythm; T=$R/summary_111026/strip_eval
until grep -q CHAIN2_DONE $J/ft2_1m.log; do sleep 30; done
pick=$(python3 -c "
import json
a=json.load(open('$T/tuned_dual1m_ft2_best.json'))['validation']; b=json.load(open('$T/tuned_dual1m_ft2_last.json'))['validation']
print('last' if (b['met'], -b['shortfall']) > (a['met'], -a['shortfall']) else 'best')")
if [ "$pick" = last ]; then ck=$(ls $R/121026_rhythm_dual_1m_ftstrip2/checkpoints/dualunet_rhythm_1m/epochs/epoch_*.keras | sort | tail -1); else ck=$R/121026_rhythm_dual_1m_ftstrip2/checkpoints/dualunet_rhythm_1m/best_model.keras; fi
echo "picked ft2_$pick ($ck)"
cp $ck $T/dual1m_ft2_${pick}_checkpoint.keras
python3 -u $J/final_strip.py dual1m_ft2_$pick $T/dual1m_ft2_${pick}_checkpoint.keras $T/test_npz/dual1m_ft2_$pick $T/val_npz/dual1m_ft2_$pick $T/tuned_dual1m_ft2_$pick.json
