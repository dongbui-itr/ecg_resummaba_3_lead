#!/bin/bash
# re-decode EC57 (5 DBs + bxb) with a Cách 2 decode from stored probabilities (symlinks), no inference
cd /media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead
export ECGR_RHYTHM_RUN_TAG=summary_111026 ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
tag=$1; dec=$2; shift 2      # remaining: db=srcdir pairs
S=/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/ec57
for pair in "$@"; do db=${pair%%=*}; src=${pair#*=}; mkdir -p $S/$tag/_ann/$db; for f in $src/*.npz; do ln -sf $f $S/$tag/_ann/$db/; done; done
python3 -u -c "
import json
from ecgr.rhythm import ec57
from ecgr.evaluation.report import summarize
d = json.load(open('$dec'))['decode']
ec57.run(None, '$tag', dbs=['mitdb','nstdb','escdb','ahadb','afdb'], skip_rhythm_eval=True, decode_only=True, decode=d)
summarize('$S/$tag/beats')
" 2>&1 | grep -E "mitdb .*E_Se=|escdb .*E_Se=|afdb .*E_Se=|beats .*Q_Se|Traceback"
echo EC57H_DONE
