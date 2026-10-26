"""final_strip.py <key> <checkpoint or -> <test npz dir> <val npz dir> <decode json>
Score one model ONCE on rhythm_eval with the decode chosen on validation (real epicmp), and on
validation for reference. Predicts the test records first when the npz dir is incomplete."""
import sys, json, os
sys.path.insert(0, '/media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead')
os.environ.setdefault('ECGR_RHYTHM_PREDICT_HOP', '5'); os.environ.setdefault('ECGR_RHYTHM_PREDICT_TTA', 'id,flip,swap')
from ecgr.rhythm import strip_eval as SE
key, ckpt, test_npz, val_npz, dec_json = sys.argv[1:6]
W = '/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval'
decode = json.load(open(dec_json))['decode']
test = SE.test_events()
if ckpt != '-' and (not os.path.isdir(test_npz) or len(os.listdir(test_npz)) < len(test)):
    from ecgr.training.train import setup_gpus; setup_gpus()
    SE.predict(ckpt, test, test_npz)
val = json.load(open(os.environ.get('STRIP_VAL_EVENTS', f'{W}/strip_val_events.json')))
out = {}
for name, evs, npz in (('validation', val, val_npz), ('rhythm_eval', test, test_npz)):
    tr = SE.all_tracks(evs, npz, decode)
    m = SE.score(tr, work=f'{W}/_work_final_{key}_{name}')
    out[name] = m
    txt = SE.format_report(m, f'{key} - {name}, decode tuned on validation ({os.path.basename(dec_json)})')
    open(f'{W}/final_{key}_{name}.md', 'w').write(txt)
    print(txt)
json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'sample_cm'} for k, v in out.items()},
          open(f'{W}/final_{key}.json', 'w'), indent=1, default=float)
print('FINAL_DONE', key)
