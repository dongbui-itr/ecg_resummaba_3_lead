"""raw_eval.py <set: val|test> key=ckpt ... : strict raw protocol (one pass per mark window, no
post-processing) -> strip_eval/raw_<set>_<key>.json"""
import sys, os, json
sys.path.insert(0, '/media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead')
from ecgr.training.train import setup_gpus; setup_gpus()
import tensorflow as tf
from ecgr.rhythm import strip_eval as SE
from ecgr.rhythm import dualunet, model as _m  # noqa: F401  (registers the custom layers)
T = '/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval'
which = sys.argv[1]
events = json.load(open(f'{T}/strip_val2_events.json')) if which == 'val' else SE.test_events()
x, index = SE.raw_windows(events, cache=f'{T}/raw_windows_{which}.npz')
print(which, len(events), 'events', len(x), 'windows', flush=True)
for spec in sys.argv[2:]:
    key, ckpt = spec.split('=', 1)
    m = tf.keras.models.load_model(ckpt, compile=False)
    cm, rows, rec = SE.raw_confusion(m, events, x, index)
    json.dump(dict(key=key, ckpt=ckpt, set=which, rows=rows.tolist(), cm=cm.tolist(), recall=rec,
                   margin=SE.raw_margin(rec)), open(f'{T}/raw_{which}_{key}.json', 'w'), indent=1)
    print(key, ' '.join(f"{t} {rec[t]:.2f}" for t in SE.REF_TYPES), f"margin {SE.raw_margin(rec):+.2f}", flush=True)
print('RAW_DONE', flush=True)
