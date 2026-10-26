"""val_predict.py <key> <checkpoint>: strip_val inference -> strip_eval/val_npz/<key>"""
import sys, json, os
sys.path.insert(0, '/media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead')
os.environ.setdefault('ECGR_RHYTHM_PREDICT_HOP', '5'); os.environ.setdefault('ECGR_RHYTHM_PREDICT_TTA', 'id,flip,swap')
from ecgr.training.train import setup_gpus
setup_gpus()
from ecgr.rhythm import strip_eval as SE
W = '/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval'
key, ckpt = sys.argv[1:3]
ev = json.load(open(os.environ.get('STRIP_VAL_EVENTS', f'{W}/strip_val_events.json')))
SE.predict(ckpt, ev, f'{W}/val_npz/{key}')
print('PREDICT_DONE', key)
