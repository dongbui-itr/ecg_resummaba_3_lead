"""source_confusion.py : raw per-step confusion (argmax, no TTA, no post-process) of each model on
every data source of npy_250hz_10s_3lead_10sec (eval split = held-out studies; test = rhythm_eval
whole windows) -> strip_eval/source_confusion.json"""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, '/media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead')
from ecgr.training.train import setup_gpus; setup_gpus()
import tensorflow as tf
from ecgr.rhythm import config as rc
from ecgr.rhythm.labels import to_current_classes
from ecgr.rhythm import dualunet, model as _m  # noqa: F401
R = '/media/Project/ECG/Model_Dong/ecgr_rhythm'; T = f'{R}/summary_111026/strip_eval'
N = os.environ.get('SC_NPY', f'{R}/npy_250hz_10s_3lead_10sec')
SPLITS = tuple(os.environ.get('SC_SPLITS', 'eval,test').split(','))
MODELS = {
    '1m_base': f'{R}/release_dualunet_1m_e26/checkpoint/best_model.keras',
    '100k_base': f'{R}/091026_rhythm_dual_small/checkpoints/dualunet_rhythm_100k/best_model.keras',
    '30k_base': f'{R}/091026_rhythm_dual_small/checkpoints/dualunet_rhythm_30k/best_model.keras',
    '1m_raw_ft': f'{T}/raw_1m_final.keras',
    '100k_raw_ft': f'{T}/raw_100k_final.keras',
    '30k_raw_ft': f'{T}/raw_30k_final.keras',
}
if sys.argv[1:]:                                  # key=ckpt ... : score these instead
    MODELS = dict(a.split('=', 1) for a in sys.argv[1:])
K = rc.NUM_CLASSES; IG = 255
out_path = os.environ.get('SC_OUT', f'{T}/source_confusion.json')
res = json.load(open(out_path)) if os.path.exists(out_path) else {}
for key, ck in MODELS.items():
    if key in res:
        continue
    m = tf.keras.models.load_model(ck, compile=False)
    cms = {}
    for split in SPLITS:
        shards = sorted(glob.glob(f'{N}/{split}/segments_*.npy'), key=lambda p: int(p.rsplit('_', 1)[1][:-4]))
        for p in shards:
            x = np.load(p); lab = np.load(p.replace('segments_', 'labels_samples_'))
            ev = json.load(open(p.replace('segments_', 'events_').replace('.npy', '.json')))
            prob = m.predict(x.astype(np.float32), batch_size=256, verbose=0)['rhythm']
            prob = to_current_classes(prob)
            pred = prob.argmax(-1)                                     # (n, steps)
            stride = rc.SEGMENT_SAMPLES // pred.shape[1]
            ref = lab[:, stride // 2::stride]
            srcs = np.array(['rhythm_eval' if split == 'test' else e[0] for e in ev])
            for s in np.unique(srcs):
                sel = srcs == s; r, q = ref[sel].ravel(), pred[sel].ravel(); ok = r != IG
                cm = np.bincount(r[ok].astype(np.int64) * K + q[ok], minlength=K * K).reshape(K, K)
                name = s if split == 'test' else s
                cms[name] = (np.array(cms.get(name, np.zeros((K, K), np.int64))) + cm).tolist()
            print(key, os.path.basename(p), flush=True)
    res[key] = dict(ckpt=ck, step_hz=pred.shape[1] / rc.OUTPUT_SECONDS, cm=cms)
    json.dump(res, open(out_path, 'w'))
print('SOURCE_CM_DONE', flush=True)
