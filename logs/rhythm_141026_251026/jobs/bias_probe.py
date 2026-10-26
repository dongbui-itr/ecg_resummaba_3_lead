"""bias_probe.py ckpt ... : raw validation recall before/after the output-bias search (no saving)."""
import sys, json
import numpy as np
sys.path.insert(0, '/media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead')
from ecgr.training.train import setup_gpus; setup_gpus()
import tensorflow as tf
from ecgr.rhythm import strip_eval as SE, config as rc
from ecgr.rhythm import dualunet, model as _m  # noqa: F401
T = '/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval'
val = json.load(open(f'{T}/strip_val2_events.json'))
x, idx = SE.raw_windows(val, cache=f'{T}/raw_windows_val.npz')
K = rc.NUM_CLASSES
for ck in sys.argv[1:]:
    m = tf.keras.models.load_model(ck, compile=False)
    out = m.predict(x.astype(np.float32), batch_size=128, verbose=0)['rhythm']
    b = np.zeros(K); _, _, r = SE.raw_confusion(m, val, x, idx, bias=b, out=out); bm = SE.raw_margin(r); br = r
    print(ck.split('/')[-5], ck.split('/')[-1], 'raw ', ' '.join(f"{t} {r[t]:.1f}" for t in SE.REF_TYPES), f"{bm:+.2f}", flush=True)
    for it in range(2):
        for c in range(1, K):
            for v in np.round(np.arange(-3.0, 2.01, 0.25), 2):
                bb = b.copy(); bb[c] = v
                _, _, rr = SE.raw_confusion(m, val, x, idx, bias=bb, out=out); mm = SE.raw_margin(rr)
                if mm > bm + 1e-9: b, bm, br = bb, mm, rr
    print('   bias', b.round(2).tolist(), ' '.join(f"{t} {br[t]:.1f}" for t in SE.REF_TYPES), f"{bm:+.2f}", flush=True)
