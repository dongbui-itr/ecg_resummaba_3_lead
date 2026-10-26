"""finalize_raw.py <size> <run checkpoints dir>: pick the epoch with the best raw validation margin,
calibrate the rhythm head's output bias on validation, bake it into the model, score rhythm_eval once."""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, '/media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead')
from ecgr.training.train import setup_gpus; setup_gpus()
import tensorflow as tf
from ecgr.rhythm import strip_eval as SE, config as rc
from ecgr.rhythm import dualunet, model as _m  # noqa: F401
T = '/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval'
size, ckdir = sys.argv[1:3]
val = json.load(open(f'{T}/strip_val2_events.json'))
xv, iv = SE.raw_windows(val, cache=f'{T}/raw_windows_val.npz')
# 1. every epoch on validation
res = []
for ck in sorted(glob.glob(f'{ckdir}/epochs/epoch_*.keras')):
    m = tf.keras.models.load_model(ck, compile=False)
    _, _, rec = SE.raw_confusion(m, val, xv, iv)
    res.append((SE.raw_margin(rec), ck, rec))
    print(os.path.basename(ck), ' '.join(f"{t} {rec[t]:.2f}" for t in SE.REF_TYPES), f"margin {SE.raw_margin(rec):+.2f}", flush=True)
best = max(res, key=lambda r: r[0])
print('picked', best[1], f'{best[0]:+.2f}', flush=True)
m = tf.keras.models.load_model(best[1], compile=False)
# 2. output-bias calibration on validation (coordinate search on log-prob offsets, SINUS = 0)
out = m.predict(xv, batch_size=128, verbose=0)['rhythm']
K = rc.NUM_CLASSES
def margin(b):
    _, _, rec = SE.raw_confusion(m, val, xv, iv, bias=b, out=out)
    return SE.raw_margin(rec), rec
b = np.zeros(K); bm, brec = margin(b)
for it in range(3):
    for c in range(1, K):
        for v in np.round(np.arange(-3.0, 2.01, 0.2), 2):
            bb = b.copy(); bb[c] = v
            mm, rr = margin(bb)
            if mm > bm + 1e-9:
                b, bm, brec = bb, mm, rr
print('bias', b.tolist(), 'val margin', f'{bm:+.2f}', ' '.join(f"{t} {brec[t]:.2f}" for t in SE.REF_TYPES), flush=True)
# 3. bake the bias into the rhythm head: logits + b  ==  softmax(conv(x) + bias + b)
layer = m.get_layer('rhythm')
w, bias = layer.get_weights()
layer.set_weights([w, bias + b.astype(bias.dtype)])
outp = f'{T}/raw_{size}_final.keras'
m.save(outp)
# 4. rhythm_eval once, raw protocol
test = SE.test_events()
xt, it_ = SE.raw_windows(test, cache=f'{T}/raw_windows_test.npz')
m2 = tf.keras.models.load_model(outp, compile=False)
_, _, recv = SE.raw_confusion(m2, val, xv, iv)
cm, rows, rect = SE.raw_confusion(m2, test, xt, it_)
json.dump(dict(size=size, epoch_ckpt=best[1], bias=b.tolist(), model=outp, val_recall=recv,
               val_margin=SE.raw_margin(recv), test_recall=rect, test_margin=SE.raw_margin(rect),
               test_rows=rows.tolist(), test_cm=cm.tolist(), epochs=[(r[1], r[0], r[2]) for r in res]),
          open(f'{T}/raw_{size}_final.json', 'w'), indent=1, default=float)
print('VAL ', ' '.join(f"{t} {recv[t]:.2f}" for t in SE.REF_TYPES), f"margin {SE.raw_margin(recv):+.2f}")
print('TEST', ' '.join(f"{t} {rect[t]:.2f}" for t in SE.REF_TYPES), f"margin {SE.raw_margin(rect):+.2f}")
print('FINAL_RAW_DONE')
