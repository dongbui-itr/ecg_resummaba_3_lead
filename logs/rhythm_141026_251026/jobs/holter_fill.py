"""holter_fill.py key=ckpt ...: predict the missing long-Holter validation records, one model at a time."""
import os, sys, traceback
sys.path.insert(0, '/media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead')
os.environ.update(ECGR_RHYTHM_RUN_TAG='summary_111026', ECGR_PHYSIONET_DIR='/media/Project/ECG/PhysionetData/',
                  ECGR_RHYTHM_PREDICT_HOP='5', ECGR_RHYTHM_PREDICT_TTA='id,flip,swap', ECGR_RHYTHM_PREDICT_TAPER='0')
from ecgr.training.train import setup_gpus; setup_gpus()
import tensorflow as tf
from ecgr.rhythm import ec57
S = '/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/ec57'
for spec in sys.argv[1:]:
    key, ckpt = spec.split('=', 1)
    model = None
    for db in ('ltafdb', 'nsrdb', 'incartdb', 'svdb'):
        src, recs = ec57.physionet_records(db)
        ann = f'{S}/holterval_{key}/_ann/{db}'
        os.makedirs(ann, exist_ok=True)
        for r in recs:
            if os.path.exists(ec57.probs_path(ann, r)):
                continue
            model = model or tf.keras.models.load_model(ckpt, compile=False)
            try:
                *probs, fs, sl, hz, beats = ec57.predict_record_probs(model, os.path.join(src, r))
                ec57.save_probs(ann, r, *probs, fs, sl, hz, beats)
                print(key, db, r, 'ok', flush=True)
            except Exception:
                print(key, db, r, 'FAILED', flush=True); traceback.print_exc()
    print(key, 'FILL_DONE', flush=True)
