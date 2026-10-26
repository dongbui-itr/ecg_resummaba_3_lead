"""eval_dataset_v4.py <key> <checkpoint> : score a rhythm checkpoint on
/media/MegaDataSet/DATA_4TINYML/Holter_report_strip/dataset-eval/v4 with the deployed pipeline
(hop 5 s + TTA inference, OFFICIAL decode incl. the beat post-process):

  beat-eval-dataset   (5,227 strips, reference .atr): bxb (reviewed mark window = the beat team's
                      protocol, and whole strip) + beat confusion N/S/V/missed/false of the
                      written .bti beats inside the mark (150 ms match)
  rhythm-eval-dataset (4,678 events = those event ids of rhythm_eval): epicmp per class inside the
                      reviewer mark (strip_eval.run_epicmp) + sample confusion + event confusion
-> summary_111026/ec57/<key>/dataset_eval_v4/{result.json, beats/, rhythm/}   (resumable)"""
import sys, os, glob, json, shutil
import numpy as np
sys.path.insert(0, '/media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead')
if __name__ == '__main__':
    from ecgr.training.train import setup_gpus; setup_gpus()
    import tensorflow as tf, wfdb
    from ecgr.rhythm import config as rc, strip_eval as SE
    from ecgr.rhythm import ec57, dualunet, model as _m  # noqa: F401
    from ecgr.evaluation import bxb
    from ecgr.evaluation.report import parse_report
    key, ckpt = sys.argv[1:3]
    DEC = json.load(open(os.environ['TUNED_DECODE']))['decode'] if os.environ.get('TUNED_DECODE') else None
    print('decode:', 'official' if DEC is None else os.environ['TUNED_DECODE'], '| overlap', os.environ.get('ECGR_RHYTHM_PREDICT_OVERLAP', 'default'), flush=True)
    EV = '/media/MegaDataSet/DATA_4TINYML/Holter_report_strip/dataset-eval/v4'
    OUT = f'/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/ec57/{key}/dataset_eval_v4'
    os.makedirs(OUT, exist_ok=True)
    res_path = f'{OUT}/result.json'; res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    # ================= beats =================
    bdir = f'{EV}/beat-eval-dataset'; ann = f'{OUT}/beats/ann'; os.makedirs(ann, exist_ok=True)
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{bdir}/*.hea'))
    todo = [n for n in names if not os.path.exists(ec57.probs_path(ann, n))]
    if todo:
        model = tf.keras.models.load_model(ckpt, compile=False)
        for i, n in enumerate(todo, 1):
            *probs, fs, sig_len, step_hz, beats = ec57.predict_record_probs(model, f'{bdir}/{n}')
            ec57.save_probs(ann, n, *probs, fs, sig_len, step_hz, beats)
            if i % 500 == 0 or i == len(todo): print(f'beats: {i}/{len(todo)} predicted', flush=True)
    for n in names: ec57.write_hypothesis(ann, n, [], DEC)           # .bti beats + .rhi
    # bxb, two protocols
    work = f'{OUT}/beats/_work'; shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
    for n in names:
        for ext in ('hea', 'dat', 'atr'):
            os.symlink(f'{bdir}/{n}.{ext}', f'{work}/{n}.{ext}')
        os.symlink(f'{ann}/{n}.{rc.RHYTHM_BEAT_AI_EXTENSION}', f'{work}/{n}.{rc.RHYTHM_BEAT_AI_EXTENSION}')
    bx = {}
    for label, script in (('mark_window', bxb.SCRIPT_MARK_WINDOW), ('whole_strip', bxb.SCRIPT_SHORT)):
        root = f'{OUT}/beats/bxb_{label}'
        bxb.run_bxb('beat_eval_v4', work, root, 'atr', rc.RHYTHM_BEAT_AI_EXTENSION, script=script)
        line = f'{root}/beat_eval_v4/beat_eval_v4_QRS_report_line.out'
        bx[label] = parse_report(line) if os.path.exists(line) else {}
        print('bxb', label, bx[label], flush=True)
    # beat confusion of the written beats inside the mark (N/S/V/missed ; false detections)
    M = {'N': 0, 'S': 1, 'V': 2}; tol = rc.BEAT_MATCH_TOLERANCE_SECONDS
    cm = np.zeros((4, 4), np.int64)
    for n in names:
        h = wfdb.rdheader(f'{bdir}/{n}'); fs = h.fs
        cm_ = {c.split(':')[0].strip(): c.split(':')[1].strip() for c in h.comments if ':' in c}
        a, b = int(cm_.get('startMarkSample', 0)), int(cm_.get('stopMarkSample', h.sig_len))
        ref = wfdb.rdann(f'{bdir}/{n}', 'atr'); hyp = wfdb.rdann(f'{ann}/{n}', rc.RHYTHM_BEAT_AI_EXTENSION)
        rs = [(s, rc.BEAT_SYMBOL_TO_CLASS[y] - 1) for s, y in zip(ref.sample, ref.symbol) if y in rc.BEAT_SYMBOL_TO_CLASS and a <= s < b]
        hs = [(s, M[y]) for s, y in zip(hyp.sample, hyp.symbol) if y in M and a <= s < b]
        used = set()
        for s, c in rs:
            best, bj = tol * fs + 1, None
            for j, (t, d) in enumerate(hs):
                if j not in used and abs(t - s) < best: best, bj = abs(t - s), j
            if bj is None: cm[c, 3] += 1
            else: used.add(bj); cm[c, hs[bj][1]] += 1
        for j, (t, d) in enumerate(hs):
            if j not in used: cm[3, d] += 1
    res['beats'] = dict(records=len(names), bxb=bx, confusion=cm.tolist(), labels=['N', 'S', 'V', 'missed/false'])
    json.dump(res, open(res_path, 'w'), indent=1)
    # ================= rhythm =================
    ev_ids = {os.path.basename(d) for d in glob.glob(f'{EV}/rhythm-eval-dataset/*/*')}
    events = [e for e in SE.test_events() if e['name'].split('_', 1)[1] in ev_ids]
    print('rhythm events', len(events), 'of', len(ev_ids), flush=True)
    rann = f'{OUT}/rhythm/ann'; os.makedirs(rann, exist_ok=True)
    SE.predict(ckpt, events, rann)
    tr = SE.all_tracks(events, rann, decode=DEC)
    m = SE.score(tr, work=f'{OUT}/rhythm/_epicmp')
    res['rhythm'] = dict(events=len(tr), types={t: sum(e['type'] == t for e, *_ in tr) for t in SE.REF_TYPES},
                         epicmp=m['epicmp'], sample_cm=m['sample_cm'], sample_rows=m['sample_rows'],
                         event_cm=m['event_cm'], event_none=m['event_none'], event_stats=m['event'],
                         ref_types=SE.REF_TYPES, pred_names=SE.PRED_NAMES)
    json.dump(res, open(res_path, 'w'), indent=1)
    print('epicmp', m['epicmp'], flush=True); print('EVAL_V4_DONE', key, flush=True)
