#!/usr/bin/env python3
"""Score a rhythm checkpoint (the one a training run is writing, or any other) on EC57.

    python3 eval_rhythm.py          # edit the CONFIG block first

The rhythm counterpart of evaluate.py: no command-line arguments on purpose - the CONFIG
block is the record of what was measured. It is the front door of the same machinery
`python3 -m ecgr.rhythm ec57` uses, in the order the project's rules prescribe:

  1. snapshot   the checkpoint is COPIED first. The file a running training job overwrites
                at the next best epoch cannot change under the evaluation.
  2. validation the held-out ltafdb / nsrdb / incartdb / svdb records are predicted and the
                decoding + beat post-processing are chosen on them (ecgr.rhythm.tune) - never
                on the EC57 databases. Reused when TUNED_DECODE points to an existing file.
  3. EC57       mitdb / afdb / escdb: inference -> npz -> decode with the chosen parameters
                (per-beat post-processing from the analysis of the production code, defects
                fixed: ecgr/rhythm/beats.py) -> epicmp per rhythm class (AAMI EC57 episode and
                duration Se / +P) -> bxb on the beat hypotheses (QRS / V / S Se and +P).
  4. report     the class x (Duration | Episode) table against the product targets, the bxb
                table, an A/B without the beat stage, and a per-record audit CSV so every
                number traces back to a record.

RECORDS = ['mitdb/201', ...] scores just those records (prediction, decoded episodes and beat
symbols printed, epicmp + bxb on that record alone) - the quickest look at what the
post-processing does to one record.
"""
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import time

# ===========================================================================
# CONFIG - edit this block
# ===========================================================================

WORK_DIR = '/media/Project/ECG/Model_Dong/ecgr_rhythm'
# The run whose checkpoint is scored. Output goes under <WORK_DIR>/<RUN_TAG>/ec57/<TAG>.
RUN_TAG = '071026_rhythm_dual_1m_s1'
CHECKPOINT = f'{WORK_DIR}/{RUN_TAG}/checkpoints/dualunet_rhythm_1m/best_model.keras'
TRAIN_CONFIG = f'{WORK_DIR}/{RUN_TAG}/checkpoints/dualunet_rhythm_1m/train_config.json'
TAG = 'ec57_dual_s1'                 # EC57 output folder
VALIDATION_TAG = 'val_dual_s1'       # validation output folder (inference + tuning)

PHYSIONET_DIR = '/media/Project/ECG/PhysionetData/'
DATABASES = ['mitdb', 'afdb', 'escdb']      # EC57 databases ([] = skip)
SCORE_RHYTHM_EVAL = False                   # the 4,909-event portal holdout as well (slow)
RECORDS = None                              # e.g. ['mitdb/201', 'afdb/04015'] = single-record mode

# --- validation tuning ------------------------------------------------------------------
TUNE_ON_VALIDATION = True                   # choose the decode / beat PP on validation first
VALIDATION_DBS = ['ltafdb', 'nsrdb', 'incartdb', 'svdb']
TUNED_DECODE = None                         # an existing tuned_decode.json = skip the tuning
AF_FP_BUDGET = 0.4                          # false AF episodes per non-AF hour allowed
TUNE_WORKERS = 12
PRIOR_ALPHA = 0.5                           # prior correction when not tuning (0 = off)

# --- inference-time layer settings (from the XAI 'experiment' sweep: python3 -m ecgr.rhythm xai)
# The dual U-Net's QRST cancellation window / peak threshold are layer settings, not weights:
# {'post': 0.35} writes a variant of the snapshot with that setting and scores it. Tune and
# score the variant like any checkpoint (validation first) - never adopt it from EC57.
QRST_OVERRIDES = {}                         # e.g. {'pre': 0.10, 'post': 0.35, 'min_prob': 0.3}

# --- run control --------------------------------------------------------------------------
DECODE_ONLY = False                         # re-decode + re-score stored npz, no inference
VALIDATION_DECODE_ONLY = False              # same for the validation run
AB_NO_BEAT_PP = True                        # also report EC57 without the beat stage
MAX_RECORDS = None                          # first N records per database = smoke test
GPU_MEMORY_MB = 6000                        # cap: the training job shares the card (None = growth)
PREDICT_HOP_SECONDS = 5                     # window hop (10 = no overlap); TTA id,flip,swap
PREDICT_TTA = 'id,flip,swap'                # 'swap' is a no-op for the dual U-Net (lead pooling)
PREDICT_TAPER = 0.0                         # position weighting of overlapping windows (0 = off;
                                            # e.g. 0.1 with PREDICT_HOP_SECONDS = 2, see config)

# ===========================================================================
# End of CONFIG
# ===========================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def export_environment():
    """CONFIG -> the ECGR_* variables ecgr.rhythm.config reads at ITS import. Called from
    main() before anything from ecgr is imported, so a wrapper that changes the CONFIG
    constants after importing this file (eval_rhythm.TAG = ...) is honoured."""
    if 'ecgr.rhythm.config' in sys.modules:
        raise RuntimeError("ecgr.rhythm.config was imported before eval_rhythm.main(): the "
                           "CONFIG block would be ignored")
    os.environ.setdefault('ECGR_PHYSIONET_DIR', PHYSIONET_DIR)
    os.environ['ECGR_RHYTHM_WORK_DIR'] = WORK_DIR
    os.environ['ECGR_RHYTHM_RUN_TAG'] = RUN_TAG
    os.environ['ECGR_RHYTHM_TRAIN_CONFIG'] = TRAIN_CONFIG
    os.environ['ECGR_RHYTHM_PREDICT_HOP'] = str(PREDICT_HOP_SECONDS)
    os.environ['ECGR_RHYTHM_PREDICT_TTA'] = PREDICT_TTA
    os.environ['ECGR_RHYTHM_PREDICT_TAPER'] = str(PREDICT_TAPER)


class Tee(io.TextIOBase):
    """stdout duplicated into the run's log file."""
    def __init__(self, path):
        self.f = open(path, 'a')
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()
        return len(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def limit_gpu(memory_mb):
    """Cap this process's GPU memory so a training job on the same card keeps running; the
    ecgr setup_gpus memory-growth call is then a no-op instead of an error."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and memory_mb:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=int(memory_mb))])
        orig = tf.config.experimental.set_memory_growth

        def quiet(device, enable):
            try:
                orig(device, enable)
            except ValueError:
                pass
        tf.config.experimental.set_memory_growth = quiet


def preflight():
    from ecgr.evaluation import bxb, epicmp
    problems = []
    if not os.path.exists(CHECKPOINT):
        problems.append(f"checkpoint not found: {CHECKPOINT}")
    if not (bxb.have_wfdb_tools() and epicmp.have_wfdb_tools()):
        problems.append("bxb / epicmp / sumstats are not on PATH (WFDB applications)")
    for db in DATABASES:
        d = os.path.join(os.environ['ECGR_PHYSIONET_DIR'], db)
        if not os.path.isdir(d):
            problems.append(f"database not on disk: {d}")
    if TUNE_ON_VALIDATION and not TUNED_DECODE:
        from ecgr.rhythm import config as rc
        for db in VALIDATION_DBS:
            d = rc.PHYSIONET_TRAIN_DBS.get(db, {}).get('dir')
            if not d or not os.path.isdir(d):
                problems.append(f"validation database not on disk: {db} ({d})")
    if TUNED_DECODE and not os.path.exists(TUNED_DECODE):
        problems.append(f"TUNED_DECODE not found: {TUNED_DECODE}")
    if problems:
        for p in problems:
            print("preflight:", p)
        sys.exit(2)


def snapshot_checkpoint(out_dir):
    """Copy the checkpoint under the output folder and record what was copied."""
    dst = os.path.join(out_dir, 'checkpoint_snapshot.keras')
    if DECODE_ONLY and os.path.exists(dst):
        return dst
    shutil.copy2(CHECKPOINT, dst)
    h = hashlib.md5()
    with open(dst, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    info = dict(source=CHECKPOINT, md5=h.hexdigest(), bytes=os.path.getsize(dst),
                source_mtime=time.strftime('%Y-%m-%d %H:%M:%S',
                                           time.localtime(os.path.getmtime(CHECKPOINT))),
                copied=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(os.path.join(out_dir, 'checkpoint_snapshot.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(f"checkpoint snapshot: {dst}\n  from {CHECKPOINT} (saved {info['source_mtime']}, "
          f"md5 {info['md5'][:12]})")
    return dst


def apply_layer_overrides(ckpt, out_dir):
    """QRST_OVERRIDES -> a variant checkpoint next to the snapshot (the setting is part of the
    saved layer config, so every later load - ec57.run included - sees it)."""
    if not QRST_OVERRIDES:
        return ckpt
    import keras
    from ecgr.rhythm import dualunet  # noqa: F401
    model = keras.models.load_model(ckpt, compile=False)
    layer = model.get_layer('qrst_cancel')
    for k, v in QRST_OVERRIDES.items():
        if not hasattr(layer, k):
            raise ValueError(f"qrst_cancel has no setting {k!r}")
        setattr(layer, k, float(v))
    tag = '_'.join(f"{k}{v}" for k, v in sorted(QRST_OVERRIDES.items()))
    path = os.path.join(out_dir, f"checkpoint_qrst_{tag}.keras")
    model.save(path)
    print(f"layer overrides {QRST_OVERRIDES} -> {path}")
    return path


def default_decode():
    from ecgr.rhythm.ec57 import prior_scale
    return dict(class_scale=prior_scale(PRIOR_ALPHA, TRAIN_CONFIG))


def choose_decode(ckpt, ec57_root):
    """The decode dict for the EC57 run: tuned on validation, loaded, or the defaults."""
    from ecgr.rhythm import ec57, tune
    if TUNED_DECODE:
        d = tune.load_decode(TUNED_DECODE)
        print(f"decode from {TUNED_DECODE}")
        return d
    if not TUNE_ON_VALIDATION:
        print("decode: project defaults (rc.DECODE_* / rc.BEAT_PP_*)")
        return default_decode()
    val_out = os.path.join(ec57_root, VALIDATION_TAG)
    tuned_path = os.path.join(val_out, 'tuned_decode.json')
    print(f"\n########## validation: {VALIDATION_DBS} -> {val_out} ##########")
    ec57.run(ckpt, VALIDATION_TAG, dbs=VALIDATION_DBS, skip_rhythm_eval=True,
             decode_only=VALIDATION_DECODE_ONLY, max_records=MAX_RECORDS,
             decode=default_decode())
    print(f"\n########## tuning on validation ##########")
    return tune.tune(os.path.join(val_out, '_ann'), budget=AF_FP_BUDGET, workers=TUNE_WORKERS,
                     train_config=TRAIN_CONFIG, max_records=MAX_RECORDS, out_path=tuned_path)


def describe_decode(decode):
    d = {k: v for k, v in decode.items() if k != 'class_scale'}
    scale = decode.get('class_scale') or {}
    if scale:
        d['class_scale'] = {k: round(v, 3) for k, v in scale.items()}
    return json.dumps(d, indent=1)


# ---------------------------------------------------------------------------
# Single-record mode
# ---------------------------------------------------------------------------

def score_records(model, out_dir, decode):
    from ecgr.rhythm import config as rc
    from ecgr.rhythm import ec57, wfdb_ann
    from ecgr.evaluation import epicmp, report
    by_db = {}
    for item in RECORDS:
        db, name = item.split('/', 1)
        by_db.setdefault(db, []).append(name)
    for db, names in by_db.items():
        src_dir = os.path.join(os.environ['ECGR_PHYSIONET_DIR'], db)
        ann_dir = ec57.annotation_dir(out_dir, db)
        os.makedirs(ann_dir, exist_ok=True)
        classes = ec57.classes_for_db(db)
        for name in names:
            path = os.path.join(src_dir, name)
            if not DECODE_ONLY or ec57.load_probs(ann_dir, name) is None:
                t0 = time.time()
                *probs, fs, sig_len, step_hz, beats = ec57.predict_record_probs(model, path)
                ec57.save_probs(ann_dir, name, *probs, fs, sig_len, step_hz, beats)
                print(f"\n{db}/{name}: {sig_len / fs:.0f} s predicted in {time.time() - t0:.0f} s")
            episodes = ec57.write_hypothesis(ann_dir, name, classes, decode)
            ref = ec57.write_physionet_reference(src_dir, ann_dir, name, classes)
            beats = ec57.load_beats(ann_dir, name)
            print(f"  reference episodes (not SINUS): " + ', '.join(
                f"{e['rhythm']} {e['start']:.0f}-{e['stop']:.0f}" for e in ref
                if e['rhythm'] != 'SINUS') or '  reference: all SINUS')
            print(f"  predicted episodes (not SINUS):")
            for e in episodes:
                if e['rhythm'] not in ('SINUS',):
                    print(f"    {e['start']:8.1f}-{e['stop']:8.1f}  {e['rhythm']:6s} p={e['prob']:.2f}")
            if beats is not None:
                _, sym = ec57.decode_record(*ec57.load_probs(ann_dir, name)[:2],
                                            ec57.load_probs(ann_dir, name)[4], beats, decode)
                counts = {s: int((sym == c).sum()) for s, c in (('N', 1), ('S', 2), ('V', 3))}
                print(f"  beats: {len(beats['t'])} picked, symbols after PP {counts}")
        linker = ec57.physionet_linker(src_dir)
        ec57.score_classes(db, out_dir, names, classes, linker, epicmp.SCRIPT_FULL)
        ec57._print_db_rows(out_dir, db)
        try:
            ec57.score_beats(db, out_dir, names, linker, src_dir)
        except Exception as e:
            print(f"  beat scoring skipped: {e}")
    report.summarize_episodes(out_dir, classes=rc.EC57_CLASSES, dbs=list(by_db),
                              target=ec57.TARGET_TABLE)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def acceptance(rows):
    """Mark every target cell: * meets the product F1, ! below it."""
    from ecgr.rhythm.ec57 import TARGET_TABLE
    from ecgr.evaluation.report import _num
    by = {(r['db'], r['class']): r for r in rows}
    lines, below = [], []
    for (db, cls), t in TARGET_TABLE.items():
        r = by.get((db, cls))
        if not r:
            continue
        for key, kind in (('D', 'Duration'), ('E', 'Episode')):
            se, pp = t[key]
            tf1 = 2 * se * pp / (se + pp)
            f1 = _num(r.get(f'{key}_F1'))
            mark = '?' if f1 is None else ('*' if f1 >= tf1 else '!')
            lines.append(f"  {db:6s} {cls:5s} {kind:9s} F1 {str(r.get(f'{key}_F1')):>5s}{mark} "
                         f"(target {tf1:.1f} = Se {se} / +P {pp})")
            if mark == '!':
                below.append((db, cls, kind, f1, tf1))
    print("\nacceptance against the product table (* meets, ! below):")
    print('\n'.join(lines))
    if below:
        print("below target: " + ', '.join(f"{db} {c} {k} {f:.1f} < {t:.1f}"
                                            for db, c, k, f, t in below))


def per_record_audit(out_dir, dbs):
    """One CSV with every record's epicmp line per class (+ its bxb line) - the trace from
    the Gross numbers back to the records that produced them."""
    from ecgr.evaluation.report import EPISODE_REPORT_SUFFIX
    path = os.path.join(out_dir, 'records_audit.csv')
    rows = []
    for db in dbs:
        d = os.path.join(out_dir, db)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(EPISODE_REPORT_SUFFIX):
                continue
            cls = f[len(db) + 1:-len(EPISODE_REPORT_SUFFIX)]
            with open(os.path.join(d, f)) as fh:
                for line in fh:
                    t = line.split()
                    if len(t) >= 11 and t[0] not in ('Record', 'Gross', 'Average', 'Summary'):
                        rows.append(dict(db=db, cls=cls, record=t[0], TPs=t[1], FN=t[2],
                                         TPp=t[3], FP=t[4], ESe=t[5], EPP=t[6], DSe=t[7],
                                         DPP=t[8], ref_dur=t[9], test_dur=t[10]))
    if rows:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"per-record audit ({len(rows)} rows) -> {path}")
    return rows


def worst_records(rows, top=8):
    """The records with the most false-positive seconds per class - where +P is lost."""
    def seconds(s):
        try:
            if ':' in s:
                parts = [float(x) for x in s.split(':')]
                return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
            return float(s)
        except ValueError:
            return 0.0
    by = {}
    for r in rows:
        fp = max(0.0, seconds(r['test_dur']) - seconds(r['ref_dur']))
        by.setdefault((r['db'], r['cls']), []).append((fp, int(r['FP']), r['record']))
    print("\nrecords with the most excess predicted duration (test - ref, s) per class:")
    for key in sorted(by):
        worst = sorted(by[key], reverse=True)[:top]
        worst = [w for w in worst if w[0] > 0 or w[1] > 0]
        if worst:
            print(f"  {key[0]:6s} {key[1]:5s} " + ', '.join(f"{n} (+{fp:.0f}s, {e} FP ep)"
                                                        for fp, e, n in worst))


def main():
    export_environment()
    import tensorflow as tf
    limit_gpu(GPU_MEMORY_MB)
    from ecgr.rhythm import config as rc
    assert rc.PREDICT_HOP_SECONDS == float(PREDICT_HOP_SECONDS) and \
        ','.join(rc.PREDICT_TTA) == PREDICT_TTA and rc.PREDICT_TAPER_FLOOR == float(PREDICT_TAPER)
    from ecgr.rhythm import ec57
    from ecgr.evaluation import report

    preflight()
    out_dir = os.path.join(rc.EC57_DIR, TAG)
    os.makedirs(out_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(out_dir, 'eval_rhythm.log'))
    print(f"===== eval_rhythm {time.strftime('%Y-%m-%d %H:%M:%S')} -> {out_dir} =====")
    print(f"hop {rc.PREDICT_HOP_SECONDS:g} s, TTA {','.join(rc.PREDICT_TTA)}, taper "
          f"{rc.PREDICT_TAPER_FLOOR:g} (as ecgr.rhythm.config resolved them), GPU cap "
          f"{GPU_MEMORY_MB} MB")
    ckpt = apply_layer_overrides(snapshot_checkpoint(out_dir), out_dir)

    decode = choose_decode(ckpt, rc.EC57_DIR)
    with open(os.path.join(out_dir, 'decode_used.json'), 'w') as f:
        json.dump(decode, f, indent=2)
    print(f"\ndecode used for EC57:\n{describe_decode(decode)}")

    if RECORDS:
        from ecgr.training.train import setup_gpus
        setup_gpus()
        model = tf.keras.models.load_model(ckpt, compile=False)
        score_records(model, out_dir, decode)
        return

    print(f"\n########## EC57: {DATABASES} -> {out_dir} ##########")
    rows = ec57.run(ckpt, TAG, dbs=DATABASES, max_records=MAX_RECORDS, decode_only=DECODE_ONLY,
                    skip_physionet=not DATABASES, skip_rhythm_eval=not SCORE_RHYTHM_EVAL,
                    decode=decode)
    acceptance(rows)
    beat_rows = report.summarize(os.path.join(out_dir, 'beats'))
    audit = per_record_audit(out_dir, DATABASES)
    if audit:
        worst_records(audit)

    if AB_NO_BEAT_PP and decode.get('beat_pp', rc.DECODE_BEAT_POSTPROCESS) and DATABASES:
        print(f"\n########## A/B: the same decode WITHOUT the beat stage ##########")
        nobeat = dict(decode, beat_pp=False)
        ab = ec57.score_decode(out_dir, nobeat, DATABASES)
        report.print_episode_table(list(ab.values()), classes=rc.EC57_CLASSES, dbs=DATABASES)
        with open(os.path.join(out_dir, 'rhythm_ec57_summary_nobeat.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=report.EPISODE_COLUMNS, extrasaction='ignore')
            w.writeheader()
            w.writerows(ab.values())
        ec57.score_decode(out_dir, decode, DATABASES)        # leave the files as reported
        print("\nbeat stage effect (F1 with - without):")
        by = {(r['db'], r['class']): r for r in rows}
        for key, r0 in sorted(ab.items()):
            r1 = by.get(key)
            if r1:
                d = [(m, report._num(r1.get(m)), report._num(r0.get(m))) for m in ('E_F1', 'D_F1')]
                print(f"  {key[0]:6s} {key[1]:5s} " + '  '.join(
                    f"{m} {a if a is not None else '-'} vs {b if b is not None else '-'}"
                    f" ({a - b:+.1f})" if a is not None and b is not None else f"{m} -"
                    for m, a, b in d))
    print(f"\n===== done {time.strftime('%Y-%m-%d %H:%M:%S')} - log: "
          f"{os.path.join(out_dir, 'eval_rhythm.log')} =====")


if __name__ == '__main__':
    main()
