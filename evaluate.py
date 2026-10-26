#!/usr/bin/env python3
"""Score the rhythm checkpoint on AAMI EC57 (epicmp rhythm + bxb beats) and the rhythm holdout.

    python3 evaluate.py          # edit the CONFIG block first

No command-line arguments on purpose: the CONFIG block is the record of what was measured. Every
relative path is read from THIS file's folder (the project root), so it runs from anywhere.

What one run does (ecgr.rhythm.ec57.run):

    per record    inference with 10 s windows overlapping by PREDICT_OVERLAP_SECONDS (the
                  probabilities of every step averaged over the windows covering it, x the TTA
                  variants) -> <out>/_ann/<db>/<rec>.npz -> decode + beat post-processing
                  (DECODE: None = the config defaults, or a decode json) -> the annotations
                  (.rhi episodes, .bti beats, one hypothesis file per rhythm class) + references
    per database  the WFDB tools only: epicmp per class -> <out>/<db>/<db>_<CLASS>_report_line.out,
                  bxb -> <out>/beats/<db>/<db>_QRS_report_line.out
    summary       summarize_ec57.py -> <out>/summary/ec57_summary.{csv,md,xlsx} (+ per record),
                  side by side with BASELINE when one is given

<out> = <WORK_DIR>/<RUN_TAG>/ec57/<TAG>. DECODE_ONLY = True re-decodes and re-scores the npz of
an earlier run under the same TAG without the model (minutes instead of ~40). The web viewer
(webservice/app.py) reads the same <out> folders.
"""
import json
import os
import sys
import time

# ===========================================================================
# CONFIG - edit this block
# ===========================================================================

# The checkpoint, and the decode / post-processing applied to its output.
#   DECODE = None  -> the config defaults (ecgr/rhythm/config.py DECODE_* / BEAT_PP_*): the
#                     OFFICIAL setting of this checkpoint (checkpoints/<ckpt>/manifest.json)
#   DECODE = '<json>' -> e.g. 'checkpoints/dualunet_rhythm_1m_e26/decode/overlap1_af_context.json'
CHECKPOINT = 'checkpoints/dualunet_rhythm_1m_e26/best_model.keras'
TRAIN_CONFIG = 'checkpoints/dualunet_rhythm_1m_e26/train_config.json'
DECODE = None

# Output: <WORK_DIR>/<RUN_TAG>/ec57/<TAG>/. TAG None = ec57_ov<overlap>_<yymmdd_HHMM>.
# eval_results/ is git-ignored.
WORK_DIR = 'eval_results'
RUN_TAG = 'dualunet_rhythm_1m_e26'
TAG = None

# Data: the PhysioNet root (mitdb/ nstdb/ escdb/ ahadb/ afdb/) and what to score.
PHYSIONET_DIR = '/media/Project/ECG/PhysionetData/'
DATABASES = ['mitdb', 'nstdb', 'escdb', 'ahadb', 'afdb']     # [] = skip PhysioNet
SCORE_RHYTHM_EVAL = True                                     # the 4,909-event portal holdout

# --- whole-record inference (ecgr/rhythm/predict.py) --------------------------------------
# PREDICT_OVERLAP_SECONDS = seconds shared by two consecutive 10 s windows: 0 = back-to-back,
# 5 = half (every step averaged over 2 windows, the official setting), 9.5 = windows 0.5 s
# apart (~19 windows per step, ~10x the inference). The step between windows is rounded to
# the 40 ms probability grid; the effective values are printed at start-up.
PREDICT_OVERLAP_SECONDS = 5
PREDICT_TTA = 'id,flip,swap'            # test-time variants averaged: lead sign flip, lead 1<->2
PREDICT_TAPER = 0.0                     # position weighting of overlapping windows (0 = off)

# --- run control --------------------------------------------------------------------------
MAX_RECORDS = None              # first N records per database = smoke test
DECODE_ONLY = False             # re-decode + re-score the npz already under TAG, no inference
INCLUDE_EXCLUDED = False        # also score the paced records excluded by EC57 (mitdb 102/104/107/217)
GPU = None                      # e.g. '0'; None = leave CUDA_VISIBLE_DEVICES alone
GPU_MEMORY_MB = None            # cap this process (a training job shares the card); None = growth
WRITE_SUMMARY = True            # summary tables under <out>/summary
BASELINE = None                 # an earlier <out> folder to put side by side in the summary

# ===========================================================================
# End of CONFIG
# ===========================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def resolve(path):
    """CONFIG paths are read relative to the project root (this file's folder)."""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def export_environment():
    """CONFIG -> the ECGR_* variables ecgr.rhythm.config reads at ITS import, so this must run
    before anything from ecgr is imported (a wrapper may change the CONFIG constants after
    importing this file and before main())."""
    if 'ecgr.rhythm.config' in sys.modules:
        raise RuntimeError("ecgr.rhythm.config was imported before evaluate.main(): the CONFIG "
                           "block would be ignored")
    if GPU is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ['ECGR_PHYSIONET_DIR'] = PHYSIONET_DIR
    os.environ['ECGR_RHYTHM_WORK_DIR'] = resolve(WORK_DIR)
    os.environ['ECGR_RHYTHM_RUN_TAG'] = RUN_TAG
    os.environ['ECGR_RHYTHM_TRAIN_CONFIG'] = resolve(TRAIN_CONFIG)
    os.environ['ECGR_RHYTHM_PREDICT_OVERLAP'] = str(PREDICT_OVERLAP_SECONDS)
    os.environ['ECGR_RHYTHM_PREDICT_TTA'] = PREDICT_TTA
    os.environ['ECGR_RHYTHM_PREDICT_TAPER'] = str(PREDICT_TAPER)


def limit_gpu(memory_mb):
    """Cap this process's GPU memory so a training job on the same card keeps running; the
    ecgr setup_gpus memory-growth call then becomes a no-op instead of an error."""
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
    """Fail on what would otherwise fail silently after the model is loaded."""
    problems = []
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        problems.append(f"tensorflow is not importable by {sys.executable}")
    if not DECODE_ONLY and not os.path.exists(resolve(CHECKPOINT)):
        problems.append(f"checkpoint not found: {resolve(CHECKPOINT)}")
    if DECODE and not os.path.exists(resolve(DECODE)):
        problems.append(f"DECODE not found: {resolve(DECODE)}")
    if not 0 <= float(PREDICT_OVERLAP_SECONDS) < 10:
        problems.append(f"PREDICT_OVERLAP_SECONDS must be in [0, 10), got {PREDICT_OVERLAP_SECONDS}")
    if BASELINE and not os.path.isdir(resolve(BASELINE)):
        problems.append(f"BASELINE is not a folder: {resolve(BASELINE)}")
    from ecgr.evaluation import bxb, epicmp
    if not (bxb.have_wfdb_tools() and epicmp.have_wfdb_tools()):
        problems.append("bxb / epicmp / sumstats are not on PATH - install the WFDB applications")
    for db in DATABASES:
        d = os.path.join(PHYSIONET_DIR, db)
        if not os.path.isdir(d):
            problems.append(f"database not on disk: {d}")
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        raise SystemExit(2)


def describe(rc, out_dir, decode):
    from ecgr.rhythm.predict import window_hop_samples
    step = window_hop_samples() / rc.SECOND_SAMPLES
    print(f"checkpoint   : {resolve(CHECKPOINT)}")
    print(f"decode       : {DECODE or '(config defaults - official)'}")
    if decode:
        print("               " + json.dumps(decode))
    print(f"output       : {out_dir}")
    print(f"scoring      : {', '.join(DATABASES) if DATABASES else '(no physionet)'}"
          f"{' + rhythm_eval' if SCORE_RHYTHM_EVAL else ''}")
    print(f"inference    : window overlap {rc.SEGMENT_SECONDS - step:g} s (step {step:g} s, "
          f"~{rc.SEGMENT_SECONDS / step:.0f} windows averaged per step), TTA "
          f"{','.join(rc.PREDICT_TTA)}, taper {rc.PREDICT_TAPER_FLOOR:g}")
    print(f"run control  : max-records {MAX_RECORDS}, decode-only {DECODE_ONLY}, "
          f"include-excluded {INCLUDE_EXCLUDED}, GPU cap {GPU_MEMORY_MB}\n")


def write_summary(out_dir):
    import summarize_ec57
    argv = ([resolve(BASELINE)] if BASELINE else []) + [out_dir, '--records',
                                                        '--out', os.path.join(out_dir, 'summary')]
    summarize_ec57.main(argv)


def main():
    export_environment()
    preflight()
    if GPU_MEMORY_MB:
        limit_gpu(GPU_MEMORY_MB)

    from ecgr.rhythm import config as rc
    from ecgr.rhythm import ec57, tune
    assert rc.PREDICT_OVERLAP_SECONDS == float(PREDICT_OVERLAP_SECONDS), \
        "ecgr.rhythm.config did not pick up the CONFIG block"

    tag = TAG or time.strftime(f'ec57_ov{PREDICT_OVERLAP_SECONDS:g}_%y%m%d_%H%M')
    out_dir = os.path.join(rc.EC57_DIR, tag)
    decode = tune.load_decode(resolve(DECODE)) if DECODE else None
    describe(rc, out_dir, decode)

    t0 = time.time()
    rows = ec57.run(resolve(CHECKPOINT), tag, dbs=DATABASES or None, max_records=MAX_RECORDS,
                    decode_only=DECODE_ONLY, skip_physionet=not DATABASES,
                    skip_rhythm_eval=not SCORE_RHYTHM_EVAL,
                    include_excluded=INCLUDE_EXCLUDED, decode=decode)
    if not rows:
        print("\nno reports were produced - see the messages above", file=sys.stderr)
        return 1
    with open(os.path.join(out_dir, 'evaluate_config.json'), 'w') as f:
        json.dump(dict(checkpoint=resolve(CHECKPOINT), decode=DECODE, decode_dict=decode,
                       databases=DATABASES, rhythm_eval=SCORE_RHYTHM_EVAL,
                       overlap_s=PREDICT_OVERLAP_SECONDS, tta=PREDICT_TTA, taper=PREDICT_TAPER,
                       max_records=MAX_RECORDS, decode_only=DECODE_ONLY,
                       include_excluded=INCLUDE_EXCLUDED, minutes=round((time.time() - t0) / 60, 1)),
                  f, indent=2, default=str)
    if WRITE_SUMMARY:
        print()
        write_summary(out_dir)
    print(f"\nresults : {out_dir}")
    print(f"viewer  : python3 {os.path.join(HERE, 'webservice', 'app.py')}   "
          f"(EVAL_ROOT = {os.path.join(resolve(WORK_DIR), RUN_TAG, 'ec57')})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
