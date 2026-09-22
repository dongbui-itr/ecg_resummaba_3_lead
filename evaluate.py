#!/usr/bin/env python
"""Score a trained checkpoint on EC57 and the v4 beat-eval set.

    ~/miniconda3/envs/beat/bin/python evaluate.py        # the env with tensorflow

Everything this script does is decided by the CONFIG block below - edit it, run it. There
are no command-line arguments on purpose: an evaluation is a record of what was measured,
and a block of named constants in the file is one. A shell line scrolls out of the history
and leaves no trace of whether that table came from one lead or three.

It is the local front door to the same machinery `python -m ecgr ec57` uses, minus the
pipeline: no ECGR_RUN_TAG, no guessing which checkpoint a training run left behind, no
scoring of the portal train/eval splits (those are a training-run tool). Every path in
CONFIG is read relative to THIS FILE, so it writes under <repo>/OUT_DIR whatever directory it
is started from; it checks its prerequisites before spending GPU time, and marks the result
against the acceptance targets.

The .ain predictions and the raw WFDB reports are kept, so any number here can be traced
back to a record.
"""
import csv
import os
import shutil
import sys

# ===========================================================================
# CONFIG - edit this block
# ===========================================================================

# The checkpoint(s) to score. Several = an ENSEMBLE: their softmax outputs are averaged per
# step, which is how the ensemble rows in README section 8 were produced.
CHECKPOINTS = [
    # 'checkpoints/resumamba_2m.keras',
    'checkpoints/resumamba_1m.keras',
    # 'checkpoints/resumamba_100k.keras',
    # 'checkpoints/resumamba_30k.keras'
]

# Output folder name under OUT_DIR. None = the first checkpoint's file stem.
NAME = None
OUT_DIR = 'eval_results'

# EC57 databases to score. [] = skip Physionet entirely.
DATABASES = ['mitdb', 'nstdb', 'escdb', 'ahadb', 'afdb']
SCORE_V4 = True                 # the held-out v4 beat-eval set (5,227 records)

# --- what actually reaches the model -------------------------------------------------
# Two separate questions. LEAD_MODE: how many of the record's REAL leads to use.
#   'native' - all of them (the default). Every EC57 database has two, and reading both is
#              what the 3-lead model is for: on resumamba_2m it moves mitdb S from
#              45.79/61.17 to 56.87/65.64 - both measured with LEAD_FILL 'duplicate', README
#              section 8f - because record 232's non-premature APCs are only visible as a P
#              wave on V5.
#   'single' - only the annotated lead, the strict single-lead control. 'duplicate' is
#              accepted as its deprecated alias.
#   'auto'   - 'native' only where the record ALREADY has 3 real leads, 'single' otherwise.
#              All five EC57 databases have two, so there 'auto' means 'single'.
LEAD_MODE = 'native'
# LEAD_FILL: what occupies the channels the record has no lead for.
#   'zero'      - silence (the default). What an electrode coming off looks like, and what
#                 training produces on purpose (config.AUGMENT_LEAD_DROP_PROB). Measured on
#                 resumamba_2m at LEAD_MODE 'native' against 'duplicate': S +P +1.9 to +4.1
#                 on mitdb/nstdb/escdb, but S sensitivity -5.4 to -5.9 on mitdb/escdb and
#                 nstdb V +P -6.1. It is a trade, not a free win - the argument for taking
#                 it is in README section 3.
#   'duplicate' - repeat the annotated lead. The other case training covers
#                 (config.AUGMENT_LEAD_DUPLICATE_PROB), and how every EC57 table up to
#                 2026-09-20 was produced.
LEAD_FILL = 'zero'

# --- decoding ------------------------------------------------------------------------
MIN_RUN = 1                     # drop decoded runs shorter than this many 20 ms steps
# Multiplies the S probability before the argmax, i.e. slides the model along its own
# sensitivity / positive-predictivity curve. CALIBRATE ON PORTAL DATA ONLY - tuning it
# against mitdb makes the EC57 benchmark self-scoring. It saturates: on the best ensemble,
# 1.0 -> 0.45 moves mitdb S +P only 68.3 -> 71.7 while sensitivity falls 56.0 -> 45.0.
S_BOOST = 1.0

# --- run control ---------------------------------------------------------------------
MAX_RECORDS = None              # first N records per database; a small int = smoke test
BATCH_SIZE = None               # None = config.BATCH_SIZE
GPU = None                      # e.g. '0'; None = leave CUDA_VISIBLE_DEVICES alone
BXB_ONLY = False                # True = re-score the predictions already under OUT_DIR
WHOLE_RECORD = False            # v4: score the whole 60 s strip, not the reviewed window
BASELINE = None                 # an ec57_summary.csv to diff against, e.g. a previous run

# ===========================================================================
# End of CONFIG
# ===========================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Targets the run is accepted against, per database. A database with no entry is not marked.
TARGETS = {
    'mitdb': {'Q_Se': 99.95, 'Q_+P': 99.95, 'S_Se': 43.0, 'S_+P': 80.0},
    'dataset-v4-beat': {'S_Se': 88.0, 'S_+P': 92.0},
}
METRICS = ['Q_Se', 'Q_+P', 'V_Se', 'V_+P', 'S_Se', 'S_+P']


def resolve(path):
    """CONFIG paths are read relative to this file, so the script runs from anywhere."""
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def preflight(checkpoints=None, databases=None, score_v4=None, bxb_only=None):
    """Fail on the things that otherwise fail silently an hour later.

    Every one of these has cost a wasted run at some point: an interpreter without
    tensorflow, bxb not installed (the shell drivers write no report and say nothing), a
    database directory that is not where config points, a mistyped checkpoint path.
    """
    checkpoints = CHECKPOINTS if checkpoints is None else checkpoints
    databases = DATABASES if databases is None else databases
    score_v4 = SCORE_V4 if score_v4 is None else score_v4
    bxb_only = BXB_ONLY if bxb_only is None else bxb_only

    problems = []
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        problems.append(
            f"tensorflow is not importable by {sys.executable}. This project runs in the "
            f"conda env `beat`; invoke it with that interpreter, e.g.\n"
            f"       ~/miniconda3/envs/beat/bin/python evaluate.py")

    if not checkpoints:
        problems.append("CHECKPOINTS is empty - name at least one .keras file")
    for path in checkpoints:
        if not os.path.exists(resolve(path)):
            problems.append(f"checkpoint not found: {path}")
    if LEAD_MODE not in ('native', 'single', 'auto', 'duplicate'):
        problems.append(f"LEAD_MODE must be native/single/auto ('duplicate' is the "
                        f"deprecated alias of single), got {LEAD_MODE!r}")
    if LEAD_FILL not in ('zero', 'duplicate'):
        problems.append(f"LEAD_FILL must be zero/duplicate, got {LEAD_FILL!r}")
    if not bxb_only and not all(shutil.which(t) for t in ('bxb', 'sumstats')):
        problems.append("bxb / sumstats are not on PATH - install the WFDB applications")

    from ecgr import config
    missing = [d for d in databases
               if not os.path.isdir(os.path.join(config.PHYSIONET_DIR, d))]
    if missing:
        problems.append(f"databases not under {config.PHYSIONET_DIR}: {', '.join(missing)} "
                        f"(set ECGR_PHYSIONET_DIR)")
    if score_v4:
        for name, src in config.PORTAL_EVAL_SETS.items():
            if not os.path.isdir(src):
                problems.append(f"v4 beat-eval set not found: {src} (set ECGR_DATA_DIR, or "
                                f"set SCORE_V4 = False)")
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        raise SystemExit(2)
    return list(databases)


def load_baseline(path=None):
    path = BASELINE if path is None else path
    if not path:
        return {}
    with open(resolve(path)) as f:
        return {row['db']: row for row in csv.DictReader(f)}


def print_table(rows, baseline=None):
    """One table: every metric, targets marked, deltas against a baseline when given."""
    baseline = baseline or {}
    width = max([len(r['db']) for r in rows] + [8])
    head = f"{'database':{width}s} {'records':>7s} " + ' '.join(f"{m:>7s}" for m in METRICS)
    print("\n" + head)
    print('-' * len(head))

    missed = []
    for row in rows:
        cells = []
        for metric in METRICS:
            try:
                value = float(row.get(metric, '-'))
            except (TypeError, ValueError):
                cells.append(f"{'-':>7s}")       # bxb writes '-' where a class has no beats
                continue
            target = TARGETS.get(row['db'], {}).get(metric)
            mark = ''
            if target is not None:
                mark = '*' if value >= target else '!'
                if value < target:
                    missed.append((row['db'], metric, value, target))
            try:
                delta = value - float(baseline.get(row['db'], {}).get(metric))
                cells.append(f"{value:6.2f}{mark}{delta:+6.2f}" if baseline
                             else f"{value:6.2f}{mark}")
            except (TypeError, ValueError):
                cells.append(f"{value:6.2f}{mark}")
        print(f"{row['db']:{width}s} {row.get('records', '?'):>7s} " + ' '.join(cells))

    print("\n* = meets the acceptance target, ! = below it")
    if missed:
        print("below target:")
        for db, metric, value, target in missed:
            print(f"  {db:16s} {metric:5s} {value:6.2f}  (target {target})")
    else:
        print("every applicable target met")


def main():
    if GPU is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    # Before importing TensorFlow: XLA needs a libdevice path or the first compiled op dies
    # (see ecgr/xla.py).
    from ecgr import xla
    xla.ensure_libdevice()

    databases = preflight()
    from ecgr import config
    from ecgr.evaluation import ec57

    checkpoints = [resolve(p) for p in CHECKPOINTS]
    name = NAME or os.path.splitext(os.path.basename(checkpoints[0]))[0]
    out_dir = os.path.abspath(os.path.join(resolve(OUT_DIR), name))
    os.makedirs(out_dir, exist_ok=True)

    # The evaluation stages read these from config; point them at this local output rather
    # than at a run directory, and apply this run's decode options.
    config.EC57_DIR = os.path.dirname(out_dir)
    config.DECODE_MIN_RUN_STEPS = MIN_RUN

    print(f"checkpoint   : {', '.join(CHECKPOINTS)}"
          f"{'  (ensemble: outputs averaged)' if len(checkpoints) > 1 else ''}")
    print(f"output       : {out_dir}")
    print(f"leads        : mode {LEAD_MODE}, fill {LEAD_FILL}")
    print(f"decoding     : min-run {MIN_RUN}, s-boost {S_BOOST}")
    print(f"scoring      : {', '.join(databases) if databases else '(no physionet)'}"
          f"{' + dataset-v4-beat' if SCORE_V4 else ''}")

    rows = ec57.run(checkpoints if len(checkpoints) > 1 else checkpoints[0],
                    tag=name, dbs=databases or None, max_records=MAX_RECORDS,
                    s_boost=S_BOOST, bxb_only=BXB_ONLY,
                    skip_physionet=not databases, skip_portal=not SCORE_V4,
                    mark_window=not WHOLE_RECORD, lead_mode=LEAD_MODE,
                    fill_mode=LEAD_FILL,
                    splits=())          # the portal train/eval splits are a training tool
    if not rows:
        print("\nno bxb reports were produced - see the messages above", file=sys.stderr)
        return 1

    print_table(rows, load_baseline())
    print(f"\nsummary  : {os.path.join(out_dir, 'ec57_summary.csv')}")
    print(f"reports  : {out_dir}/<database>/*_QRS_report_line.out")
    print(f"predictions kept in {out_dir}/_ann/ - set BXB_ONLY = True to re-score them")
    return 0


if __name__ == '__main__':
    sys.exit(main())
