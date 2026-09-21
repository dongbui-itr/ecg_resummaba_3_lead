#!/usr/bin/env python
"""Score a trained checkpoint on EC57 and the v4 beat-eval set, from one local command.

    ./evaluate.py --checkpoint checkpoints/resumamba_2m.keras --lead-mode native

This is the thin, self-contained front door to the same machinery `python -m ecgr ec57`
uses. It exists because that command is part of a pipeline: it writes into a RUN directory
named by ECGR_RUN_TAG, defaults to whichever checkpoint a training run happened to leave
behind, and scores the portal train/eval splits as well. For "score this file, here, now"
none of that helps. This script:

  * takes the checkpoint(s) explicitly - several means an ensemble, their softmax outputs
    averaged per step, exactly as the EC57 tables in README section 8 were produced;
  * writes everything under ./eval_results/<name>/ in the working directory, so nothing
    depends on a run tag or on where the training data lives;
  * checks the prerequisites (bxb on PATH, databases present, checkpoint shape) BEFORE
    spending an hour of GPU, because each of those failures otherwise surfaces as an empty
    report an hour in;
  * prints one table, and marks it against the acceptance targets.

Everything it measures is what bxb measures - the .ain predictions and the raw WFDB reports
are kept under the output directory, so any number here can be traced back to a record.
"""
import argparse
import csv
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Targets a run is accepted against. None = not applicable to that database.
# 'ge' is a lower bound on the metric, in percent.
TARGETS = {
    'mitdb': {'Q_Se': 99.95, 'Q_+P': 99.95, 'S_Se': 43.0, 'S_+P': 80.0},
    'dataset-v4-beat': {'S_Se': 88.0, 'S_+P': 92.0},
}
METRICS = ['Q_Se', 'Q_+P', 'V_Se', 'V_+P', 'S_Se', 'S_+P']


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', nargs='+', required=True,
                   help='one .keras file, or several to average their outputs (ensemble)')
    p.add_argument('--name', default=None,
                   help='output folder under --out-dir (default: the first checkpoint stem)')
    p.add_argument('--out-dir', default=os.path.join(os.getcwd(), 'eval_results'))
    p.add_argument('--dbs', nargs='*', default=None,
                   help='EC57 databases (default: all five). Empty = skip Physionet.')
    p.add_argument('--skip-v4', action='store_true', help='skip the v4 beat-eval set')
    p.add_argument('--lead-mode', choices=['auto', 'duplicate', 'native'], default='native',
                   help="how to fill the 3-lead input. 'native' (default) uses the record's "
                        "own leads - every EC57 database has two, and reading both is what "
                        "the 3-lead model is for; 'duplicate' repeats the annotated lead, "
                        "the strict single-lead control; 'auto' picks per record")
    p.add_argument('--min-run', type=int, default=1,
                   help='drop decoded runs shorter than this many 20 ms steps (default 1)')
    p.add_argument('--s-boost', type=float, default=1.0,
                   help='multiply the S probability before argmax. Calibrate on portal data '
                        'only - tuning it on mitdb makes the benchmark self-scoring')
    p.add_argument('--max-records', type=int, default=None,
                   help='first N records per database (smoke test)')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--gpu', default=None, help='CUDA_VISIBLE_DEVICES for this run')
    p.add_argument('--bxb-only', action='store_true',
                   help='re-score the predictions already under --out-dir, no inference')
    p.add_argument('--whole-record', action='store_true',
                   help='v4: score the whole 60 s strip, not just the reviewed window')
    p.add_argument('--baseline', default=None,
                   help='an ec57_summary.csv to diff against, e.g. a previous run of this '
                        'script; every metric is then shown with its delta')
    return p.parse_args(argv)


def preflight(args):
    """Fail on the things that otherwise fail silently an hour later."""
    problems = []
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        problems.append(
            f"tensorflow is not importable by {sys.executable}. This project runs in the "
            f"conda env `beat`; invoke it with that interpreter, e.g.\n"
            f"       ~/miniconda3/envs/beat/bin/python evaluate.py ...")
    for path in args.checkpoint:
        if not os.path.exists(path):
            problems.append(f"checkpoint not found: {path}")
    if not args.bxb_only and not all(shutil.which(t) for t in ('bxb', 'sumstats')):
        problems.append("bxb / sumstats are not on PATH - install the WFDB applications")

    from ecgr import config
    dbs = config.EC57_DBS if args.dbs is None else args.dbs
    missing = [d for d in dbs if not os.path.isdir(os.path.join(config.PHYSIONET_DIR, d))]
    if missing:
        problems.append(f"databases not under {config.PHYSIONET_DIR}: {', '.join(missing)} "
                        f"(set ECGR_PHYSIONET_DIR)")
    if not args.skip_v4:
        for name, src in config.PORTAL_EVAL_SETS.items():
            if not os.path.isdir(src):
                problems.append(f"v4 beat-eval set not found: {src} (set ECGR_DATA_DIR, or "
                                f"pass --skip-v4)")
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        raise SystemExit(2)
    return dbs


def load_baseline(path):
    if not path:
        return {}
    with open(path) as f:
        return {r['db']: r for r in csv.DictReader(f)}


def print_table(rows, baseline=None):
    """One table, with targets marked and (optionally) deltas against a baseline."""
    baseline = baseline or {}
    width = max([len(r['db']) for r in rows] + [8])
    head = f"{'database':{width}s} {'records':>7s} " + ' '.join(f"{m:>7s}" for m in METRICS)
    print("\n" + head)
    print('-' * len(head))

    missed = []
    for row in rows:
        cells = []
        for m in METRICS:
            raw = row.get(m, '-')
            try:
                value = float(raw)
            except (TypeError, ValueError):
                cells.append(f"{'-':>7s}")
                continue
            mark = ''
            target = TARGETS.get(row['db'], {}).get(m)
            if target is not None:
                mark = '*' if value >= target else '!'
                if value < target:
                    missed.append((row['db'], m, value, target))
            base = baseline.get(row['db'], {}).get(m)
            try:
                delta = value - float(base)
                cells.append(f"{value:6.2f}{mark}" if not baseline else
                             f"{value:6.2f}{mark}{delta:+6.2f}")
            except (TypeError, ValueError):
                cells.append(f"{value:6.2f}{mark}")
        print(f"{row['db']:{width}s} {row.get('records', '?'):>7s} " + ' '.join(cells))

    print("\n* = meets the acceptance target, ! = below it")
    if missed:
        print("below target:")
        for db, m, value, target in missed:
            print(f"  {db:16s} {m:5s} {value:6.2f}  (target {target})")
    else:
        print("every applicable target met")


def main(argv=None):
    args = parse_args(argv)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    # Before importing TensorFlow: XLA needs a libdevice path or every stage dies at its
    # first compiled op (see ecgr/xla.py).
    from ecgr import xla
    xla.ensure_libdevice()

    dbs = preflight(args)
    from ecgr import config
    from ecgr.evaluation import ec57, report

    name = args.name or os.path.splitext(os.path.basename(args.checkpoint[0]))[0]
    out_dir = os.path.abspath(os.path.join(args.out_dir, name))
    os.makedirs(out_dir, exist_ok=True)

    # The evaluation stages read these from config; point them at this local output instead
    # of at a run directory, and apply the decode options for this invocation.
    config.EC57_DIR = os.path.dirname(out_dir)
    config.DECODE_MIN_RUN_STEPS = args.min_run

    print(f"checkpoint   : {', '.join(args.checkpoint)}"
          f"{'  (ensemble: outputs averaged)' if len(args.checkpoint) > 1 else ''}")
    print(f"output       : {out_dir}")
    print(f"decoding     : lead-mode {args.lead_mode}, min-run {args.min_run}, "
          f"s-boost {args.s_boost}")
    print(f"databases    : {', '.join(dbs) if dbs else '(none)'}"
          f"{'' if args.skip_v4 else ' + dataset-v4-beat'}")

    rows = ec57.run(args.checkpoint if len(args.checkpoint) > 1 else args.checkpoint[0],
                    tag=name, dbs=dbs or None, max_records=args.max_records,
                    s_boost=args.s_boost, bxb_only=args.bxb_only,
                    skip_physionet=not dbs, skip_portal=args.skip_v4,
                    mark_window=not args.whole_record, lead_mode=args.lead_mode,
                    splits=())          # the portal train/eval splits are a training-run tool
    if not rows:
        print("\nno bxb reports were produced - see the messages above", file=sys.stderr)
        return 1

    print_table(rows, load_baseline(args.baseline))
    print(f"\nsummary  : {os.path.join(out_dir, 'ec57_summary.csv')}")
    print(f"reports  : {out_dir}/<database>/*_QRS_report_line.out")
    print(f"predictions kept in {out_dir}/_ann/ - rerun with --bxb-only to re-score them")
    return 0


if __name__ == '__main__':
    sys.exit(main())
