"""Pick a checkpoint by BEAT-level scoring on portal-eval, not by step-level F1.

`ecgr train` with SAVE_EVERY_EPOCH keeps every epoch from CKPT_START_EPOCH on under
<ckpt>/epochs/. This scores each of them with bxb on the deterministic portal-eval sample
(evaluation/ec57.score_portal_split) and picks the one with the highest S F1 among those
that regress no Q/V/S sensitivity or positive predictivity against a reference report - by
default the run's own step-F1 pick, so the answer to "did beat-level selection change
anything?" is in the table. The reference can be any bxb report (e.g. another run's).

Shipping caveat, learned the hard way on run 260917_3lead: a candidate that passes here can
still regress on an EC57 database whose rhythm regime portal does not stress (mitdb record
213, sinus tachycardia). This is a selection step, not a shipping criterion - the full EC57
comes after it, and it is the full EC57 that decides.
"""
import glob
import json
import os
import shutil

import keras

from .. import config, models
from ..evaluation import ec57, report
from .train import setup_gpus

METRICS = ('Q_Se', 'Q_+P', 'V_Se', 'V_+P', 'S_Se', 'S_+P')


def numbers(path):
    row = report.parse_report(path)
    out = {}
    for m in METRICS:
        try:
            out[m] = float(row[m])
        except (KeyError, ValueError, TypeError):
            out[m] = None
    return out


def s_f1(n):
    se, pp = n.get('S_Se'), n.get('S_+P')
    return 2 * se * pp / (se + pp) if se and pp and se + pp else 0.0


def score_candidates(paths, select_root, records=None, min_run=None):
    """bxb on portal-eval for every checkpoint path; returns one dict per candidate.

    A candidate whose report already exists under select_root/<name>/ is not re-predicted,
    so an interrupted selection resumes where it stopped.
    """
    records = config.PORTAL_SPLIT_RECORDS if records is None else records
    if min_run is not None:
        config.DECODE_MIN_RUN_STEPS = min_run
    table = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        done = os.path.join(select_root, name, 'portal-eval',
                            'portal-eval_QRS_report_line.out')
        if os.path.exists(done):
            rep = done
        else:
            model = ec57.load_checkpoints(path)
            rep = ec57.score_portal_split(model, 'eval', os.path.join(select_root, name),
                                          max_records=records)
            keras.backend.clear_session()
        n = numbers(rep) if rep else {m: None for m in METRICS}
        table.append({'name': name, 'path': path, **n, 'S_F1': s_f1(n)})
    return table


def choose(table, reference, tolerance=None):
    """Mark feasibility against `reference` (a metrics dict) and return the winner."""
    tol = config.REFINE_TOLERANCE_PP if tolerance is None else tolerance
    for row in table:
        row['regressions'] = [m for m in METRICS
                              if reference.get(m) is not None and row.get(m) is not None
                              and row[m] < reference[m] - tol]
        row['feasible'] = not row['regressions']
    feasible = [r for r in table if r['feasible']]
    pool = feasible or table
    return max(pool, key=lambda r: r['S_F1']), len(feasible)


def print_table(table, reference, winner, ref_label='reference'):
    print(f"\n{'candidate':16s} " + ' '.join(f"{m:>6s}" for m in METRICS) + f" {'S_F1':>6s}  status")
    print(f"{ref_label:16s} " + ' '.join(f"{reference[m]:6.2f}" if reference.get(m) is not None
                                          else f"{'-':>6s}" for m in METRICS)
          + f" {s_f1(reference):6.2f}")
    for r in table:
        vals = ' '.join(f"{r[m]:6.2f}" if r.get(m) is not None else f"{'-':>6s}" for m in METRICS)
        status = ('WINNER' if r is winner else '') + \
                 ('' if r['feasible'] else f"  regresses {','.join(r['regressions'])}")
        print(f"{r['name']:16s} {vals} {r['S_F1']:6.2f}  {status}")


def select_epochs(model_name, epochs_dir=None, reference_report=None, records=None,
                  tolerance=None, min_run=None, out_name='selected_by_bxb.keras'):
    """Score <ckpt>/epochs/*.keras (or `epochs_dir`) and copy the winner next to them."""
    setup_gpus()
    keras_name = models.keras_name(model_name)
    epochs_dir = epochs_dir or os.path.join(config.CHECKPOINT_DIR, keras_name, 'epochs')
    paths = sorted(glob.glob(os.path.join(epochs_dir, '*.keras')))
    if not paths:
        raise FileNotFoundError(f"no .keras under {epochs_dir}")

    select_root = os.path.join(config.EC57_DIR, f'{model_name}_select')
    table = score_candidates(paths, select_root, records=records, min_run=min_run)

    if reference_report is None:
        # the run's own step-F1 pick, scored on the same sample, is the natural reference
        from .. import checkpoints
        step_pick = checkpoints.best_checkpoint(keras_name)
        ref_rows = score_candidates([step_pick], select_root, records=records, min_run=min_run)
        reference, ref_label = ref_rows[0], 'step-F1 pick'
    else:
        reference, ref_label = numbers(reference_report), 'reference'

    winner, n_feasible = choose(table, reference, tolerance)
    print_table(table, reference, winner, ref_label)

    out = os.path.join(epochs_dir, out_name)
    shutil.copy(winner['path'], out)
    with open(os.path.join(epochs_dir, 'selection.json'), 'w') as f:
        json.dump({'model': model_name, 'reference': {**reference, 'label': ref_label},
                   'winner': winner, 'feasible': n_feasible, 'candidates': table,
                   'records': records or config.PORTAL_SPLIT_RECORDS,
                   'min_run': config.DECODE_MIN_RUN_STEPS}, f, indent=2)
    print(f"\n{out_name} <- {winner['name']} (S F1 {s_f1(reference):.2f} -> {winner['S_F1']:.2f}, "
          f"feasible {n_feasible}/{len(table)})\n-> {out}")
    return out, table
