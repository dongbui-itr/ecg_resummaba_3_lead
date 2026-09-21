"""Train the temporal refinement head on a frozen base, then pick the epoch that regresses
nothing on portal-eval.

Two steps, both under <CHECKPOINT_DIR>/<model>/refined/:

  train   the best base checkpoint of this run is frozen, models/refine.attach_refinement
          puts the head on it, and every epoch is saved as epoch_NN.keras - epoch_00 being
          the untrained head, i.e. the base model itself. The step-level confusion report is
          written per epoch like any training run.

  select  each epoch is scored with bxb on the deterministic 5,000-record portal-eval sample
          (evaluation/ec57.score_portal_split, the same sample the base was scored on) and
          the winner is the epoch with the highest S F1 AMONG THOSE whose Q, V and S
          sensitivity and positive predictivity are all >= the base's, within a tolerance
          for scoring noise. Because epoch_00 is the base, that set is never empty. If no
          trained epoch qualifies, the base is what gets picked, and selection.json says so.

Selection runs on portal-eval and not on the beat-eval set or mitdb on purpose: those are
the holdouts the result is reported on, and choosing on them would make them self-scoring.
"""
import glob
import json
import os
import shutil

import keras
import tensorflow as tf

from .. import checkpoints, config, models
from ..data import pipeline
from ..evaluation import ec57, report, step_metrics
from ..models.refine import attach_refinement, head_parameters
from .callbacks import WeightedF1Checkpoint
from .losses import LOSSES
from .train import JIT_COMPILE, model_dirs, setup_gpus

METRICS = ('Q_Se', 'Q_+P', 'V_Se', 'V_+P', 'S_Se', 'S_+P')


def refined_dir(keras_name, tag=None):
    """<CHECKPOINT_DIR>/<model>/refined[_<tag>]: one folder per head configuration."""
    return os.path.join(config.CHECKPOINT_DIR, keras_name, 'refined' + (f'_{tag}' if tag else ''))


def train_refinement(model_name, base_checkpoint=None, epochs=None, lr=None,
                     batch_size=None, db_names=None, class_weights=None, tag=None, mode=None):
    """Attach and train the head. Returns the directory holding epoch_NN.keras.

    `class_weights` are the loss weights for the HEAD, default config.REFINE_CLASS_WEIGHTS -
    not the base's. The base's [0.3, 1, 2, 2.5] exist to fight the None/beat imbalance the
    head never sees (p_None is fixed), and their 2.5x on S turned the head into a threshold
    shift: on portal-eval every trained epoch of the 2m head raised S_Se and lowered S_+P.
    """
    setup_gpus()
    keras_name = models.keras_name(model_name)
    epochs = config.REFINE_EPOCHS if epochs is None else epochs
    lr = config.REFINE_LEARNING_RATE if lr is None else lr
    batch_size = batch_size or config.BATCH_SIZE
    class_weights = list(config.REFINE_CLASS_WEIGHTS if class_weights is None else class_weights)
    mode = config.REFINE_MODE if mode is None else mode

    base_checkpoint = base_checkpoint or checkpoints.best_checkpoint(keras_name)
    print(f"base         : {base_checkpoint}")
    base = keras.models.load_model(base_checkpoint, compile=False)
    model = attach_refinement(base, width=config.REFINE_WIDTH, blocks=config.REFINE_BLOCKS,
                              state_dim=config.REFINE_STATE_DIM,
                              kernel_len=config.REFINE_KERNEL_LEN, mode=mode)
    print(f"mode         : {mode} ({'N<->S only, p_V and p_None fixed' if mode == 's_only' else 'N/V/S, p_None fixed'})")
    print(f"head         : {head_parameters(model):,} trainable params on a frozen "
          f"{base.count_params():,}-param base ({config.REFINE_BLOCKS} SSM blocks, kernel "
          f"{config.REFINE_KERNEL_LEN} steps = {config.REFINE_KERNEL_LEN * 20 / 1000:.1f} s each way)")

    f1_metric = step_metrics.StepConfusion(name='weighted_f1')
    print(f"loss         : {config.LOSS}, head class weights {class_weights}")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=LOSSES[config.LOSS](class_weights),
                  metrics=['accuracy', f1_metric], jit_compile=JIT_COMPILE)

    pipeline.check_manifest()
    train_ds = pipeline.load_split('train', batch_size, db_names)
    eval_ds = pipeline.load_split('eval', batch_size, db_names)

    _, report_dir, logs_dir = model_dirs(keras_name)
    out_dir = refined_dir(keras_name, tag)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    report_dir = os.path.join(report_dir, os.path.basename(out_dir))
    logs_dir = os.path.join(logs_dir, os.path.basename(out_dir))
    with open(os.path.join(out_dir, 'head_config.json'), 'w') as f:
        json.dump({'base_checkpoint': base_checkpoint, 'class_weights': class_weights, 'mode': mode,
                   'epochs': epochs, 'lr': lr, 'width': config.REFINE_WIDTH,
                   'blocks': config.REFINE_BLOCKS, 'kernel_len': config.REFINE_KERNEL_LEN}, f,
                  indent=2)

    # epoch_00 = the base: the fallback that makes the no-regression rule always satisfiable
    model.save(os.path.join(out_dir, 'epoch_00.keras'))
    print(f"refined      : {out_dir} (epoch_00 = base, then one file per epoch)\n")

    model.fit(train_ds, validation_data=eval_ds, epochs=epochs, callbacks=[
        WeightedF1Checkpoint(f1_metric, out_dir, report_dir, save_start_epoch=1),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(out_dir, 'epoch_{epoch:02d}.keras'), save_best_only=False),
        tf.keras.callbacks.TensorBoard(log_dir=logs_dir),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_weighted_f1', mode='max',
                                             factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ])
    return out_dir


def _numbers(path):
    row = report.parse_report(path)
    out = {}
    for m in METRICS:
        try:
            out[m] = float(row[m])
        except (KeyError, ValueError, TypeError):
            out[m] = None
    return out


def _s_f1(n):
    se, pp = n.get('S_Se'), n.get('S_+P')
    return 2 * se * pp / (se + pp) if se and pp and se + pp else 0.0


def base_portal_eval_report(model_name):
    """The base model's portal-eval report from this run's ec57 stage."""
    return os.path.join(config.EC57_DIR, model_name, 'portal-eval',
                        'portal-eval_QRS_report_line.out')


def select_refinement(model_name, out_dir=None, records=None, tolerance=None, tag=None):
    """Score every epoch_NN.keras on portal-eval and copy the no-regression winner to
    refined_best.keras. Returns (path, table) where table is a list of dicts."""
    setup_gpus()
    keras_name = models.keras_name(model_name)
    out_dir = out_dir or refined_dir(keras_name, tag)
    records = config.PORTAL_SPLIT_RECORDS if records is None else records
    tol = config.REFINE_TOLERANCE_PP if tolerance is None else tolerance

    base_path = base_portal_eval_report(model_name)
    select_root = os.path.join(config.EC57_DIR,
                               f'{model_name}_refine_select' + (f'_{tag}' if tag else ''))
    if not os.path.exists(base_path):
        print(f"no base portal-eval report at {base_path} - scoring epoch_00 as the base")
        base_path = None

    candidates = sorted(glob.glob(os.path.join(out_dir, 'epoch_*.keras')))
    if not candidates:
        raise FileNotFoundError(f"no epoch_*.keras under {out_dir} - run `ecgr refine` first")

    table = []
    for path in candidates:
        tag = os.path.splitext(os.path.basename(path))[0]
        model = keras.models.load_model(path, compile=False)
        rep = ec57.score_portal_split(model, 'eval', os.path.join(select_root, tag),
                                      max_records=records)
        n = _numbers(rep) if rep else {m: None for m in METRICS}
        table.append({'epoch': tag, 'path': path, **n, 'S_F1': _s_f1(n)})
        keras.backend.clear_session()

    base = _numbers(base_path) if base_path else dict(table[0])
    base_f1 = _s_f1(base)

    def regressions(n):
        return [m for m in METRICS
                if base.get(m) is not None and n.get(m) is not None and n[m] < base[m] - tol]

    for row in table:
        row['regressions'] = regressions(row)
        row['feasible'] = not row['regressions']

    feasible = [r for r in table if r['feasible']]
    winner = max(feasible, key=lambda r: r['S_F1']) if feasible else max(table, key=lambda r: r['S_F1'])
    improved = winner['S_F1'] > base_f1 + 1e-9

    best_path = os.path.join(out_dir, 'refined_best.keras')
    shutil.copy(winner['path'], best_path)
    summary = {'model': model_name, 'records_per_candidate': records, 'tolerance_pp': tol,
               'base': {**base, 'S_F1': base_f1, 'report': base_path},
               'winner': winner, 'improved_over_base': improved, 'candidates': table}
    with open(os.path.join(out_dir, 'selection.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'epoch':10s} " + ' '.join(f"{m:>6s}" for m in METRICS) + f" {'S_F1':>6s}  status")
    print(f"{'base':10s} " + ' '.join(f"{base[m]:6.2f}" if base[m] is not None else f"{'-':>6s}"
                                       for m in METRICS) + f" {base_f1:6.2f}")
    for r in table:
        vals = ' '.join(f"{r[m]:6.2f}" if r[m] is not None else f"{'-':>6s}" for m in METRICS)
        status = ('WINNER' if r is winner else '') + \
                 ('' if r['feasible'] else f"  regresses {','.join(r['regressions'])}")
        print(f"{r['epoch']:10s} {vals} {r['S_F1']:6.2f}  {status}")
    print(f"\nrefined_best.keras <- {winner['epoch']} "
          f"({'improves' if improved else 'does NOT improve'} S F1 on portal-eval: "
          f"{base_f1:.2f} -> {winner['S_F1']:.2f}; feasible candidates: {len(feasible)}/{len(table)})")
    print(f"-> {best_path}")
    return best_path, summary
