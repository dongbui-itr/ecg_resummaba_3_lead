"""The rhythm fit loop.

    <RUN_DIR>/checkpoints/<model>/best_model.keras        best val_rhythm_f1
    <RUN_DIR>/checkpoints/<model>/epochs/epoch_NN.keras   every epoch from CKPT_START_EPOCH
    <RUN_DIR>/eval/<model>/confusion_log.txt              per epoch: the per-second rhythm
                                                          matrix and the per-window lead one
    <RUN_DIR>/logs/<model>/                               tensorboard + history.csv

Validation runs on the eval split corrupted with the SAME noise every epoch (pipeline mode
'noisy'), so val_rhythm_f1 measures reading rhythm through artefact, val_lead_acc picking the
right lead (or NOISE) and val_lead_noise_f1 recognising an unreadable window - on a benchmark
that does not move between epochs.
"""
import json
import os

import numpy as np
import tensorflow as tf

from .. import models as beat_models
from ..training.callbacks import DelayedModelCheckpoint
from ..training.train import setup_gpus
from . import config as rc
from . import model as rmodel
from . import pipeline
from .objectives import (BeatF1, LeadConfusion, NoiseF1, RhythmF1, beat_loss, lead_loss,
                         noise_loss, per_class_table, rhythm_loss)


def model_dirs(keras_name):
    dirs = [os.path.join(d, keras_name) for d in (rc.CHECKPOINT_DIR, rc.REPORT_DIR,
                                                  rc.LOGS_DIR)]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def format_confusion(cm, title, names=None, unit='seconds'):
    names = names or rc.CLASS_NAMES
    width = max(len(n) for n in names) + 2
    lines = [title, '=' * len(title), f'rows = reference, columns = predicted ({unit})',
             ' ' * width + ''.join(f"{n:>9s}" for n in names)]
    for i, n in enumerate(names):
        lines.append(f"{n:<{width}s}" + ''.join(f"{int(v):>9d}" for v in cm[i]))
    lines += ['', f"{'class':<8s}{'support':>10s}{'Se':>9s}{'+P':>9s}{'F1':>9s}"]
    for r in per_class_table(cm, names):
        lines.append(f"{r['cls']:<8s}{r['support']:>10d}{100 * r['se']:>9.2f}"
                     f"{100 * r['ppv']:>9.2f}{100 * r['f1']:>9.2f}")
    return '\n'.join(lines)


class ConfusionLog(tf.keras.callbacks.Callback):
    def __init__(self, rhythm_metric, lead_metric, report_dir, quality='lead',
                 beat_metric=None):
        super().__init__()
        self.rhythm_metric, self.lead_metric = rhythm_metric, lead_metric
        self.quality = quality
        self.beat_metric = beat_metric
        self.path = os.path.join(report_dir, 'confusion_log.txt')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        cm = self.rhythm_metric.matrix()
        if cm.sum() == 0:
            return
        text = format_confusion(cm, f"EPOCH {epoch + 1}  val_rhythm_f1="
                                    f"{logs.get('val_rhythm_f1', float('nan')):.4f}")
        if self.quality == 'channel':
            text += '\n\n' + format_confusion(
                self.lead_metric.matrix(),
                f"EPOCH {epoch + 1}  channel: acc={logs.get('val_channel_acc', float('nan')):.4f}"
                f"  noise_f1={logs.get('val_channel_noise_f1', float('nan')):.4f}",
                names=rc.LEAD_CLASSES, unit='2 s segments')
        elif self.quality == 'noise':
            text += '\n\n' + format_confusion(
                self.lead_metric.matrix(),
                f"EPOCH {epoch + 1}  noise: acc={logs.get('val_noise_acc', float('nan')):.4f}"
                f"  noise_f1={logs.get('val_noise_f1', float('nan')):.4f}",
                names=rc.NOISE_CLASSES, unit='2 s segments')
        else:
            text += '\n\n' + format_confusion(
                self.lead_metric.matrix(),
                f"EPOCH {epoch + 1}  lead: acc={logs.get('val_lead_acc', float('nan')):.4f}  "
                f"noise_f1={logs.get('val_lead_noise_f1', float('nan')):.4f}",
                names=rc.LEAD_CLASSES, unit='windows')
        if self.beat_metric is not None:
            text += '\n\n' + format_confusion(
                self.beat_metric.matrix(),
                f"EPOCH {epoch + 1}  beat: f1(N,S,V)={logs.get('val_beat_f1', float('nan')):.4f}",
                names=rc.BEAT_CLASSES, unit='8 ms steps')
        print('\n' + text)
        with open(self.path, 'a') as f:
            f.write('\n' + text + '\n')


def init_backbone(model, source):
    """Start the rhythm backbone from a beat model's (or an SSL run's) backbone weights.

    The backbone is label-independent and resolution-agnostic in its weights - its pooling
    plan is the only thing BACKBONE_STEPS changes, and pooling has no weights - so a beat
    checkpoint of the same size is a legitimate starting point: it already knows QRS, P and T
    morphology on this very portal data.
    """
    target = beat_models.sub_model(model, 'backbone')
    if source.endswith('.keras'):
        other = tf.keras.models.load_model(source, compile=False)
        weights = beat_models.sub_model(other, 'backbone').get_weights()
        shapes = [w.shape for w in target.get_weights()]
        if [w.shape for w in weights] != shapes:
            raise ValueError(f"{source}: its backbone does not match this model's - use a "
                             f"beat checkpoint of the same size")
        target.set_weights(weights)
    else:
        target.load_weights(source)
    print(f"backbone     : initialised from {source}")


def class_weights(repeat=None):
    """The manifest's class weights (~ inverse sqrt second frequency), divided by
    sqrt(oversampling ratio) of the stratified sampler (`repeat` = windows drawn per epoch /
    windows present, per category) so the prior is not corrected twice."""
    if rc.CLASS_WEIGHTS is not None:
        weights = list(rc.CLASS_WEIGHTS)
    else:
        weights = pipeline.manifest_class_weights(pipeline.read_manifest())
    if repeat:
        # only ever LOWER a weight: a class drawn more often than SINUS has part of its prior
        # correction done by the sampler; one drawn less often keeps the manifest's weight
        base = repeat.get('SINUS', 1.0)
        weights = [w / np.sqrt(max(repeat.get(n, base) / base, 1.0))
                   for w, n in zip(weights, rc.CLASS_NAMES)]
    return [float(w) for w in weights]


def _all_layers(model):
    """Leaf layers of a model, recursing into nested models (the 'backbone' sub-model)."""
    out = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            out.extend(_all_layers(layer))
        else:
            out.append(layer)
    return out


def init_matching(model, source):
    """Warm start: every weight tensor whose path (layer/variable name, nested models
    included) and shape exist in the `source` checkpoint takes its value - rhythm_unet1250_1m
    -> rhythm_unet1250b_1m moves 169 of 271 tensors: the whole encoder, the SSM path, the
    context / descriptor branch, the noise head. The rhythm decoder's first conv and the
    beat-conditioning layers see a different input width and start fresh. Returns the names
    of the leaf layers whose EVERY weight was transferred (the ones --freeze-epochs freezes)."""
    other = tf.keras.models.load_model(source, compile=False)
    theirs = {w.path: w for w in other.weights}
    moved = skipped = 0
    full = set()
    for layer in _all_layers(model):
        if not layer.weights:
            continue
        ok = True
        for w in layer.weights:
            src = theirs.get(w.path)
            if src is not None and tuple(src.shape) == tuple(w.shape):
                w.assign(src)
                moved += 1
            else:
                ok = False
                skipped += 1
        if ok:
            full.add(layer.name)
    fresh = [l.name for l in _all_layers(model) if l.weights and l.name not in full]
    print(f"init         : {moved} tensors from {source}; {skipped} tensors fresh; frozen "
          f"candidates {len(full)} layers; fresh/partial layers: {fresh}")
    return full


def cosine_schedule(lr, epochs, warmup=None, floor=None):
    """Epoch -> learning rate: linear warm-up over `warmup` epochs, then cosine to
    floor x lr at the last epoch."""
    warmup = rc.LR_WARMUP_EPOCHS if warmup is None else warmup
    floor = rc.LR_FLOOR_FRACTION if floor is None else floor

    def at(epoch, _lr=None):
        if epoch < warmup:
            return lr * (epoch + 1) / (warmup + 1)
        span = max(1, epochs - warmup - 1)
        t = min(1.0, (epoch - warmup) / span)
        return lr * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * t)))
    return at


class LogLR(tf.keras.callbacks.Callback):
    """Puts the optimizer's learning rate into the epoch logs (history.csv) - the plateau
    callback did that implicitly, the cosine scheduler does not."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['learning_rate'] = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))


class BeatMatchLog(tf.keras.callbacks.Callback):
    """Beat-level validation the way bxb scores it: the 'beat' output of a fixed set of eval
    windows -> beats.pick_beats -> matched to the annotated R samples within
    rc.BEAT_MATCH_TOLERANCE_SECONDS. Logs val_beat_se / val_beat_pp (QRS) and
    val_beat_type_f1 (macro F1 of N/S/V on matched beats) - the numbers the post-processing
    depends on - and appends them to confusion_log.txt."""

    def __init__(self, segments, beat_labels, report_dir, batch_size=64):
        super().__init__()
        self.x = segments
        self.ref = beat_labels
        self.path = os.path.join(report_dir, 'confusion_log.txt')
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        from .beats import pick_beats
        from .predict import beat_step_hz
        logs = logs if logs is not None else {}
        hz = beat_step_hz()
        tol = rc.BEAT_MATCH_TOLERANCE_SECONDS
        pred = self.model.predict(self.x.astype(np.float32), batch_size=self.batch_size,
                                  verbose=0)
        beat = pred['beat'] if isinstance(pred, dict) else pred[-1]
        cm = np.zeros((4, 4), np.int64)
        hits = fn = fp = 0
        for probs, lab in zip(beat, self.ref):
            if np.all(lab == rc.IGNORE):
                continue
            ref_s = np.flatnonzero((lab > 0) & (lab != rc.IGNORE))
            ref_t = ref_s / rc.SAMPLING_RATE
            ref_c = lab[ref_s]
            ign = lab == rc.IGNORE
            got = pick_beats(probs, hz)
            t, c = got['t'], got['cls']
            used = np.zeros(len(t), bool)
            for rt, rcl in zip(ref_t, ref_c):
                j = np.searchsorted(t, rt)
                cand = [k for k in (j - 1, j) if 0 <= k < len(t) and abs(t[k] - rt) <= tol
                        and not used[k]]
                if cand:
                    k = min(cand, key=lambda k: abs(t[k] - rt))
                    used[k] = True
                    hits += 1
                    cm[rcl, c[k]] += 1
                else:
                    fn += 1
                    cm[rcl, 0] += 1
            for k in np.flatnonzero(~used):      # a beat in an IGNORE zone is not a false one
                s = int(round(t[k] * rc.SAMPLING_RATE))
                if not (0 <= s < len(lab) and ign[s]):
                    fp += 1
        se = hits / max(hits + fn, 1)
        pp = hits / max(hits + fp, 1)
        f1s = []
        for k in (1, 2, 3):
            if cm[k].sum() == 0:
                continue
            tp = cm[k, k]
            p = tp / max(cm[:, k].sum(), 1)
            r = tp / max(cm[k].sum(), 1)
            f1s.append(0.0 if tp == 0 else 2 * p * r / (p + r))
        type_f1 = float(np.mean(f1s)) if f1s else 0.0
        logs['val_beat_se'], logs['val_beat_pp'], logs['val_beat_type_f1'] = se, pp, type_f1
        text = format_confusion(cm, f"EPOCH {epoch + 1}  beats matched at {tol * 1000:.0f} ms: "
                                    f"QRS Se={100 * se:.2f} +P={100 * pp:.2f}  "
                                    f"type F1(N,S,V)={type_f1:.4f}",
                                names=rc.BEAT_CLASSES, unit='beats (col 0 = missed)')
        print(f"\n{text}")
        with open(self.path, 'a') as f:
            f.write('\n' + text + '\n')


def train(model_name, epochs=None, batch_size=None, lr=None, patience=None,
          steps_per_epoch=None, max_windows=None, init_from=None, backbone_from=None,
          init_matching_from=None, freeze_epochs=0, lr2=None, sampler=None, schedule=None):
    """init_matching_from + freeze_epochs: warm start (init_matching), train only the layers
    it did not fill for `freeze_epochs` epochs at `lr`, then everything at `lr2` (default
    lr / 5) for the remaining epochs. sampler: rc.SAMPLER; schedule: rc.LR_SCHEDULE."""
    setup_gpus()
    batch_size = batch_size or rc.BATCH_SIZE
    epochs = epochs or rc.EPOCHS
    patience = rc.PATIENCE if patience is None else patience
    sampler = sampler or rc.SAMPLER
    schedule = schedule or rc.LR_SCHEDULE
    lr = lr or rc.LEARNING_RATE
    lr2 = lr2 or lr / 5

    model = rmodel.build(model_name)
    warm = set()
    if init_from:
        model.load_weights(init_from)
        print(f"init         : all weights from {init_from}")
    elif backbone_from:
        init_backbone(model, backbone_from)
    if init_matching_from:
        warm = init_matching(model, init_matching_from)
    if freeze_epochs and not warm:
        raise ValueError("--freeze-epochs needs --init-matching")

    steps = rmodel.rhythm_steps(model)
    outputs = rmodel.output_names(model)
    train_ds, train_arrays = pipeline.load_split(
        'train', batch_size, 'train', max_windows, label_steps=steps, outputs=outputs,
        sampler=sampler, steps_per_epoch=steps_per_epoch, return_arrays=True)
    repeat = None
    if sampler == 'stratified':
        cats = pipeline.window_categories(train_arrays[1])
        _plan, repeat, _steps = pipeline.stratified_plan(cats, batch_size, steps_per_epoch)
    weights = class_weights(repeat)
    rhythm_f1 = RhythmF1(name='f1')
    quality = next(q for q in ('noise', 'channel', 'lead') if q in outputs)
    # Keras logs these as rhythm_f1 / lead_acc / lead_noise_f1, or noise_acc / noise_f1
    # (output name + metric name)
    if quality == 'noise':
        lead_acc = NoiseF1(name='acc', mode='accuracy')
        quality_loss, quality_metrics = noise_loss(), [lead_acc, NoiseF1(name='f1')]
        quality_weight = rc.NOISE_LOSS_WEIGHT
    else:
        # 'lead' per window, or 'channel' (dual U-Net) per 2 s segment - same classes
        lead_acc = LeadConfusion(name='acc', mode='accuracy')
        quality_loss = lead_loss()
        quality_metrics = [lead_acc, LeadConfusion(name='noise_f1')]
        quality_weight = rc.CHANNEL_LOSS_WEIGHT if quality == 'channel' else \
            rc.LEAD_LOSS_WEIGHT
    losses = {'rhythm': rhythm_loss(weights), quality: quality_loss}
    loss_weights = {'rhythm': 1.0, quality: quality_weight}
    metrics = {'rhythm': [rhythm_f1], quality: quality_metrics}
    beat_f1 = None
    if 'beat' in outputs:
        beat_f1 = BeatF1(name='f1')
        losses['beat'] = beat_loss(list(rc.BEAT_CLASS_WEIGHTS))
        loss_weights['beat'] = rc.BEAT_LOSS_WEIGHT
        metrics['beat'] = [beat_f1]
    def compile_model(rate):
        opt = tf.keras.optimizers.AdamW(rate, weight_decay=rc.WEIGHT_DECAY, clipnorm=1.0) \
            if schedule == 'cosine' else tf.keras.optimizers.Adam(rate, clipnorm=1.0)
        model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=metrics,
                      jit_compile=False)

    compile_model(lr)
    model.summary()
    print(f"\n{rc.describe()}")
    print(f"model        : {model.name}, {model.count_params():,} parameters")
    print(f"class weights: {dict(zip(rc.CLASS_NAMES, np.round(weights, 3)))}"
          + (f" (after sampler repeats {dict((k, round(v, 2)) for k, v in repeat.items())})"
             if repeat else ""))
    print(f"schedule     : {schedule}, lr {lr:g}" + (f" -> {lr2:g} after {freeze_epochs} "
          f"frozen epochs" if freeze_epochs else ""))

    eval_ds, eval_arrays = pipeline.load_split('eval', batch_size, 'noisy', max_windows,
                                               label_steps=steps, outputs=outputs,
                                               return_arrays=True)
    if steps_per_epoch and sampler != 'stratified':
        train_ds = train_ds.repeat()          # a finite dataset would run dry after epoch 1

    ckpt_dir, report_dir, logs_dir = model_dirs(model.name)
    os.makedirs(os.path.join(ckpt_dir, 'epochs'), exist_ok=True)
    with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
        json.dump({'model': model_name, 'params': model.count_params(),
                   'class_weights': weights, 'sampler': sampler,
                   'sampler_repeat': repeat, 'batch_size': batch_size, 'lr': lr, 'lr2': lr2,
                   'schedule': schedule, 'freeze_epochs': freeze_epochs,
                   'init_from': init_from, 'backbone_from': backbone_from,
                   'init_matching_from': init_matching_from, 'npy_dir': rc.NPY_DIR},
                  f, indent=2)

    monitor = rc.MONITOR
    callbacks = [
        LogLR(),
        ConfusionLog(rhythm_f1, lead_acc, report_dir, quality, beat_metric=beat_f1),
        DelayedModelCheckpoint(os.path.join(ckpt_dir, 'best_model.keras'),
                               start_epoch=rc.CKPT_START_EPOCH, monitor=monitor, mode='max',
                               save_best_only=True, verbose=1),
        DelayedModelCheckpoint(os.path.join(ckpt_dir, 'epochs', 'epoch_{epoch:02d}.keras'),
                               start_epoch=rc.CKPT_START_EPOCH, save_best_only=False),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, mode='max', patience=patience,
                                         start_from_epoch=max(0, rc.CKPT_START_EPOCH - 1),
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=logs_dir),
        tf.keras.callbacks.CSVLogger(os.path.join(logs_dir, 'history.csv'), append=True),
    ]
    if 'beat' in outputs:
        # the beat matcher must run before ConfusionLog / CSVLogger so its numbers are logged
        callbacks.insert(0, BeatMatchLog(eval_arrays[0][:rc.BEAT_MATCH_WINDOWS],
                                         eval_arrays[3][:rc.BEAT_MATCH_WINDOWS], report_dir,
                                         batch_size))
    if schedule == 'plateau':
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode='max', factor=0.5, patience=max(2, patience // 3),
            min_lr=1e-5, verbose=1))

    per_epoch = steps_per_epoch or None
    first = 0
    if freeze_epochs:
        for layer in _all_layers(model):
            if layer.name in warm:
                layer.trainable = False
        compile_model(lr)
        print(f"phase 1      : {freeze_epochs} epochs, {len(model.trainable_weights)} trainable "
              f"tensors (warm-started layers frozen)")
        cbs = list(callbacks)
        if schedule == 'cosine':
            cbs.append(tf.keras.callbacks.LearningRateScheduler(
                cosine_schedule(lr, freeze_epochs, warmup=0), verbose=0))
        model.fit(train_ds, validation_data=eval_ds, epochs=freeze_epochs,
                  steps_per_epoch=per_epoch, callbacks=cbs)
        for layer in _all_layers(model):
            layer.trainable = True
        compile_model(lr2)
        first = freeze_epochs
        print(f"phase 2      : epochs {first + 1}-{epochs}, everything trainable at {lr2:g}")
    cbs = list(callbacks)
    if schedule == 'cosine':
        rate = lr2 if freeze_epochs else lr
        sched = cosine_schedule(rate, epochs - first, warmup=0 if freeze_epochs else None)
        cbs.append(tf.keras.callbacks.LearningRateScheduler(
            lambda e, _lr=None: sched(e - first), verbose=0))
    model.fit(train_ds, validation_data=eval_ds, epochs=epochs, initial_epoch=first,
              steps_per_epoch=per_epoch, callbacks=cbs)
    print(f"\ntraining finished; checkpoints in {ckpt_dir}")
    return model
