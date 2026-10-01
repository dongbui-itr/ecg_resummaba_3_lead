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
from .objectives import (LeadConfusion, NoiseF1, RhythmF1, lead_loss, noise_loss,
                         per_class_table, rhythm_loss)


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
    def __init__(self, rhythm_metric, lead_metric, report_dir, quality='lead'):
        super().__init__()
        self.rhythm_metric, self.lead_metric = rhythm_metric, lead_metric
        self.quality = quality
        self.path = os.path.join(report_dir, 'confusion_log.txt')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        cm = self.rhythm_metric.matrix()
        if cm.sum() == 0:
            return
        text = format_confusion(cm, f"EPOCH {epoch + 1}  val_rhythm_f1="
                                    f"{logs.get('val_rhythm_f1', float('nan')):.4f}")
        if self.quality == 'noise':
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


def class_weights():
    if rc.CLASS_WEIGHTS is not None:
        return list(rc.CLASS_WEIGHTS)
    manifest = pipeline.read_manifest()
    weights = manifest.get('class_weights') or [1.0] * rc.NUM_CLASSES
    return [float(w) for w in weights]


def train(model_name, epochs=None, batch_size=None, lr=None, patience=None,
          steps_per_epoch=None, max_windows=None, init_from=None, backbone_from=None):
    setup_gpus()
    batch_size = batch_size or rc.BATCH_SIZE
    epochs = epochs or rc.EPOCHS
    patience = rc.PATIENCE if patience is None else patience

    model = rmodel.build(model_name)
    if init_from:
        model.load_weights(init_from)
        print(f"init         : all weights from {init_from}")
    elif backbone_from:
        init_backbone(model, backbone_from)

    weights = class_weights()
    rhythm_f1 = RhythmF1(name='f1')
    outputs = rmodel.output_names(model)
    quality = 'noise' if 'noise' in outputs else 'lead'
    # Keras logs these as rhythm_f1 / lead_acc / lead_noise_f1, or noise_acc / noise_f1
    # (output name + metric name)
    if quality == 'noise':
        lead_acc = NoiseF1(name='acc', mode='accuracy')
        quality_loss, quality_metrics = noise_loss(), [lead_acc, NoiseF1(name='f1')]
        quality_weight = rc.NOISE_LOSS_WEIGHT
    else:
        lead_acc = LeadConfusion(name='acc', mode='accuracy')
        quality_loss = lead_loss()
        quality_metrics = [lead_acc, LeadConfusion(name='noise_f1')]
        quality_weight = rc.LEAD_LOSS_WEIGHT
    model.compile(optimizer=tf.keras.optimizers.Adam(lr or rc.LEARNING_RATE, clipnorm=1.0),
                  loss={'rhythm': rhythm_loss(weights), quality: quality_loss},
                  loss_weights={'rhythm': 1.0, quality: quality_weight},
                  metrics={'rhythm': [rhythm_f1], quality: quality_metrics},
                  jit_compile=False)
    model.summary()
    print(f"\n{rc.describe()}")
    print(f"model        : {model.name}, {model.count_params():,} parameters")
    print(f"class weights: {dict(zip(rc.CLASS_NAMES, np.round(weights, 3)))}")

    steps = rmodel.rhythm_steps(model)
    train_ds = pipeline.load_split('train', batch_size, 'train', max_windows,
                                   label_steps=steps, outputs=outputs)
    eval_ds = pipeline.load_split('eval', batch_size, 'noisy', max_windows,
                                  label_steps=steps, outputs=outputs)
    if steps_per_epoch:
        train_ds = train_ds.repeat()          # a finite dataset would run dry after epoch 1

    ckpt_dir, report_dir, logs_dir = model_dirs(model.name)
    os.makedirs(os.path.join(ckpt_dir, 'epochs'), exist_ok=True)
    with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
        json.dump({'model': model_name, 'params': model.count_params(),
                   'class_weights': weights, 'batch_size': batch_size,
                   'lr': lr or rc.LEARNING_RATE, 'init_from': init_from,
                   'backbone_from': backbone_from, 'npy_dir': rc.NPY_DIR}, f, indent=2)

    monitor = rc.MONITOR
    callbacks = [
        ConfusionLog(rhythm_f1, lead_acc, report_dir, quality),
        DelayedModelCheckpoint(os.path.join(ckpt_dir, 'best_model.keras'),
                               start_epoch=rc.CKPT_START_EPOCH, monitor=monitor, mode='max',
                               save_best_only=True, verbose=1),
        DelayedModelCheckpoint(os.path.join(ckpt_dir, 'epochs', 'epoch_{epoch:02d}.keras'),
                               start_epoch=rc.CKPT_START_EPOCH, save_best_only=False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode='max', factor=0.5,
                                             patience=max(2, patience // 3), min_lr=1e-5,
                                             verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, mode='max', patience=patience,
                                         start_from_epoch=max(0, rc.CKPT_START_EPOCH - 1),
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=logs_dir),
        tf.keras.callbacks.CSVLogger(os.path.join(logs_dir, 'history.csv'), append=True),
    ]
    model.fit(train_ds, validation_data=eval_ds, epochs=epochs,
              steps_per_epoch=steps_per_epoch or None, callbacks=callbacks)
    print(f"\ntraining finished; checkpoints in {ckpt_dir}")
    return model
