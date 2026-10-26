"""Small dual U-Nets: self-supervised encoder pretraining, then teacher -> student training.

Same input (2500, 3) and the same three outputs as the teacher - 'beat' (1250, 4),
'rhythm' (1250, NUM_CLASSES), 'channel' (5, 4) - at 100k / 30k parameters
(dualunet.SIZES['rhythm_dual_100k' | 'rhythm_dual_30k']).

Stage 1 - SSL of the shared encoder (`pretrain_encoder`), no labels:
    clean window x --(mask time spans + one whole lead)--> x_masked
    encoder(x_masked, p_beat) -> z (250), v1 (1250), a1 (1250) -> small head -> x_hat
    loss = MSE(x_hat, x) on the HIDDEN samples only (copying the visible ones scores nothing)
  The atrial branch reads the signal after QRST cancellation, which needs beat positions.
  A student whose position decoder has never seen a label has none, so the cancellation is
  driven by the frozen teacher's p(beat) of the UNMASKED window - a stop-gradient input; the
  objective itself stays reconstruction only. Saved: <run>/checkpoints/<model>/ssl_encoder.keras
  (the encoder-only graph; its layer names are the student's, so it loads by name).

Stage 2 - teacher -> student (`distill`):
    loss = sum over heads w_h [ (1 - a) * label_loss_h + a * T^2 * KL(teacher_T || student_T) ]
  label losses are the project's own (rhythm CE with class weights, beat heat + type, channel
  CE); the KL is per step / segment on the temperature-softened distributions, over EVERY step
  - the teacher's opinion is available where the label is IGNORE, which is most of what a
  small student gains from it. The teacher is frozen and runs on the same augmented batch.
  The student starts from the SSL encoder (train.init_matching) and is fine-tuned whole.
  Saved like `train.train`: <run>/checkpoints/<model>/best_model.keras = the student alone.
"""
import json
import os

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from . import config as rc
from . import dualunet
from . import model as rmodel
from . import pipeline
from .objectives import BeatF1, LeadConfusion, RhythmF1, beat_loss, lead_loss, rhythm_loss


def load_teacher(path=None):
    path = path or rc.DISTILL_TEACHER
    teacher = keras.models.load_model(path, compile=False)
    teacher.trainable = False
    return teacher


# =====================================================================================
# Stage 1: masked-reconstruction SSL of the encoder
# =====================================================================================

def build_encoder(model_name):
    """The encoder-only graph of a dual U-Net size: inputs [signal, p_beat]."""
    kw = dict(rmodel.SIZES[model_name])
    kw.pop('family', None)
    return dualunet.build_dual_unet_model(**kw, name=f"{rmodel.keras_name(model_name)}_enc",
                                          external_pos=True, features_only=True)


def build_ssl_head(encoder, width=32):
    """{'z','v1','a1'} -> (2500, C) reconstruction. Small on purpose: a strong head would
    learn the reconstruction itself and leave the encoder untrained."""
    out = encoder.output
    z, v1, a1 = out['z'], out['v1'], out['a1']
    zi = keras.Input(z.shape[1:], name='ssl_z')
    vi = keras.Input(v1.shape[1:], name='ssl_v1')
    ai = keras.Input(a1.shape[1:], name='ssl_a1')
    h = layers.UpSampling1D(vi.shape[1] // zi.shape[1], name='ssl_up')(zi)
    h = layers.Concatenate(name='ssl_cat')([h, vi, ai])
    h = layers.Conv1D(width, 7, padding='same', activation='gelu', name='ssl_mix')(h)
    stride = rc.SEGMENT_SAMPLES // vi.shape[1]
    h = layers.Conv1D(stride * rc.IN_CHANNELS, 1, name='ssl_out')(h)          # (1250, 2C)
    h = layers.Reshape((rc.SEGMENT_SAMPLES, rc.IN_CHANNELS), name='ssl_signal')(h)
    return keras.Model([zi, vi, ai], h, name='ssl_head')


class MaskedEncoderSSL(keras.Model):
    """Encoder + head trained to fill in masked spans and a masked lead."""

    def __init__(self, encoder, teacher, mask_ratio=None, span_seconds=None,
                 lead_mask_prob=None, **kw):
        super().__init__(**kw)
        self.encoder = encoder
        self.teacher = teacher
        self.head = build_ssl_head(encoder)
        self.mask_ratio = rc.SSL_MASK_RATIO if mask_ratio is None else float(mask_ratio)
        span = rc.SSL_SPAN_SECONDS if span_seconds is None else float(span_seconds)
        self.span = max(1, int(round(span * rc.SAMPLING_RATE)))
        self.lead_mask_prob = rc.SSL_LEAD_MASK_PROB if lead_mask_prob is None \
            else float(lead_mask_prob)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.nmse_tracker = keras.metrics.Mean(name='nmse')

    @property
    def metrics(self):
        return [self.loss_tracker, self.nmse_tracker]

    def call(self, x, training=False):
        p = self.teacher_positions(x)
        f = self.encoder([x, p], training=training)
        return self.head([f['z'], f['v1'], f['a1']], training=training)

    def teacher_positions(self, x):
        beat = self.teacher(x, training=False)['beat']
        return tf.stop_gradient(1.0 - beat[..., :1])                         # (B, 1250, 1)

    def masks(self, batch):
        """(time mask (B, n, 1), lead keep (B, 1, C)); 1 in the time mask = hidden."""
        n, c = rc.SEGMENT_SAMPLES, rc.IN_CHANNELS
        n_spans = -(-n // self.span)
        hidden = tf.cast(tf.random.uniform([batch, n_spans]) < self.mask_ratio, tf.float32)
        time = tf.repeat(hidden, self.span, axis=1)[:, :n, None]
        victim = tf.random.uniform([batch], 0, c, dtype=tf.int32)
        drop = tf.cast(tf.random.uniform([batch]) < self.lead_mask_prob, tf.float32)
        keep = (1.0 - tf.one_hot(victim, c) * drop[:, None])[:, None, :]
        return time, keep

    def _reconstruction(self, x, training):
        x = tf.cast(x, tf.float32)
        time, keep = self.masks(tf.shape(x)[0])
        p = self.teacher_positions(x)                       # from the UNMASKED window
        corrupted = x * (1.0 - time) * keep
        f = self.encoder([corrupted, p], training=training)
        x_hat = self.head([f['z'], f['v1'], f['a1']], training=training)
        hidden = tf.maximum(time, 1.0 - keep)               # (B, n, C)
        denom = tf.reduce_sum(hidden) + 1e-6
        loss = tf.reduce_sum(tf.square(x_hat - x) * hidden) / denom
        mean = tf.reduce_sum(x * hidden) / denom
        var = tf.reduce_sum(tf.square(x - mean) * hidden) / denom
        return loss, loss / (var + 1e-6)

    def train_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        with tf.GradientTape() as tape:
            loss, nmse = self._reconstruction(x, training=True)
        variables = self.encoder.trainable_variables + self.head.trainable_variables
        grads = tape.gradient(loss, variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, variables))
        self.loss_tracker.update_state(loss)
        self.nmse_tracker.update_state(nmse)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        loss, nmse = self._reconstruction(x, training=False)
        self.loss_tracker.update_state(loss)
        self.nmse_tracker.update_state(nmse)
        return {m.name: m.result() for m in self.metrics}


def ssl_path(model_name):
    return os.path.join(rc.CHECKPOINT_DIR, rmodel.keras_name(model_name), 'ssl_encoder.keras')


def pretrain_encoder(model_name, teacher_path=None, epochs=None, steps_per_epoch=None,
                     batch_size=None, lr=None, max_windows=None, val_steps=200):
    """Stage 1. Returns the path of the saved encoder."""
    from ..training.train import setup_gpus
    setup_gpus()
    epochs = rc.SSL_EPOCHS if epochs is None else epochs
    steps_per_epoch = rc.SSL_STEPS_PER_EPOCH if steps_per_epoch is None else steps_per_epoch
    batch_size = batch_size or rc.BATCH_SIZE
    lr = lr or rc.SSL_LEARNING_RATE
    teacher = load_teacher(teacher_path)
    encoder = build_encoder(model_name)
    trainer = MaskedEncoderSSL(encoder, teacher)
    trainer.compile(optimizer=keras.optimizers.AdamW(lr, weight_decay=rc.WEIGHT_DECAY),
                    jit_compile=False)    # as train.py: XLA fails on this graph
    print(f"SSL          : {model_name} encoder {encoder.count_params():,} params, teacher "
          f"{teacher.count_params():,} (frozen, beat positions only)")
    print(f"masking      : {100 * trainer.mask_ratio:.0f}% of time in {trainer.span}-sample "
          f"spans + one lead with p={trainer.lead_mask_prob}")

    # labels are not used: the 'train' pipeline only for its augmentation (noise, gain, flip,
    # lead drop, permutation) - the SSL input is the same kind of window the student will see
    train_ds = pipeline.load_split('train', batch_size, 'train', max_windows,
                                   label_steps=rc.BEAT_STEPS, outputs=('rhythm',),
                                   sampler='uniform').repeat()
    val_ds = pipeline.load_split('eval', batch_size, 'noisy', max_windows,
                                 label_steps=rc.BEAT_STEPS, outputs=('rhythm',))
    if val_steps:
        val_ds = val_ds.take(val_steps)
    out = ssl_path(model_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    trainer.fit(train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
                validation_data=val_ds,
                callbacks=[
                    keras.callbacks.LearningRateScheduler(
                        _cosine(lr, epochs), verbose=0),
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs,
                                                  restore_best_weights=True, verbose=1),
                    keras.callbacks.CSVLogger(os.path.join(os.path.dirname(out),
                                                           'ssl_history.csv')),
                ])
    encoder.save(out)
    print(f"\nSSL encoder -> {out}\nnmse = masked reconstruction error / variance of what was "
          f"hidden: 1.0 = no better than the mean, lower = structure learned")
    return out


def _cosine(lr, epochs, floor=0.05):
    def at(epoch, _lr=None):
        t = min(1.0, epoch / max(1, epochs - 1))
        return float(lr * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * t))))
    return at


# =====================================================================================
# Stage 2: teacher -> student
# =====================================================================================

def kd_kl(p_teacher, p_student, temperature):
    """Per-position KL(teacher_T || student_T) of two probability tensors (..., K), * T^2."""
    t = float(temperature)
    lt = tf.math.log(tf.clip_by_value(p_teacher, 1e-7, 1.0)) / t
    ls = tf.math.log(tf.clip_by_value(p_student, 1e-7, 1.0)) / t
    qt = tf.nn.softmax(lt, axis=-1)
    kl = tf.reduce_sum(qt * (tf.nn.log_softmax(lt, axis=-1) - tf.nn.log_softmax(ls, axis=-1)),
                       axis=-1)
    return kl * (t * t)


HEADS = ('rhythm', 'beat', 'channel')


class Distiller(keras.Model):
    """Frozen teacher + trainable student. call() is the student, so predict() / the beat
    matcher / saving the student all see the student's outputs."""

    def __init__(self, student, teacher, class_weights, alpha=None, temperature=None, **kw):
        super().__init__(**kw)
        self.student = student
        self.teacher = teacher
        self.alpha = rc.KD_ALPHA if alpha is None else float(alpha)
        self.temperature = rc.KD_TEMPERATURE if temperature is None else float(temperature)
        self.label_losses = {'rhythm': rhythm_loss(class_weights),
                             'beat': beat_loss(list(rc.BEAT_CLASS_WEIGHTS)),
                             'channel': lead_loss()}
        self.head_weights = {'rhythm': 1.0, 'beat': rc.BEAT_LOSS_WEIGHT,
                             'channel': rc.CHANNEL_LOSS_WEIGHT}
        self.trackers = {k: keras.metrics.Mean(name=k) for k in
                         ('loss', 'rhythm_loss', 'beat_loss', 'channel_loss', 'kd_loss')}
        self.rhythm_f1 = RhythmF1(name='rhythm_f1')
        self.beat_f1 = BeatF1(name='beat_f1')
        self.channel_acc = LeadConfusion(name='channel_acc', mode='accuracy')
        self.channel_noise_f1 = LeadConfusion(name='channel_noise_f1')

    @property
    def metrics(self):
        return list(self.trackers.values()) + [self.rhythm_f1, self.beat_f1, self.channel_acc,
                                               self.channel_noise_f1]

    def call(self, x, training=False):
        return self.student(x, training=training)

    def _kd_step(self, x, y, training):
        t_out = self.teacher(x, training=False)
        s_out = self.student(x, training=training)
        total, parts, kd_total = 0.0, {}, 0.0
        for h in HEADS:
            label = tf.reduce_mean(self.label_losses[h](y[h], s_out[h]))
            kd = tf.reduce_mean(kd_kl(tf.stop_gradient(t_out[h]), s_out[h], self.temperature))
            parts[h] = label
            kd_total += self.head_weights[h] * kd
            total += self.head_weights[h] * ((1.0 - self.alpha) * label + self.alpha * kd)
        return total, parts, kd_total, s_out

    def _update(self, total, parts, kd_total, y, s_out):
        self.trackers['loss'].update_state(total)
        for h in HEADS:
            self.trackers[f'{h}_loss'].update_state(parts[h])
        self.trackers['kd_loss'].update_state(kd_total)
        self.rhythm_f1.update_state(y['rhythm'], s_out['rhythm'])
        self.beat_f1.update_state(y['beat'], s_out['beat'])
        self.channel_acc.update_state(y['channel'], s_out['channel'])
        self.channel_noise_f1.update_state(y['channel'], s_out['channel'])
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            total, parts, kd_total, s_out = self._kd_step(x, y, training=True)
        variables = self.student.trainable_variables
        grads = tape.gradient(total, variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, variables))
        return self._update(total, parts, kd_total, y, s_out)

    def test_step(self, data):
        x, y = data
        total, parts, kd_total, s_out = self._kd_step(x, y, training=False)
        return self._update(total, parts, kd_total, y, s_out)


class SaveBestStudent(keras.callbacks.Callback):
    """best_model.keras = the STUDENT at the best val_rhythm_f1 (from start_epoch)."""

    def __init__(self, path, monitor=None, start_epoch=None):
        super().__init__()
        self.path = path
        self.monitor = monitor or rc.MONITOR
        self.start_epoch = rc.CKPT_START_EPOCH if start_epoch is None else start_epoch
        self.best = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get(self.monitor)
        if v is None or epoch + 1 < self.start_epoch or v <= self.best:
            return
        print(f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.5f} to {v:.5f}, "
              f"saving student to {self.path}")
        self.best = float(v)
        self.model.student.save(self.path)


class SaveEveryStudent(keras.callbacks.Callback):
    """epochs/epoch_<n>.keras = the STUDENT after every epoch from start_epoch (the epoch pick is
    made later on validation, not by the val_rhythm_f1 of the training monitor)."""

    def __init__(self, directory, start_epoch=None):
        super().__init__()
        self.directory = directory
        self.start_epoch = rc.CKPT_START_EPOCH if start_epoch is None else start_epoch
        os.makedirs(directory, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 >= self.start_epoch:
            self.model.student.save(os.path.join(self.directory, f'epoch_{epoch + 1:02d}.keras'))


def distill(model_name, teacher_path=None, ssl_from=None, epochs=None, batch_size=None,
            lr=None, patience=None, steps_per_epoch=None, max_windows=None, alpha=None,
            temperature=None, init_from=None, train_only=None):
    """Stage 2. ssl_from: an ssl_encoder.keras (default: this run's, if it exists; 'none' =
    random init). init_from: a full student checkpoint (.keras) of the SAME model: every weight
    (encoder + decoders) starts there and trains again under the teacher (a fine-tune; it
    replaces ssl_from). train_only: regex on student layer names; every other layer is frozen
    (incl. BatchNorm statistics), e.g. '^(rhythm|rr_)' = the rhythm decoder only, leaving the
    encoder, the beat head and the channel head bit-identical. Returns the student."""
    from ..training.train import setup_gpus
    from .train import (BeatMatchLog, ConfusionLog, LogLR, class_weights, cosine_schedule,
                        init_matching, model_dirs)
    setup_gpus()
    batch_size = batch_size or rc.BATCH_SIZE
    epochs = epochs or rc.EPOCHS
    patience = rc.PATIENCE if patience is None else patience
    lr = lr or rc.LEARNING_RATE

    teacher = load_teacher(teacher_path)
    student = rmodel.build(model_name)
    if init_from:
        student.load_weights(init_from)
        ssl_from = 'none'
    if ssl_from is None and os.path.exists(ssl_path(model_name)):
        ssl_from = ssl_path(model_name)
    if ssl_from and ssl_from != 'none':
        init_matching(student, ssl_from)

    if train_only:
        import re
        kept = [l.name for l in student.layers if re.search(train_only, l.name)]
        for layer in student.layers:
            layer.trainable = layer.name in kept
        print(f"train-only   : /{train_only}/ -> {len(kept)} layers, "
              f"{sum(l.count_params() for l in student.layers if l.name in kept):,} params trainable")
    outputs = rmodel.output_names(student)
    steps = rmodel.rhythm_steps(student)
    train_ds, train_arrays = pipeline.load_split(
        'train', batch_size, 'train', max_windows, label_steps=steps, outputs=outputs,
        sampler='stratified', steps_per_epoch=steps_per_epoch, return_arrays=True)
    if steps_per_epoch:
        train_ds = train_ds.repeat()   # Keras keeps ONE iterator across epochs when steps are set
    cats = pipeline.window_categories(train_arrays[1])
    _plan, repeat, _n = pipeline.stratified_plan(cats, batch_size, steps_per_epoch)
    weights = class_weights(repeat)
    eval_ds, eval_arrays = pipeline.load_split('eval', batch_size, 'noisy', max_windows,
                                               label_steps=steps, outputs=outputs,
                                               return_arrays=True)

    trainer = Distiller(student, teacher, weights, alpha=alpha, temperature=temperature)
    trainer.compile(optimizer=keras.optimizers.AdamW(lr, weight_decay=rc.WEIGHT_DECAY,
                                                     clipnorm=1.0),
                    jit_compile=False)
    print(f"\n{rc.describe()}")
    print(f"student      : {student.name}, {student.count_params():,} params"
          + (f", all weights from {init_from}" if init_from else
             f", encoder from {ssl_from}" if ssl_from and ssl_from != 'none' else ", random init"))
    print(f"teacher      : {teacher.name}, {teacher.count_params():,} params (frozen)")
    print(f"KD           : alpha {trainer.alpha}, T {trainer.temperature}; class weights "
          f"{dict(zip(rc.CLASS_NAMES, np.round(weights, 3)))}")

    ckpt_dir, report_dir, logs_dir = model_dirs(student.name)
    with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
        json.dump({'model': model_name, 'params': student.count_params(), 'stage': 'distill',
                   'teacher': teacher_path or rc.DISTILL_TEACHER, 'ssl_from': ssl_from,
                   'init_from': init_from, 'train_only': train_only,
                   'kd_alpha': trainer.alpha, 'kd_temperature': trainer.temperature,
                   'class_weights': weights, 'sampler': 'stratified', 'sampler_repeat': repeat,
                   'batch_size': batch_size, 'lr': lr, 'npy_dir': rc.NPY_DIR}, f, indent=2)

    callbacks = [
        BeatMatchLog(eval_arrays[0][:rc.BEAT_MATCH_WINDOWS],
                     eval_arrays[3][:rc.BEAT_MATCH_WINDOWS], report_dir, batch_size),
        LogLR(),
        ConfusionLog(trainer.rhythm_f1, trainer.channel_acc, report_dir, 'channel',
                     beat_metric=trainer.beat_f1),
        SaveBestStudent(os.path.join(ckpt_dir, 'best_model.keras')),
        SaveEveryStudent(os.path.join(ckpt_dir, 'epochs')),
        keras.callbacks.EarlyStopping(monitor=rc.MONITOR, mode='max', patience=patience,
                                      start_from_epoch=max(0, rc.CKPT_START_EPOCH - 1),
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.LearningRateScheduler(cosine_schedule(lr, epochs), verbose=0),
        keras.callbacks.CSVLogger(os.path.join(logs_dir, 'history.csv'), append=True),
    ]
    trainer.fit(train_ds, validation_data=eval_ds, epochs=epochs,
                steps_per_epoch=steps_per_epoch or None, callbacks=callbacks)
    print(f"\ndistillation finished; student checkpoints in {ckpt_dir}")
    return student
