"""Self-supervised pretraining of the BACKBONE, by masked reconstruction.

This is the stage that pretrains where the parameters actually are. The CPC stage
(training/cpc.py) trains the context encoder, which is 5-9% of the model; everything else -
the stem and both paths of the dual-path backbone - was starting from random init and had
only the labelled beat loss to learn from.

The objective, on unlabeled signal from the same tfrecords:

    corrupt the 3-lead strip -> backbone -> a small decoder -> reconstruct what was hidden

Two kinds of corruption, and the second one is the point of doing this on a 3-lead input:

  1. **Span masking.** SSL_MASK_RATIO of the output steps, in contiguous spans of
     SSL_MASK_SPAN_STEPS (~240 ms, about one beat), are zeroed on every lead. Filling a
     masked beat back in from its neighbours is exactly the multi-beat context the
     state-space path exists to provide, so the pretext task and the real task want the same
     receptive field.
  2. **Lead masking.** With probability SSL_LEAD_MASK_PROB one whole lead is zeroed and has
     to be reconstructed from the other two. This teaches the cross-lead redundancy that
     makes three leads worth having - and it is the same operation the EC57 stage performs
     when it hands the model one lead repeated three times, so the model arrives at the
     benchmark already fluent in reading a degenerate montage.

The target is the input at STEP resolution: the 5 raw samples behind each output step, on
every lead, i.e. 5*C numbers per step. Predicting the pooled patch rather than the single
pooled mean keeps the fine morphology in the objective - a reconstruction that only matched
the 20 ms average could smooth every QRS away and still score well.

The loss is computed on the masked steps ONLY. Averaging over all steps would let the model
score by copying the 65% of the input it can still see.

Written next to the model's checkpoints as `ssl_backbone.weights.h5`; `ecgr train` loads it
and (by default) fine-tunes it, which is the usual pretrain/fine-tune split rather than the
paper's frozen patient encoder - that freeze is right for the context encoder, whose job is
to describe nuisance variation, and wrong for the feature extractor the classifier reads.
"""
import os

import keras
import tensorflow as tf

from .. import config, models
from ..data import pipeline


def build_decoder(steps, width, in_channels, step_samples, hidden=None, name='ssl_decoder'):
    """(steps, width) -> (steps, step_samples * in_channels), the reconstruction head.

    Deliberately small - two 1x1 convolutions with one 5-wide conv between them. A decoder
    with real capacity of its own can reconstruct from a weaker representation, which moves
    the learning out of the backbone and into a head that is then thrown away.
    """
    hidden = hidden or max(32, width)
    inp = keras.Input(shape=(steps, width), name='ssl_dec_input')
    h = keras.layers.Conv1D(hidden, 1, activation='gelu', name='ssl_dec_in')(inp)
    h = keras.layers.Conv1D(hidden, 5, padding='same', activation='gelu',
                            name='ssl_dec_mix')(h)
    out = keras.layers.Conv1D(step_samples * in_channels, 1, name='ssl_dec_out')(h)
    return keras.Model(inp, out, name=name)


class MaskedReconstruction(keras.Model):
    """Backbone + decoder trained to fill in masked spans and masked leads.

    Keeps `backbone` as an attribute so `pretrain` can save exactly those weights - the
    decoder never leaves this class.
    """

    def __init__(self, backbone, in_channels, steps, step_samples,
                 mask_ratio=None, span_steps=None, lead_mask_prob=None, **kw):
        super().__init__(**kw)
        self.backbone = backbone
        self.in_channels = int(in_channels)
        self.steps = int(steps)
        self.step_samples = int(step_samples)
        self.mask_ratio = config.SSL_MASK_RATIO if mask_ratio is None else float(mask_ratio)
        self.span_steps = (config.SSL_MASK_SPAN_STEPS if span_steps is None
                           else int(span_steps))
        self.lead_mask_prob = (config.SSL_LEAD_MASK_PROB if lead_mask_prob is None
                               else float(lead_mask_prob))
        self.decoder = build_decoder(self.steps, backbone.output_shape[-1],
                                     self.in_channels, self.step_samples)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.rel_tracker = keras.metrics.Mean(name='nmse')

    @property
    def metrics(self):
        return [self.loss_tracker, self.rel_tracker]

    def _step_mask(self, batch):
        """(B, steps) in {0, 1}, 1 = masked, in contiguous spans of self.span_steps.

        Built by masking whole SPANS and then upsampling, which is what makes the spans
        contiguous without a loop: draw one Bernoulli per span, repeat it span_steps times.
        """
        n_spans = -(-self.steps // self.span_steps)          # ceil
        masked = tf.random.uniform([batch, n_spans]) < self.mask_ratio
        mask = tf.repeat(tf.cast(masked, tf.float32), self.span_steps, axis=1)
        return mask[:, :self.steps]

    def _lead_mask(self, batch):
        """(B, 1, C) in {0, 1}, 0 on the one lead (if any) dropped for this sample."""
        if self.in_channels < 2:
            return tf.ones([batch, 1, self.in_channels])
        victim = tf.random.uniform([batch], 0, self.in_channels, dtype=tf.int32)
        active = tf.cast(tf.random.uniform([batch]) < self.lead_mask_prob, tf.float32)
        return (1.0 - tf.one_hot(victim, self.in_channels) * active[:, None])[:, None, :]

    def _corrupt(self, signal):
        """(corrupted signal, per-step loss weight, per-lead loss weight)."""
        batch = tf.shape(signal)[0]
        step_mask = self._step_mask(batch)                               # (B, steps)
        lead_mask = self._lead_mask(batch)                               # (B, 1, C)

        sample_mask = tf.repeat(step_mask, self.step_samples, axis=1)[:, :, None]
        corrupted = signal * (1.0 - sample_mask) * lead_mask
        return corrupted, step_mask, lead_mask

    def _target(self, signal):
        """(B, steps, step_samples * C) - the raw samples behind each output step."""
        batch = tf.shape(signal)[0]
        return tf.reshape(signal, [batch, self.steps,
                                   self.step_samples * self.in_channels])

    def _reconstruct(self, signal, training):
        corrupted, step_mask, lead_mask = self._corrupt(signal)
        features = self.backbone(corrupted, training=training)
        prediction = self.decoder(features, training=training)
        target = self._target(signal)

        # An element of the target counts when its step was masked in time OR its lead was
        # dropped. The target lays (step_samples, leads) out samples-major, so tiling the
        # (B, 1, C) lead mask step_samples times along the last axis reproduces that order
        # exactly - element j*C + c is lead c of sample j.
        lead_w = tf.tile(lead_mask, [1, 1, self.step_samples])   # (B, 1, C*step_samples)
        hidden = tf.maximum(step_mask[..., None], 1.0 - lead_w)
        denom = tf.reduce_sum(hidden) + 1e-6
        loss = tf.reduce_sum(tf.square(prediction - target) * hidden) / denom

        # nmse = the same error over the VARIANCE of what was hidden, so 1.0 means exactly
        # "no better than predicting the mean of the hidden samples". The mean is computed
        # over the hidden elements rather than assumed to be zero: the windows are z-scored
        # per lead over the whole 10 s, which says nothing about the mean of one masked
        # 240 ms span. That is the number worth watching - the raw MSE moves with whatever
        # amplitude happens to be in the batch.
        mean = tf.reduce_sum(target * hidden) / denom
        variance = tf.reduce_sum(tf.square(target - mean) * hidden) / denom
        return loss, loss / (variance + 1e-6)

    def train_step(self, signal):
        with tf.GradientTape() as tape:
            loss, nmse = self._reconstruct(signal, training=True)
        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.rel_tracker.update_state(nmse)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, signal):
        loss, nmse = self._reconstruct(signal, training=False)
        self.loss_tracker.update_state(loss)
        self.rel_tracker.update_state(nmse)
        return {m.name: m.result() for m in self.metrics}


def weights_path(keras_name, run_dir=None):
    return os.path.join(run_dir or config.CHECKPOINT_DIR, keras_name,
                        'ssl_backbone.weights.h5')


def pretrain(model_name, epochs=None, batch_size=None, lr=None, steps_per_epoch=None,
             val_steps=40, db_names=None):
    """Pretrain and save the backbone of `model_name`. Returns the weights path."""
    from .train import setup_gpus
    setup_gpus()
    batch_size = batch_size or config.BATCH_SIZE
    epochs = config.SSL_EPOCHS if epochs is None else epochs
    lr = config.SSL_LEARNING_RATE if lr is None else lr
    steps_per_epoch = (config.SSL_STEPS_PER_EPOCH if steps_per_epoch is None
                       else steps_per_epoch)

    model = models.build(model_name)
    backbone = models.sub_model(model, 'backbone')
    step_samples = config.SEGMENT_SAMPLES // config.OUTPUT_STEPS
    print(f"{model_name}: backbone {backbone.count_params():,} params "
          f"({100 * backbone.count_params() / model.count_params():.0f}% of the model), "
          f"features {backbone.output_shape[1:]}")
    print(f"masking      : {100 * config.SSL_MASK_RATIO:.0f}% of steps in spans of "
          f"{config.SSL_MASK_SPAN_STEPS} ({20 * config.SSL_MASK_SPAN_STEPS} ms), "
          f"one lead dropped with p={config.SSL_LEAD_MASK_PROB}")

    trainer = MaskedReconstruction(backbone, config.IN_CHANNELS, config.OUTPUT_STEPS,
                                   step_samples)
    from .train import JIT_COMPILE
    trainer.compile(optimizer=keras.optimizers.Adam(lr),
                    jit_compile=JIT_COMPILE)

    pipeline.check_manifest()
    train_ds = pipeline.load_split('train', batch_size, db_names, signal_only=True)
    val_ds = pipeline.load_split('eval', batch_size, db_names, signal_only=True)
    if steps_per_epoch:
        train_ds = train_ds.repeat()
    if val_steps:
        val_ds = val_ds.take(val_steps)

    out = weights_path(model.name)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    trainer.fit(train_ds, epochs=epochs, validation_data=val_ds,
                steps_per_epoch=steps_per_epoch or None,
                callbacks=[
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
                    # restore_best_weights is what makes `out` the BEST backbone rather than
                    # the last one. Without it a final epoch that regressed - and the last
                    # epoch of a pretraining run regresses often, because the LR schedule is
                    # not tied to the end of training - is what gets handed to `ecgr train`.
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=epochs, restore_best_weights=True,
                        verbose=1),
                ])

    backbone.save_weights(out)
    print(f"\nSSL-pretrained backbone -> {out}")
    print("nmse is the masked reconstruction error over the variance of what was hidden: "
          "1.0 = no better than predicting the mean, so lower is real structure learned")
    return out
