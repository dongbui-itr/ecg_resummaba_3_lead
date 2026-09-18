"""Self-supervised (CPC) pretraining of the context encoder.

This is the SECOND self-supervised stage. training/ssl.py pretrains the backbone - where
nearly every parameter is - by masked reconstruction; this one trains the small context
encoder whose embedding drives the AdaIN conditioning. They are separate because the two
sub-networks want opposite things from a representation: the backbone must keep the beat
morphology, the context encoder must summarise the nuisance variation AdaIN then divides out.

Section 3.3 of the paper: a convolutional embedding network maps short ECG windows to
latents v_i, an autoregressive model turns the past into context vectors c_i, and InfoNCE
trains c_i to predict the latent v_{i+j} that actually follows it - against negatives drawn
from other time indices and other recordings in the batch. No labels are involved, which is
the whole point: the encoder learns what is stable about a recording (baseline morphology,
amplitude, rhythm regime) rather than what a beat looks like. That is exactly the nuisance
variation AdaIN then divides out.

What differs from the paper is the calibration length. The paper has 60 s of unlabeled
calibration per patient; these tfrecords are 10 s strips with no patient identity, so the
windows come from inside one strip: 2 s windows at 50% overlap give M = 9 per strip, and
with horizon j in {1, 2} that is 15 positive pairs per strip. The strip itself is then the
unit that has to be told apart, which is the signal AdaIN consumes downstream.

Labels in the tfrecords are parsed away - only the signal is read - so this stage runs over
the same training split without touching the eval studies.

After training, the weights are written next to the model's checkpoints as
`cpc_context.weights.h5`; `ecgr train` loads them and freezes the encoder, the way the paper
keeps its patient encoder frozen for both training and inference.
"""
import os

import keras
import tensorflow as tf

from .. import config, models
from ..data import pipeline

HORIZONS = (1, 2)      # the paper fixes j = 2; predicting j = 1 too costs one matrix


def context_encoder_of(model):
    """The context-encoder sub-model inside a built ResUMamba model."""
    return models.sub_model(model, 'context_encoder')


def weights_path(keras_name, run_dir=None):
    return os.path.join(run_dir or config.CHECKPOINT_DIR, keras_name,
                        'cpc_context.weights.h5')


class CPCTrainer(keras.Model):
    """InfoNCE over (context vector, future latent) pairs.

    score_j(c_i, v) = (W_j c_i)^T v  (paper eq. 5), and for each (i, j) the true v_{i+j} has
    to beat every other latent in the batch - other time indices AND other strips (eq. 6).
    The negatives are what force the embedding to be strip-specific: a representation that
    described "an ECG window" in general could not pick its own strip's future out of the
    pile.
    """

    def __init__(self, encoder, dim, horizons=HORIZONS, temperature=1.0, **kw):
        super().__init__(**kw)
        self.encoder = encoder
        self.dim = int(dim)
        # Window count is a property of the ARCHITECTURE (strip length / window / hop), not
        # of the batch, so take it from the encoder's output signature and slice with it.
        # Taking it from tf.shape() instead loses the static last dimension once the batch
        # axis is dynamic, and the predictors below then cannot be built.
        self.n_win = int(encoder.output_shape[1][1])
        self.horizons = tuple(j for j in horizons if j < self.n_win)
        self.temperature = float(temperature)
        self.predictors = [keras.layers.Dense(dim, use_bias=False, name=f'W{j}')
                           for j in self.horizons]
        for predictor in self.predictors:
            predictor.build((None, None, self.dim))
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.acc_tracker = keras.metrics.Mean(name='top1')

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def _info_nce(self, signal, training):
        _, v_seq, c_seq = self.encoder(signal, training=training)   # (B, M, D) each
        batch = tf.shape(v_seq)[0]
        n_win, dim = self.n_win, self.dim

        # L2-normalise: without it the score is dominated by embedding norm, and the loss can
        # be driven down by inflating ||v|| instead of by predicting anything.
        v_n = tf.math.l2_normalize(v_seq, axis=-1)
        flat_v = tf.reshape(v_n, [batch * n_win, dim])              # the negative pool

        total_loss, total_acc, terms = 0.0, 0.0, 0
        for j, predictor in zip(self.horizons, self.predictors):
            ctx = c_seq[:, :n_win - j]                              # contexts with a future
            pred = tf.math.l2_normalize(predictor(ctx), axis=-1)    # (B, M-j, D)
            n_ctx = n_win - j

            logits = tf.matmul(tf.reshape(pred, [batch * n_ctx, dim]), flat_v,
                               transpose_b=True) / self.temperature
            # index of v_{i+j} inside flat_v, for every (b, i)
            rows = tf.range(batch)[:, None] * n_win + tf.range(n_ctx)[None, :] + j
            target = tf.reshape(rows, [-1])

            total_loss += tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(target, logits))
            total_acc += tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(logits, axis=-1, output_type=tf.int32), target), tf.float32))
            terms += 1
        return total_loss / terms, total_acc / terms

    def train_step(self, signal):
        with tf.GradientTape() as tape:
            loss, acc = self._info_nce(signal, training=True)
        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, signal):
        loss, acc = self._info_nce(signal, training=False)
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {m.name: m.result() for m in self.metrics}


def pretrain(model_name, epochs=5, batch_size=None, lr=1e-3, steps_per_epoch=400,
             val_steps=40, db_names=None):
    """Pretrain and save the context encoder of `model_name`. Returns the weights path."""
    from .train import setup_gpus
    setup_gpus()
    batch_size = batch_size or config.BATCH_SIZE

    model = models.build(model_name)
    encoder = context_encoder_of(model)
    dim = encoder.output_shape[0][-1]
    n_win = encoder.output_shape[1][1]
    print(f"{model_name}: context encoder {encoder.count_params():,} params, dim {dim}, "
          f"{n_win} windows per strip")
    print(f"negatives per step: batch {batch_size} x windows -> the pool the true future "
          f"latent has to win against")

    trainer = CPCTrainer(encoder, dim)
    from .train import JIT_COMPILE
    trainer.compile(optimizer=keras.optimizers.Adam(lr),
                    jit_compile=JIT_COMPILE)

    pipeline.check_manifest()
    # lead_jitter=True: the encoder has to describe a strip whose montage is degenerate (one
    # lead repeated, or one lead flat) as readily as a clean 3-lead one, because that is what
    # the EC57 stage feeds it.
    train_ds = pipeline.load_split('train', batch_size, db_names, signal_only=True,
                                   lead_jitter=True)
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
                    # Same reason as in ssl.py: save the best encoder, not the last one.
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=epochs, restore_best_weights=True,
                        verbose=1),
                ])

    encoder.save_weights(out)
    print(f"\nCPC-pretrained context encoder -> {out}")
    print("top1 well above chance (1 / (batch * windows)) means the encoder learned "
          "strip-specific structure, which is what AdaIN consumes")
    return out
