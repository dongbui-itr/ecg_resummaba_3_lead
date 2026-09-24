"""Loss functions.

The class imbalance here is extreme - 'None' is ~84% of all LABELLED steps and S ~1% - and
the paper this model comes from handles it by resampling (SMOTE-Tomek). That has no analogue
in a seq2seq contract: you cannot interpolate between two strips and keep a 3000-step label
sequence meaningful. So the imbalance is handled in the loss instead.

Every beat loss here is MASKED: a target row with zero mass (config.IGNORE_LABEL - outside
the reviewed span of a strip, or padding) contributes nothing, and the mean runs over the
labelled steps only. Averaging over all steps instead would let the 50 unlabelled seconds of
a 60 s strip dilute every gradient by 5x, and (worse) would score a "None" prediction on
steps where nobody knows whether a beat is there.
"""
import tensorflow as tf

from .. import config


def _masked_mean(per_step, mass):
    """Mean of `per_step` over the steps whose target has mass, per batch."""
    labelled = tf.cast(mass > 0.5, per_step.dtype)
    return tf.reduce_sum(per_step * labelled) / tf.maximum(tf.reduce_sum(labelled), 1.0)


def weighted_categorical_crossentropy(weights=None):
    """Per-class weighted CE. `weights` indexes the same way as config.CLASS_NAMES."""
    w = tf.constant(weights if weights is not None else config.CLASS_WEIGHTS, tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        mass = tf.reduce_sum(y_true, axis=-1)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred) * w, axis=-1)
        return _masked_mean(ce, mass)
    return loss


def weighted_poly_crossentropy(weights=None, eps1=1.0, eps2=0.0):
    """PolyLoss (Leng et al., ICLR 2022): CE plus the first terms of its Taylor expansion.

    CE = sum_j (1-pt)^j / j; adding eps1*(1-pt) re-weights the leading term, which lifts the
    gradient on examples the model is already fairly sure about. On this data that is where
    the hard S beats live, so it behaves like a gentler focal loss.
    """
    w = tf.constant(weights if weights is not None else config.CLASS_WEIGHTS, tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        mass = tf.reduce_sum(y_true, axis=-1)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1) / tf.maximum(mass, 1e-7)
        weight = tf.reduce_sum(y_true * w, axis=-1)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        poly = ce + eps1 * (1.0 - pt) + eps2 * tf.square(1.0 - pt)
        return _masked_mean(weight * poly, mass)
    return loss


def weighted_poly2_crossentropy(weights=None, eps1=None, eps2=None):
    """Poly-2 loss exactly as the ResUMamba paper states it (sec. 3.6).

        L = CE + eps1*(1 - Pt) + eps2*(1 - Pt)^2,   eps1 = 0.3, eps2 = -0.5

    This is NOT the same thing as `poly` above. PolyLoss adds a positive first-order term;
    Poly-2 adds a second-order term with a NEGATIVE coefficient, which is what the paper's
    grid search landed on (flat optimum over the whole [-1.5, 1.5]^2 grid, so the values are
    not load-bearing magic numbers).

    Two consequences worth knowing before using it:

    * It trains well - it is what the shipped checkpoints were trained with.
    * Its VALUE is a bad ranking signal. d/du (0.3u - 0.5u^2) = 0.3 - u with u = 1 - Pt, so
      below Pt = 0.7 a worse prediction LOWERS the term. Measured on resumamba_1m: val_loss
      bottoms out at epoch 1 (0.0989) and rises monotonically while the weighted F1 keeps
      climbing to epoch 7. Stop on the weighted F1, never on val_loss, with this loss.

    The class weights multiply the polynomial terms too, so the correction keeps the same
    per-class emphasis as the CE term instead of quietly flattening it.
    """
    w = tf.constant(weights if weights is not None else config.CLASS_WEIGHTS, tf.float32)
    e1 = config.POLY2_EPS[0] if eps1 is None else eps1
    e2 = config.POLY2_EPS[1] if eps2 is None else eps2

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred) * w, axis=-1)
        mass = tf.reduce_sum(y_true, axis=-1)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1) / tf.maximum(mass, 1e-7)
        cw = tf.reduce_sum(y_true * w, axis=-1)
        return _masked_mean(ce + cw * (e1 * (1.0 - pt) + e2 * tf.square(1.0 - pt)), mass)
    return loss


def lead_quality_loss(y_true, y_pred):
    """Binary cross-entropy between the sigmoid lead-quality output and its soft target.

    The target is in [0, 1] (data/pipeline.quality_from_corruption), not {0, 1}, so this is
    the cross-entropy against a soft label: the minimiser is the target itself, and a lead
    the pipeline degraded "a little" is asked to score "a little" lower. Every step and lead
    has a target - quality is defined everywhere the signal is - so nothing is masked.
    """
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
    bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(bce)


LOSSES = {
    'wce': weighted_categorical_crossentropy,
    'poly': weighted_poly_crossentropy,
    'poly2': weighted_poly2_crossentropy,      # config.LOSS, the paper's
}
