"""Loss and metrics of the rhythm model - one set per output.

'rhythm'  target (batch, 10, NUM_CLASSES + 1) = one-hot rhythm (all zero on an IGNORE second)
          | loss weight of that second (augment.targets); prediction (batch, 10, NUM_CLASSES)
'lead'    target (batch, NUM_LEAD_CLASSES) one-hot; prediction the same shape
'noise'   target (batch, NOISE_SEGMENTS, 2) one-hot CLEAN / NOISE; prediction the same shape
'channel' target (batch, NOISE_SEGMENTS, NUM_LEAD_CLASSES) one-hot NOISE / CH1..CH3 per 2 s
          segment (the dual U-Net); loss lead_loss, metric LeadConfusion

The weight rides inside the rhythm target rather than as a Keras sample_weight because it is
per SECOND: 0 on an IGNORE second, NOISY_SECOND_WEIGHT on a noisy one, 1 otherwise.
"""
import keras
import numpy as np
import tensorflow as tf

from . import config as rc

K = rc.NUM_CLASSES
PKG = 'ecgr_rhythm'


def rhythm_loss(class_weights=None, label_smoothing=None):
    w = tf.constant(class_weights if class_weights is not None else [1.0] * K, tf.float32)
    ls = rc.LABEL_SMOOTHING if label_smoothing is None else label_smoothing

    def loss(y_true, y_pred):
        onehot, weight = y_true[..., :K], y_true[..., K]
        p = tf.clip_by_value(y_pred, 1e-7, 1.0)
        target = onehot * (1.0 - ls) + ls / K
        ce = -tf.reduce_sum(target * tf.math.log(p), axis=-1) * \
            tf.reduce_sum(onehot * w, axis=-1)
        return weight * ce
    loss.__name__ = 'rhythm_loss'
    return loss


def lead_loss(label_smoothing=None):
    ls = rc.LABEL_SMOOTHING if label_smoothing is None else label_smoothing
    return keras.losses.CategoricalCrossentropy(label_smoothing=ls, name='lead_loss')


def beat_loss(class_weights=None, label_smoothing=None):
    """'beat' target (batch, BEAT_STEPS, 6) = [heat | N S V | w_heat | w_type]; prediction
    (batch, BEAT_STEPS, 4) softmax none/N/S/V. Per step:
      w_heat * BCE(heat, p(beat) = 1 - p(none))  (positives x BEAT_HEAT_POS_WEIGHT)
    + w_type * BEAT_TYPE_LOSS_WEIGHT * class_weight * CE(N/S/V one-hot, p(N/S/V) renormalised)
    Where and what are learnt by different terms, so neither fights the other over 8 ms."""
    nb = len(rc.BEAT_CLASSES)
    cw = class_weights if class_weights is not None else [1.0] * nb
    w = tf.constant(list(cw)[1:], tf.float32)                            # N, S, V
    ls = rc.LABEL_SMOOTHING if label_smoothing is None else label_smoothing
    pos_w = float(rc.BEAT_HEAT_POS_WEIGHT)
    type_w = float(rc.BEAT_TYPE_LOSS_WEIGHT)

    def loss(y_true, y_pred):
        heat, onehot = y_true[..., 0], y_true[..., 1:nb]
        w_heat, w_type = y_true[..., nb], y_true[..., nb + 1]
        p = tf.clip_by_value(y_pred, 1e-6, 1.0)
        p_none = p[..., 0]
        p_beat = tf.clip_by_value(1.0 - p_none, 1e-6, 1.0)
        bce = -(pos_w * heat * tf.math.log(p_beat) + (1.0 - heat) * tf.math.log(p_none))
        q = p[..., 1:] / tf.reduce_sum(p[..., 1:], axis=-1, keepdims=True)
        q = tf.clip_by_value(q, 1e-6, 1.0)
        target = onehot * (1.0 - ls) + ls / (nb - 1)
        ce = -tf.reduce_sum(target * tf.math.log(q), axis=-1) * tf.reduce_sum(onehot * w, axis=-1)
        return w_heat * bce + type_w * w_type * ce
    loss.__name__ = 'beat_loss'
    return loss


def noise_loss(label_smoothing=None):
    """'noise' (batch, NOISE_SEGMENTS, 2): categorical CE per 2 s segment."""
    ls = rc.LABEL_SMOOTHING if label_smoothing is None else label_smoothing
    return keras.losses.CategoricalCrossentropy(label_smoothing=ls, name='noise_loss')


def _f1(cm):
    cm = tf.cast(cm, tf.float32)
    diag = tf.linalg.diag_part(cm)
    precision = tf.math.divide_no_nan(diag, tf.reduce_sum(cm, axis=0))
    recall = tf.math.divide_no_nan(diag, tf.reduce_sum(cm, axis=1))
    return tf.math.divide_no_nan(2 * precision * recall, precision + recall)


@keras.saving.register_keras_serializable(package=PKG)
class RhythmF1(keras.metrics.Metric):
    """Per-second confusion over the six rhythm classes; result() is the macro F1.

    Only labelled CLEAN seconds count (weight 1 in the target): the rhythm of a second of pure
    artefact is not something the model can be asked to read, which is exactly why its loss
    weight is reduced. Macro, not support-weighted: SINUS is most of the seconds, and a
    weighted F1 would be a SINUS score. Classes absent from the split are left out of the mean;
    a present class that is never predicted scores 0.
    """

    def __init__(self, name='f1', **kw):
        super().__init__(name=name, **kw)
        self.cm = self.add_weight(shape=(K, K), initializer='zeros', name='cm', dtype='int64')

    def update_state(self, y_true, y_pred, sample_weight=None):
        valid = y_true[..., K] > 0.5
        truth = tf.boolean_mask(tf.argmax(y_true[..., :K], axis=-1), valid)
        pred = tf.boolean_mask(tf.argmax(y_pred, axis=-1), valid)
        self.cm.assign_add(tf.math.confusion_matrix(truth, pred, num_classes=K,
                                                    dtype='int64'))

    def result(self):
        support = tf.reduce_sum(self.cm, axis=1) > 0
        return tf.reduce_mean(tf.boolean_mask(_f1(self.cm), support))

    def matrix(self):
        return self.cm.numpy()

    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))


@keras.saving.register_keras_serializable(package=PKG)
class LeadConfusion(keras.metrics.Metric):
    """Confusion of the 'lead' output (per window) or the 'channel' output (per 2 s segment:
    every segment is one count). result(): F1 of NOISE, or the accuracy."""

    def __init__(self, name='noise_f1', mode='noise_f1', **kw):
        super().__init__(name=name, **kw)
        self.mode = mode
        n = rc.NUM_LEAD_CLASSES
        self.cm = self.add_weight(shape=(n, n), initializer='zeros', name='cm', dtype='int64')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.cm.assign_add(tf.math.confusion_matrix(
            tf.reshape(tf.argmax(y_true, axis=-1), [-1]),
            tf.reshape(tf.argmax(y_pred, axis=-1), [-1]),
            num_classes=rc.NUM_LEAD_CLASSES, dtype='int64'))

    def result(self):
        if self.mode == 'accuracy':
            cm = tf.cast(self.cm, tf.float32)
            return tf.math.divide_no_nan(tf.linalg.trace(cm), tf.reduce_sum(cm))
        noisy = tf.stack([
            tf.stack([tf.reduce_sum(self.cm[1:, 1:]), tf.reduce_sum(self.cm[1:, :1])]),
            tf.stack([tf.reduce_sum(self.cm[:1, 1:]), self.cm[0, 0]])])
        return _f1(noisy)[1]

    def matrix(self):
        return self.cm.numpy()

    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))

    def get_config(self):
        return {**super().get_config(), 'mode': self.mode}


@keras.saving.register_keras_serializable(package=PKG)
class BeatF1(keras.metrics.Metric):
    """Type confusion of the 'beat' output on the TYPED steps (w_type > 0, +-40 ms around each
    annotated beat): reference N/S/V against the argmax of the N/S/V probabilities. result() is
    the macro F1 over N, S, V (the classes present) - what the beat is, not where to 8 ms;
    the beat-level Se / +P at 150 ms come from train.BeatMatchLog. Row / column 0 stay 0."""

    def __init__(self, name='f1', **kw):
        super().__init__(name=name, **kw)
        nb = len(rc.BEAT_CLASSES)
        self.cm = self.add_weight(shape=(nb, nb), initializer='zeros', name='cm', dtype='int64')

    def update_state(self, y_true, y_pred, sample_weight=None):
        nb = len(rc.BEAT_CLASSES)
        valid = y_true[..., nb + 1] > 0
        truth = tf.boolean_mask(tf.argmax(y_true[..., 1:nb], axis=-1), valid) + 1
        pred = tf.boolean_mask(tf.argmax(y_pred[..., 1:], axis=-1), valid) + 1
        self.cm.assign_add(tf.math.confusion_matrix(truth, pred, num_classes=nb, dtype='int64'))

    def result(self):
        f1 = _f1(self.cm)[1:]
        support = tf.reduce_sum(self.cm, axis=1)[1:] > 0
        return tf.math.divide_no_nan(tf.reduce_sum(tf.boolean_mask(f1, support)),
                                     tf.reduce_sum(tf.cast(support, tf.float32)))

    def matrix(self):
        return self.cm.numpy()

    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))


@keras.saving.register_keras_serializable(package=PKG)
class NoiseF1(keras.metrics.Metric):
    """Segment-level CLEAN / NOISE confusion of the 'noise' output; result() is the F1 of
    NOISE (mode='noise_f1') or the accuracy (mode='accuracy')."""

    def __init__(self, name='f1', mode='noise_f1', **kw):
        super().__init__(name=name, **kw)
        self.mode = mode
        self.cm = self.add_weight(shape=(2, 2), initializer='zeros', name='cm', dtype='int64')

    def update_state(self, y_true, y_pred, sample_weight=None):
        t = tf.reshape(tf.argmax(y_true, axis=-1), [-1])
        p = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])
        self.cm.assign_add(tf.math.confusion_matrix(t, p, num_classes=2, dtype='int64'))

    def result(self):
        if self.mode == 'accuracy':
            cm = tf.cast(self.cm, tf.float32)
            return tf.math.divide_no_nan(tf.linalg.trace(cm), tf.reduce_sum(cm))
        return _f1(self.cm)[1]

    def matrix(self):
        return self.cm.numpy()

    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))

    def get_config(self):
        return {**super().get_config(), 'mode': self.mode}


# ---------------------------------------------------------------------------
# numpy side, for reports
# ---------------------------------------------------------------------------

def per_class_table(cm, names):
    cm = np.asarray(cm, dtype=np.float64)
    rows = []
    for i, name in enumerate(names):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        se = tp / (tp + fn) if tp + fn else float('nan')
        ppv = tp / (tp + fp) if tp + fp else float('nan')
        f1 = 2 * se * ppv / (se + ppv) if se + ppv and not np.isnan(se + ppv) else float('nan')
        rows.append(dict(cls=name, support=int(cm[i, :].sum()), se=se, ppv=ppv, f1=f1))
    return rows
