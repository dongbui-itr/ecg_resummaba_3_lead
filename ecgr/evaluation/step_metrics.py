"""Step-level evaluation: the confusion matrix over the 500-step label grid.

This is the cheap metric that drives checkpoint selection during training. It is NOT the
metric the project is judged on - a step-level F1 says nothing about whether a beat was
detected once, twice or not at all - so it exists to rank checkpoints, and EC57 (bxb) decides.

Two ways in:

  * `StepConfusion`, a Keras metric, accumulates the matrix DURING the validation pass
    Keras already runs, and returns the weighted F1 as its scalar. That is what training
    uses. The previous arrangement computed the same number in a callback with its own full
    pass over the eval split, so every epoch walked ~600k segments twice.
  * `evaluate(model, dataset)`, a plain loop, for the standalone `stepeval` stage where no
    fit() is running.
"""
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from .. import config


def confusion(model, dataset):
    """Step-level confusion matrix of `model` on `dataset`, accumulated batch by batch.

    Accumulated rather than collected: the eval split is ~600k segments x 500 steps = 300M
    labels, and two Python lists that size cost tens of GB.
    """
    n = len(config.CLASS_NAMES)
    cm = np.zeros((n, n), dtype=np.int64)
    predict = tf.function(lambda x: model(x, training=False),
                          reduce_retracing=True)
    for x, y in dataset:
        preds = np.argmax(predict(x).numpy(), axis=-1).reshape(-1)
        truth = np.argmax(y.numpy(), axis=-1).reshape(-1)
        # bincount over truth*n + pred is this batch's confusion matrix, flattened
        cm += np.bincount(truth * n + preds, minlength=n * n).reshape(n, n)
    return cm


def weighted_f1_from_confusion(cm):
    """Support-weighted F1 over the beat classes only, from a (C, C) matrix.

    'None' is ~84% of all steps; including it would make the score a measure of background
    detection, which is not the task.
    """
    cm = np.asarray(cm, dtype=np.float64)
    f1s, support = [], []
    for i in range(1, cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        support.append(cm[i, :].sum())
    total = sum(support)
    return float(np.average(f1s, weights=support)) if total else 0.0


def metrics_from_confusion(cm):
    """(confusion DataFrame, per-class DataFrame, weighted F1 over N/V/S)."""
    names = config.CLASS_NAMES
    cm = np.asarray(cm, dtype=np.int64)
    rows = []
    for i, name in enumerate(names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({'Class': name, 'Support': int(cm[i, :].sum()),
                     'Precision': precision, 'Recall': recall, 'F1': f1})
    per_class = pd.DataFrame(rows)
    return (pd.DataFrame(cm, index=names, columns=names), per_class,
            weighted_f1_from_confusion(cm))


def evaluate(model, dataset):
    return metrics_from_confusion(confusion(model, dataset))


@keras.saving.register_keras_serializable(package='resumamba_seq2seq')
class StepConfusion(keras.metrics.Metric):
    """Accumulates the step-level confusion matrix; `result()` is the weighted F1 over N/V/S.

    Registered as a metric so Keras runs it inside the validation pass it already does and
    publishes the value as `val_weighted_f1` in `logs` - which is what every callback that
    monitors the F1 (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint) reads. Two things
    that used to matter stop mattering: callback ORDER (a callback publishing into another
    callback's `logs` dict had to be listed first) and the second eval pass.

    The full matrix stays available in `.matrix()` for the per-epoch confusion report.
    """

    def __init__(self, num_classes=None, name='weighted_f1', **kw):
        super().__init__(name=name, **kw)
        self.num_classes = int(num_classes or config.NUM_CLASSES)
        self.total = self.add_weight(shape=(self.num_classes, self.num_classes),
                                     initializer='zeros', name='cm', dtype='int64')

    def update_state(self, y_true, y_pred, sample_weight=None):
        truth = tf.reshape(tf.argmax(y_true, axis=-1), [-1])
        pred = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])
        cm = tf.math.confusion_matrix(truth, pred, num_classes=self.num_classes,
                                      dtype='int64')
        self.total.assign_add(cm)

    def result(self):
        cm = tf.cast(self.total, tf.float32)
        # Beat classes only, same reason as weighted_f1_from_confusion
        diag = tf.linalg.diag_part(cm)[1:]
        predicted = tf.reduce_sum(cm, axis=0)[1:]
        actual = tf.reduce_sum(cm, axis=1)[1:]
        precision = tf.math.divide_no_nan(diag, predicted)
        recall = tf.math.divide_no_nan(diag, actual)
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        return tf.math.divide_no_nan(tf.reduce_sum(f1 * actual), tf.reduce_sum(actual))

    def matrix(self):
        """The accumulated matrix as a numpy array."""
        return self.total.numpy()

    def reset_state(self):
        self.total.assign(tf.zeros_like(self.total))

    def get_config(self):
        return {**super().get_config(), 'num_classes': self.num_classes}


def format_report(df_cm, per_class, weighted_f1, title=None):
    lines = ([f"{title}", "=" * len(title), ""] if title else [])
    lines += ["Confusion matrix (steps):", df_cm.to_string(), "",
              "Per-class metrics:",
              per_class.to_string(index=False,
                                  float_format=lambda v: f"{v:.4f}"), "",
              f"Weighted F1 (N/V/S): {weighted_f1:.4f}"]
    return "\n".join(lines)
