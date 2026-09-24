"""Step-level evaluation: the confusion matrix over the label grid.

This is the cheap metric that drives checkpoint selection during training. It is NOT the
metric the project is judged on - a step-level F1 says nothing about whether a beat was
detected once, twice or not at all - so it exists to rank checkpoints, and EC57 (bxb) decides.

Steps with no trusted label (an all-zero one-hot row, config.IGNORE_LABEL) are left out of
the matrix entirely: the 50 unlabelled seconds of a 60 s strip are neither right nor wrong.

Two ways in:

  * `StepConfusion`, a Keras metric, accumulates the matrix DURING the validation pass
    Keras already runs, and returns the weighted F1 as its scalar. That is what training
    uses.
  * `evaluate(model, dataset)`, a plain loop, for the standalone `stepeval` stage where no
    fit() is running.
"""
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from .. import config


def beat_output(outputs):
    """The beat softmax of a model call: the first output of a two-output model, or the only
    output of a legacy single-output one."""
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    if isinstance(outputs, dict):
        return outputs['beat_cls']
    return outputs


def beat_target(targets):
    """The one-hot beat labels of a dataset element: the 'beat_cls' entry, or the tensor."""
    if isinstance(targets, dict):
        return targets['beat_cls']
    return targets


def confusion(model, dataset):
    """Step-level confusion matrix of `model` on `dataset`, accumulated batch by batch.

    Accumulated rather than collected: the eval split is ~100k strips x 3000 steps = 300M
    labels, and two Python lists that size cost tens of GB.
    """
    n = len(config.CLASS_NAMES)
    cm = np.zeros((n, n), dtype=np.int64)
    predict = tf.function(lambda x: beat_output(model(x, training=False)),
                          reduce_retracing=True)
    for x, y in dataset:
        y = beat_target(y).numpy()
        labelled = y.sum(axis=-1).reshape(-1) > 0.5
        preds = np.argmax(predict(x).numpy(), axis=-1).reshape(-1)[labelled]
        truth = np.argmax(y, axis=-1).reshape(-1)[labelled]
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
    publishes the value in `logs` - which is what every callback that monitors the F1
    (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint) reads. Steps whose target row has no
    mass are dropped before the matrix is updated.

    The full matrix stays available in `.matrix()` for the per-epoch confusion report.
    """

    def __init__(self, num_classes=None, name='weighted_f1', **kw):
        super().__init__(name=name, **kw)
        self.num_classes = int(num_classes or config.NUM_CLASSES)
        self.total = self.add_weight(shape=(self.num_classes, self.num_classes),
                                     initializer='zeros', name='cm', dtype='int64')

    def update_state(self, y_true, y_pred, sample_weight=None):
        labelled = tf.reshape(tf.reduce_sum(y_true, axis=-1), [-1]) > 0.5
        truth = tf.boolean_mask(tf.reshape(tf.argmax(y_true, axis=-1), [-1]), labelled)
        pred = tf.boolean_mask(tf.reshape(tf.argmax(y_pred, axis=-1), [-1]), labelled)
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


def lead_quality_report(model, dataset, corrupt=True, batches=8, seed=1234):
    """How well output 2 ranks the leads, measured where the answer is KNOWN.

    Takes `batches` batches of `dataset` (an evaluation split), corrupts them with the very
    pipeline the target is defined by (data/pipeline.corrupt) under a fixed seed, and scores
    the lead_quality head against that target:

        lead_acc   fraction of samples whose predicted best lead is the target's best lead,
                   over samples where the target has a clear winner (margin >= 0.2)
        mae        mean |q_pred - q_target| over every step and lead
        sep        mean q_pred on steps the target calls readable (>= 0.5) minus the mean on
                   steps it calls unreadable - the head's contrast

    Returns a dict (all None for a model without the head).
    """
    from ..data import pipeline
    if not model_has_quality(model):
        return {'lead_acc': None, 'mae': None, 'sep': None, 'samples': 0}
    tf.random.set_seed(seed)
    acc_n = acc_hit = 0
    abs_err = abs_n = 0.0
    good_sum = good_n = bad_sum = bad_n = 0.0
    for x, _ in dataset.take(batches):
        if corrupt:
            x, target = pipeline.corrupt(x)
        else:
            target = pipeline.clean_quality_target(x)
        pred = model(x, training=False)[1].numpy()
        target = target.numpy()
        abs_err += float(np.abs(pred - target).sum())
        abs_n += target.size
        readable = target >= 0.5
        good_sum += float(pred[readable].sum())
        good_n += float(readable.sum())
        bad_sum += float(pred[~readable].sum())
        bad_n += float((~readable).sum())
        t_mean, p_mean = target.mean(axis=1), pred.mean(axis=1)          # (b, leads)
        order = np.sort(t_mean, axis=1)
        clear = (order[:, -1] - order[:, -2]) >= 0.2
        acc_n += int(clear.sum())
        acc_hit += int((t_mean.argmax(1) == p_mean.argmax(1))[clear].sum())
    return {'lead_acc': acc_hit / acc_n if acc_n else None,
            'mae': abs_err / abs_n if abs_n else None,
            'sep': ((good_sum / good_n) if good_n else 0.0) - ((bad_sum / bad_n) if bad_n else 0.0),
            'samples': acc_n}


def model_has_quality(model):
    """True when the model emits the lead_quality output as its second output."""
    outputs = getattr(model, 'outputs', None)
    return bool(outputs) and len(outputs) >= 2


def format_report(df_cm, per_class, weighted_f1, title=None, quality=None):
    lines = ([f"{title}", "=" * len(title), ""] if title else [])
    lines += ["Confusion matrix (labelled steps):", df_cm.to_string(), "",
              "Per-class metrics:",
              per_class.to_string(index=False,
                                  float_format=lambda v: f"{v:.4f}"), "",
              f"Weighted F1 (N/V/S): {weighted_f1:.4f}"]
    if quality and quality.get('lead_acc') is not None:
        lines += ["", f"Lead quality (output 2) on a corrupted probe of {quality['samples']} "
                      f"samples: best-lead accuracy {quality['lead_acc']:.3f}, "
                      f"MAE {quality['mae']:.3f}, readable-vs-unreadable separation "
                      f"{quality['sep']:.3f}"]
    return "\n".join(lines)
