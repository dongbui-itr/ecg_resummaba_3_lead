"""Callbacks, chiefly the one that turns step metrics into checkpoint selection."""
import os

import numpy as np
import tensorflow as tf

from ..evaluation import step_metrics


def resolve_monitor(logs, monitor):
    """The key of `monitor` in `logs`, tolerant of Keras's output-name prefix.

    With two outputs Keras publishes `val_beat_cls_weighted_f1`; with one (a legacy model)
    `val_weighted_f1`. Asking for either finds the other, so a config written for the
    two-output family still drives a single-output run - and vice versa.
    """
    if monitor in logs:
        return monitor
    tail = monitor.split('_', 1)[1] if monitor.startswith('val_') else monitor
    for key in logs:
        if key.endswith(tail) and key.startswith('val_') == monitor.startswith('val_'):
            return key
    return monitor


class WeightedF1Checkpoint(tf.keras.callbacks.Callback):
    """Writes the per-epoch confusion report and keeps the best-F1 weights.

    The F1 itself is no longer computed here: `step_metrics.StepConfusion` is a compiled
    metric, so Keras produces the weighted F1 and the full confusion matrix inside the
    validation pass it already runs. This callback reads both.

    That matters for two reasons beyond tidiness:

      * The eval split is walked ONCE per epoch instead of twice.
      * The monitor is in `logs` before any callback runs, so EarlyStopping,
        ReduceLROnPlateau and ModelCheckpoint can monitor it whatever order they are listed
        in. Publishing a monitor key FROM a callback made the list order load-bearing, and a
        monitor key that is missing early makes those callbacks skip silently.

    Why the F1 over N/V/S and not val_loss: 'None' is ~84% of the label steps, so val_loss
    and val_accuracy are both dominated by background detection and move very little with
    beat quality - and with the default poly2 loss val_loss moves the WRONG WAY (see
    training/losses.py).
    """

    def __init__(self, metric, ckpt_dir, report_dir, interval=1, save_start_epoch=1,
                 monitor='val_beat_cls_weighted_f1', quality_probe=None):
        super().__init__()
        self.metric = metric
        self.monitor = monitor
        self.best_dir = os.path.join(ckpt_dir, 'BEST_F1')
        self.report_dir = report_dir
        self.interval = interval
        # First epoch (1-indexed) whose weights may be written out. Measuring and saving are
        # separate on purpose: the F1 has to be computed EVERY epoch because it is the
        # monitor, while an early epoch's weights are not worth keeping. Measured on this
        # family: the 1M size swings between 0.8267 and 0.8519 over its first nine epochs
        # and then settles into 0.8467-0.8523 for the next fourteen, so a "best" picked from
        # epoch 4 records the noise, not the model.
        self.save_start_epoch = save_start_epoch
        # A dataset to probe output 2 on each epoch (step_metrics.lead_quality_report), or
        # None. The eval split's own quality target only knows about flat leads, so the head
        # is measured on a fixed-seed corrupted probe where the answer is known.
        self.quality_probe = quality_probe
        self.best_f1 = 0.0
        os.makedirs(self.best_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        key = resolve_monitor(logs, self.monitor)
        f1 = logs.get(key)
        if f1 is None:
            raise RuntimeError(
                f"{self.monitor} is not in this epoch's logs ({sorted(logs)}). The model has "
                f"to be compiled with step_metrics.StepConfusion(name='weighted_f1') on its "
                f"beat output and given validation data for it to exist.")
        if (epoch + 1) % self.interval:
            return

        matrix = self.metric.matrix()
        if matrix.sum() == 0:
            return                                  # no validation pass this epoch
        df_cm, per_class, cm_f1 = step_metrics.metrics_from_confusion(matrix)
        quality = None
        if self.quality_probe is not None:
            quality = step_metrics.lead_quality_report(self.model, self.quality_probe)
            for name in ('lead_acc', 'mae', 'sep'):
                if quality.get(name) is not None:
                    logs[f'val_quality_{name}'] = float(quality[name])
        report = step_metrics.format_report(df_cm, per_class, cm_f1, f"EPOCH {epoch + 1}",
                                            quality=quality)
        with open(os.path.join(self.report_dir, 'confusion_log.txt'), 'a') as f:
            f.write("\n" + report + "\n")
        print(f"\n{report}\n")
        if not np.isclose(cm_f1, float(f1), atol=1e-4):
            # The matrix and the logged scalar come from the same accumulator, so a
            # disagreement means the metric was reset between them - worth saying out loud
            # rather than silently reporting two different numbers.
            print(f"  note: logged {key}={float(f1):.4f} but the matrix gives {cm_f1:.4f}")

        if epoch + 1 < self.save_start_epoch:
            print(f"  epoch {epoch + 1} < save_start_epoch {self.save_start_epoch}: "
                  f"measured only, no checkpoint")
            return

        if cm_f1 > self.best_f1:
            self.best_f1 = cm_f1
            base = f"best_model_f1_{cm_f1:.4f}_epoch_{epoch + 1}"
            path = os.path.join(self.best_dir, base + ".keras")
            self.model.save(path)
            with open(os.path.join(self.report_dir, base + "_metrics.txt"), 'w') as f:
                f.write(f"Checkpoint: {path}\n\n" + report + "\n")
            print(f"new best weighted F1 -> {path}")

    def on_train_end(self, logs=None):
        with open(os.path.join(self.report_dir, 'best_model_summary.txt'), 'w') as f:
            f.write(f"best weighted F1 (N/V/S): {self.best_f1:.4f}\n{self.best_dir}\n")


class DelayedModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """ModelCheckpoint that writes nothing before `start_epoch` (1-indexed).

    Keras's own ModelCheckpoint has no such option, and simply suppressing the write is not
    enough: its `best` is seeded on the first epoch it sees, so a warm-up epoch would still
    set the bar every later epoch has to beat. Skipping the epoch entirely leaves `best` at
    its initial infinity, which is what makes `start_epoch` the true first candidate.
    """

    def __init__(self, *args, start_epoch=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 < self.start_epoch:
            return
        super().on_epoch_end(epoch, logs)


class UnfreezeBackbone(tf.keras.callbacks.Callback):
    """Make the SSL-pretrained backbone trainable again at epoch `at_epoch` (1-indexed).

    Optional warm-up (config.FREEZE_BACKBONE_EPOCHS, 0 = off): with a pretrained backbone and
    a random head, the first gradients are dominated by the head's error and can undo the
    pretraining before the head is any good. Holding the backbone still for a few epochs
    lets the head catch up first. Recompiling is required for Keras to rebuild the train
    function against the new trainable set.

    The recompile builds a FRESH optimizer, so Adam's moment estimates for the head are
    discarded at the unfreeze. That is the cost of the warm-up and the reason it is off by
    default: a few hundred steps of re-accumulation against a whole run's worth of epochs.
    """

    def __init__(self, backbone, at_epoch, recompile):
        super().__init__()
        self.backbone = backbone
        self.at_epoch = int(at_epoch)
        self.recompile = recompile

    def on_epoch_begin(self, epoch, logs=None):
        if epoch + 1 == self.at_epoch and not self.backbone.trainable:
            self.backbone.trainable = True
            self.recompile()
            print(f"\nepoch {epoch + 1}: backbone unfrozen "
                  f"({self.backbone.count_params():,} params now training)")
