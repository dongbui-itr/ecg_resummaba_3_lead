"""Distil a CfC student from the trained state-space teacher of the same size.

Why distillation and not just training the CfC model on the labels. The swap is
architectural, not a new task: `resucfc_2m` and `resumamba_2m` are the same graph with one
layer per block exchanged, and the teacher is already a converged solution to the exact
problem the student is being asked to solve. Hard labels at 20 ms resolution carry one
number per step; the teacher's softmax carries its whole uncertainty structure - where the
QRS onset is ambiguous, which beats it is unsure about, how it hedges on the N/S boundary -
and that is far more signal per step than the one-hot. For a student with a *smaller
receptive mechanism* that matters most exactly where the old long FIR was doing the work.

Two things make it cheap here:

  * **Shared weights transfer.** Only the mixer layer differs inside each block; in_proj,
    the depthwise 3, both BatchNorms, gate_proj and out_proj have the same name and shape in
    both families, as do the stem, the whole ResU path, the AdaIN conditioning and the head.
    `transfer_shared_weights` copies all of them, so the student starts as the teacher with
    its time mixer replaced by an untrained one, and only that has to be learned.
  * **No second data pass.** The teacher runs inside `train_step` on the same batch, so the
    cost is one extra forward, not a precomputed 2 GB of soft targets.

The loss is the usual pair:

    L = alpha * T^2 * KL(teacher_soft || student_soft) + (1 - alpha) * hard_loss(y, student)

The project's models emit PROBABILITIES, not logits, so temperature is applied as
softmax(log p / T) - the same thing, up to the constant the softmax normalises away. The
T^2 factor keeps the soft-term gradient the same magnitude as the hard one when T changes.
"""
import os

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from .. import config, models
from ..data import pipeline
from ..evaluation import step_metrics
from .callbacks import DelayedModelCheckpoint, WeightedF1Checkpoint
from .losses import LOSSES
from .train import JIT_COMPILE, model_dirs, setup_gpus


def transfer_shared_weights(student, teacher, verbose=True):
    """Copy every weight the two models share by name AND shape. Returns (copied, skipped).

    Name-and-shape rather than position: the two graphs differ by one layer per block, so a
    positional copy would silently shift every weight after the first mixer. A name that
    matches with a different shape is the mixer itself (DiagSSM1D's banks against CfC1D's
    cells) and is left alone - that is what distillation is for.
    """
    have = {w.path: w for w in teacher.weights}
    copied, skipped = [], []
    for w in student.weights:
        src = have.get(w.path)
        if src is not None and tuple(src.shape) == tuple(w.shape):
            w.assign(src)
            copied.append(w.path)
        else:
            skipped.append(w.path)
    if verbose:
        print(f"transfer      : {len(copied)} weights copied from the teacher, "
              f"{len(skipped)} left to train")
        for path in skipped:
            print(f"                cold: {path}")
    return copied, skipped


def distill_loss(hard_loss, temperature=3.0, alpha=0.7):
    """alpha * T^2 * KL(teacher || student) + (1 - alpha) * hard_loss, on probabilities.

    `y_pred` is the (steps, 2*NUM_CLASSES) output of build_distiller: the student's
    distribution followed by the teacher's. Carrying the soft targets in the PREDICTION
    rather than in the label is what lets the teacher run inside the graph, on the GPU, next
    to the student - see build_distiller for why that is not a detail.
    """
    num = config.NUM_CLASSES
    t = float(temperature)
    a = float(alpha)

    def loss(y_true, y_pred):
        student, teacher = y_pred[..., :num], y_pred[..., num:]
        eps = 1e-7
        # softmax(log p / T): the temperature-softened version of a distribution that
        # arrives already normalised.
        ps = tf.nn.softmax(tf.math.log(tf.clip_by_value(student, eps, 1.0)) / t, axis=-1)
        pt = tf.nn.softmax(tf.math.log(tf.clip_by_value(teacher, eps, 1.0)) / t, axis=-1)
        pt = tf.stop_gradient(pt)
        kl = tf.reduce_sum(pt * (tf.math.log(tf.clip_by_value(pt, eps, 1.0))
                                 - tf.math.log(tf.clip_by_value(ps, eps, 1.0))), axis=-1)
        return a * (t * t) * tf.reduce_mean(kl) + (1.0 - a) * hard_loss(y_true, student)
    return loss


class StudentConfusion(step_metrics.StepConfusion):
    """StepConfusion scored on the student half of the distiller's output.

    The monitor stays exactly what every other run in this project monitors - the weighted
    F1 of the student against the REAL labels. The teacher's opinion is a training signal,
    never a yardstick, so it must not touch the number that selects checkpoints.
    """

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, y_pred[..., :config.NUM_CLASSES], sample_weight)


class SaveStudent:
    """Callback mixin: write the STUDENT, not the distiller wrapper.

    `fit` runs the two-branch graph, but what ec57 and stepeval have to load afterwards is a
    plain (2500,3) -> (500,4) model. Pointing the callback's `self.model` at the student
    makes every checkpoint it writes exactly that, with no teacher inside it and no
    unpickling of a subclassed wrapper.
    """

    def __init__(self, *args, student=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._student = student

    def set_model(self, model):
        super().set_model(self._student if self._student is not None else model)


class StudentF1Checkpoint(SaveStudent, WeightedF1Checkpoint):
    pass


class StudentModelCheckpoint(SaveStudent, DelayedModelCheckpoint):
    pass


def build_distiller(student, teacher):
    """x -> concat([student(x), teacher(x)]) on the class axis, one functional graph.

    The teacher has to be evaluated somewhere, and WHERE turned out to matter more than
    anything else in this file. The obvious place is the input pipeline - map the teacher
    over the batch and hand its output along as part of the label - and it is wrong: tf.data
    executes on the CPU, so a 2M-parameter forward per batch lands there. Measured: 6 s per
    step with the GPU 19% busy, against 132 ms for the student alone. Inside the graph the
    teacher runs on the same device as the student and overlaps with its backward pass.

    The teacher is frozen and its branch is wrapped in stop_gradient, so nothing flows back
    into it even though it is part of the same graph.
    """
    # Enforced here rather than trusted to the caller: a teacher that is still trainable
    # doubles the optimiser state and lets the distillation quietly move its target.
    teacher.trainable = False
    inp = keras.Input(shape=student.input_shape[1:], name='input')
    soft = layers.Lambda(tf.stop_gradient, name='teacher_detach')(teacher(inp))
    out = layers.Concatenate(name='student_teacher')([student(inp), soft])
    return keras.Model(inp, out, name=f'distill_{student.name}')


def distill(model_name, teacher_checkpoint=None, epochs=None, batch_size=None, lr=None,
            loss=None, monitor=None, patience=None, cm_interval=1, db_names=None,
            ckpt_start_epoch=None, temperature=3.0, alpha=0.7, no_transfer=False,
            save_interval=None):
    """Train one CfC size against its state-space teacher. Returns the student model."""
    setup_gpus()
    teacher_size = models.teacher_of(model_name)
    if not teacher_size:
        raise ValueError(f"{model_name} has no teacher - distillation is for the resucfc_* "
                         f"family; {sorted(models.TEACHER_OF)} have one.")
    teacher_checkpoint = teacher_checkpoint or os.path.join(
        config.ENCODER_CHECKPOINT_DIR, f"{teacher_size}.keras")
    if not os.path.exists(teacher_checkpoint):
        raise FileNotFoundError(
            f"teacher checkpoint not found: {teacher_checkpoint}. Distillation needs the "
            f"trained {teacher_size}; pass --teacher to point at it.")

    batch_size = batch_size or config.BATCH_SIZE
    epochs = epochs or config.EPOCHS
    patience = config.PATIENCE if patience is None else patience
    loss = loss or config.LOSS
    monitor = monitor or config.MONITOR
    ckpt_start = config.CKPT_START_EPOCH if ckpt_start_epoch is None else ckpt_start_epoch
    every = max(1, int(config.SAVE_EPOCH_INTERVAL if save_interval is None else save_interval))

    student = models.build(model_name)
    teacher = tf.keras.models.load_model(teacher_checkpoint, compile=False)
    teacher.trainable = False
    print(f"teacher       : {teacher.name} from {teacher_checkpoint} "
          f"({teacher.count_params():,} params, frozen)")
    print(f"student       : {student.name} ({student.count_params():,} params)")
    if not no_transfer:
        transfer_shared_weights(student, teacher)
    else:
        print("transfer      : off, the student starts from scratch")

    f1_metric = StudentConfusion(name='weighted_f1')
    hard = LOSSES[loss](config.CLASS_WEIGHTS)
    wrapper = build_distiller(student, teacher)
    wrapper.compile(optimizer=tf.keras.optimizers.Adam(lr or config.LEARNING_RATE),
                    loss=distill_loss(hard, temperature=temperature, alpha=alpha),
                    metrics=[f1_metric], jit_compile=JIT_COMPILE)
    student.summary()
    print(f"\n{config.describe()}")
    print(f"distill       : T={temperature}, alpha={alpha} "
          f"(hard half: {loss}, weights {config.CLASS_WEIGHTS})")
    print(f"trainable     : {sum(int(np.prod(w.shape)) for w in wrapper.trainable_weights):,} "
          f"of {student.count_params():,} student params "
          f"(the teacher branch is frozen)")

    pipeline.check_manifest()
    train_ds = pipeline.load_split('train', batch_size, db_names)
    eval_ds = pipeline.load_split('eval', batch_size, db_names)

    ckpt_dir, report_dir, logs_dir = model_dirs(student.name)
    os.makedirs(os.path.join(ckpt_dir, 'epochs'), exist_ok=True)
    print(f"checkpoints   : {ckpt_dir} (epoch {every}, {2 * every}, ...; best from {ckpt_start})")

    mode = 'max' if ('acc' in monitor or 'f1' in monitor) else 'min'
    callbacks = [
        StudentF1Checkpoint(f1_metric, ckpt_dir, report_dir, interval=cm_interval,
                            save_start_epoch=ckpt_start, monitor=monitor, student=student),
        StudentModelCheckpoint(os.path.join(ckpt_dir, 'best_model.keras'),
                               start_epoch=ckpt_start, monitor=monitor, mode=mode,
                               save_best_only=True, verbose=1, student=student),
        StudentModelCheckpoint(os.path.join(ckpt_dir, 'epochs', 'epoch_{epoch:02d}.keras'),
                               start_epoch=every, interval=every, save_best_only=False,
                               verbose=0, student=student),
        tf.keras.callbacks.TensorBoard(log_dir=logs_dir),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode=mode, factor=0.5,
                                             patience=max(3, patience // 3), min_lr=1e-5,
                                             verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience,
                                         start_from_epoch=max(0, ckpt_start - 1),
                                         restore_best_weights=True, verbose=1),
    ]
    wrapper.fit(train_ds, validation_data=eval_ds, epochs=epochs, callbacks=callbacks)
    print(f"\ndistillation finished; checkpoints in {ckpt_dir}")
    return student
