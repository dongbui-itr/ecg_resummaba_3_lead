"""The fit loop.

Three stages feed into it, and the first two never see a label:

    ssl  (training/ssl.py)  masked reconstruction  -> backbone weights
    cpc  (training/cpc.py)  InfoNCE                -> context-encoder weights
    train                   poly2 over 4 classes   -> the beat classifier

Output layout, one folder per kind of output and one subfolder per model, so several models
trained side by side never mix:

    <RUN_DIR>/checkpoints/<model>/best_model.keras          best on --monitor
    <RUN_DIR>/checkpoints/<model>/BEST_F1/*.keras           best step-level weighted F1
    <RUN_DIR>/checkpoints/<model>/ssl_backbone.weights.h5   SSL-pretrained backbone
    <RUN_DIR>/checkpoints/<model>/cpc_context.weights.h5    CPC-pretrained context encoder
    <RUN_DIR>/eval/<model>/confusion_log.txt                one block per exported epoch
    <RUN_DIR>/logs/<model>/                                 tensorboard --logdir here
"""
import os

import tensorflow as tf

from .. import config, models, xla
from ..data import pipeline
from ..evaluation import step_metrics
from .callbacks import DelayedModelCheckpoint, UnfreezeBackbone, WeightedF1Checkpoint
from .losses import LOSSES

# See compile_model() below for why this is off by default.
JIT_COMPILE = os.environ.get("ECGR_JIT", "0") not in ("0", "", "false", "False")


def setup_gpus():
    """Memory growth on every visible GPU, and XLA pointed at libdevice.

    Without memory growth TF grabs the whole card at import and a second job on the same
    card dies with an allocator error. Without the libdevice path every stage dies at its
    first step instead - see ecgr/xla.py for why that is not optional.
    """
    xla.ensure_libdevice()
    devices = tf.config.list_physical_devices('GPU')
    for device in devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as e:                 # already initialised - harmless
            print(f"memory growth for {device}: {e}")
    print(f"{len(devices)} GPU(s)" if devices else "no GPU, running on CPU")
    return devices


def model_dirs(model_name):
    ckpt = os.path.join(config.CHECKPOINT_DIR, model_name)
    report = os.path.join(config.EVAL_DIR, model_name)
    logs = os.path.join(config.LOGS_DIR, model_name)
    for d in (ckpt, report, logs):
        os.makedirs(d, exist_ok=True)
    return ckpt, report, logs


# The CLI flag that supplies each sub-model's weights, for the error message below.
_WEIGHTS_FLAG = {'backbone': '--ssl-weights', 'context_encoder': '--ctx-weights'}


def load_pretrained(model, name, weights_path, freeze, required=False):
    """Load self-supervised weights into the `name` sub-model. Returns it, or None.

    `required` separates the two cases that look alike. A DEFAULT path that does not exist
    just means the self-supervised stage has not been run for this run, which is a legitimate
    way to train (from scratch) and only worth a line of output. A path the caller asked for
    explicitly (--ssl-weights / --ctx-weights) that does not exist is a mistake - silently
    training from scratch there would look like a successful run of something it is not.
    """
    if not weights_path or not os.path.exists(weights_path):
        if required:
            raise FileNotFoundError(
                f"{_WEIGHTS_FLAG.get(name, name)} points at a file that does not exist: "
                f"{weights_path}")
        print(f"{name:15s}: no pretrained weights, training from scratch")
        return None
    sub = models.sub_model(model, name)
    sub.load_weights(weights_path)
    sub.trainable = not freeze
    print(f"{name:15s}: loaded {weights_path} "
          f"({sub.count_params():,} params, {'frozen' if freeze else 'trainable'})")
    return sub


def train(model_name, epochs=None, batch_size=None, lr=None, loss=None,
          monitor=None, patience=None, cm_interval=1, db_names=None,
          ckpt_start_epoch=None, ctx_weights=None, ssl_weights=None,
          freeze_ctx=True, freeze_backbone_epochs=None):
    """Train one model of models.BUILDERS end to end.

    monitor defaults to config.MONITOR (val_weighted_f1, produced by the StepConfusion
    metric) rather than val_loss, and with this family that is not merely a preference.
    'None' is ~84% of the label steps, so val_loss barely moves with beat quality; and with
    the default poly2 loss it moves the WRONG WAY - measured on the 1M size it bottoms out
    at epoch 1 and rises monotonically while the weighted F1 climbs for six more epochs
    (see losses.py).

    The two self-supervised sub-models are treated differently on purpose:

      * the **context encoder** is frozen, as the paper freezes its patient encoder, so the
        beat loss can never reshape a description of nuisance variation into a beat detector;
      * the **backbone** is fine-tuned, because it IS the feature extractor the classifier
        reads. `freeze_backbone_epochs` > 0 holds it still for the first few epochs so the
        random head cannot wash out the pretraining before it is any good.

    ckpt_start_epoch delays every checkpoint write, so no epoch before it can be kept as
    "best". Three pieces have to agree for that to mean anything, see below.
    """
    setup_gpus()
    batch_size = batch_size or config.BATCH_SIZE
    epochs = epochs or config.EPOCHS
    patience = config.PATIENCE if patience is None else patience
    loss = loss or config.LOSS
    monitor = monitor or config.MONITOR
    ckpt_start = config.CKPT_START_EPOCH if ckpt_start_epoch is None else ckpt_start_epoch
    warmup = (config.FREEZE_BACKBONE_EPOCHS if freeze_backbone_epochs is None
              else freeze_backbone_epochs)

    model = models.build(model_name)
    keras_name = model.name
    ssl_asked, ctx_asked = ssl_weights is not None, ctx_weights is not None
    if not ssl_asked:
        ssl_weights = config.ssl_weights_dir(keras_name)
    if not ctx_asked:
        ctx_weights = config.cpc_weights_dir(keras_name)

    backbone = load_pretrained(model, 'backbone', ssl_weights, freeze=warmup > 0,
                               required=ssl_asked)
    load_pretrained(model, 'context_encoder', ctx_weights, freeze=freeze_ctx,
                    required=ctx_asked)

    f1_metric = step_metrics.StepConfusion(name='weighted_f1')

    def compile_model():
        # jit_compile=False, not Keras's "auto": XLA needs CUDA's libdevice.10.bc, which the
        # pip CUDA wheels do not ship, so "auto" turns a working install into
        # `libdevice not found at ./libdevice.10.bc` at the first training step. It is also
        # not obviously faster here - the FFT in DiagSSM1D is already one fused op. To use
        # it anyway, point XLA at a real CUDA toolkit:
        #   XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda  ECGR_JIT=1 python -m ecgr train ...
        model.compile(optimizer=tf.keras.optimizers.Adam(lr or config.LEARNING_RATE),
                      loss=LOSSES[loss](config.CLASS_WEIGHTS),
                      metrics=['accuracy', f1_metric],
                      jit_compile=JIT_COMPILE)

    compile_model()
    model.summary()
    print(f"\n{config.describe()}")
    print(f"model        : {model.name}, {model.count_params():,} parameters")
    print(f"loss         : {loss}, class weights {config.CLASS_WEIGHTS}")

    pipeline.check_manifest()
    train_ds = pipeline.load_split('train', batch_size, db_names)
    eval_ds = pipeline.load_split('eval', batch_size, db_names)

    ckpt_dir, report_dir, logs_dir = model_dirs(model.name)
    print(f"checkpoints  : {ckpt_dir}\nreports      : {report_dir}\nlogs         : {logs_dir}")

    mode = 'max' if ('acc' in monitor or 'f1' in monitor) else 'min'
    print(f"monitor      : {monitor} ({mode}), patience {patience}")
    if ckpt_start > 1:
        print(f"checkpoints  : measured every epoch, SAVED from epoch {ckpt_start} on; "
              f"EarlyStopping is held off until then too")
    if warmup > 0 and backbone is not None:
        print(f"backbone     : frozen for {warmup} epoch(s), then fine-tuned")
    print()

    callbacks = [
        WeightedF1Checkpoint(f1_metric, ckpt_dir, report_dir, interval=cm_interval,
                             save_start_epoch=ckpt_start, monitor=monitor),
        DelayedModelCheckpoint(
            os.path.join(ckpt_dir, 'best_model.keras'), start_epoch=ckpt_start,
            monitor=monitor, mode=mode, save_best_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=logs_dir),
        # Patience well under EarlyStopping's, so at least two LR drops can happen before
        # the run is allowed to stop - a plateau at a fixed LR reads as convergence otherwise.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode=mode, factor=0.5,
            patience=max(3, patience // 3), min_lr=1e-5, verbose=1),
        # start_from_epoch is the third piece of the delayed-checkpoint rule: without it a
        # short-patience run can stop before ckpt_start and finish with no checkpoint at all.
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode=mode, patience=patience,
            start_from_epoch=max(0, ckpt_start - 1),
            restore_best_weights=True, verbose=1),
    ]
    if warmup > 0 and backbone is not None:
        callbacks.insert(0, UnfreezeBackbone(backbone, warmup + 1, compile_model))

    model.fit(train_ds, validation_data=eval_ds, epochs=epochs, callbacks=callbacks)
    print(f"\ntraining finished; checkpoints in {ckpt_dir}")
    return model


def evaluate_checkpoint(model_path, batch_size=None, db_names=None):
    """Step-level metrics of a saved checkpoint on the eval split."""
    setup_gpus()
    print(f"loading {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    pipeline.check_manifest()
    eval_ds = pipeline.load_split('eval', batch_size or config.BATCH_SIZE, db_names)

    df_cm, per_class, f1 = step_metrics.evaluate(model, eval_ds)
    report = step_metrics.format_report(df_cm, per_class, f1,
                                        f"STEP EVAL - {os.path.basename(model_path)}")
    print("\n" + report)

    out_dir = os.path.join(config.EVAL_DIR, model.name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                            os.path.splitext(os.path.basename(model_path))[0] + "_step_eval.txt")
    with open(out_path, 'w') as f:
        f.write(f"Checkpoint: {model_path}\n\n" + report + "\n")
    print(f"\nreport -> {out_path}")
    return f1
