"""The tf.data pipeline, the losses, the step metric and both self-supervised objectives.

Everything runs on a tfrecord written into tmp_path, so no dataset has to exist.
"""
import keras
import numpy as np
import pytest
import tensorflow as tf

from ecgr import config, models
from ecgr.data import build_tfrecord, pipeline
from ecgr.evaluation import step_metrics
from ecgr.training import losses


@pytest.fixture(scope='module')
def tiny_tfrecord(tmp_path_factory):
    """256 synthetic segments as one tfrecord, plus the manifest pipeline.py checks."""
    root = tmp_path_factory.mktemp('tfr')
    out_dir = root / 'db' / 'train'
    out_dir.mkdir(parents=True)

    rng = np.random.default_rng(3)
    n = 256
    segments = rng.standard_normal(
        (n, config.SEGMENT_SAMPLES, config.IN_CHANNELS)).astype('float32')
    labels = np.zeros((n, config.OUTPUT_STEPS), dtype=np.uint8)
    for i in range(n):
        for step in range(10, config.OUTPUT_STEPS - 11, 40):
            labels[i, step:step + 11] = rng.integers(1, config.NUM_CLASSES + 1) - 1 + 1
    labels = np.clip(labels, 0, config.NUM_CLASSES - 1)

    with tf.io.TFRecordWriter(str(out_dir / 'db_train_batch_0.tfrecord')) as writer:
        for i in range(n):
            writer.write(build_tfrecord._example(segments[i], labels[i]))
    return str(root), segments, labels


def _dataset(root, batch=32, training=False, **kw):
    files = tf.io.gfile.glob(f"{root}/db/train/*.tfrecord")
    return pipeline.make_dataset(files, batch, training=training, cache=False, **kw)


def test_parse_round_trips_the_bytes_exactly(tiny_tfrecord):
    root, segments, labels = tiny_tfrecord
    for x, y in _dataset(root, batch=8).take(1):
        assert x.shape == (8, config.SEGMENT_SAMPLES, config.IN_CHANNELS)
        assert y.shape == (8, config.OUTPUT_STEPS, config.NUM_CLASSES)
        # raw bytes, so this is exact, not approximate
        assert np.array_equal(x.numpy(), segments[:8])
        assert np.array_equal(np.argmax(y.numpy(), -1).astype('uint8'), labels[:8])


def test_every_label_step_carries_exactly_one_class(tiny_tfrecord):
    root, _, _ = tiny_tfrecord
    for _, y in _dataset(root, batch=32, training=True).take(3):
        mass = tf.reduce_sum(y, axis=-1).numpy()
        assert np.allclose(mass, 1.0), "augmentation left steps with no class at all"


def test_augmentation_keeps_the_static_shape(tiny_tfrecord):
    root, _, _ = tiny_tfrecord
    ds = _dataset(root, batch=32, training=True)
    assert ds.element_spec[0].shape[1:] == (config.SEGMENT_SAMPLES, config.IN_CHANNELS)
    assert ds.element_spec[1].shape[1:] == (config.OUTPUT_STEPS, config.NUM_CLASSES)


def test_lead_duplication_augmentation_produces_exact_copies(tiny_tfrecord):
    """The EC57 case is one lead repeated EXACTLY. If the per-lead gain is applied after the
    duplication the leads come out merely proportional, and the case is never trained."""
    root, _, _ = tiny_tfrecord
    duplicated = total = 0
    for x, _ in _dataset(root, batch=64, training=True).repeat(6).take(6):
        arr = x.numpy()
        total += len(arr)
        for i in range(len(arr)):
            if all(np.array_equal(arr[i, :, 0], arr[i, :, c])
                   for c in range(1, config.IN_CHANNELS)):
                duplicated += 1
    assert total > 0
    rate = duplicated / total
    assert 0.05 < rate < 0.5, f"lead duplication fired on {rate:.1%} of samples"


def test_manifest_mismatch_is_refused(tiny_tfrecord, tmp_path):
    import json
    root, _, _ = tiny_tfrecord
    path = tmp_path / config.DATASET_MANIFEST
    manifest = {'segment_samples': config.SEGMENT_SAMPLES, 'in_channels': 1,
                'output_steps': config.OUTPUT_STEPS, 'num_classes': config.NUM_CLASSES,
                'signal_dtype': 'float32', 'labels_dtype': 'uint8', 'built': 'now',
                'totals': {'train': {'segments': 1}, 'eval': {'segments': 1}}}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='in_channels'):
        pipeline.check_manifest(str(tmp_path))


def test_benchmark_data_is_refused():
    with pytest.raises(RuntimeError, match='EC57'):
        pipeline.assert_no_benchmark_data(['/data/mitdb/train/x.tfrecord'])
    pipeline.assert_no_benchmark_data(['/data/dataset-1/train/x.tfrecord'])


# --- losses and metrics ----------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(losses.LOSSES))
def test_a_perfect_prediction_scores_better_than_a_wrong_one(name):
    y = tf.one_hot([[1, 2, 3, 0]], config.NUM_CLASSES)
    good = tf.clip_by_value(y, 1e-6, 1.0)
    bad = tf.clip_by_value(tf.one_hot([[0, 0, 0, 1]], config.NUM_CLASSES), 1e-6, 1.0)
    loss = losses.LOSSES[name]()
    assert float(loss(y, good)) < float(loss(y, bad))


def test_poly2_value_is_not_monotone_in_the_error():
    """Documented trap: d/du (0.3u - 0.5u^2) = 0.3 - u, so below Pt = 0.7 a WORSE prediction
    lowers the polynomial term. This is why the monitor is the F1 and not val_loss."""
    loss = losses.weighted_poly2_crossentropy(weights=[1.0] * config.NUM_CLASSES)
    y = tf.one_hot([[1]], config.NUM_CLASSES)

    def at(pt):
        rest = (1.0 - pt) / (config.NUM_CLASSES - 1)
        probs = [[[rest, pt, rest, rest]]]
        return float(loss(y, tf.constant(probs)))

    # over Pt = 0.7 the loss behaves; under it the polynomial term works against us
    assert at(0.9) < at(0.75)
    poly = lambda u: 0.3 * u - 0.5 * u ** 2          # noqa: E731
    assert poly(0.5) > poly(0.9), "the documented non-monotonicity is gone - recheck MONITOR"


def test_step_confusion_matches_the_numpy_reference():
    rng = np.random.default_rng(4)
    truth = rng.integers(0, config.NUM_CLASSES, (8, config.OUTPUT_STEPS))
    pred = rng.integers(0, config.NUM_CLASSES, (8, config.OUTPUT_STEPS))
    y_true = tf.one_hot(truth, config.NUM_CLASSES)
    y_pred = tf.one_hot(pred, config.NUM_CLASSES)

    metric = step_metrics.StepConfusion()
    metric.update_state(y_true, y_pred)
    matrix = metric.matrix()
    assert matrix.sum() == truth.size

    reference = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=np.int64)
    np.add.at(reference, (truth.reshape(-1), pred.reshape(-1)), 1)
    assert np.array_equal(matrix, reference)
    assert np.isclose(float(metric.result()),
                      step_metrics.weighted_f1_from_confusion(matrix), atol=1e-6)


def test_step_confusion_ignores_the_background_class():
    """'None' is ~84% of all steps; scoring it would make the metric a background detector."""
    cm = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=np.int64)
    cm[0, 0] = 10 ** 6                      # a perfect background, no beats at all
    assert step_metrics.weighted_f1_from_confusion(cm) == 0.0
    cm[1, 1] = 100
    assert step_metrics.weighted_f1_from_confusion(cm) == 1.0


# --- the self-supervised objectives ----------------------------------------------------

def test_masked_reconstruction_learns_something_learnable(tiny_tfrecord):
    """A structureless input gives nmse ~1 (no better than the mean); a strongly structured
    one has to come out under that, or the objective is not training the backbone at all."""
    from ecgr.training.ssl import MaskedReconstruction
    root, _, _ = tiny_tfrecord
    model = models.build('resumamba_30k')
    backbone = models.sub_model(model, 'backbone')
    trainer = MaskedReconstruction(backbone, config.IN_CHANNELS, config.OUTPUT_STEPS,
                                   config.SEGMENT_SAMPLES // config.OUTPUT_STEPS)
    trainer.compile(optimizer='adam', jit_compile=False)

    # a pure low-frequency signal: whatever is masked is predictable from its neighbours
    t = np.arange(config.SEGMENT_SAMPLES) / config.SAMPLING_RATE
    wave = np.sin(2 * np.pi * 1.2 * t)[None, :, None]
    x = np.repeat(np.repeat(wave, 64, axis=0), config.IN_CHANNELS, axis=-1).astype('float32')
    ds = tf.data.Dataset.from_tensor_slices(x).batch(16)

    history = trainer.fit(ds, epochs=6, verbose=0)
    assert history.history['nmse'][-1] < history.history['nmse'][0]
    assert np.isfinite(history.history['loss']).all()


def test_cpc_beats_chance_on_distinguishable_strips():
    """InfoNCE has to pick a strip's own future latent out of the batch. Give every strip its
    own frequency and it must do better than 1/(batch*windows)."""
    from ecgr.training.cpc import CPCTrainer
    model = models.build('resumamba_30k')
    encoder = models.sub_model(model, 'context_encoder')
    trainer = CPCTrainer(encoder, encoder.output_shape[0][-1])
    trainer.compile(optimizer='adam', jit_compile=False)

    t = np.arange(config.SEGMENT_SAMPLES) / config.SAMPLING_RATE
    rng = np.random.default_rng(5)
    x = np.stack([np.stack([np.sin(2 * np.pi * f * t + p)] * config.IN_CHANNELS, -1)
                  for f, p in zip(rng.uniform(0.5, 8, 32), rng.uniform(0, 6, 32))])
    ds = tf.data.Dataset.from_tensor_slices(x.astype('float32')).batch(8)

    history = trainer.fit(ds, epochs=15, verbose=0)
    chance = 1.0 / (8 * trainer.n_win)
    assert history.history['top1'][-1] > chance, "InfoNCE did not get above chance"


def test_manifest_merge_keeps_untouched_datasets(tmp_path, monkeypatch):
    """Rebuilding one dataset (`--db dataset-2`, as the duplicate-recording fix needed) must
    not drop the others from the manifest: check_manifest prints the totals it reads, so a
    manifest listing one dataset out of eight is worse than no manifest at all."""
    from ecgr.data import build_tfrecord

    monkeypatch.setattr(config, 'TFRECORD_DIR', str(tmp_path))
    for db in ('db-a', 'db-b'):
        (tmp_path / db).mkdir()

    def one(db, segments):
        return {db: {'train': {'segments': segments, 'files': 1,
                               'class_steps': [segments] + [0] * (config.NUM_CLASSES - 1)}}}

    build_tfrecord.write_manifest({**one('db-a', 10), **one('db-b', 20)})
    assert pipeline.read_manifest(str(tmp_path))['totals']['train']['segments'] == 30

    # rebuild only db-b, with a different count
    build_tfrecord.write_manifest(one('db-b', 5))
    manifest = pipeline.read_manifest(str(tmp_path))
    assert set(manifest['per_dataset']) == {'db-a', 'db-b'}
    assert manifest['totals']['train']['segments'] == 15, "db-a was dropped or db-b not updated"

    # a dataset whose tfrecords are gone is not carried over
    import shutil
    shutil.rmtree(tmp_path / 'db-a')
    build_tfrecord.write_manifest(one('db-b', 5))
    assert set(pipeline.read_manifest(str(tmp_path))['per_dataset']) == {'db-b'}


def test_self_supervised_stages_save_the_best_epoch_not_the_last(tmp_path, monkeypatch):
    """`pretrain` writes the weights AFTER fit(), so without restore_best_weights it hands
    `ecgr train` whatever the final epoch happened to be. Measured on the real run: the 30k
    backbone's val_nmse was 0.6895 at epoch 7 and 0.7067 at epoch 8, and epoch 8 is what got
    saved. A pretraining LR schedule is not tied to the end of training, so a regressing
    last epoch is normal rather than exceptional."""
    import inspect
    from ecgr.training import cpc, ssl

    for module in (ssl, cpc):
        source = inspect.getsource(module.pretrain)
        assert 'restore_best_weights=True' in source, (
            f"{module.__name__}.pretrain would save the last epoch, not the best")
        # and the restore has to happen BEFORE the save, or it changes nothing
        assert source.index('restore_best_weights') < source.index('save_weights(out)')


# --- noise augmentation, checkpoint averaging, ensembles, decoder run filter -------------

def test_noise_augmentation_touches_the_signal_but_never_the_labels(monkeypatch):
    rng = np.random.default_rng(6)
    x = tf.constant(rng.standard_normal((32, config.SEGMENT_SAMPLES, config.IN_CHANNELS))
                    .astype('float32'))
    y = tf.one_hot(rng.integers(0, config.NUM_CLASSES, (32, config.OUTPUT_STEPS)),
                   config.NUM_CLASSES)
    monkeypatch.setattr(config, 'AUGMENT_NOISE', True)
    noisy = pipeline._noise(x)
    assert noisy.shape == x.shape
    changed = tf.reduce_max(tf.abs(noisy - x), axis=[1, 2]) > 1e-6
    assert 0.3 < float(tf.reduce_mean(tf.cast(changed, tf.float32))) < 1.0, \
        "about half the batch should carry noise, not none and not all"
    assert float(tf.math.reduce_std(noisy - x)) < 1.5, "noise must stay below QRS scale"
    sig, lab = pipeline.augment(x, y)
    assert np.allclose(tf.reduce_sum(lab, -1).numpy(), 1.0), "labels must be untouched"

    monkeypatch.setattr(config, 'AUGMENT_NOISE', False)
    assert np.array_equal(pipeline._noise(x).numpy(), x.numpy()), "the switch must switch it off"


def test_checkpoint_averaging_is_the_elementwise_mean(tmp_path):
    from ecgr.training import swa
    a = models.build('resumamba_30k')
    b = models.build('resumamba_30k')
    for wa, wb in zip(a.weights, b.weights):
        if np.issubdtype(wa.numpy().dtype, np.floating):
            wa.assign(np.full(wa.shape, 1.0, dtype='float32'))
            wb.assign(np.full(wb.shape, 3.0, dtype='float32'))
    pa, pb = str(tmp_path / 'a.keras'), str(tmp_path / 'b.keras')
    a.save(pa); b.save(pb)
    out = swa.average_checkpoints([pa, pb], str(tmp_path / 'avg.keras'))
    avg = keras.models.load_model(out, compile=False)
    floats = [w.numpy() for w in avg.weights if np.issubdtype(w.numpy().dtype, np.floating)]
    assert floats and all(np.allclose(v, 2.0) for v in floats)
    with pytest.raises(ValueError):
        swa.average_checkpoints([pa], str(tmp_path / 'one.keras'))


def test_ensemble_predicts_the_mean_of_its_members():
    from ecgr.evaluation.ec57 import Ensemble, predict_segments
    m1, m2 = models.build('resumamba_30k'), models.build('resumamba_30k')
    x = np.random.default_rng(7).standard_normal(
        (3, config.SEGMENT_SAMPLES, config.IN_CHANNELS)).astype('float32')
    ens = Ensemble([m1, m2])
    expected = (predict_segments(m1, x) + predict_segments(m2, x)) / 2
    assert np.allclose(predict_segments(ens, x), expected, atol=1e-6)
    assert ens.count_params() == m1.count_params() + m2.count_params()
    assert tuple(ens.input_shape[1:]) == (config.SEGMENT_SAMPLES, config.IN_CHANNELS)


def test_min_run_steps_drops_only_short_runs():
    from ecgr.labels import decode_beats
    segments = np.zeros((1, config.SEGMENT_SAMPLES, config.IN_CHANNELS), np.float32)
    segments[0, 100 * 5 + 2, 0] = 1.0      # a peak inside the long run
    segments[0, 300 * 5 + 2, 0] = 1.0      # a peak inside the 1-step run
    preds = np.zeros((1, config.OUTPUT_STEPS, config.NUM_CLASSES), np.float32)
    preds[..., 0] = 1.0
    preds[0, 95:106] = [0, 1, 0, 0]        # 11-step run: a beat
    preds[0, 300] = [0, 1, 0, 0]           # 1-step flicker
    starts = np.array([0])
    both, _ = decode_beats(preds, segments, starts, min_run_steps=1)
    one, _ = decode_beats(preds, segments, starts, min_run_steps=2)
    assert len(both) == 2 and len(one) == 1
    assert one[0] == 100 * 5 + 2, "the long run must survive at the right position"


def test_select_choose_prefers_feasible_then_s_f1():
    from ecgr.training import select
    ref = {'Q_Se': 99.0, 'Q_+P': 99.0, 'V_Se': 90.0, 'V_+P': 90.0, 'S_Se': 80.0, 'S_+P': 80.0}
    rows = [
        {'name': 'a', **ref, 'S_Se': 84.0, 'S_+P': 79.0},        # best F1 but S_+P regresses
        {'name': 'b', **ref, 'S_Se': 81.0, 'S_+P': 81.0},        # feasible, smaller gain
        {'name': 'c', **ref},                                    # the reference itself
    ]
    for r in rows:
        r['S_F1'] = select.s_f1(r)
    winner, n_feasible = select.choose(rows, ref, tolerance=0.1)
    assert winner['name'] == 'b' and n_feasible == 2
    # with a tolerance that forgives the regression the higher-F1 candidate wins
    winner, _ = select.choose(rows, ref, tolerance=1.5)
    assert winner['name'] == 'a'
