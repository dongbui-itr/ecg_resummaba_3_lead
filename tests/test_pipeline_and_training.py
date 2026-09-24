"""The tf.data pipeline, the losses, the step metric and both self-supervised objectives.

Everything runs on a tfrecord written into tmp_path, so no dataset has to exist. The
windows are the real 60 s geometry (15000 x 3), so the tiny corpus is kept small.
"""
import keras
import numpy as np
import pytest
import tensorflow as tf

from ecgr import config, models
from ecgr.data import build_tfrecord, pipeline
from ecgr.evaluation import step_metrics
from ecgr.training import losses

SMALL = 'resumamba_100k'
IGN = config.IGNORE_LABEL


@pytest.fixture(scope='module')
def tiny_tfrecord(tmp_path_factory):
    """64 synthetic strips as one tfrecord: beats every 40 steps, labelled only inside a
    'reviewed span' of steps [1000, 1500), IGNORE elsewhere - like a real strip."""
    root = tmp_path_factory.mktemp('tfr')
    out_dir = root / 'db' / 'train'
    out_dir.mkdir(parents=True)

    rng = np.random.default_rng(3)
    n = 64
    segments = rng.standard_normal(
        (n, config.SEGMENT_SAMPLES, config.IN_CHANNELS)).astype('float32')
    labels = np.full((n, config.OUTPUT_STEPS), IGN, dtype=np.uint8)
    for i in range(n):
        labels[i, 1000:1500] = 0
        for step in range(1010, 1490, 40):
            labels[i, step:step + 11] = rng.integers(1, config.NUM_CLASSES)

    with tf.io.TFRecordWriter(str(out_dir / 'db_train_batch_0.tfrecord')) as writer:
        for i in range(n):
            writer.write(build_tfrecord._example(segments[i], labels[i]))
    return str(root), segments, labels


def _dataset(root, batch=32, training=False, **kw):
    files = tf.io.gfile.glob(f"{root}/db/train/*.tfrecord")
    return pipeline.make_dataset(files, batch, training=training, cache=False, **kw)


def test_parse_round_trips_the_bytes_exactly(tiny_tfrecord):
    root, segments, labels = tiny_tfrecord
    for x, y in _dataset(root, batch=8, targets=False).take(1):
        assert x.shape == (8, config.SEGMENT_SAMPLES, config.IN_CHANNELS)
        assert y.shape == (8, config.OUTPUT_STEPS, config.NUM_CLASSES)
        # raw bytes, so this is exact, not approximate
        assert np.array_equal(x.numpy(), segments[:8])
        y = y.numpy()
        labelled = labels[:8] != IGN
        assert np.array_equal(y.sum(-1) > 0.5, labelled), "IGNORE must become a zero row"
        assert np.array_equal(np.argmax(y, -1)[labelled], labels[:8][labelled])


def test_ignored_steps_are_zero_rows_and_labelled_steps_one_hot(tiny_tfrecord):
    root, _, _ = tiny_tfrecord
    for x, y in _dataset(root, batch=32, training=True).take(3):
        mass = tf.reduce_sum(y['beat_cls'], axis=-1).numpy()
        assert set(np.unique(np.round(mass, 5))) <= {0.0, 1.0}, "a step has 0 or 1 class"
        assert 0.05 < (mass > 0.5).mean() < 0.3, "roughly the 500 of 3000 reviewed steps"


def test_targets_are_a_dict_with_both_heads(tiny_tfrecord):
    root, _, _ = tiny_tfrecord
    ds = _dataset(root, batch=32, training=True)
    assert ds.element_spec[0].shape[1:] == (config.SEGMENT_SAMPLES, config.IN_CHANNELS)
    spec = ds.element_spec[1]
    assert set(spec) == {'beat_cls', 'lead_quality'}
    assert spec['beat_cls'].shape[1:] == (config.OUTPUT_STEPS, config.NUM_CLASSES)
    assert spec['lead_quality'].shape[1:] == (config.OUTPUT_STEPS, config.IN_CHANNELS)
    for _, y in _dataset(root, batch=8).take(1):        # the eval map, no corruption
        q = y['lead_quality'].numpy()
        assert q.shape == (8, config.OUTPUT_STEPS, config.IN_CHANNELS)
        assert np.allclose(q, 1.0), "nothing injected, nothing flat: every lead is readable"


def test_lead_duplication_augmentation_produces_exact_copies(tiny_tfrecord):
    """The EC57 case is one lead repeated EXACTLY. If the per-lead gain is applied after the
    duplication the leads come out merely proportional, and the case is never trained."""
    root, _, _ = tiny_tfrecord
    duplicated = total = 0
    for x, y in _dataset(root, batch=64, training=True).repeat(6).take(6):
        arr, q = x.numpy(), y['lead_quality'].numpy()
        total += len(arr)
        for i in range(len(arr)):
            if all(np.array_equal(arr[i, :, 0], arr[i, :, c])
                   for c in range(1, config.IN_CHANNELS)):
                duplicated += 1
                assert all(np.allclose(q[i, :, 0], q[i, :, c]) for c in range(1, 3)), \
                    "copies of lead 0 are exactly as readable as lead 0"
    assert total > 0
    rate = duplicated / total
    assert 0.05 < rate < 0.5, f"lead duplication fired on {rate:.1%} of samples"


def test_manifest_mismatch_is_refused(tiny_tfrecord, tmp_path):
    import json
    path = tmp_path / config.DATASET_MANIFEST
    manifest = {'segment_samples': config.SEGMENT_SAMPLES, 'in_channels': 1,
                'output_steps': config.OUTPUT_STEPS, 'num_classes': config.NUM_CLASSES,
                'signal_dtype': 'float32', 'labels_dtype': 'uint8', 'built': 'now',
                'totals': {'train': {'segments': 1}, 'eval': {'segments': 1}}}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='in_channels'):
        pipeline.check_manifest(str(tmp_path))
    manifest['in_channels'], manifest['segment_samples'] = config.IN_CHANNELS, 2500
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='segment_samples'):
        pipeline.check_manifest(str(tmp_path))       # a 10 s tree under a 60 s run


def test_benchmark_data_is_refused():
    with pytest.raises(RuntimeError, match='EC57'):
        pipeline.assert_no_benchmark_data(['/data/mitdb/train/x.tfrecord'])
    pipeline.assert_no_benchmark_data(['/data/dataset-1/train/x.tfrecord'])


def test_tfrecord_histogram_counts_labelled_steps_only():
    labels = np.full((2, config.OUTPUT_STEPS), IGN, np.uint8)
    labels[0, :10] = 1
    labels[1, :5] = 3
    counts, ignored = build_tfrecord.label_histogram(labels)
    assert counts.tolist() == [0, 10, 0, 5]
    assert ignored == 2 * config.OUTPUT_STEPS - 15


# --- losses and metrics ----------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(losses.LOSSES))
def test_a_perfect_prediction_scores_better_than_a_wrong_one(name):
    y = tf.one_hot([[1, 2, 3, 0]], config.NUM_CLASSES)
    good = tf.clip_by_value(y, 1e-6, 1.0)
    bad = tf.clip_by_value(tf.one_hot([[0, 0, 0, 1]], config.NUM_CLASSES), 1e-6, 1.0)
    loss = losses.LOSSES[name]()
    assert float(loss(y, good)) < float(loss(y, bad))


@pytest.mark.parametrize('name', sorted(losses.LOSSES))
def test_ignored_steps_do_not_move_the_loss(name):
    """Zero-mass target rows contribute nothing, and the mean runs over the labelled steps:
    the same labelled steps with 0 or 2000 ignored neighbours give the same loss."""
    loss = losses.LOSSES[name]()
    rng = np.random.default_rng(0)
    y_lab = tf.one_hot(rng.integers(0, 4, (2, 40)), config.NUM_CLASSES)
    p_lab = tf.nn.softmax(rng.standard_normal((2, 40, 4)).astype('float32'))
    y_ign = tf.zeros((2, 2000, 4))
    p_ign = tf.nn.softmax(rng.standard_normal((2, 2000, 4)).astype('float32'))
    alone = float(loss(y_lab, p_lab))
    padded = float(loss(tf.concat([y_lab, y_ign], 1), tf.concat([p_lab, p_ign], 1)))
    assert np.isclose(alone, padded, rtol=1e-5), f"{name}: {alone} vs {padded}"
    # a fully ignored batch is finite (no division by zero), and zero
    assert float(loss(y_ign, p_ign)) == 0.0


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


def test_lead_quality_loss_is_minimised_at_the_soft_target():
    target = tf.constant([[[0.9, 0.1, 0.5]]])
    at_target = float(losses.lead_quality_loss(target, target))
    for other in ([[[0.5, 0.5, 0.5]]], [[[0.99, 0.01, 0.9]]], [[[0.1, 0.9, 0.5]]]):
        assert float(losses.lead_quality_loss(target, tf.constant(other))) > at_target


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


def test_step_confusion_skips_ignored_steps():
    """An all-zero target row must not land in the matrix - not as 'None', not at all."""
    truth = np.array([[1, 2, 3, 0, 1]])
    pred = np.array([[1, 2, 0, 0, 1]])
    y_true = tf.one_hot(truth, config.NUM_CLASSES).numpy()
    y_true[0, 3:] = 0.0                              # the last two steps are IGNORE
    metric = step_metrics.StepConfusion()
    metric.update_state(tf.constant(y_true), tf.one_hot(pred, config.NUM_CLASSES))
    matrix = metric.matrix()
    assert matrix.sum() == 3, "only the three labelled steps count"
    assert matrix[3, 0] == 1 and matrix[1, 1] == 1 and matrix[2, 2] == 1
    assert matrix[0].sum() == 0, "an IGNORE row must not be read as background"


def test_step_confusion_ignores_the_background_class():
    """'None' is ~84% of all steps; scoring it would make the metric a background detector."""
    cm = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=np.int64)
    cm[0, 0] = 10 ** 6                      # a perfect background, no beats at all
    assert step_metrics.weighted_f1_from_confusion(cm) == 0.0
    cm[1, 1] = 100
    assert step_metrics.weighted_f1_from_confusion(cm) == 1.0


def test_confusion_loop_reads_dict_targets_and_two_output_models(tiny_tfrecord):
    root, _, labels = tiny_tfrecord
    model = models.build(SMALL)
    cm = step_metrics.confusion(model, _dataset(root, batch=16).take(1))
    assert cm.sum() == int((labels[:16] != IGN).sum()), "only labelled steps are scored"


# --- output 2: the lead-quality target -------------------------------------------------

def _clean_batch(n=8, seed=0):
    """Structured 3-lead strips: sinusoids of a few Hz, z-scored, one lead flat in sample 0."""
    rng = np.random.default_rng(seed)
    t = np.arange(config.SEGMENT_SAMPLES) / config.SAMPLING_RATE
    x = np.stack([np.stack([np.sin(2 * np.pi * f * t + p) for p in (0, 1, 2)], -1)
                  for f in rng.uniform(1, 4, n)]).astype('float32')
    x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    x[0, :, 2] = 0.0                                        # a lead-off, in the ORIGINAL
    return tf.constant(x)


def test_clean_quality_target_is_one_except_on_flat_leads():
    q = pipeline.clean_quality_target(_clean_batch()).numpy()
    assert q.shape == (8, config.OUTPUT_STEPS, config.IN_CHANNELS)
    assert np.allclose(q[1:], 1.0) and np.allclose(q[0, :, :2], 1.0)
    assert np.allclose(q[0, :, 2], 0.0), "a flat lead is unreadable everywhere"


def test_quality_target_follows_the_injected_corruption(monkeypatch):
    """Wreck lead 1 of every sample hard, nothing else: lead 1's target must drop where the
    noise is, the other leads must stay readable, and the corrupted lead must be the one the
    signal actually changed on."""
    monkeypatch.setattr(config, 'AUGMENT_NOISE', True)
    monkeypatch.setattr(config, 'AUGMENT_LEAD_DUPLICATE_PROB', 0.0)
    monkeypatch.setattr(config, 'AUGMENT_LEAD_DROP_PROB', 0.0)
    x = _clean_batch(n=6)[1:]                               # no flat lead in this batch
    added = np.zeros(x.shape, np.float32)
    n = config.SEGMENT_SAMPLES
    added[:, n // 4: 3 * n // 4, 1] = np.random.default_rng(1).normal(0, 2.5, (5, n // 2))
    q = pipeline.quality_from_corruption(tf.constant(added), x).numpy()
    steps = config.OUTPUT_STEPS
    middle, edges = slice(steps // 4 + 60, 3 * steps // 4 - 60), np.r_[0:steps // 4 - 60, 3 * steps // 4 + 60:steps]
    assert (q[:, middle, 1] < 0.05).all(), "a swamped lead is unreadable"
    assert np.allclose(q[:, edges, 1], 1.0), "outside the burst the same lead is fine"
    assert np.allclose(q[:, :, [0, 2]], 1.0), "untouched leads stay readable: exactly 1"
    # a mild degradation is a mild drop, not a cliff
    mild = np.zeros(x.shape, np.float32)
    mild[:, :, 0] = np.random.default_rng(2).normal(0, 0.5, (5, n))
    q_mild = pipeline.quality_from_corruption(tf.constant(mild), x).numpy()
    assert (0.6 < q_mild[:, 100:-100, 0]).all() and (q_mild[:, 100:-100, 0] < 0.95).all()


def test_corrupt_returns_a_target_that_names_the_lead_it_wrecked(monkeypatch):
    monkeypatch.setattr(config, 'AUGMENT_NOISE', True)
    monkeypatch.setattr(config, 'AUGMENT_LEAD_NOISE_PROB', 1.0)
    monkeypatch.setattr(config, 'AUGMENT_LEAD_NOISE_SPAN', 0.9)
    monkeypatch.setattr(config, 'AUGMENT_WANDER_PROB', 0.0)
    monkeypatch.setattr(config, 'AUGMENT_NOISE_PROB', 0.0)
    monkeypatch.setattr(config, 'AUGMENT_MOTION_PROB', 0.0)
    monkeypatch.setattr(config, 'AUGMENT_LEAD_DUPLICATE_PROB', 0.0)
    monkeypatch.setattr(config, 'AUGMENT_LEAD_DROP_PROB', 0.0)
    tf.random.set_seed(5)
    x = _clean_batch(n=64, seed=9)[1:]
    corrupted, q = pipeline.corrupt(x)
    corrupted, q, x = corrupted.numpy(), q.numpy(), x.numpy()
    assert q.shape == (63, config.OUTPUT_STEPS, config.IN_CHANNELS)
    # per lead: how much the signal changed (beyond the per-lead gain) vs its quality
    gain_free = corrupted / (np.abs(corrupted).mean(1, keepdims=True) + 1e-6) - \
        x / (np.abs(x).mean(1, keepdims=True) + 1e-6)
    change = np.abs(gain_free).mean(1)                                   # (b, leads)
    worst_by_signal, worst_by_target = change.argmax(1), q.mean(1).argmin(1)
    strong = change.max(1) > 0.5 * change.max()
    agree = (worst_by_signal == worst_by_target)[strong].mean()
    assert agree > 0.9, f"the target names the wrecked lead on only {agree:.0%} of samples"
    assert (q.mean(1).max(1) > 0.9).all(), "two leads are always left readable"


def test_dropped_and_duplicated_leads_get_the_right_quality(monkeypatch):
    monkeypatch.setattr(config, 'AUGMENT_NOISE', False)
    x = _clean_batch(n=4)[1:]
    added = tf.zeros_like(x)
    dup = tf.constant([[[True]], [[False]], [[False]]])
    drop = tf.constant(np.array([[[0, 0, 0]], [[0, 1, 0]], [[0, 0, 0]]], np.float32))
    q = pipeline.quality_from_corruption(added, x, dup, drop).numpy()
    assert np.allclose(q[0], 1.0)                      # copies of a clean lead 0: all readable
    assert np.allclose(q[1, :, 1], 0.0) and np.allclose(q[1, :, [0, 2]], 1.0)
    assert np.allclose(q[2], 1.0)


def test_augment_never_touches_the_beat_labels(tiny_tfrecord):
    root, _, labels = tiny_tfrecord
    for x, y in _dataset(root, batch=64, training=True).take(1):
        beats = y['beat_cls'].numpy()
        mass = beats.sum(-1)
        assert set(np.unique(np.round(mass, 5))) <= {0.0, 1.0}
        # time-scaling moves the labelled span by at most 10%, never changes its class mix
        assert 0.6 * 500 < (mass > 0.5).sum(1).mean() < 1.4 * 500


# --- the self-supervised objectives ----------------------------------------------------

def test_masked_reconstruction_learns_something_learnable():
    """A structureless input gives nmse ~1 (no better than the mean); a strongly structured
    one has to come out under that, or the objective is not training the backbone at all."""
    from ecgr.training.ssl import MaskedReconstruction
    model = models.build(SMALL)
    backbone = models.sub_model(model, 'backbone')
    trainer = MaskedReconstruction(backbone, config.IN_CHANNELS, config.OUTPUT_STEPS,
                                   config.SEGMENT_SAMPLES // config.OUTPUT_STEPS)
    trainer.compile(optimizer='adam', jit_compile=False)

    # a pure low-frequency signal: whatever is masked is predictable from its neighbours
    t = np.arange(config.SEGMENT_SAMPLES) / config.SAMPLING_RATE
    wave = np.sin(2 * np.pi * 1.2 * t)[None, :, None]
    x = np.repeat(np.repeat(wave, 32, axis=0), config.IN_CHANNELS, axis=-1).astype('float32')
    ds = tf.data.Dataset.from_tensor_slices(x).batch(8)

    history = trainer.fit(ds, epochs=4, verbose=0)
    assert history.history['nmse'][-1] < history.history['nmse'][0]
    assert np.isfinite(history.history['loss']).all()


def test_cpc_beats_chance_on_distinguishable_strips():
    """InfoNCE has to pick a strip's own future latent out of the batch. Give every strip its
    own frequency and it must do better than 1/(batch*windows)."""
    from ecgr.training.cpc import CPCTrainer
    model = models.build(SMALL)
    encoder = models.sub_model(model, 'context_encoder')
    trainer = CPCTrainer(encoder, encoder.output_shape[0][-1])
    trainer.compile(optimizer='adam', jit_compile=False)
    assert trainer.n_win == 59

    t = np.arange(config.SEGMENT_SAMPLES) / config.SAMPLING_RATE
    rng = np.random.default_rng(5)
    x = np.stack([np.stack([np.sin(2 * np.pi * f * t + p)] * config.IN_CHANNELS, -1)
                  for f, p in zip(rng.uniform(0.5, 8, 16), rng.uniform(0, 6, 16))])
    ds = tf.data.Dataset.from_tensor_slices(x.astype('float32')).batch(8)

    history = trainer.fit(ds, epochs=8, verbose=0)
    chance = 1.0 / (8 * trainer.n_win)
    assert history.history['top1'][-1] > chance, "InfoNCE did not get above chance"


def test_manifest_merge_keeps_untouched_datasets(tmp_path, monkeypatch):
    """Rebuilding one dataset (`--db dataset-2`) must not drop the others from the manifest:
    check_manifest prints the totals it reads, so a manifest listing one dataset out of five
    is worse than no manifest at all."""
    from ecgr.data import build_tfrecord

    monkeypatch.setattr(config, 'TFRECORD_DIR', str(tmp_path))
    for db in ('db-a', 'db-b'):
        (tmp_path / db).mkdir()

    def one(db, segments):
        return {db: {'train': {'segments': segments, 'files': 1,
                               'class_steps': [segments] + [0] * (config.NUM_CLASSES - 1),
                               'ignored_steps': 7}}}

    build_tfrecord.write_manifest({**one('db-a', 10), **one('db-b', 20)})
    manifest = pipeline.read_manifest(str(tmp_path))
    assert manifest['totals']['train']['segments'] == 30
    assert manifest['totals']['train']['ignored_steps'] == 14
    assert manifest['ignore_label'] == IGN and manifest['segment_seconds'] == 60

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


def test_self_supervised_stages_save_the_best_epoch_not_the_last():
    """`pretrain` writes the weights AFTER fit(), so without restore_best_weights it hands
    `ecgr train` whatever the final epoch happened to be."""
    import inspect
    from ecgr.training import cpc, ssl

    for module in (ssl, cpc):
        source = inspect.getsource(module.pretrain)
        assert 'restore_best_weights=True' in source, (
            f"{module.__name__}.pretrain would save the last epoch, not the best")
        # and the restore has to happen BEFORE the save, or it changes nothing
        assert source.index('restore_best_weights') < source.index('save_weights(out)')


# --- the supervised fit loop, end to end on the tiny corpus -----------------------------

def test_two_output_training_step_runs_and_publishes_the_monitor(tiny_tfrecord, tmp_path,
                                                                 monkeypatch):
    """compile_targets + the pipeline dict + StepConfusion must fit together: one training
    epoch on the tiny corpus, and the F1 monitor comes out under the name config names."""
    from ecgr.training import train as trainer
    from ecgr.training.losses import LOSSES
    root, _, _ = tiny_tfrecord
    model = models.build(SMALL)
    f1 = step_metrics.StepConfusion(name='weighted_f1')
    loss, weights, metrics = trainer.compile_targets(model, LOSSES['poly2'](config.CLASS_WEIGHTS), f1)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, loss_weights=weights,
                  metrics=metrics)
    ds = _dataset(root, batch=8, training=True).take(2)
    history = model.fit(ds, validation_data=_dataset(root, batch=8).take(1), epochs=1,
                        verbose=0)
    assert trainer.monitor_key(model) == config.MONITOR == 'val_beat_cls_weighted_f1'
    assert config.MONITOR in history.history, sorted(history.history)
    assert 'val_lead_quality_mae' in history.history
    assert np.isfinite(history.history['loss']).all()
    # the legacy layout resolves to the unprefixed name
    assert trainer.monitor_key(models.build(SMALL, use_quality=False)) == 'val_weighted_f1'


def test_lead_quality_report_scores_the_head_where_the_answer_is_known(tiny_tfrecord):
    root, _, _ = tiny_tfrecord
    report = step_metrics.lead_quality_report(models.build(SMALL), _dataset(root, batch=8),
                                              batches=2)
    assert set(report) == {'lead_acc', 'mae', 'sep', 'samples'}
    assert report['mae'] is not None and 0 <= report['mae'] <= 1
    assert step_metrics.lead_quality_report(models.build(SMALL, use_quality=False),
                                            _dataset(root, batch=8))['lead_acc'] is None


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
    sig, targets = pipeline.augment(x, y)
    assert np.allclose(tf.reduce_sum(targets['beat_cls'], -1).numpy().max(), 1.0)
    assert set(targets) == {'beat_cls', 'lead_quality'}

    monkeypatch.setattr(config, 'AUGMENT_NOISE', False)
    assert np.array_equal(pipeline._noise(x).numpy(), x.numpy()), "the switch must switch it off"


def test_lead_noise_hits_exactly_one_lead_and_spares_the_labelled_one(monkeypatch):
    """The point of _lead_noise is a lead the OTHER leads can be read against, so it must
    corrupt exactly one - and lead 0, which the labels refer to, only mildly."""
    rng = np.random.default_rng(11)
    x = tf.constant(rng.standard_normal((128, config.SEGMENT_SAMPLES, config.IN_CHANNELS))
                    .astype('float32'))
    monkeypatch.setattr(config, 'AUGMENT_NOISE', True)
    monkeypatch.setattr(config, 'AUGMENT_LEAD_NOISE_PROB', 1.0)

    delta = (pipeline._lead_noise(x) - x).numpy()
    touched = np.abs(delta).max(axis=1) > 1e-6                  # (batch, channels)
    assert set(touched.sum(axis=1)) <= {1}, "never more than one lead per sample"
    assert touched.sum() > 0, "with prob 1.0 something must be corrupted"
    assert touched[:, 0].any() and touched[:, 1:].any(), "every lead must be reachable"

    per_lead_std = delta.std(axis=1)                            # (batch, channels)
    primary = per_lead_std[touched[:, 0], 0]
    if len(primary):
        assert primary.max() < config.AUGMENT_LEAD_NOISE_PRIMARY_AMP + 0.05, \
            "lead 0 carries the labels - it must stay below the QRS scale"

    monkeypatch.setattr(config, 'AUGMENT_LEAD_NOISE_PROB', 0.0)
    assert np.array_equal(pipeline._lead_noise(x).numpy(), x.numpy()), \
        "the switch must switch it off"


def test_checkpoint_averaging_is_the_elementwise_mean(tmp_path):
    from ecgr.training import swa
    a = models.build(SMALL)
    b = models.build(SMALL)
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


def test_ensemble_predicts_the_mean_of_its_members_on_both_outputs():
    from ecgr.evaluation.ec57 import Ensemble, predict_segments, predict_segments_full
    m1, m2 = models.build(SMALL), models.build(SMALL)
    x = np.random.default_rng(7).standard_normal(
        (3, config.SEGMENT_SAMPLES, config.IN_CHANNELS)).astype('float32')
    ens = Ensemble([m1, m2])
    expected = (predict_segments(m1, x) + predict_segments(m2, x)) / 2
    assert np.allclose(predict_segments(ens, x), expected, atol=1e-6)
    q1, q2 = predict_segments_full(m1, x)[1], predict_segments_full(m2, x)[1]
    assert np.allclose(predict_segments_full(ens, x)[1], (q1 + q2) / 2, atol=1e-6)
    assert models.has_quality_output(ens)
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


def test_min_peak_prob_drops_only_the_unconfident_run():
    """The noise gate keys on the PEAK beat probability of a run, not its mean: a run that is
    certain anywhere along it is a beat, and one that is never certain is artefact."""
    from ecgr.labels import decode_beats
    segments = np.zeros((1, config.SEGMENT_SAMPLES, config.IN_CHANNELS), np.float32)
    segments[0, 100 * 5 + 2, 0] = 1.0
    segments[0, 300 * 5 + 2, 0] = 1.0
    preds = np.zeros((1, config.OUTPUT_STEPS, config.NUM_CLASSES), np.float32)
    preds[..., 0] = 1.0
    preds[0, 95:106] = [0.02, 0.98, 0, 0]      # confident run, peak 1 - p_None = 0.98
    preds[0, 295:306] = [0.40, 0.60, 0, 0]     # hesitant run, peak 0.60
    starts = np.array([0])

    both, _ = decode_beats(preds, segments, starts, min_peak_prob=0.0)
    one, _ = decode_beats(preds, segments, starts, min_peak_prob=0.9)
    none, _ = decode_beats(preds, segments, starts, min_peak_prob=0.99)
    assert len(both) == 2 and len(one) == 1 and len(none) == 0
    assert one[0] == 100 * 5 + 2, "the confident run must survive at the right position"

    # a run that is hesitant on average but certain at its peak is still a beat
    preds[0, 295:306] = [0.40, 0.60, 0, 0]
    preds[0, 300] = [0.05, 0.95, 0, 0]
    kept, _ = decode_beats(preds, segments, starts, min_peak_prob=0.9)
    assert len(kept) == 2, "the gate must read the peak, not the mean"

    monkey = config.DECODE_MIN_PEAK_PROB
    try:
        config.DECODE_MIN_PEAK_PROB = 0.0
        assert len(decode_beats(preds, segments, starts)[0]) == 2, "0 must switch it off"
    finally:
        config.DECODE_MIN_PEAK_PROB = monkey


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
