"""The two self-supervised stages must be label-free, proved rather than asserted.

`ssl` (masked reconstruction) and `cpc` (InfoNCE) are the project's claim to learning from
unlabeled signal. A docstring saying so is worth nothing: the tfrecords carry labels in the
same record as the signal, so it would take one line to start using them by accident, and
nothing downstream would complain - the stage would simply stop being self-supervised.

So these tests attack it three ways:
  * structural  - the dataset those stages read yields ONE tensor, so labels are absent from
                  the graph rather than merely unused;
  * lexical     - neither module's code mentions a label, a class or a class weight;
  * causal      - the same signals with two completely different label sets give bit-identical
                  losses. This is the one that cannot be fooled by an indirect path.
"""
import inspect
import os
import re
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

from ecgr import config, models
from ecgr.data import build_tfrecord, pipeline
from ecgr.training import cpc as cpc_mod
from ecgr.training import ssl as ssl_mod

SEED = 1234
N = 16


def _signals():
    """Structured signals: each strip its own frequency, so both objectives have something
    learnable and a loss that is not a constant."""
    rng = np.random.default_rng(0)
    t = np.arange(config.SEGMENT_SAMPLES) / config.SAMPLING_RATE
    return np.stack([np.stack([np.sin(2 * np.pi * f * t + p)] * config.IN_CHANNELS, -1)
                     for f, p in zip(rng.uniform(0.5, 8, N), rng.uniform(0, 6, N))]
                    ).astype('float32')


def _write(signals, labels, root, tag):
    out = os.path.join(root, tag, 'db', 'train')
    os.makedirs(out)
    path = os.path.join(out, 'x.tfrecord')
    with tf.io.TFRecordWriter(path) as writer:
        for i in range(len(signals)):
            writer.write(build_tfrecord._example(signals[i], labels[i]))
    return path


def _make_trainer(kind):
    model = models.build('resumamba_100k')
    if kind == 'ssl':
        return ssl_mod.MaskedReconstruction(
            models.sub_model(model, 'backbone'), config.IN_CHANNELS, config.OUTPUT_STEPS,
            config.SEGMENT_SAMPLES // config.OUTPUT_STEPS)
    encoder = models.sub_model(model, 'context_encoder')
    return cpc_mod.CPCTrainer(encoder, encoder.output_shape[0][-1])


def _losses(kind, path, epochs=3, steps=2):
    """Train `kind` for a few steps on `path` and return its per-epoch losses.

    Pinned to the CPU: the causal test below demands BIT-identical losses, and GPU
    reductions over a 15000 x 3 strip are not deterministic at the last float32 digit
    (measured: 0.5576407 vs 0.5576411 for the same signals). On the CPU the same code is
    exactly reproducible, so a difference there can only be a label reaching the objective.
    """
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)          # same init AND same masking/negatives draw
    with tf.device('/CPU:0'):
        dataset = pipeline.make_dataset([path], 8, training=False, cache=False,
                                        signal_only=True)
        trainer = _make_trainer(kind)
        trainer.compile(optimizer=keras.optimizers.Adam(1e-3), jit_compile=False)
        history = trainer.fit(dataset.repeat(), epochs=epochs, steps_per_epoch=steps,
                              verbose=0)
    return dataset.element_spec, [float(v) for v in history.history['loss']]


# --- structural ------------------------------------------------------------------------

def test_the_signal_only_dataset_carries_no_labels(tmp_path):
    signals = _signals()
    path = _write(signals, np.zeros((N, config.OUTPUT_STEPS), 'uint8'), str(tmp_path), 'a')
    signal_only = pipeline.make_dataset([path], 8, cache=False, signal_only=True)
    assert isinstance(signal_only.element_spec, tf.TensorSpec), \
        "a label-free stage must receive one tensor, not a (signal, labels) pair"
    assert tuple(signal_only.element_spec.shape[1:]) == (config.SEGMENT_SAMPLES,
                                                         config.IN_CHANNELS)
    # ... while the supervised path does hand over both, so the flag is what makes it so
    supervised = pipeline.make_dataset([path], 8, cache=False, signal_only=False)
    assert isinstance(supervised.element_spec, tuple) and len(supervised.element_spec) == 2
    assert 'beat_cls' in supervised.element_spec[1]


@pytest.mark.parametrize('module', [ssl_mod, cpc_mod])
def test_the_trainers_take_a_single_tensor(module):
    trainer_cls = (module.MaskedReconstruction if module is ssl_mod else module.CPCTrainer)
    for method in ('train_step', 'test_step'):
        params = list(inspect.signature(getattr(trainer_cls, method)).parameters)
        assert params == ['self', 'signal'], f"{method} signature is {params}"


# --- lexical ---------------------------------------------------------------------------

@pytest.mark.parametrize('module', [ssl_mod, cpc_mod])
def test_no_label_machinery_is_referenced_in_the_code(module):
    """Docstrings may discuss labels; the code must not touch them."""
    source = inspect.getsource(module)
    code = '\n'.join(line for line in source.splitlines()
                     if not line.strip().startswith('#'))
    code = re.sub(r'"""(?:.|\n)*?"""', '', code)          # strip docstrings
    # Not 'one_hot': ssl._lead_mask one-hots a CHANNEL index to pick which lead to mask,
    # which has nothing to do with class labels. The list names label machinery only.
    for forbidden in ('y_true', 'CLASS_WEIGHTS', 'SYMBOL_TO_LABEL', 'CLASS_NAMES',
                      'NUM_CLASSES', 'LOSSES', 'labels_from_annotations'):
        assert forbidden not in code, f"{module.__name__} references {forbidden}"


# --- causal: the one that matters ------------------------------------------------------

@pytest.mark.parametrize('kind', ['ssl', 'cpc'])
def test_two_different_label_sets_give_identical_losses(kind, tmp_path):
    """Same signals, labels all-zero vs uniformly random over the classes. If a label ever
    reached either objective - directly, or through the augment path, or through a class
    weight - these two runs would diverge."""
    signals = _signals()
    rng = np.random.default_rng(1)
    zeros = np.zeros((N, config.OUTPUT_STEPS), 'uint8')
    random = rng.integers(0, config.NUM_CLASSES, (N, config.OUTPUT_STEPS)).astype('uint8')
    assert (zeros != random).sum() > 0.5 * zeros.size, "the two label sets must really differ"

    root = str(tmp_path)
    spec_a, losses_a = _losses(kind, _write(signals, zeros, root, 'a'))
    spec_b, losses_b = _losses(kind, _write(signals, random, root, 'b'))

    assert spec_a == spec_b
    assert losses_a == losses_b, f"labels changed the {kind} loss: {losses_a} vs {losses_b}"
    # and the objective is actually training, or two equal constants would prove nothing
    assert len(losses_a) > 1 and len(set(losses_a)) > 1, \
        f"the {kind} loss never moved across epochs ({losses_a}) - the test proves nothing"


def test_the_lead_quality_target_never_reads_a_label(tmp_path):
    """Output 2 is label-free too: the quality target of the very same signals is identical
    whatever the beat labels say, because it is built from the signal and from what the
    pipeline did to it (data/pipeline.corrupt), never from the annotation stream."""
    signals = _signals()
    rng = np.random.default_rng(2)
    zeros = np.zeros((N, config.OUTPUT_STEPS), 'uint8')
    random = rng.integers(0, config.NUM_CLASSES, (N, config.OUTPUT_STEPS)).astype('uint8')
    root = str(tmp_path)
    targets = []
    for tag, labels in (('a', zeros), ('b', random)):
        path = _write(signals, labels, root, tag)
        keras.utils.set_random_seed(SEED)
        for _, y in pipeline.make_dataset([path], N, cache=False, training=True).take(1):
            targets.append(y['lead_quality'].numpy())
    assert np.array_equal(*targets), "the beat labels changed the lead-quality target"
    assert 0.0 < targets[0].mean() < 1.0, "something was corrupted, or the test is vacuous"
    # and the code that builds it does not name the label machinery
    source = inspect.getsource(pipeline.quality_from_corruption) + \
        inspect.getsource(pipeline.corrupt) + inspect.getsource(pipeline.flat_mask)
    for forbidden in ('beat_cls', 'labels', 'CLASS_NAMES', 'SYMBOL_TO_LABEL', 'NUM_CLASSES'):
        assert forbidden not in source.replace('# ', ''), f"quality target reads {forbidden}"


def test_the_self_supervised_path_reads_a_corpus_with_no_labels_at_all(tmp_path):
    """The point of self-supervision is pretraining on UNLABELLED recordings.

    Parsing the labels and then dropping them still makes the field mandatory, so a tfrecord
    written from unlabelled data could not be read at all - the stage was label-free in its
    objective but not in its input requirement. parse_signal declares only the signal.
    """
    out = tmp_path / 'db' / 'train'
    out.mkdir(parents=True)
    path = str(out / 'x.tfrecord')
    signals = _signals()[:4]
    with tf.io.TFRecordWriter(path) as writer:
        for i in range(len(signals)):          # NO 'labels' feature is written
            example = tf.train.Example(features=tf.train.Features(feature={
                'signal': tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[signals[i].tobytes()]))}))
            writer.write(example.SerializeToString())

    dataset = pipeline.make_dataset([path], 2, cache=False, signal_only=True)
    batches = list(dataset)
    assert len(batches) == 2
    assert tuple(batches[0].shape) == (2, config.SEGMENT_SAMPLES, config.IN_CHANNELS)
    assert np.allclose(np.concatenate([b.numpy() for b in batches]), signals)

    # the supervised path, by contrast, needs the labels and must say so rather than guess
    with pytest.raises(tf.errors.InvalidArgumentError):
        list(pipeline.make_dataset([path], 2, cache=False, signal_only=False, targets=False))
