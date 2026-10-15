"""Small dual U-Nets: encoder-only graph, SSL trainer, teacher -> student distiller."""
import os

import keras
import numpy as np
import pytest
import tensorflow as tf

from ecgr.rhythm import augment as A
from ecgr.rhythm import config as rc
from ecgr.rhythm import distill as D
from ecgr.rhythm import model as rmodel


@pytest.fixture(scope='module')
def teacher():
    t = rmodel.build('rhythm_dual_30k')        # stands in for the 1M teacher
    t.trainable = False
    return t


def _batch(n=4, seed=0):
    rng = np.random.default_rng(seed)
    x = tf.constant(rng.normal(size=(n, rc.SEGMENT_SAMPLES, 3)).astype(np.float16))
    y = np.zeros((n, rc.BEAT_STEPS), np.uint8)
    y[:, 600:700] = 1
    bt = np.zeros((n, rc.SEGMENT_SAMPLES), np.uint8)
    bt[:, 200] = 1; bt[:, 900] = 3; bt[:, 1600] = 2
    return A.augment(x, tf.constant(y), tf.constant([3, 3], tf.int64), beats=tf.constant(bt))


@pytest.mark.parametrize('name', ['rhythm_dual_100k', 'rhythm_dual_30k'])
def test_small_models_keep_the_teacher_contract(name):
    m = rmodel.build(name)
    assert 0.7 * rmodel.BUDGETS[name] < m.count_params() < rmodel.BUDGETS[name]
    shapes = {k: tuple(v.shape[1:]) for k, v in m.output.items()}
    assert shapes == {'beat': (rc.BEAT_STEPS, 4), 'rhythm': (rc.BEAT_STEPS, rc.NUM_CLASSES),
                      'channel': (rc.NOISE_SEGMENTS, rc.NUM_LEAD_CLASSES)}
    assert m.input_shape[1:] == (rc.SEGMENT_SAMPLES, rc.IN_CHANNELS)


def test_encoder_graph_shares_the_student_layer_names():
    enc = D.build_encoder('rhythm_dual_30k')
    student = rmodel.build('rhythm_dual_30k')
    enc_paths = {w.path for w in enc.weights}
    student_paths = {w.path: tuple(w.shape) for w in student.weights}
    assert enc_paths and all(p in student_paths for p in enc_paths)
    assert all(tuple(w.shape) == student_paths[w.path] for w in enc.weights)
    # the position decoder is not part of the SSL graph: it is driven by the teacher
    assert not any(p.startswith(('pos', 'beat_pos')) for p in enc_paths)
    assert set(enc.output) == {'z', 'v1', 'a1'}


def test_ssl_step_learns_only_the_encoder_and_head(teacher):
    enc = D.build_encoder('rhythm_dual_30k')
    trainer = D.MaskedEncoderSSL(enc, teacher)
    trainer.compile(optimizer=keras.optimizers.Adam(1e-3))
    x, _ = _batch()
    before = [w.numpy().copy() for w in teacher.weights]
    logs = trainer.train_step(x)
    assert np.isfinite(float(logs['loss'])) and float(logs['nmse']) > 0
    assert all(np.array_equal(a, w.numpy()) for a, w in zip(before, teacher.weights))
    time, keep = trainer.masks(256)
    assert 0.2 < float(tf.reduce_mean(time)) < 0.5                 # ~ SSL_MASK_RATIO
    assert 0.1 < float(tf.reduce_mean(tf.cast(tf.reduce_min(keep, -1) == 0, tf.float32))) < 0.5


def test_ssl_encoder_initialises_the_student(teacher, tmp_path):
    from ecgr.rhythm.train import init_matching
    enc = D.build_encoder('rhythm_dual_30k')
    trainer = D.MaskedEncoderSSL(enc, teacher)
    trainer.compile(optimizer=keras.optimizers.Adam(1e-2))
    x, _ = _batch()
    trainer.train_step(x)
    path = os.path.join(tmp_path, 'ssl_encoder.keras')
    enc.save(path)
    student = rmodel.build('rhythm_dual_30k')
    full = init_matching(student, path)
    assert {'v1_a_conv', 'a1_d8_conv', 'neck3_a_conv', 'stem0_conv'} <= full
    got = {w.path: w.numpy() for w in student.weights}
    for w in enc.weights:
        np.testing.assert_array_equal(got[w.path], w.numpy())


def test_kd_kl_is_zero_for_identical_and_positive_otherwise():
    p = tf.constant([[0.7, 0.2, 0.1]])
    assert float(D.kd_kl(p, p, 2.0)[0]) == pytest.approx(0.0, abs=1e-6)
    q = tf.constant([[0.1, 0.2, 0.7]])
    assert float(D.kd_kl(p, q, 2.0)[0]) > 0


def test_distiller_step_trains_the_student_only(teacher, tmp_path):
    student = rmodel.build('rhythm_dual_30k')
    trainer = D.Distiller(student, teacher, [1.0] * rc.NUM_CLASSES)
    trainer.compile(optimizer=keras.optimizers.Adam(1e-3))
    x, t = _batch(8)
    y = {k: t[k] for k in ('rhythm', 'beat', 'channel')}
    t_before = [w.numpy().copy() for w in teacher.weights]
    s_before = [w.numpy().copy() for w in student.trainable_weights]
    logs = trainer.train_step((x, y))
    assert set(logs) >= {'loss', 'kd_loss', 'rhythm_loss', 'beat_loss', 'channel_loss',
                         'rhythm_f1', 'beat_f1', 'channel_acc', 'channel_noise_f1'}
    assert all(np.isfinite(float(v)) for v in logs.values())
    assert all(np.array_equal(a, w.numpy()) for a, w in zip(t_before, teacher.weights))
    assert any(not np.array_equal(a, w.numpy())
               for a, w in zip(s_before, student.trainable_weights))
    out = trainer.predict(x.numpy().astype('float32'), verbose=0)       # = the student
    assert out['rhythm'].shape == (8, rc.BEAT_STEPS, rc.NUM_CLASSES)
    cb = D.SaveBestStudent(os.path.join(tmp_path, 'best.keras'), start_epoch=1)
    cb.set_model(trainer)
    cb.on_epoch_end(0, {rc.MONITOR: 0.5})
    again = keras.models.load_model(os.path.join(tmp_path, 'best.keras'), compile=False)
    assert again.count_params() == student.count_params()
