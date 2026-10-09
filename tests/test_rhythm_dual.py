"""The dual U-Net (ecgr/rhythm/dualunet.py): contract, lead symmetry, QRST cancellation,
gradient isolation of the beat head, the 'channel' target and the predict path."""
import os

import numpy as np
import pytest
import tensorflow as tf

from ecgr.rhythm import augment as A
from ecgr.rhythm import config as rc
from ecgr.rhythm import model as rmodel
from ecgr.rhythm.dualunet import QRSTCancel
from ecgr.rhythm.objectives import BeatF1, LeadConfusion, RhythmF1, beat_loss, lead_loss, \
    rhythm_loss

NAME = 'rhythm_dual_30k'


@pytest.fixture(scope='module')
def dual():
    return rmodel.build(NAME)


def _x(n=2, seed=0):
    return np.random.default_rng(seed).normal(size=(n, rc.SEGMENT_SAMPLES, 3)).astype('float32')


def test_contract(dual):
    assert dual.name == 'dualunet_rhythm_30k'
    assert dual.input_shape[1:] == (rc.SEGMENT_SAMPLES, rc.IN_CHANNELS)
    shapes = {k: tuple(v.shape[1:]) for k, v in dual.output.items()}
    assert shapes == {'beat': (rc.BEAT_STEPS, len(rc.BEAT_CLASSES)),
                      'rhythm': (rc.BEAT_STEPS, rc.NUM_CLASSES),
                      'channel': (rc.NOISE_SEGMENTS, rc.NUM_LEAD_CLASSES)}
    y = dual(_x())
    for k in y:
        np.testing.assert_allclose(y[k].numpy().sum(-1), 1.0, atol=1e-5)
    names = {l.name for l in dual.layers}
    # two branches, a shared neck, no attention anywhere
    assert {'v1_a_conv', 'vb_a_conv', 'a1_d8_conv', 'ab_a_conv', 'qrst_cancel',
            'neck_ssm0_ssm', 'beat_pos', 'beat_type', 'rhythm_film'} <= names
    assert not any('mha' in n or 'attention' in n for n in names)
    assert not any(isinstance(l, tf.keras.layers.MultiHeadAttention) for l in dual.layers)


def test_lead_order_symmetry(dual):
    """beat and rhythm read the leads through symmetric pooling: invariant to their order.
    channel: CH1..CH3 move with the leads, NOISE does not."""
    x = _x(2, 1)
    perm = [2, 0, 1]
    a = dual(x, training=False)
    b = dual(x[..., perm], training=False)
    for k in ('beat', 'rhythm'):
        np.testing.assert_allclose(a[k].numpy(), b[k].numpy(), atol=1e-4)
    ca, cb = a['channel'].numpy(), b['channel'].numpy()
    np.testing.assert_allclose(cb[..., 0], ca[..., 0], atol=1e-4)
    np.testing.assert_allclose(cb[..., 1:], ca[..., 1:][..., perm], atol=1e-4)


def _synthetic_beats(f_amp=0.0, p_amp=0.0):
    """10 s at 250 Hz, a QRST complex every 0.8 s on 3 leads (+ an optional P wave 180 ms
    before each R, + optional 6 Hz 'f waves' everywhere). Returns (x, R samples)."""
    fs, n = rc.SAMPLING_RATE, rc.SEGMENT_SAMPLES
    t = np.arange(n) / fs
    r = np.arange(int(0.4 * fs), n - int(0.5 * fs), int(0.8 * fs))
    x = np.zeros(n)
    for s in r:
        x += 1.5 * np.exp(-0.5 * ((t - s / fs) / 0.012) ** 2)              # QRS
        x += 0.3 * np.exp(-0.5 * ((t - s / fs - 0.28) / 0.05) ** 2)        # T
        x += p_amp * np.exp(-0.5 * ((t - s / fs + 0.18) / 0.025) ** 2)     # P
    x += f_amp * np.sin(2 * np.pi * 6.0 * t)
    return np.stack([x, 0.7 * x, -0.5 * x], -1)[None].astype('float32'), r


def _heat(r):
    p = np.zeros((1, rc.BEAT_STEPS, 1), np.float32)
    for s in r:
        k = s // 2
        p[0, k - 3:k + 4, 0] = np.exp(-0.5 * (np.arange(-3, 4) / 1.5) ** 2)
    return p


def test_qrst_cancellation_keeps_p_and_f_waves():
    layer = QRSTCancel()
    x, r = _synthetic_beats(f_amp=0.08, p_amp=0.2)
    residual, qrs = (v.numpy() for v in layer([tf.constant(x), tf.constant(_heat(r))]))
    assert residual.shape == (1, rc.BEAT_STEPS, 3) and qrs.shape == (1, rc.BEAT_STEPS, 1)
    xd = x.reshape(1, rc.BEAT_STEPS, 2, 3).mean(2)
    # the QRST complexes go: the residual's peak is a fraction of the signal's
    assert np.abs(residual[0, :, 0]).max() < 0.25 * np.abs(xd[0, :, 0]).max()
    # the P wave stays (it is outside the -100 / +450 ms template)
    p_steps = [(s - int(0.18 * rc.SAMPLING_RATE)) // 2 for s in r]
    assert np.mean([residual[0, k, 0] for k in p_steps]) > 0.12
    # the 6 Hz f waves stay: correlation with the injected sinusoid
    t = (np.arange(rc.BEAT_STEPS) * 2 + 0.5) / rc.SAMPLING_RATE
    f = np.sin(2 * np.pi * 6.0 * t)
    _, r2 = _synthetic_beats(f_amp=0.08)
    x2, _ = _synthetic_beats(f_amp=0.08)
    res2 = layer([tf.constant(x2), tf.constant(_heat(r2))])[0].numpy()[0, :, 0]
    assert np.corrcoef(res2, f)[0, 1] > 0.8
    # the QRS mask marks the beats
    assert all(qrs[0, s // 2, 0] == 1.0 for s in r) and qrs.mean() < 0.2


def test_qrst_cancellation_without_beats_is_identity():
    x, _ = _synthetic_beats(f_amp=0.1)
    residual, qrs = QRSTCancel()([tf.constant(x),
                                  tf.zeros((1, rc.BEAT_STEPS, 1), tf.float32)])
    xd = x.reshape(1, rc.BEAT_STEPS, 2, 3).mean(2)
    np.testing.assert_allclose(residual.numpy(), xd, atol=1e-5)
    assert float(tf.reduce_sum(qrs)) == 0.0


def test_rhythm_loss_does_not_reach_the_beat_head(dual):
    x = _x(2, 2)
    with tf.GradientTape() as tape:
        out = dual(x, training=True)
        loss = tf.reduce_mean(out['rhythm'][..., 1])
    beat_only = [v for v in dual.trainable_variables
                 if v.path.split('/')[0] in ('beat_pos', 'beat_type', 'type_a_conv',
                                             'pos1_a_conv')]
    assert len(beat_only) >= 4
    assert all(g is None for g in tape.gradient(loss, beat_only))


def test_channel_target_follows_noise_and_leads():
    rng = np.random.default_rng(3)
    x = tf.constant(rng.normal(size=(64, rc.SEGMENT_SAMPLES, 3)).astype(np.float16))
    y = tf.zeros((64, rc.BEAT_STEPS), tf.uint8)
    _, t = A.augment(x, y, tf.constant([4, 4], tf.int64))
    ch, noise = t['channel'].numpy(), t['noise'].numpy()
    assert ch.shape == (64, rc.NOISE_SEGMENTS, rc.NUM_LEAD_CLASSES)
    np.testing.assert_allclose(ch.sum(-1), 1.0)
    # NOISE in 'channel' is exactly the 'noise' head's NOISE
    np.testing.assert_array_equal(ch[..., rc.LEAD_NOISE], noise[..., 1])
    assert 0 < noise[..., 1].mean() < 1
    # no noise, no permutation: a clean segment picks a real lead
    _, t0 = A.augment(x, y, tf.constant([5, 5], tf.int64), noise_prob=0.0, wreck_prob=0.0,
                      permute_prob=0.0, drop_prob=0.0)
    assert (np.argmax(t0['channel'].numpy(), -1) > 0).all()
    # the clean stored windows get the target too
    _, tc = A.no_augment(x, y)
    assert tc['channel'].shape == (64, rc.NOISE_SEGMENTS, rc.NUM_LEAD_CLASSES)


def test_channel_scores_rank_the_noisy_lead_last():
    n = rc.SEGMENT_SAMPLES
    rng = np.random.default_rng(4)
    clean = tf.constant(rng.normal(size=(1, n, 3)).astype(np.float32))
    sig_rms = tf.ones((1, 1, 3))
    noise = np.zeros((1, n, 3), np.float32)
    noise[0, :1000, 1] = 5.0 * rng.normal(size=1000)          # lead 2 noisy in segments 0-1
    key = A.channel_scores(clean, sig_rms, tf.constant(noise), sig_rms > 0.05).numpy()[0]
    assert key.shape == (rc.NOISE_SEGMENTS, 3)
    assert (key[:2, 1] < key[:2, 0]).all() and (key[:2, 1] < key[:2, 2]).all()
    label = A.channel_label(tf.constant(key[None]), tf.ones((1, rc.NOISE_SEGMENTS))).numpy()
    assert (label[0, :2] != 2).all() and (label > 0).all()


def test_losses_and_metrics_run(dual):
    rng = np.random.default_rng(5)
    x = tf.constant(rng.normal(size=(16, rc.SEGMENT_SAMPLES, 3)).astype(np.float16))
    y = np.zeros((16, rc.BEAT_STEPS), np.uint8)
    y[:, 600:700] = 1
    bt = np.zeros((16, rc.SEGMENT_SAMPLES), np.uint8)
    bt[:, 200] = 1; bt[:, 900] = 3; bt[:, 1600] = 2
    xa, t = A.augment(x, tf.constant(y), tf.constant([6, 6], tf.int64), beats=tf.constant(bt))
    p = dual(xa)
    assert np.all(np.isfinite(rhythm_loss()(t['rhythm'], p['rhythm']).numpy()))
    assert np.all(np.isfinite(beat_loss(list(rc.BEAT_CLASS_WEIGHTS))(t['beat'],
                                                                      p['beat']).numpy()))
    assert np.isfinite(float(lead_loss()(t['channel'], p['channel'])))
    acc = LeadConfusion(mode='accuracy')
    acc.update_state(t['channel'], p['channel'])
    assert acc.matrix().sum() == 16 * rc.NOISE_SEGMENTS
    for m, k in ((RhythmF1(), 'rhythm'), (BeatF1(), 'beat'), (LeadConfusion(), 'channel')):
        m.update_state(t[k], p[k])
        assert 0.0 <= float(m.result()) <= 1.0


def test_save_load_roundtrip(dual, tmp_path):
    path = os.path.join(tmp_path, 'dual.keras')
    dual.save(path)
    again = tf.keras.models.load_model(path, compile=False)
    x = _x(1, 7)
    for k in rmodel.output_names(dual):
        np.testing.assert_allclose(again(x)[k].numpy(), dual(x)[k].numpy(), atol=1e-5)


def test_predict_signal_reads_noise_and_channel(dual):
    from ecgr.rhythm.predict import predict_signal, probs_step_hz
    x, _ = _synthetic_beats(f_amp=0.05)
    sig = np.concatenate([x[0], x[0], x[0]])[:7300]                      # 29.2 s
    rhythm, p_noise, windows, beats = predict_signal(dual, sig, with_beats=True)
    hz = probs_step_hz(dual)
    assert rhythm.shape == (29 * hz, rc.NUM_CLASSES) and p_noise.shape == (29 * hz,)
    assert np.all((p_noise >= 0) & (p_noise <= 1))
    assert all(len(w['channel']) == rc.NOISE_SEGMENTS and len(w['noise']) == rc.NOISE_SEGMENTS
               and set(w['channel']) <= set(rc.LEAD_CLASSES) for w in windows)
    assert beats is not None and set(beats) >= {'t', 'cls'}
