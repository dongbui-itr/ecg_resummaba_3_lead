"""XAI module (ecgr.rhythm.xai) on a freshly built, untrained small dual U-Net."""
import numpy as np
import pytest
import tensorflow as tf

from ecgr.rhythm import config as rc
from ecgr.rhythm import xai
from ecgr.rhythm.model import build


@pytest.fixture(scope='module')
def model():
    return build('rhythm_dual_30k')


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, rc.SEGMENT_SAMPLES, 3)).astype(np.float32)
    labels = np.repeat(np.arange(rc.NUM_CLASSES), 2)[:, None].repeat(rc.OUTPUT_SECONDS, 1)
    return x, labels, labels[:, 0]


def test_patch_changes_and_restores(model, data):
    x = data[0][:2]
    base = xai.run(model, x)['rhythm']
    with xai.patches(model, [xai._map_patch('qrst_cancel', tf.zeros_like, index=0)]):
        off = xai.run(model, x)['rhythm']
    again = xai.run(model, x)['rhythm']
    assert np.allclose(base, again, atol=1e-6)
    assert off.shape == base.shape


def test_donor_with_own_values_is_identity(model, data):
    x = data[0][:3]
    rr = np.asarray(xai.sub_model(model, ['rr_ac'])(x, training=False))
    base = xai.run(model, x)['rhythm']
    same = xai._run_with_donor(model, x, rr, 'rr_ac', None, batch=4)
    assert np.allclose(base, same, atol=1e-5)


def test_scores_perfect_prediction():
    labels = np.repeat(np.arange(rc.NUM_CLASSES), 2)[:, None].repeat(rc.OUTPUT_SECONDS, 1)
    probs = np.eye(rc.NUM_CLASSES)[labels]
    s = xai.scores(probs, labels)
    assert s['macro_f1'] == pytest.approx(1.0)
    assert all(s[n]['se'] == pytest.approx(1.0) for n in rc.CLASS_NAMES)


def test_f_waves_shape_and_band():
    w = xai.f_waves(2, 0.1, np.random.default_rng(0))
    assert w.shape == (2, rc.SEGMENT_SAMPLES, 3)
    spec = np.abs(np.fft.rfft(w[0, :, 0])) ** 2
    f = np.fft.rfftfreq(rc.SEGMENT_SAMPLES, 1 / rc.SAMPLING_RATE)
    assert spec[(f >= 3.5) & (f <= 9)].sum() > 0.8 * spec.sum()


def test_layer_parts_and_suggest(model, data):
    x, labels, cat = data
    ssm = xai.ssm_kernels(model)
    assert set(ssm) <= set(xai.SSM_LAYERS) and ssm
    q = xai.qrst_stats(model, x, cat)
    assert 'SINUS' in q and 0 <= q['SINUS']['share_4_9hz'] <= 1
    lw = xai.lead_weights(model, x)
    assert 0 <= lw['v_stem']['entropy'] <= 1.0001
    sh = xai.branch_shares(model, x, labels, cat, batch=2)
    for per in sh.values():
        for v in per.values():
            assert sum(v.values()) == pytest.approx(1.0, abs=1e-4)
    rep = dict(layers=dict(ssm=ssm, qrst=q, lead_weights=lw, dead=xai.dead_units(model, x[:2]),
                           branches=sh))
    assert isinstance(xai.suggest(rep), list)


def test_experiment_knobs_restore(model, data):
    x, labels, _ = data
    q = model.get_layer('qrst_cancel')
    before = (q.pre, q.post, q.min_prob)
    rows = xai.experiment(model, x[:2], labels[:2], log=lambda *a: None)
    assert rows[0]['knob'] == 'trained'
    assert (q.pre, q.post, q.min_prob) == before
