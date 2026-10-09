"""XAI web service engine (ecgr.rhythm.xai_web) on a freshly built, untrained small dual U-Net."""
import base64
import json

import numpy as np
import pytest

from ecgr.rhythm import config as rc
from ecgr.rhythm import xai_web as W
from ecgr.rhythm.model import build


@pytest.fixture(scope='module')
def engine():
    return W.Engine('untrained', model=build('rhythm_dual_30k'), physionet_dir='/nonexistent')


@pytest.fixture(scope='module')
def raw():
    t = np.arange(rc.SEGMENT_SAMPLES) / rc.SAMPLING_RATE
    beat = np.exp(-((t % 0.8) - 0.4) ** 2 / 2e-4)
    return np.stack([beat, 0.5 * beat, 0.2 * beat], 1).astype(np.float32)


def test_edits_shape_and_identity(raw):
    rng = np.random.default_rng(0)
    assert np.array_equal(W.apply_edits(raw, {}, rng), raw)
    e = dict(f_wave=dict(on=True, amp=0.1), white=dict(on=True, snr_db=0, leads=[0]),
             band=dict(on=True, snr_db=6), wander=dict(on=True), mains=dict(on=True),
             flat=dict(on=True, **{'from': 2, 'to': 3}), gain=[1, 0, -1], order=[1, 0, 2], flip=True)
    x = W.apply_edits(raw, e, rng)
    assert x.shape == raw.shape and np.isfinite(x).all()
    assert np.allclose(W.apply_edits(raw, dict(order=[1, 0, 2]), rng), raw[:, [1, 0, 2]])
    assert np.allclose(W.apply_edits(raw, dict(gain=[1, 0, 1]), rng)[:, 1], 0)


def test_patches_compose_and_restore(engine, raw):
    x = W.normalize(raw)[None]
    base = engine.forward_batch(x, {}, batch=1)
    iv = dict(atrial_off=True, ssm_identity=engine.ssm, v_lead_off=[0], lead_temperature=2.0,
              ssm_kernel_frac=0.5, qrst_pre=0.08, rr_mean=False)
    items, _ = engine.intervention_items(iv)
    mod = engine.forward_batch(x, iv, batch=1)
    assert mod.shape == base.shape
    again = engine.forward_batch(x, {}, batch=1)
    assert np.allclose(base, again, atol=1e-6)
    assert engine.model.get_layer('qrst_cancel').pre != 0.08


def test_analyze_array_source(engine, raw):
    req = dict(source=dict(kind='array', data=raw.tolist(), fs=rc.SAMPLING_RATE),
               edits=dict(flip=True), interventions=dict(atrial_off=True),
               target=dict(cls='auto', seconds=[4, 5]), ig_steps=4)
    r = W._clean(engine.analyze(req))
    json.dumps(r, allow_nan=False)
    for run in ('orig', 'mod'):
        R = r['runs'][run]
        assert np.array(R['per_second']).shape == (rc.OUTPUT_SECONDS, rc.NUM_CLASSES)
        assert np.array(R['ig']).shape == (3, rc.SEGMENT_SAMPLES)
        assert all(len(L['rel_profile']) == W.OVERVIEW_BINS for L in R['layers'] if not L['vector'])
    names = [L['name'] for L in r['runs']['mod']['layers'] if not L['vector']]
    d = engine.layer_detail(r['id'], names[0])
    assert len(base64.b64decode(d['act'])) == d['T'] * d['C']


def test_resample_time():
    a = np.arange(10, dtype=np.float32)
    assert W._resample_time(a, 5).tolist() == [0.5, 2.5, 4.5, 6.5, 8.5]
    assert W._resample_time(a[:2], 4).tolist() == [0, 0, 1, 1]
