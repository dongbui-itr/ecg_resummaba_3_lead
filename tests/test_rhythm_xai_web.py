"""XAI web service engine (ecgr.rhythm.xai_web) on a freshly built, untrained small dual U-Net."""
import base64
import json
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')     # as the xai-web CLI does: CPU, GPU is shared

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


def test_tune_is_monotone_and_reports_check_set(engine, raw):
    from ecgr.rhythm import xai_tune as T
    rng = np.random.default_rng(1)
    wins = [raw * s + rng.normal(0, 0.01, raw.shape).astype(np.float32) for s in (0.8, 1.0, 1.2, 0.9)]
    cls = rc.CLASS_NAMES[1]
    assert all(m[2] != {} for m in T.moves(engine, {}))
    assert not any(k == 'atrial_off' and v == {} for k, _, v in T.moves(engine, {'atrial_off': True}))
    r = T.tune(engine, dict(cls=cls, windows=np.stack(wins).tolist(), max_rounds=1,
                            max_evals=12, interventions=dict(lead_temperature=2.0)))
    json.dumps(r, allow_nan=False)
    assert r['n_select'] == 2 and r['n_check'] == 2
    assert r['score']['tuned'] >= r['score']['start'] - 1e-9
    assert set(r['report']) == {'orig', 'start', 'tuned'}
    assert abs(sum(r['report']['orig']['check']['argmax_share'].values()) - 1) < 1e-6
    with pytest.raises(ValueError):
        T.tune(engine, dict(cls='nope', windows=[raw.tolist()] * 2))


def test_header_events():
    ev = W.header_events(['eventType: AVB2', '- comment: 2nd Deg HB', 'startSample: 7500',
                          'stopSample: 10000', 'eventStartSample: 7750', 'eventStopSample: 99999'], 15000)
    assert ev == [dict(a=7750, b=15000, cls=rc.CLASS_NAMES.index('AVB'), text='AVB2 · 2nd Deg HB')]
    ev = W.header_events(['eventType: AFIB', 'eventStartSample: -1', 'eventStopSample: -1'], 5000)
    assert (ev[0]['a'], ev[0]['b'], ev[0]['cls']) == (0, 5000, rc.CLASS_NAMES.index('AFIB'))
    ev = W.header_events(['Age: 60', 'Dx: 164889003,59118001'], 5000)
    assert ev[0]['cls'] == rc.CLASS_NAMES.index('AFIB') and ev[0]['b'] == 5000
    assert W.header_events(['83 M 1268 694 x1'], 100) == []


def test_file_source_any_wfdb_record(engine, tmp_path):
    import wfdb
    fs, n = 360, 360 * 30
    t = np.arange(n) / fs
    sig = np.stack([np.sin(2 * np.pi * 1.2 * t), 0.5 * np.cos(2 * np.pi * 1.2 * t)], 1)
    wfdb.wrsamp('rec', fs=fs, units=['mV', 'mV'], sig_name=['MLII', 'V1'], p_signal=sig,
                fmt=['16', '16'], write_dir=str(tmp_path), comments=['eventType: VT',
                                                                  'eventStartSample: 3600',
                                                                  'eventStopSample: 7200'])
    wfdb.wrann('rec', 'atr', np.array([360, 3700, 3800, 4000]), np.array(['N', '+', 'V', '~']),
               aux_note=['', '(VT', '', ''], write_dir=str(tmp_path))
    e = W.Engine('x', model=engine.model, physionet_dir='/nonexistent', browse_roots=[str(tmp_path)])
    b = e.browse(str(tmp_path))
    assert [r['name'] for r in b['records']] == ['rec'] and b['records'][0]['ann'] == ['atr']
    fi = e.file_info(str(tmp_path / 'rec.dat'))
    assert fi['n_sig'] == 2 and fi['ann_exts'] == ['atr'] and fi['events'][0]['t0'] == 10.0
    w, info = e.load_source(dict(kind='file', path=str(tmp_path / 'rec'), start=5.5, leads=[1, None, 0]))
    assert w.shape == (rc.SEGMENT_SAMPLES, 3) and not w[:, 1].any() and w[:, 0].any() and w[:, 2].any()
    assert info['lead_names'] == ['V1', '—', 'MLII'] and info['start'] == 5.5
    lab = info['labels']
    assert lab['ann'] == 'atr' and lab['rhythm'][-1]['code'] == '(VT'
    assert [o['sym'] for o in lab['other']] == ['~']
    assert info['ref'][-1] == rc.CLASS_NAMES.index('VT') and info['ref'][0] == rc.SINUS
    _, info = e.load_source(dict(kind='file', path=str(tmp_path / 'rec'), start=8, ann=''))
    assert info['ref'][1] == -1 and info['ref'][3] == rc.CLASS_NAMES.index('VT')   # from the .hea
    assert info['labels']['events'][0]['t0'] == 2.0
    for bad in ('/etc/hostname', str(tmp_path / '..' / 'x')):
        with pytest.raises(PermissionError):
            e.resolve_path(bad)
    ov = e.record_overview(path=str(tmp_path / 'rec'), ann='')
    assert ov['seconds'] == 30 and ov['ref'][12] == rc.CLASS_NAMES.index('VT') and ov['ref'][0] == -1
    up = e.upload(dict(files=[dict(name=f, data=base64.b64encode((tmp_path / f).read_bytes()).decode())
                              for f in ('rec.hea', 'rec.dat', 'rec.atr')]))
    assert up['name'] == 'rec' and e.resolve_path(up['path']) == up['path']
