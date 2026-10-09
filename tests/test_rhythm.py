"""Rhythm task: labels, windows, holdout, augmentation + lead target, model contract."""
import json
import os

import numpy as np
import pytest
import tensorflow as tf

from ecgr.rhythm import augment as A
from ecgr.rhythm import config as rc
from ecgr.rhythm import inventory as inv
from ecgr.rhythm import labels as L
from ecgr.rhythm import model as rmodel
from ecgr.rhythm.objectives import (BeatF1, LeadConfusion, NoiseF1, RhythmF1, beat_loss,
                                    lead_loss, noise_loss, rhythm_loss)

FS = rc.SAMPLING_RATE
AF, SVT, VT = (rc.CLASS_NAMES.index(n) for n in ('AFIB', 'SVT', 'VT'))


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def test_beat_runs_need_three_consecutive_ectopics():
    samples = np.arange(12) * 200 + 100
    symbols = list('NSSNSSSNVVVN')
    runs = L.beat_runs(samples, symbols)
    assert [(c, a, b) for c, a, b in runs] == [
        (SVT, 900 - 38, 1300 + 38), (VT, 1700 - 38, 2100 + 38)]
    # outside the reviewed range the beats are not read at all
    assert L.beat_runs(samples, symbols, lo=1500) == [(VT, 1700 - 38, 2100 + 38)]


def test_second_labels_cover_known_and_ignore():
    known = [(0, 2500)]
    # AF over seconds 2..5.6 -> 2,3,4,5 are AF (5 is 60% covered), the rest SINUS
    lab = L.second_labels(0, [(AF, 500, 1400)], known)
    assert lab.tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    # nothing known outside the span -> IGNORE there
    lab = L.second_labels(0, [(AF, 500, 1400)], [])
    assert lab.tolist() == [rc.IGNORE] * 2 + [AF] * 4 + [rc.IGNORE] * 4


def test_a_short_span_is_not_lost():
    # 0.34 s AV-block caliper straddling a second boundary: covers no second by half
    avb3 = rc.CLASS_NAMES.index('AVB')
    lab = L.second_labels(0, [(avb3, 700, 785)], [(0, 2500)])
    assert (lab == avb3).sum() == 1 and lab[2] == avb3


def _probs(track):
    """A hard (T, NUM_CLASSES) array from a list of class indices."""
    return np.eye(rc.NUM_CLASSES, dtype=np.float32)[np.asarray(track)]


def _spans(eps):
    return [(e['rhythm'], e['start'], e['stop']) for e in eps]


def test_decode_folds_short_episodes_and_marks_noise():
    r = np.zeros((10, rc.NUM_CLASSES), np.float32)
    r[:, rc.SINUS] = 1.0
    r[2:4] = np.eye(rc.NUM_CLASSES)[AF]                       # 2 s AF: below the AF minimum
    r[5:8] = np.eye(rc.NUM_CLASSES)[VT]                       # 3 s VT: at the VT minimum
    p_noise = np.zeros(10)
    p_noise[9] = 0.9
    eps = L.decode_episodes(r, p_noise, min_seconds={'AFIB': 3, 'VT': 3}, smooth_seconds=1,
                            merge_gap={})
    assert _spans(eps) == [('SINUS', 0, 5), ('VT', 5, 8), ('SINUS', 8, 9), ('NOISE', 9, 10)]
    assert L.decode_episodes(np.zeros((0, rc.NUM_CLASSES))) == []


def test_decode_bridges_gaps_by_priority():
    S = rc.SINUS
    #        0  1  2   3   4   5  6   7   8   9  10  11  12  13
    track = [AF, AF, AF, S, S, AF, AF, VT, VT, SVT, S, VT, VT, AF]
    gaps = {'AFIB': 3, 'SVT': 0, 'VT': 2, 'AVB': 0}
    cls = L.bridge_gaps(track, gaps, rc.DECODE_PRIORITY)
    # VT (top priority) bridges over the SVT + SINUS gap of 2 s; AFIB bridges the 2 s SINUS
    assert cls.tolist() == [AF, AF, AF, AF, AF, AF, AF, VT, VT, VT, VT, VT, VT, AF]
    # a higher-priority class in the gap is never overwritten
    cls = L.bridge_gaps([AF, S, VT, S, AF], {'AFIB': 5, 'VT': 0}, rc.DECODE_PRIORITY)
    assert cls.tolist() == [AF, S, VT, S, AF]
    # NOISE in the gap is bridged like SINUS; a gap longer than the limit is not
    N = L.NOISE
    assert L.bridge_gaps([AF, N, S, AF], {'AFIB': 2}, ['AFIB']).tolist() == [AF] * 4
    assert L.bridge_gaps([AF, N, S, S, AF], {'AFIB': 2}, ['AFIB']).tolist() == [AF, N, S, S, AF]


def test_decode_min_duration_takes_agreeing_neighbours_else_sinus():
    S = rc.SINUS
    mins = {'AFIB': 7, 'SVT': 3, 'VT': 3, 'AVB': 2}
    # a 2 s SVT blip inside AFIB becomes AFIB; a 2 s VT between SINUS and AFIB becomes SINUS
    track = [AF] * 8 + [SVT, SVT] + [AF] * 8 + [S, S, VT, VT] + [AF] * 8
    cls = L.enforce_min_duration(track, mins)
    assert cls.tolist() == [AF] * 18 + [S, S, S, S] + [AF] * 8
    # at the record edge there is only one neighbour: SINUS
    assert L.enforce_min_duration([SVT, SVT] + [AF] * 7, mins).tolist() == [S, S] + [AF] * 7
    # a NOISE neighbour pair does not absorb the blip
    N = L.NOISE
    assert L.enforce_min_duration([N, VT, N], mins).tolist() == [N, S, N]


def test_decode_smoothing_and_second_bridging_pass():
    S = rc.SINUS
    # one flipped second in a long AF run: the 3 s moving average votes it back to AF
    track = [AF] * 6 + [SVT] + [AF] * 6
    r = _probs(track)
    r[6] = [0.1, 0.4, 0.5, 0, 0]                               # SVT by a hair
    eps = L.decode_episodes(r, smooth_seconds=3, merge_gap={}, min_seconds={})
    assert _spans(eps) == [('AFIB', 0, 13)]
    sm = L.smooth_probs(r, 3)
    np.testing.assert_allclose(sm.sum(1), 1.0, atol=1e-5)      # edges average what exists

    # a 2 s VT blip inside AF is AF (agreeing neighbours), with or without bridging
    track = [AF] * 8 + [VT, VT] + [AF] * 8
    for gap in ({'AFIB': 2}, {}):
        eps = L.decode_episodes(_probs(track), smooth_seconds=1, merge_gap=gap,
                                min_seconds={'VT': 3}, priority=['VT', 'AFIB'])
        assert _spans(eps) == [('AFIB', 0, 18)]
    # the second bridging pass: AF - VT blip - SINUS - AF: the blip becomes SINUS (neighbours
    # disagree), which turns the gap into 3 s of SINUS that AF may then bridge
    track = [AF] * 8 + [VT] + [S, S] + [AF] * 8
    eps = L.decode_episodes(_probs(track), smooth_seconds=1, merge_gap={'AFIB': 3},
                            min_seconds={'VT': 3}, priority=['VT', 'AFIB'])
    assert _spans(eps) == [('AFIB', 0, 19)]
    eps = L.decode_episodes(_probs(track), smooth_seconds=1, merge_gap={'AFIB': 2},
                            min_seconds={'VT': 3}, priority=['VT', 'AFIB'])
    assert _spans(eps) == [('AFIB', 0, 8), ('SINUS', 8, 11), ('AFIB', 11, 19)]
    # neighbours disagree -> SINUS, and the AF minimum then decides the leftovers
    track = [S] * 3 + [AF] * 8 + [VT, VT] + [SVT] * 4
    eps = L.decode_episodes(_probs(track), smooth_seconds=1, merge_gap={},
                            min_seconds={'AFIB': 7, 'VT': 3, 'SVT': 3})
    assert _spans(eps) == [('SINUS', 0, 3), ('AFIB', 3, 11), ('SINUS', 11, 13), ('SVT', 13, 17)]
    # the defaults run end to end on a real-shaped input
    eps = L.decode_episodes(_probs(track), np.zeros(len(track)))
    assert eps[0]['start'] == 0 and eps[-1]['stop'] == len(track)


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------

def test_window_starts_strip_and_long_spans():
    n = 15000
    assert inv.window_starts([(7500, 10000)], n) == [7500]
    assert inv.window_starts([(7500, 9999)], n) == [7500]          # the 2499-sample strip
    assert inv.window_starts([(8000, 9000)], n) == [7250]          # short: centred
    starts = inv.window_starts([(0, 15000)], n)
    assert starts[0] == 0 and starts[-1] == 12500
    assert all(b - a >= rc.WINDOW_HOP_SECONDS * FS // 2 for a, b in zip(starts, starts[1:]))
    # 2559 samples: one centred window, not two windows 59 samples apart
    assert inv.window_starts([(8157, 10716)], n) == [8186]
    # nested spans do not add windows
    assert inv.window_starts([(2303, 7335), (4410, 6423)], n) == \
        inv.window_starts([(2303, 7335)], n)


def test_event_region_policy():
    af = dict(spans=[(AF, 1000, 4000)], strip=None, types=['AFIB'], whole=True)
    spans, region, known = inv.event_region(af, 15000)
    assert region == [(1000, 4000)] and known == []                 # outside AF = unknown
    vt = dict(spans=[(VT, 1000, 1600)], strip=None, types=['VT'], whole=True)
    assert inv.event_region(vt, 15000)[2] == [(0, 15000)]           # outside VT = sinus
    strip = dict(spans=[(VT, 8750, 10037)], strip=(7500, 10000), types=['VE_RUN'],
                 whole=False)
    spans, region, known = inv.event_region(strip, 15000)
    assert region == [(7500, 10000)] and known == [(7500, 10000)]


# ---------------------------------------------------------------------------
# PTB-XL
# ---------------------------------------------------------------------------

def test_ptbxl_label_rule():
    from ecgr.rhythm import ptbxl
    assert ptbxl.record_label(['AFIB', 'IMI']) == ('AFIB', None)
    assert ptbxl.record_label(['SR', 'PSVT']) == ('SVT', None)
    assert ptbxl.record_label(['2AVB', 'CRBBB']) == ('AVB', None)
    assert ptbxl.record_label(['3AVB']) == ('AVB', None)
    assert ptbxl.record_label(['2AVB', '3AVB']) == ('AVB', None)        # one class now
    # flutter counts as AF (rc.EC57_AFL_AS_AF): alone or with AFIB it is an AFIB record
    assert ptbxl.record_label(['AFLT']) == ('AFIB', None)
    assert ptbxl.record_label(['AFIB', 'AFLT']) == ('AFIB', None)
    assert ptbxl.record_label(['SR', 'SVARR']) == (None, 'ambiguous')
    assert ptbxl.record_label(['AFIB', '3AVB']) == (None, 'several_classes')
    for hard in ('PACE', 'CLBBB', 'STACH', 'SBRAD', 'PVC', '1AVB', 'WPW'):
        assert ptbxl.record_label(['NORM', hard]) == ('SINUS', None)
    assert ptbxl.record_label(['NORM', 'SR']) == (None, 'plain')
    assert ptbxl.record_label(['IMI', 'LVH']) == (None, 'plain')


def test_ptbxl_lead_subsets_and_events():
    from ecgr.rhythm import ptbxl
    limb, chest = len(rc.PTBXL_LIMB_LEADS), len(ptbxl.LEAD_NAMES)
    for ecg_id in range(1, 300):
        subsets = ptbxl.lead_subsets(ecg_id)
        assert len(subsets) == rc.PTBXL_WINDOWS_PER_RECORD
        assert len(set(subsets)) == len(subsets)
        for s in subsets:
            assert len(s) == len(set(s)) == rc.IN_CHANNELS and all(0 <= i < chest for i in s)
        assert any(i < limb for i in subsets[0]) and any(i >= limb for i in subsets[0])
        assert subsets == ptbxl.lead_subsets(ecg_id)                 # a hash, not a draw
    assert ptbxl.lead_subsets(1) != ptbxl.lead_subsets(2)
    assert 0.0 <= ptbxl.hash01(5) < 1.0 and ptbxl.hash01(5) == ptbxl.hash01(5)

    row = dict(ecg_id=7, patient_id=42.0, filename_hr='records500/00000/00007_hr',
               codes=['AFIB'])
    events = ptbxl.make_events(row, 'AFIB', ptbxl_dir='/x')
    assert [e['event_id'] for e in events] == ['7_0', '7_1']
    ev = events[0]
    assert ev['study_id'] == str(rc.PTBXL_STUDY_OFFSET + 42) and ev['source'] == 'ptbxl'
    assert ev['spans'] == [(AF, 0, 5000)] and ev['strip'] == (0, 5000)
    assert ev['record'] == '/x/records500/00000/00007_hr' and len(ev['leads']) == 3
    assert ev['lead_names'] == [ptbxl.LEAD_NAMES[i] for i in ev['leads']]
    # the contract build.process_event consumes
    spans, region, known = inv.event_region(dict(ev, spans=[(AF, 0, 2500)], strip=(0, 2500)),
                                            2500)
    assert region == [(0, 2500)] and known == [(0, 2500)]
    assert inv.window_starts(region, 2500) == [0]


def test_read_leads_selects_columns_and_resamples(tmp_path):
    import wfdb
    from ecgr.rhythm.build import read_leads
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5000, 12))
    x[:, 4] = 5.0 * np.sin(np.arange(5000) * 2 * np.pi * 5 / 500)    # a 5 Hz tone on AVL
    wfdb.wrsamp('r12', fs=500, units=['mV'] * 12, sig_name=[f"L{i}" for i in range(12)],
                p_signal=x, write_dir=str(tmp_path))
    leads, ratio = read_leads(str(tmp_path / 'r12'), leads=[4, 7, 9])
    assert leads.shape == (2500, 3) and ratio == 0.5
    # channel 0 is the tone, band-passed but intact: one clear 5 Hz line
    spec = np.abs(np.fft.rfft(leads[:, 0]))
    assert np.argmax(spec) == 50                                        # 5 Hz x 10 s
    assert leads[:, 1].std() > 0 and leads[:, 2].std() > 0
    # default: the first three columns
    first, _ = read_leads(str(tmp_path / 'r12'))
    assert first.shape == (2500, 3) and not np.allclose(first[:, 0], leads[:, 0])


# ---------------------------------------------------------------------------
# holdout
# ---------------------------------------------------------------------------

def test_verify_split_refuses_any_overlap(tmp_path):
    inv.verify_split(['1', '2'], ['3'], ['9'], ['9', '8'], out_path=str(tmp_path / 'v.json'))
    assert json.load(open(tmp_path / 'v.json'))['verified']
    for train, evaluation, test in ((['1', '3'], ['3'], ['9']),
                                    (['1', '9'], ['3'], ['9']),
                                    (['1'], ['3', '9'], ['9'])):
        with pytest.raises(ValueError):
            inv.verify_split(train, evaluation, test, test)


def test_rhythm_test_studies_read_from_folder_and_json(tmp_path):
    (tmp_path / '111' / 'evt').mkdir(parents=True)
    (tmp_path / '222' / 'evt').mkdir(parents=True)
    with open(tmp_path / '222' / 'evt' / 'x.json', 'w') as f:
        json.dump({'studyId': '333'}, f)
    assert inv.rhythm_test_studies(str(tmp_path)) == {'111', '222', '333'}
    with pytest.raises(FileNotFoundError):
        inv.rhythm_test_studies(str(tmp_path / 'missing'))


# ---------------------------------------------------------------------------
# augmentation and the clean target
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def batch():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, rc.SEGMENT_SAMPLES, 3)).astype(np.float16)
    y = np.zeros((64, 10), np.uint8)
    y[:, 3] = rc.IGNORE
    y[:, 6] = AF
    return tf.constant(x), tf.constant(y)


def test_augment_is_deterministic_per_seed_and_keeps_shapes(batch):
    x, y = batch
    a = A.augment(x, y, tf.constant([1, 2], tf.int64))
    b = A.augment(x, y, tf.constant([1, 2], tf.int64))
    c = A.augment(x, y, tf.constant([1, 3], tf.int64))
    assert a[0].shape == (64, rc.SEGMENT_SAMPLES, 3)
    assert a[1]['rhythm'].shape == (64, 10, rc.NUM_CLASSES + 1)
    assert a[1]['lead'].shape == (64, rc.NUM_LEAD_CLASSES)
    np.testing.assert_array_equal(a[0].numpy(), b[0].numpy())
    assert not np.allclose(a[0].numpy(), c[0].numpy())


def test_targets_encode_ignore_and_noisy_weight(batch):
    x, y = batch
    _, t = A.augment(x, y, tf.constant([4, 2], tf.int64))
    r, lead = t['rhythm'].numpy(), t['lead'].numpy()
    assert r.shape == (64, 10, rc.NUM_CLASSES + 1) and lead.shape == (64, rc.NUM_LEAD_CLASSES)
    assert np.all(r[:, 3, :rc.NUM_CLASSES] == 0) and np.all(r[:, 3, -1] == 0)
    assert np.all(r[:, 6, AF] == 1)
    np.testing.assert_allclose(lead.sum(-1), 1.0)
    w = r[:, [i for i in range(10) if i != 3], -1]
    assert set(np.unique(w)) <= {1.0, rc.NOISY_SECOND_WEIGHT}
    assert (w < 1).any() and (w == 1).any()
    labels = np.argmax(lead, -1)
    assert (labels == rc.LEAD_NOISE).any() and (labels != rc.LEAD_NOISE).any()


def test_no_noise_means_a_real_lead_and_leads_only_permuted(batch):
    x, y = batch
    xa, t = A.augment(x, y, tf.constant([7, 7], tf.int64), noise_prob=0.0, wreck_prob=0.0)
    assert np.all(np.argmax(t['lead'].numpy(), -1) != rc.LEAD_NOISE)
    assert t['rhythm'].numpy()[..., -1].max() == 1.0
    # every output lead is one of the input leads (z-scored), up to gain + wander
    xs, xo = x.numpy().astype(np.float32), xa.numpy()
    for b in range(4):
        corr = np.abs(np.corrcoef(np.concatenate([xs[b].T, xo[b].T]))[:3, 3:])
        alive = xo[b].std(axis=0) > 0
        assert np.all(corr.max(axis=0)[alive] > 0.9)


def test_polarity_flip_leaves_targets_alone(batch):
    x, y = batch
    seed = tf.constant([5, 9], tf.int64)
    kw = dict(permute_prob=0.0, drop_prob=0.0)
    x0, t0 = A.augment(x, y, seed, flip_prob=0.0, **kw)
    x1, t1 = A.augment(x, y, seed, flip_prob=1.0, **kw)
    # same noise, same SNR, same kurtosis: identical lead / clean targets ...
    np.testing.assert_array_equal(t0['lead'].numpy(), t1['lead'].numpy())
    np.testing.assert_array_equal(t0['rhythm'].numpy(), t1['rhythm'].numpy())
    # ... over the sign-flipped signal (wander + noise are not flipped, hence the tolerance)
    x0, x1 = x0.numpy(), x1.numpy()
    corr = [np.corrcoef(x0[b, :, c], x1[b, :, c])[0, 1] for b in range(8) for c in range(3)]
    assert min(corr) < -0.5
    # with the default probability some leads flip and some do not
    x2, _ = A.augment(x, y, seed, noise_prob=0.0, wreck_prob=0.0, **kw)
    x3, _ = A.augment(x, y, seed, noise_prob=0.0, wreck_prob=0.0, flip_prob=0.0, **kw)
    sign = np.sign([np.corrcoef(x2.numpy()[b, :, c], x3.numpy()[b, :, c])[0, 1]
                    for b in range(64) for c in range(3)])
    assert (sign < 0).any() and (sign > 0).any()


def test_lead_target_moves_with_the_permutation(batch):
    x, y = batch
    seed = tf.constant([11, 3], tf.int64)
    x0, t0 = A.augment(x, y, seed, permute_prob=0.0)
    x1, t1 = A.augment(x, y, seed, permute_prob=1.0)
    l0, l1 = np.argmax(t0['lead'].numpy(), -1), np.argmax(t1['lead'].numpy(), -1)
    np.testing.assert_array_equal(l0 == rc.LEAD_NOISE, l1 == rc.LEAD_NOISE)
    x0, x1 = x0.numpy(), x1.numpy()
    for b in np.flatnonzero(l0 != rc.LEAD_NOISE):
        np.testing.assert_allclose(x1[b, :, l1[b] - 1], x0[b, :, l0[b] - 1], atol=1e-5)


def test_lead_scores_rank_readable_seconds_then_snr():
    n = rc.SEGMENT_SAMPLES
    rng = np.random.default_rng(1)
    clean = tf.constant(rng.normal(size=(4, n, 3)).astype(np.float32))
    sig = tf.ones([4, 1, 3])
    live = tf.constant([[[True, True, True]]] * 3 + [[[True, False, True]]])
    noise = np.zeros((4, n, 3), np.float32)
    noise[0, :, 0] = noise[0, :, 1] = 2.0           # 0: CH1, CH2 wrecked -> CH3
    noise[1, :, :] = 2.0                            # 1: all wrecked -> NOISE
    noise[1, :500, 2] = 0.0                         #    (CH3 readable 2 s: still < 8)
    noise[2, :, 0] = 0.3                            # 2: all readable, CH1 10 dB, CH2 5 s bad
    noise[2, :1250, 1] = 2.0                        #    -> CH3 (untouched) wins on SNR
    noise[3, :, 2] = 2.0                            # 3: CH2 flat, CH3 wrecked -> CH1
    key, n_read = A.lead_scores(clean, sig, tf.constant(noise), live)
    assert A.lead_label(key, n_read).numpy().tolist() == [3, rc.LEAD_NOISE, 3, 1]
    assert n_read.numpy()[1].tolist() == [0.0, 0.0, 2.0]


def test_clean_seconds_counts_readable_leads():
    n = rc.SEGMENT_SAMPLES
    sig = tf.ones([1, 1, 3])
    live = tf.constant([[[True, True, True]]])
    noise = np.zeros((1, n, 3), np.float32)
    noise[0, :250, 0] = 1.0          # second 0: lead 0 at 0 dB
    noise[0, 250:500, :2] = 1.0      # second 1: leads 0 and 1 at 0 dB
    noise[0, 500:750, :] = 0.1       # second 2: all leads at 20 dB
    clean = A.clean_seconds(sig, tf.constant(noise), live).numpy()[0]
    # one bad lead of three is still readable (CLEAN_MIN_LEADS = 2); two are not
    assert clean[:4].tolist() == [1.0, 0.0, 1.0, 1.0]
    one = tf.constant([[[True, False, False]]])
    assert A.clean_seconds(sig, tf.constant(noise), one).numpy()[0][:3].tolist() == \
        [0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# model, loss, metrics
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module', params=['rhythm_30k', 'rhythm_unet_30k', 'rhythm_unet500_30k',
                                        'rhythm_unet1250_30k', 'rhythm_unet1250b_30k'])
def small_model(request):
    return rmodel.build(request.param)


def test_unet_backbone_is_a_drop_in_for_the_resu_one():
    from ecgr.models import sub_model
    m = rmodel.build('rhythm_unet_30k')
    assert m.name == 'unetmamba_rhythm_30k'
    bb = sub_model(m, 'backbone')
    fused, enc1, stem = bb.output
    assert fused.shape[1:] == (rc.BACKBONE_STEPS, rmodel.SIZES['rhythm_unet_30k']['width'])
    assert enc1.shape[1] == 500 and stem.shape[1] == rc.SEGMENT_SAMPLES   # the head's skips
    assert rmodel.is_per_sample(m) and not rmodel.is_per_sample(rmodel.build('rhythm_30k'))
    names = {layer.name for layer in bb.layers}
    assert 'bottom_a_conv' in names and 'ssm0_ssm' in names         # U-Net || Mamba
    assert not any(n.startswith('resu') for n in names)


@pytest.mark.parametrize('name', rmodel.list_models())
def test_parameter_budget(name):
    total = rmodel.build(name).count_params()
    assert 0.7 * rmodel.BUDGETS[name] < total < rmodel.BUDGETS[name]


def _steps(model):
    return rmodel.rhythm_steps(model)


def _quality(model):
    return 'noise' if 'noise' in rmodel.output_names(model) else 'lead'


def test_model_contract(small_model):
    assert small_model.input_shape[1:] == (rc.SEGMENT_SAMPLES, rc.IN_CHANNELS)
    shapes = {k: tuple(v.shape[1:]) for k, v in small_model.output.items()}
    expected = {'resumamba_rhythm_30k': {'rhythm': (rc.OUTPUT_SECONDS, rc.NUM_CLASSES),
                                         'lead': (rc.NUM_LEAD_CLASSES,)},
                'unetmamba_rhythm_30k': {'rhythm': (rc.SEGMENT_SAMPLES, rc.NUM_CLASSES),
                                         'lead': (rc.NUM_LEAD_CLASSES,)},
                'unetmamba500_rhythm_30k': {'rhythm': (500, rc.NUM_CLASSES),
                                            'noise': (rc.NOISE_SEGMENTS, 2)},
                'unetmamba1250_rhythm_30k': {'rhythm': (1250, rc.NUM_CLASSES),
                                             'noise': (rc.NOISE_SEGMENTS, 2)},
                'unetmamba1250b_rhythm_30k': {'rhythm': (1250, rc.NUM_CLASSES),
                                              'noise': (rc.NOISE_SEGMENTS, 2),
                                              'beat': (rc.BEAT_STEPS, len(rc.BEAT_CLASSES))}}
    assert shapes == expected[small_model.name]
    y = small_model(np.random.randn(2, rc.SEGMENT_SAMPLES, 3).astype('float32'))
    for k in y:
        np.testing.assert_allclose(y[k].numpy().sum(-1), 1.0, atol=1e-5)


def test_lead_head_is_permutation_equivariant(small_model):
    if _quality(small_model) != 'lead':
        pytest.skip('no lead output')
    x = np.random.randn(2, rc.SEGMENT_SAMPLES, 3).astype('float32')
    perm = [2, 0, 1]
    a = small_model(x, training=False)['lead'].numpy()
    b = small_model(x[..., perm], training=False)['lead'].numpy()
    # CH1..CH3 come from one shared scorer, so among themselves they move EXACTLY with the
    # leads. (NOISE also reads the backbone, which mixes leads: invariant only as trained.)
    ch = lambda p: p[:, 1:] / p[:, 1:].sum(-1, keepdims=True)      # noqa: E731
    np.testing.assert_allclose(ch(b), ch(a)[:, perm], atol=1e-5)


def test_loss_ignores_unlabelled_seconds_and_metrics_run(small_model, batch):
    x, y = batch
    per = _steps(small_model) // rc.OUTPUT_SECONDS                 # 1, 50 or 250 a second
    y = tf.repeat(y, per, axis=1)
    xa, t = A.augment(x, y, tf.constant([1, 1], tf.int64))
    p = small_model(xa)
    loss = rhythm_loss([1.0] * rc.NUM_CLASSES)(t['rhythm'], p['rhythm']).numpy()
    assert loss.shape == (64, 10 * per) and np.all(np.isfinite(loss))
    assert np.all(loss[:, 3 * per:4 * per] == 0)                    # IGNORE second
    q = _quality(small_model)
    if 'beat' in rmodel.output_names(small_model):
        bt = np.zeros((64, rc.SEGMENT_SAMPLES), np.uint8)
        bt[:, 200] = 1; bt[:, 900] = 3; bt[:, 1600] = 2; bt[:5] = rc.IGNORE
        xa, t = A.augment(x, y, tf.constant([1, 1], tf.int64), beats=tf.constant(bt))
        p = small_model(xa)
        bl = beat_loss(list(rc.BEAT_CLASS_WEIGHTS))(t['beat'], p['beat']).numpy()
        assert bl.shape == (64, rc.BEAT_STEPS) and np.all(np.isfinite(bl))
        assert np.all(bl[:5] == 0)                                   # no beat annotation
        bf = BeatF1(); bf.update_state(t['beat'], p['beat'])
        assert 0.0 <= float(bf.result()) <= 1.0 and bf.matrix().sum() > 0
    if q == 'noise':
        assert np.isfinite(float(noise_loss()(t['noise'], p['noise'])))
        f1, acc, nf1 = RhythmF1(), NoiseF1(mode='accuracy'), NoiseF1()
    else:
        assert np.isfinite(float(lead_loss()(t['lead'], p['lead'])))
        f1, acc, nf1 = RhythmF1(), LeadConfusion(mode='accuracy'), LeadConfusion()
    f1.update_state(t['rhythm'], p['rhythm'])
    acc.update_state(t[q], p[q])
    nf1.update_state(t[q], p[q])
    for m in (f1, acc, nf1):
        assert 0.0 <= float(m.result()) <= 1.0
    assert acc.matrix().sum() == 64 * (rc.NOISE_SEGMENTS if q == 'noise' else 1)
    r = t['rhythm'].numpy()
    # IGNORE and noisy seconds never reach the rhythm confusion
    assert f1.matrix().sum() == int((r[..., -1] > 0.5).sum())


def test_save_load_roundtrip(small_model, tmp_path):
    path = os.path.join(tmp_path, 'm.keras')
    small_model.save(path)
    again = tf.keras.models.load_model(path, compile=False)
    x = np.random.randn(1, rc.SEGMENT_SAMPLES, 3).astype('float32')
    for k in rmodel.output_names(small_model):
        np.testing.assert_allclose(again(x)[k].numpy(), small_model(x)[k].numpy(), atol=1e-5)


def test_predict_signal_covers_every_second(small_model, synthetic_record):
    from ecgr.rhythm.predict import predict_signal, probs_step_hz, window_starts
    signal, _ = synthetic_record                                    # 30 s
    rhythm, p_noise, windows = predict_signal(small_model, signal[:7300])   # 29.2 s
    hz = probs_step_hz(small_model)
    assert hz == (1 if _steps(small_model) == rc.OUTPUT_SECONDS else rc.SAMPLE_PROBS_HZ)
    assert rhythm.shape == (29 * hz, rc.NUM_CLASSES) and p_noise.shape == (29 * hz,)
    np.testing.assert_allclose(rhythm.sum(-1), 1.0, atol=1e-3)     # overlap averaged, not summed
    assert window_starts(7300, hop=2500) == [0, 2500, 4750]          # back-to-back
    hop = int(round(rc.PREDICT_HOP_SECONDS)) * rc.SECOND_SAMPLES
    assert window_starts(7300) == window_starts(7300, hop=hop)
    assert [w['start'] for w in windows] == [s // rc.SECOND_SAMPLES for s in window_starts(7300)]
    if _quality(small_model) == 'lead':
        assert all(w['lead'] in rc.LEAD_CLASSES for w in windows)
    else:
        assert all(len(w['noise']) == rc.NOISE_SEGMENTS for w in windows)
        assert np.all((p_noise >= 0) & (p_noise <= 1))


def test_header_span_reads_the_caliper_and_refuses_minus_one(tmp_path):
    from ecgr.rhythm.build import _apply_header_span, header_span
    base = str(tmp_path / 'rec')
    with open(base + '.hea', 'w') as f:
        f.write("rec 3 250 15000\n# eventType: VT\n#- channel: 1 \n#- startSample: 7500\n"
                "#- stopSample: 10000\n#- eventStartSample: 8804\n#- eventStopSample: 9169\n")
    assert header_span(base) == (8804, 9169)
    ev = dict(types=['VE_RUN'], spans=[], strip=(7500, 10000), needs_runs=True,
              header_span=True, xlsx_span=[False])
    from collections import Counter
    out = _apply_header_span(ev, base, Counter())
    assert out['spans'] == [(VT, 8804, 9169)] and not out['needs_runs']
    with open(base + '.hea', 'w') as f:
        f.write("rec 3 250 15000\n#- eventStartSample: -1\n#- eventStopSample: -1\n")
    assert header_span(base) is None
    assert _apply_header_span(ev, base, Counter()) is ev


def test_per_strip_view_uses_raw_calls_not_record_post_processing():
    """A 2 s AFIB call inside one 10 s window counts for the strip: the 7 s AFIB minimum and
    the rest of DECODE_* belong to whole-record decoding, not to an isolated window."""
    from ecgr.rhythm.evaluate import summarize
    K, S = rc.NUM_CLASSES, rc.OUTPUT_SECONDS
    y_rhythm = np.zeros((2, S, K + 1), np.float32)
    y_rhythm[..., rc.SINUS] = 1.0
    y_rhythm[..., K] = 1.0                        # every second labelled and clean
    y_rhythm[0, 3:5, rc.SINUS], y_rhythm[0, 3:5, L.class_index('AFIB')] = 0.0, 1.0
    p_rhythm = y_rhythm[..., :K].copy()           # perfect per-second calls
    y_lead = np.zeros((2, rc.NUM_LEAD_CLASSES), np.float32)
    y_lead[:, 1] = 1.0
    p_lead = y_lead.copy()
    p_lead[1] = [1.0, 0.0, 0.0, 0.0]              # second window unreadable -> nothing called

    s = summarize({'rhythm': y_rhythm, 'lead': y_lead}, {'rhythm': p_rhythm, 'lead': p_lead})
    assert s['strip']['AFIB'] == dict(tp=1, fp=0, fn=0, se=1.0, ppv=1.0)
    assert all(v['tp'] == v['fp'] == v['fn'] == 0 for n, v in s['strip'].items() if n != 'AFIB')


def test_sample_labels_follow_the_spans_to_the_sample():
    AFv, VTv = L.class_index('AFIB'), L.class_index('VT')
    known = [(1000, 3500)]
    spans = [(AFv, 1500, 2100), (VTv, 2000, 2200)]                 # VT inside the AF tail
    lab = L.sample_labels(1000, spans, known)
    assert lab.shape == (rc.SEGMENT_SAMPLES,) and lab.dtype == np.uint8
    assert np.all(lab[:500] == rc.SINUS)
    assert np.all(lab[500:1000] == AFv)
    assert np.all(lab[1000:1200] == VTv)                           # priority: VT over AFIB
    assert np.all(lab[1200:] == rc.SINUS)
    unknown = L.sample_labels(0, [], [(0, 1000)])
    assert np.all(unknown[:1000] == rc.SINUS) and np.all(unknown[1000:] == rc.IGNORE)
    # a 0.2 s run keeps its 50 samples; the per-second grid can only give it one whole second
    short = L.sample_labels(0, [(VTv, 1010, 1060)], [(0, 2500)])
    assert int((short == VTv).sum()) == 50


def test_decode_on_a_finer_grid_keeps_durations_in_seconds():
    hz = rc.SAMPLE_PROBS_HZ
    track = np.array([0] * (10 * hz) + [1] * (8 * hz) + [0] * int(2.4 * hz) + [1] * (4 * hz)
                     + [0] * (10 * hz))
    eps = L.decode_episodes(_probs(track), smooth_seconds=1, merge_gap={'AFIB': 3},
                            min_seconds={'AFIB': 7}, step_hz=hz)
    af = [e for e in eps if e['rhythm'] == 'AFIB']
    assert [(e['start'], e['stop']) for e in af] == [(10.0, 24.4)]      # 2.4 s gap bridged
    eps = L.decode_episodes(_probs(track), smooth_seconds=1, merge_gap={'AFIB': 2},
                            min_seconds={'AFIB': 7}, step_hz=hz)
    assert [(e['start'], e['stop']) for e in eps if e['rhythm'] == 'AFIB'] == [(10.0, 18.0)]


def test_targets_repeat_the_clean_seconds_over_their_samples():
    labels = np.zeros((2, rc.SEGMENT_SAMPLES), np.uint8)
    labels[:, :250] = rc.IGNORE
    clean = np.ones((2, rc.OUTPUT_SECONDS), np.float32)
    clean[:, 4] = 0.0
    t = A.targets(tf.constant(labels), tf.constant(clean), tf.constant([1, 2]))['rhythm']
    w = t.numpy()[..., -1]
    assert t.shape == (2, rc.SEGMENT_SAMPLES, rc.NUM_CLASSES + 1)
    assert np.all(w[:, :250] == 0) and np.all(w[:, 1000:1250] == rc.NOISY_SECOND_WEIGHT)
    assert np.all(w[:, 250:1000] == 1) and np.all(w[:, 1250:] == 1)


def test_pipeline_serves_sample_labels(tmp_path):
    from ecgr.rhythm import pipeline
    split = tmp_path / 'train'
    split.mkdir()
    n = 4
    np.save(split / 'segments_0.npy', np.zeros((n, rc.SEGMENT_SAMPLES, 3), np.float16))
    np.save(split / 'labels_0.npy', np.zeros((n, rc.OUTPUT_SECONDS), np.uint8))
    np.save(split / 'studyids_0.npy', np.arange(n))
    with pytest.raises(FileNotFoundError, match='per-sample labels'):
        pipeline.load_arrays('train', str(tmp_path), label_steps=rc.SEGMENT_SAMPLES)
    per_sample = np.zeros((n, rc.SEGMENT_SAMPLES), np.uint8)
    per_sample[:, 7] = 3                                   # centre of the 2nd 20 ms block
    per_sample[:, 5] = 4                                   # not a centre: must not survive
    np.save(split / 'labels_samples_0.npy', per_sample)
    segs, labs, _ = pipeline.load_arrays('train', str(tmp_path), label_steps=rc.SEGMENT_SAMPLES)
    assert labs.shape == (n, rc.SEGMENT_SAMPLES)
    x, y = next(iter(pipeline.make_dataset(segs, labs, 2, mode='clean')))
    assert y['rhythm'].shape == (2, rc.SEGMENT_SAMPLES, rc.NUM_CLASSES + 1)
    assert set(y) == {'rhythm', 'lead'}
    _, l20, _ = pipeline.load_arrays('train', str(tmp_path), label_steps=500)
    assert l20.shape == (n, 500) and l20[0, 1] == 3 and l20[0, 0] == 0 and 4 not in l20
    x, y = next(iter(pipeline.make_dataset(segs, l20, 2, mode='noisy',
                                            outputs=('rhythm', 'noise'))))
    assert set(y) == {'rhythm', 'noise'}
    assert y['rhythm'].shape == (2, 500, rc.NUM_CLASSES + 1)
    assert y['noise'].shape == (2, rc.NOISE_SEGMENTS, 2)


def test_noise_target_marks_a_segment_noisy_when_either_second_is():
    labels = np.zeros((1, 500), np.uint8)
    clean = np.ones((1, rc.OUTPUT_SECONDS), np.float32)
    clean[0, 3] = 0.0                                  # segment 1 (seconds 2-3)
    clean[0, 8:] = 0.0                                 # segment 4 (seconds 8-9)
    t = A.targets(tf.constant(labels), tf.constant(clean), tf.constant([1]))
    assert np.argmax(t['noise'].numpy()[0], -1).tolist() == [0, 1, 0, 0, 1]
    w = t['rhythm'].numpy()[0, :, -1]                  # rhythm weight per 20 ms, 50 a second
    assert np.all(w[150:200] == rc.NOISY_SECOND_WEIGHT) and np.all(w[:150] == 1)


def test_noise_metric_counts_segments():
    t = tf.one_hot([[0, 1, 1, 0, 0]], 2)
    p = tf.one_hot([[0, 1, 0, 0, 1]], 2)
    f1, acc = NoiseF1(), NoiseF1(mode='accuracy')
    f1.update_state(t, p)
    acc.update_state(t, p)
    assert f1.matrix().tolist() == [[2, 1], [1, 1]]
    assert abs(float(f1.result()) - 0.5) < 1e-6 and abs(float(acc.result()) - 0.6) < 1e-6


def test_per_class_smoothing_keeps_short_runs_sharp():
    VTv = L.class_index('VT')
    track = [0] * 10 + [VTv] * 2 + [0] * 10                      # a 2 s VT run in sinus
    r = _probs(track)
    same = L.smooth_probs(r, 5)
    assert same[10:12, VTv].max() < 0.5 < same[10, rc.SINUS]    # smeared: sinus wins
    split = L.smooth_probs(r, {'AFIB': 5, 'VT': 1})
    assert np.all(split[10:12, VTv] == 1.0) and np.all(split[:, rc.SINUS] == r[:, rc.SINUS])
    eps = L.decode_episodes(r, smooth_seconds={'AFIB': 5, 'VT': 1}, merge_gap={},
                            min_seconds={'VT': 1})
    assert _spans(eps) == [('SINUS', 0, 10), ('VT', 10, 12), ('SINUS', 12, 22)]
    lost = L.decode_episodes(r, smooth_seconds=5, merge_gap={}, min_seconds={'VT': 1})
    assert _spans(lost) == [('SINUS', 0, 22)]
    # on the 25 Hz grid the dict is converted per class like the scalar
    hz = rc.SAMPLE_PROBS_HZ
    fine = L.decode_episodes(_probs(np.repeat(track, hz)), step_hz=hz,
                             smooth_seconds={'AFIB': 5, 'VT': 1}, merge_gap={},
                             min_seconds={'VT': 1})
    assert [(e['rhythm'], e['start'], e['stop']) for e in fine if e['rhythm'] == 'VT'] == \
        [('VT', 10.0, 12.0)]


def test_default_decode_keeps_a_three_beat_vt_run():
    """mitdb's reference VT episodes are 1.8 s long at the median: the defaults must keep a 2 s
    run inside sinus (a 3 s minimum with 5 s smoothing erased 51 of its 60 episodes), while an
    AFIB blip under the AFIB minimum still folds into the sinus around it."""
    VTv, AF = L.class_index('VT'), L.class_index('AFIB')
    track = [0] * 20 + [VTv] * 2 + [0] * 20 + [AF] * 2 + [0] * 20 + [AF] * 5 + [0] * 10
    spans = _spans(L.decode_episodes(_probs(track), class_scale={}))
    assert ('VT', 20, 22) in spans
    assert [s for s in spans if s[0] == 'AFIB'] == [('AFIB', 64, 69)]


def _long_record(tmp_path, name, fs, seconds, marks, beats):
    import wfdb
    n = fs * seconds
    sig = (0.3 * np.sin(2 * np.pi * np.arange(n) / fs)).reshape(-1, 1).repeat(2, 1)
    wfdb.wrsamp(name, fs=fs, units=['mV', 'mV'], sig_name=['ECG1', 'ECG2'], p_signal=sig,
                write_dir=str(tmp_path))
    samples = [s for s, _ in beats] + [s for s, _ in marks]
    symbols = [b for _, b in beats] + ['+'] * len(marks)
    aux = [''] * len(beats) + [c for _, c in marks]
    order = np.argsort(samples, kind='stable')
    wfdb.Annotation(record_name=name, extension='atr', sample=np.array(samples)[order],
                    symbol=[symbols[i] for i in order], aux_note=[aux[i] for i in order],
                    fs=fs).wrann(write_fs=True, write_dir=str(tmp_path))
    return str(tmp_path / name)


def test_physionet_train_labels_and_refuses_ec57(tmp_path):
    from ecgr.rhythm import physionet_train as P
    fs = 128
    beats = [(int(t * fs), 'N') for t in np.arange(0.5, 120, 0.8)]
    marks = [(0, '(N'), (40 * fs, '(AFIB'), (70 * fs, '(IVR'), (80 * fs, '(SBR')]
    _long_record(tmp_path, 'r1', fs, 120, marks, beats)
    cfg = dict(dir=str(tmp_path), labels='rhythm')
    events, counts, _ = P.record_events('ltafdb', 'r1', cfg)
    assert events and all(ev['chunk'][1] - ev['chunk'][0] == 20 * fs for ev in events)
    cats = {ev['strip'][0] + ev['chunk'][0]: ev['category'] for ev in events}
    assert cats[35 * fs] == 'AF_edge'                    # 35-45 s straddles the AF onset
    assert cats[45 * fs] == 'AF' and cats[5 * fs] == 'sinus'
    assert 65 * fs not in cats and 75 * fs not in cats   # IVR (not a class) never becomes a window
    ev = next(e for e in events if e['category'] == 'sinus')
    assert ev['spans'] == [] and ev['known'] == [(0, 20 * fs)]
    # sinus DB: V runs become VT; runs-only DB: nothing but the run is labelled
    beats_v = [b for b in beats if not 30 * fs < b[0] < 33 * fs] + \
        [(int(t * fs), 'V') for t in (30.5, 31.0, 31.5, 32.0)]
    _long_record(tmp_path, 'r2', fs, 120, [], beats_v)
    ev, _, _ = P.record_events('svdb', 'r2', dict(dir=str(tmp_path), labels='runs'))
    # the grid window holding the run plus the one centred on it
    assert {e['category'] for e in ev} == {'VT'} and all(e['known'] == [] for e in ev)
    assert any(e['strip'][0] + e['chunk'][0] == 26 * fs for e in ev)
    with pytest.raises(ValueError, match='EC57'):
        P.collect({'mitdb': dict(dir=str(tmp_path), labels='rhythm')})
    # the events go through build.process_event like any portal strip
    from ecgr.rhythm.build import process_event
    for e in events + ev:
        assert all(t in rc.EVENT_TYPE_TO_CLASS for t in e['types'])
        seg, lab, per_sample, bts, stats = process_event(e)
        assert not stats.get('errors'), stats
        assert len(seg) == 1 and len(lab) == 1 and per_sample[0].shape == (rc.SEGMENT_SAMPLES,)
        assert bts[0].shape == (rc.SEGMENT_SAMPLES,) and set(np.unique(bts[0])) <= {0, 1, 2, 3}
    vt = process_event(ev[0])[1][0]
    assert L.class_index('VT') in vt and rc.SINUS not in vt       # svdb: nothing else known


def test_read_leads_reads_only_the_chunk(tmp_path):
    from ecgr.rhythm.build import read_leads
    path = _long_record(tmp_path, 'r3', 128, 60, [], [(64, 'N')])
    full, _ = read_leads(path)
    part, ratio = read_leads(path, chunk=(128 * 10, 128 * 30))
    assert ratio == rc.SAMPLING_RATE / 128
    assert abs(len(part) - 20 * rc.SAMPLING_RATE) <= 2 and len(full) == 3 * len(part)


def test_class_scale_moves_the_decision_not_the_reported_prob():
    VTv = L.class_index('VT')
    r = np.zeros((20, rc.NUM_CLASSES), np.float32)
    r[:, rc.SINUS] = 0.45
    r[:, VTv] = 0.55                                     # VT barely wins everywhere
    kw = dict(smooth_seconds=1, merge_gap={}, min_seconds={})
    assert _spans(L.decode_episodes(r, class_scale={}, **kw)) == [('VT', 0, 20)]
    eps = L.decode_episodes(r, class_scale={'VT': 0.5}, **kw)    # odds x0.5 -> sinus
    assert _spans(eps) == [('SINUS', 0, 20)] and abs(eps[0]['prob'] - 0.45) < 1e-6


def test_low_confidence_episodes_are_folded():
    AF = L.class_index('AFIB')
    r = np.zeros((40, rc.NUM_CLASSES), np.float32)
    r[:, rc.SINUS] = 1.0
    r[5:15, rc.SINUS], r[5:15, AF] = 0.4, 0.6            # weak AF call
    r[20:30, rc.SINUS], r[20:30, AF] = 0.05, 0.95        # confident AF call
    kw = dict(smooth_seconds=1, merge_gap={}, min_seconds={}, class_scale={})
    assert [s for s in _spans(L.decode_episodes(r, min_prob={}, **kw)) if s[0] == 'AFIB'] == \
        [('AFIB', 5, 15), ('AFIB', 20, 30)]
    kept = L.decode_episodes(r, min_prob={'AFIB': 0.8}, **kw)
    assert [s for s in _spans(kept) if s[0] == 'AFIB'] == [('AFIB', 20, 30)]


def test_overlapping_windows_and_tta_keep_the_contract(small_model, synthetic_record):
    from ecgr.rhythm import predict as PR
    signal, _ = synthetic_record
    assert PR.window_starts(7300, hop=1250) == [0, 1250, 2500, 3750, 4750]
    base, _, _ = PR.predict_signal(small_model, signal[:7300])
    x = np.random.randn(2, rc.SEGMENT_SAMPLES, 3).astype('float32')
    y = PR.predict_tta(small_model, x, 2, variants=('id', 'flip', 'swap'))
    assert set(y) == set(rmodel.output_names(small_model))
    np.testing.assert_allclose(y['rhythm'].sum(-1), 1.0, atol=1e-4)
    one = PR.predict_tta(small_model, x, 2, variants=('id',))
    np.testing.assert_allclose(one['rhythm'], small_model.predict(x, verbose=0)['rhythm'], atol=1e-5)
    assert base.shape[0] == 29 * PR.probs_step_hz(small_model)


def test_challenge2020_label_rule():
    from ecgr.rhythm import challenge2020 as C
    assert C.record_label({'164889003'}) == ('AFIB', None)                 # AF
    assert C.record_label({'164890007', '59118001'}) == ('AFIB', None)     # AFL (+RBBB) = AF
    assert C.record_label({'195042002'}) == ('AVB', None)
    assert C.record_label({'27885002'}) == ('AVB', None)                   # complete block
    assert C.record_label({'426783006', '284470004'}) == ('SINUS', None)   # sinus + PACs
    assert C.record_label({'426783006'}) == (None, 'plain')
    assert C.record_label({'426761007'})[1] == 'paroxysmal_or_ambiguous'   # SVT: no location
    assert C.record_label({'164889003', '27885002'}) == (None, 'several_classes')
    assert 'ptb-xl' not in rc.CHALLENGE2020_SUBSETS and 'st_petersburg_incart' not in \
        rc.CHALLENGE2020_SUBSETS                                           # copies of sources


def test_pipeline_serves_8ms_labels(tmp_path):
    from ecgr.rhythm import pipeline
    split = tmp_path / 'train'
    split.mkdir()
    np.save(split / 'segments_0.npy', np.zeros((2, rc.SEGMENT_SAMPLES, 3), np.float16))
    np.save(split / 'studyids_0.npy', np.arange(2))
    lab = np.zeros((2, rc.SEGMENT_SAMPLES), np.uint8)
    lab[:, 3] = 3                                         # centre of the 2nd 8 ms block
    lab[:, 2] = 4                                         # not a centre
    np.save(split / 'labels_samples_0.npy', lab)
    _, l8, _ = pipeline.load_arrays('train', str(tmp_path), label_steps=1250)
    assert l8.shape == (2, 1250) and l8[0, 1] == 3 and 4 not in l8


def test_beat_labels_and_targets():
    samples = np.array([100, 600, 1100, 1700, 2300, 2600])      # last one past the window
    symbols = np.array(['N', 'A', 'V', 'F', 'L', 'N'])
    lab = L.beat_labels(0, samples, symbols)
    assert lab.shape == (rc.SEGMENT_SAMPLES,) and lab.dtype == np.uint8
    assert lab[100] == 1 and lab[600] == 2 and lab[1100] == 3 and lab[2300] == 1
    r = int(round(rc.BEAT_IGNORE_RADIUS_SECONDS * rc.SAMPLING_RATE))
    assert np.all(lab[1700 - r:1700 + r + 1] == rc.IGNORE) and lab[1700 - r - 1] == 0
    assert np.all(L.beat_labels(0, None, None) == rc.IGNORE)
    shifted = L.beat_labels(500, samples, symbols)                # window starts at sample 500
    assert shifted[100] == 2 and shifted[600] == 3 and shifted[0] == 0

    clean = np.ones((1, rc.OUTPUT_SECONDS), np.float32)
    clean[0, 2] = 0.0                                             # second 2 is noisy
    t = A.beat_targets(tf.constant(lab[None]), tf.constant(clean)).numpy()[0]
    nb = len(rc.BEAT_CLASSES)
    assert t.shape == (rc.BEAT_STEPS, nb + 2)          # heat | N S V | w_heat | w_type
    heat, typ, w_heat, w_type = t[:, 0], t[:, 1:nb], t[:, nb], t[:, nb + 1]
    # heatmap: 1 at the R step, Gaussian flanks, ~0 beyond 3 sigma
    assert heat[50] == pytest.approx(1.0, abs=1e-5) and 0.3 < heat[52] < 0.9
    assert heat[50 + int(np.ceil(3 * rc.BEAT_HEAT_SIGMA_STEPS)) + 1] == 0
    # type one-hot on +-BEAT_TYPE_RADIUS_STEPS around each beat, nothing elsewhere
    r = rc.BEAT_TYPE_RADIUS_STEPS
    assert np.all(np.argmax(typ[50 - r:50 + r + 1], -1) == 0) and np.all(w_type[50 - r:50 + r + 1] > 0)
    assert typ[50 - r - 1].sum() == 0 and w_type[50 - r - 1] == 0
    assert np.argmax(typ[300]) == 1 and np.argmax(typ[550]) == 2       # S at 300, V at 550
    assert w_heat[850] == 0 and w_heat[845] == 0 and w_heat[830] == 1.0   # IGNORE zone (F beat)
    assert w_heat[300] == rc.NOISY_SECOND_WEIGHT and w_type[300] == rc.NOISY_SECOND_WEIGHT
    assert w_heat[560] == 1.0 and w_type[700] == 0
    assert (w_type > 0).mean() < 0.1 and (heat > 0.5).mean() < 0.05


def test_beat_decoder_conditions_the_rhythm_decoder():
    from ecgr.models import sub_model
    m = rmodel.build('rhythm_unet1250b_30k')
    names = {l.name for l in m.layers}
    assert {'beat', 'beat_sg', 'beat_to_grid', 'beat_ctx_ssm', 'beat_ac', 'beat_tokens'} <= names
    x = np.random.randn(2, rc.SEGMENT_SAMPLES, 3).astype('float32')
    y = m(x, training=False)
    np.testing.assert_allclose(y['beat'].numpy().sum(-1), 1.0, atol=1e-5)
    # the gradient of the rhythm loss does not reach the beat head (stop-gradient)
    with tf.GradientTape() as tape:
        out = m(x, training=True)
        loss = tf.reduce_mean(out['rhythm'][..., 1])
    beat_vars = [v for v in m.trainable_variables if v.name.startswith('beat_dec') or
                 v.path.startswith('beat_dec') if hasattr(v, 'path')] or \
        [v for v in m.trainable_variables if 'beat_dec' in v.path]
    grads = tape.gradient(loss, beat_vars)
    assert beat_vars and all(g is None for g in grads)


def test_pipeline_serves_beat_labels(tmp_path):
    from ecgr.rhythm import pipeline
    split = tmp_path / 'train'
    split.mkdir()
    n = 3
    np.save(split / 'segments_0.npy', np.zeros((n, rc.SEGMENT_SAMPLES, 3), np.float16))
    np.save(split / 'labels_0.npy', np.zeros((n, rc.OUTPUT_SECONDS), np.uint8))
    np.save(split / 'labels_samples_0.npy', np.zeros((n, rc.SEGMENT_SAMPLES), np.uint8))
    np.save(split / 'studyids_0.npy', np.arange(n))
    with pytest.raises(FileNotFoundError, match='beat labels'):
        pipeline.load_arrays('train', str(tmp_path), label_steps=1250, beats=True)
    bt = np.zeros((n, rc.SEGMENT_SAMPLES), np.uint8); bt[:, 1000] = 3
    np.save(split / 'beats_0.npy', bt)
    segs, labs, _, bts = pipeline.load_arrays('train', str(tmp_path), label_steps=1250, beats=True)
    assert bts.shape == (n, rc.SEGMENT_SAMPLES)
    with pytest.raises(ValueError, match="'beat' output"):
        pipeline.make_dataset(segs, labs, 2, mode='clean', outputs=('rhythm', 'noise', 'beat'))
    x, y = next(iter(pipeline.make_dataset(segs, labs, 2, mode='noisy',
                                            outputs=('rhythm', 'noise', 'beat'), beats=bts)))
    assert set(y) == {'rhythm', 'noise', 'beat'}
    assert y['beat'].shape == (2, rc.BEAT_STEPS, len(rc.BEAT_CLASSES) + 2)
    assert float(y['beat'][0, 500, 0]) == pytest.approx(1.0, abs=1e-5)   # sample 1000 -> step 500
    assert int(np.argmax(y['beat'][0, 500, 1:4])) == 2                 # V


def test_window_categories_and_stratified_plan():
    from ecgr.rhythm import pipeline
    steps = 10
    lab = np.zeros((40, steps), np.uint8)
    lab[0:4, :5] = rc.CLASS_NAMES.index('AFIB')
    lab[4:6, 2:5] = rc.CLASS_NAMES.index('VT')
    lab[6, 0] = rc.CLASS_NAMES.index('VT')                   # 1 s only: not VT
    lab[6, 2:8] = rc.CLASS_NAMES.index('AFIB')
    lab[7, :5] = rc.CLASS_NAMES.index('AFIB')
    lab[7, 5:8] = rc.CLASS_NAMES.index('SVT')                # AFIB + SVT -> SVT (rarer)
    cats = pipeline.window_categories(lab)
    names = [rc.CLASS_NAMES[c] for c in cats]
    assert names[:4] == ['AFIB'] * 4 and names[4:6] == ['VT'] * 2
    assert names[6] == 'AFIB' and names[7] == 'SVT' and set(names[8:]) == {'SINUS'}
    plan, repeat, n_steps = pipeline.stratified_plan(cats, 8, steps_per_epoch=5)
    assert sum(plan.values()) == 8 and 'AVB' not in plan
    assert plan['VT'] <= int(np.ceil(2 * rc.STRAT_MAX_REPEAT / 5))     # repeat cap
    counts = {rc.CLASS_NAMES[c]: int(n) for c, n in zip(*np.unique(cats, return_counts=True))}
    for k, v in plan.items():                                           # repeat cap, per class
        if k != 'SINUS':
            assert v <= int(np.ceil(counts[k] * rc.STRAT_MAX_REPEAT / 5))
    batches = list(pipeline.stratified_batches(cats, plan, n_steps, seed=1))
    assert len(batches) == 5 and all(len(b) == 8 for b in batches)
    for b in batches:
        got = {rc.CLASS_NAMES[c]: int(n) for c, n in zip(*np.unique(cats[b], return_counts=True))}
        assert got == plan


def test_stratified_dataset_batches(tmp_path):
    from ecgr.rhythm import pipeline
    n = 30
    segs = np.zeros((n, rc.SEGMENT_SAMPLES, 3), np.float16)
    lab = np.zeros((n, rc.OUTPUT_SECONDS), np.uint8)
    lab[:3, :] = rc.CLASS_NAMES.index('VT')
    ds = pipeline.make_dataset(segs, lab, 6, mode='train', outputs=('rhythm', 'noise'),
                               sampler='stratified', steps_per_epoch=4)
    xs = list(ds)
    assert len(xs) == 4 and xs[0][0].shape == (6, rc.SEGMENT_SAMPLES, 3)


def test_class_weights_divide_by_sqrt_repeat(monkeypatch):
    from ecgr.rhythm import train as T
    monkeypatch.setattr(rc, 'CLASS_WEIGHTS', [1.0] * rc.NUM_CLASSES)
    w = T.class_weights({'SINUS': 0.5, 'VT': 2.0, 'SVT': 0.25})
    assert w[rc.SINUS] == pytest.approx(1.0) and w[rc.CLASS_NAMES.index('VT')] == pytest.approx(0.5)
    assert w[rc.CLASS_NAMES.index('AFIB')] == pytest.approx(1.0)      # absent -> SINUS rate
    assert w[rc.CLASS_NAMES.index('SVT')] == pytest.approx(1.0)       # under-sampled: never raised


def test_five_classes_and_the_legacy_six():
    assert rc.CLASS_NAMES == ['SINUS', 'AFIB', 'SVT', 'VT', 'AVB']
    assert rc.EC57_CLASSES == ['AFIB', 'SVT', 'VT', 'AVB']
    assert rc.EVENT_TYPE_TO_CLASS['AVB2'] == rc.EVENT_TYPE_TO_CLASS['AVB3'] == 'AVB'
    assert rc.PHYSIONET_AUX_TO_CLASS['(BII'] == rc.PHYSIONET_AUX_TO_CLASS['(B3'] == 'AVB'
    # labels of a 6-class build: AVB2 (4) and AVB3 (5) -> AVB (4), IGNORE untouched
    old = np.array([[0, 1, 2, 3, 4, 5, rc.IGNORE]], np.uint8)
    assert L.to_current_labels(old).tolist() == [[0, 1, 2, 3, 4, 4, rc.IGNORE]]
    # probabilities of a 6-output checkpoint: the two block columns summed
    p = np.array([[0.5, 0.1, 0.1, 0.1, 0.12, 0.08]], np.float32)
    q = L.to_current_classes(p)
    np.testing.assert_allclose(q, [[0.5, 0.1, 0.1, 0.1, 0.2]], atol=1e-6)
    assert L.to_current_classes(q) is not None and L.to_current_classes(q).shape == (1, 5)
    assert L.to_current_weights([0.2, 0.4, 0.8, 1.6, 1.7, 2.4]) == [0.2, 0.4, 0.8, 1.6, 1.7]
    with pytest.raises(ValueError):
        L.to_current_classes(np.zeros((1, 4)))


def test_legacy_manifest_weights_pool_the_blocks():
    from ecgr.rhythm import pipeline
    m = {'class_names': rc.LEGACY_CLASS_NAMES,
         'splits': {'train': {'seconds': {'SINUS': 1000, 'AFIB': 400, 'SVT': 100, 'VT': 30,
                                          'AVB2': 20, 'AVB3': 10}}}}
    assert pipeline.is_legacy(m)
    w = pipeline.manifest_class_weights(m)
    assert len(w) == rc.NUM_CLASSES
    assert w[rc.CLASS_NAMES.index('AVB')] == pytest.approx(np.sqrt(100 / 30))   # median 100
