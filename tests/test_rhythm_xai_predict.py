"""Predict-vs-label tab of the XAI web service (ecgr.rhythm.xai_predict): scores and decode settings."""
import numpy as np
import pytest

from ecgr.rhythm import config as rc
from ecgr.rhythm import xai_predict as XP


def test_match_beats_one_to_one_within_tolerance():
    ref = [1.0, 2.0, 3.0, 4.0]
    pred = [1.05, 1.1, 2.3, 3.0, 5.0]
    match, hit = XP.match_beats(ref, pred)
    assert match.tolist() == [0, -1, -1, 2, -1]          # 1.1 loses to the nearer 1.05; 2.3 too far
    assert hit.tolist() == [True, False, True, False]
    m, h = XP.match_beats([], pred)
    assert (m == -1).all() and len(h) == 0


def test_second_and_beat_scores():
    af, sinus = rc.CLASS_NAMES.index('AFIB'), rc.SINUS
    ref = np.array([-1, sinus, sinus, af, af, af])
    pred = np.array([af, sinus, af, af, af, XP.NOISE])
    s = XP.second_scores(ref, pred)
    row = {r['cls']: r for r in s['rows']}
    assert s['known'] == 5 and s['agree'] == 60.0 and s['noise_s'] == 1
    assert row['AFIB']['se'] == pytest.approx(200 / 3, abs=0.01) and row['AFIB']['ppv'] == pytest.approx(200 / 3, abs=0.01)
    assert row['SINUS']['se'] == 50.0 and row['SVT']['se'] is None
    b = XP.beat_scores('NVN', 'NVVN', np.array([0, 1, -1, 2]))
    assert b['qrs_se'] == 100.0 and b['qrs_ppv'] == 75.0
    assert b['V'] == dict(ref=1, pred=2, tp=1, se=100.0, ppv=50.0)


def test_decode_dict_overrides():
    class E:                                              # PredictLab only needs these here
        physionet_dir, checkpoint = '/nonexistent', None
    lab = XP.PredictLab(E())
    assert lab.decode_dict({}) == {}
    d = lab.decode_dict(dict(beat_pp=False, afib_gate=False, noise_threshold='0.7',
                             min_prob={'AFIB': 0.8, 'SVT': ''}, min_seconds={'VT': 3}))
    assert d['beat_pp'] is False and d['afib_gate'] is False and d['noise_threshold'] == 0.7
    assert d['min_prob'] == {**rc.DECODE_MIN_EPISODE_PROB, 'AFIB': 0.8}
    assert d['min_seconds']['VT'] == 3 and d['min_seconds']['AFIB'] == rc.DECODE_MIN_EPISODE_SECONDS['AFIB']
    assert lab.decode_dict(dict(afib_gate=True))['afib_gate'] == rc.AFIB_GATE_DEFAULT
    with pytest.raises(PermissionError):
        lab.decode_dict(dict(preset='/etc/passwd'))
