"""Beat picking and the beat-level rhythm post-processing (ecgr.rhythm.beats)."""
import numpy as np
import pytest

from ecgr.rhythm import beats as B
from ecgr.rhythm import config as rc
from ecgr.rhythm.labels import NOISE, class_index

HZ = 25           # step grid of the rhythm track
AF, SVT, VT, AVB, SINUS = (class_index(n) for n in ('AFIB', 'SVT', 'VT', 'AVB', 'SINUS'))
N, S, V = B.N_BEAT, B.S_BEAT, B.V_BEAT


def make_beats(rr_list, cls_list, start=0.5):
    t = start + np.concatenate([[0.0], np.cumsum(rr_list[:-1])])
    return dict(t=t, cls=np.asarray(cls_list, int), conf=np.ones(len(t), np.float32),
                probs=np.zeros((len(t), 3), np.float32))


def track_from(segments, hz=HZ):
    """[(class, seconds)] -> step track."""
    return np.concatenate([np.full(int(round(sec * hz)), c, int) for c, sec in segments])


def classes_in(track, hz=HZ):
    out, cur = [], None
    for c in track:
        if c != cur:
            out.append(int(c))
            cur = c
    return out


# --- pick_beats ------------------------------------------------------------------------------

def test_pick_beats_local_maxima_and_refractory():
    hz = 125
    probs = np.zeros((hz * 10, 4), np.float32)
    probs[:, 0] = 1.0
    truth = [1.0, 1.8, 2.6, 3.4]
    for i, t in enumerate(truth):
        k = int(t * hz)
        for d, w in ((0, .95), (1, .7), (-1, .7)):
            probs[k + d, 0] = 1 - w
            probs[k + d, 1 + (2 if i == 2 else 0)] = w        # third beat is V
    # a twin peak 40 ms after the first beat must be swallowed by the refractory period
    probs[int(1.04 * hz), 0] = 0.3
    probs[int(1.04 * hz), 1] = 0.7
    got = B.pick_beats(probs, hz)
    assert np.allclose(got['t'], truth, atol=1 / hz)
    assert list(got['cls']) == [N, N, V, N]


def test_pick_beats_empty_and_short():
    assert len(B.pick_beats(np.zeros((2, 4)), 125)['t']) == 0
    probs = np.zeros((500, 4), np.float32)
    probs[:, 0] = 1.0
    assert len(B.pick_beats(probs, 125)['t']) == 0


# --- criteria --------------------------------------------------------------------------------

def test_duration_counts_rr_intervals_and_num_beat_is_enforced():
    # fix A: 2 V beats in 'VT' fail num_beat even when the track says VT
    t = np.array([0.0, 0.4, 0.8, 1.2])
    sym = np.array([V, V, N, N])
    assert not B.region_valid('VT', t, sym, 0, 2)
    assert B._duration(t, 0, 4) == pytest.approx(1.2)


def test_vt_needs_fast_rate():
    # fix J: 4 V beats at 50 bpm (idioventricular) are not VT
    t = np.arange(5) * 1.2
    sym = np.full(5, V)
    assert not B.region_valid('VT', t, sym, 0, 5)
    assert B.region_valid('VT', t / 3.0, sym, 0, 5)          # 150 bpm


def test_avb_rate_is_an_upper_bound():
    # fix F: 20 bpm complete block is still AVB
    t = np.arange(4) * 3.0
    assert B.region_valid('AVB', t, np.full(4, N), 0, 4)
    assert not B.region_valid('AVB', np.arange(6) * 0.8, np.full(6, N), 0, 6)   # 75 bpm


def test_svt_accepts_n_beats_on_rate_jump():
    t = np.arange(8) * 0.4                                     # 150 bpm, all N
    assert B.region_valid('SVT', t, np.full(8, N), 0, 8, prev_hr=75.0)
    assert not B.region_valid('SVT', t, np.full(8, N), 0, 8, prev_hr=140.0)


# --- the stage on synthetic sequences ------------------------------------------------------

def test_sinus_svt_afib_sequence_keeps_three_regions():
    # SINUS 10 s (75 bpm, N) | SVT 6 s (150 bpm, S run) | AFIB 12 s (irregular, N)
    rr = [0.8] * 12 + [0.4] * 15 + list(np.tile([0.5, 0.9, 0.7, 1.1, 0.6], 4))
    cls = [N] * 12 + [S] * 15 + [N] * 20
    beats = make_beats(rr, cls)
    t = beats['t']
    track = np.full(int(t[-1] * HZ) + 2 * HZ, SINUS)
    track[int(t[12] * HZ):int(t[27] * HZ)] = SVT
    track[int(t[27] * HZ):] = AF
    out, sym = B.postprocess(track, beats, HZ)
    assert classes_in(out) == [SINUS, SVT, AF]
    assert (sym[27:] == N).all()


def test_short_sinus_island_inside_afib_is_absorbed():
    # AFIB | SINUS 3 s (< BEAT_PP_CRITERIA['SINUS'].duration) | AFIB -> one AFIB region
    rr = list(np.tile([0.5, 0.9, 0.7, 1.1, 0.6], 4)) + [0.75] * 4 + list(np.tile([0.5, 0.9, 0.7, 1.1, 0.6], 4))
    beats = make_beats(rr, [N] * len(rr))
    t = beats['t']
    track = np.full(int(t[-1] * HZ) + HZ, AF)
    track[int(t[20] * HZ) - 5:int(t[24] * HZ) - 5] = SINUS
    out, _ = B.postprocess(track, beats, HZ)
    assert classes_in(out) == [AF]


def test_long_sinus_between_afib_stays():
    rr = list(np.tile([0.5, 0.9, 0.7, 1.1, 0.6], 4)) + [0.8] * 12 + list(np.tile([0.5, 0.9, 0.7, 1.1, 0.6], 4))
    beats = make_beats(rr, [N] * len(rr))
    t = beats['t']
    track = np.full(int(t[-1] * HZ) + HZ, AF)
    track[int(t[20] * HZ) - 5:int(t[32] * HZ) - 5] = SINUS
    out, _ = B.postprocess(track, beats, HZ)
    assert classes_in(out) == [AF, SINUS, AF]


def test_vt_split_by_normal_beats_takes_surrounding_rhythm():
    # fix D: VT | N beats | VT inside an AFIB record -> the gap becomes AFIB, not SINUS,
    # and the VT is extended along the V runs the track missed.
    rr = [0.7, 0.9, 0.6] * 3 + [0.35] * 6 + [0.8, 0.7, 0.9, 0.7] + [0.35] * 6 + [0.7, 0.9, 0.6] * 3
    cls = [N] * 9 + [V] * 6 + [N] * 4 + [V] * 6 + [N] * 9
    beats = make_beats(rr, cls)
    t = beats['t']
    track = np.full(int(t[-1] * HZ) + HZ, AF)
    track[int(t[10] * HZ):int(t[24] * HZ)] = VT        # one VT blob covering the N gap
    out, _ = B.postprocess(track, beats, HZ)
    assert classes_in(out) == [AF, VT, AF, VT, AF]
    # the first V beat (t[9]) is now VT although the track started VT at t[10]
    a, b = B.beat_cells(t, len(out), HZ)
    assert out[a[9]] == VT and out[a[8]] == AF


def test_two_v_beats_are_not_vt():
    rr = [0.8] * 10 + [0.4, 0.4] + [0.8] * 10
    cls = [N] * 10 + [V, V] + [N] * 10
    beats = make_beats(rr, cls)
    t = beats['t']
    track = np.full(int(t[-1] * HZ) + HZ, SINUS)
    track[int(t[10] * HZ):int(t[12] * HZ)] = VT
    out, _ = B.postprocess(track, beats, HZ)
    assert classes_in(out) == [SINUS]


def test_invalid_between_two_spec_regions_becomes_sinus():
    # fix E: a 2-beat 'AFIB' blip between an SVT and a VT -> SINUS, never ranked by class id
    rr = [0.4] * 8 + [0.8, 0.8] + [0.35] * 8
    cls = [S] * 8 + [N, N] + [V] * 8
    beats = make_beats(rr, cls)
    t = beats['t']
    track = np.full(int(t[-1] * HZ) + HZ, SINUS)
    track[: int(t[8] * HZ)] = SVT
    track[int(t[8] * HZ):int(t[10] * HZ)] = AF
    track[int(t[10] * HZ):] = VT
    out, _ = B.postprocess(track, beats, HZ)
    assert classes_in(out) == [SVT, SINUS, VT]


def test_noise_is_preserved_and_few_beats_passthrough():
    track = track_from([(SINUS, 4), (NOISE, 3), (SINUS, 4)])
    beats = make_beats([0.8] * 14, [N] * 14)
    out, _ = B.postprocess(track, beats, HZ)
    assert (out[4 * HZ:7 * HZ] == NOISE).all()
    two = make_beats([0.8, 0.8], [N, N])
    out2, sym2 = B.postprocess(track, two, HZ)
    assert (out2 == track).all() and len(sym2) == 2


def test_afib_svt_window_arbitration(monkeypatch):
    monkeypatch.setattr(rc, 'BEAT_PP_AFIB_SVT_MERGE', True)
    rr = [0.5, 0.9, 0.7, 1.1, 0.6] + [0.4] * 40 + [0.5, 0.9, 0.7, 1.1, 0.6]
    cls = [N] * 5 + [S] * 40 + [N] * 5
    beats = make_beats(rr, cls)
    t = beats['t']
    track = np.full(int(t[-1] * HZ) + HZ, AF)
    track[int(t[5] * HZ):int(t[45] * HZ)] = SVT
    out, _ = B.postprocess(track, beats, HZ)
    assert classes_in(out) == [SVT]           # 15.6 s SVT >= 2 x (2 x 3.2 s) AF
    monkeypatch.setattr(rc, 'BEAT_PP_SVT_RATIO', 3.0)
    out, _ = B.postprocess(track, beats, HZ)
    assert classes_in(out) == [AF]


# --- the steps added from the production analysis (K, C, beat symbols) ---------------------

def test_v_run_becomes_vt_without_the_track():
    # K: 5 V beats at 170 bpm the step track called SINUS -> VT when runs_to_rhythm['VT']
    rr = [0.8] * 10 + [0.35] * 5 + [0.8] * 10
    cls = [N] * 10 + [V] * 5 + [N] * 10
    beats = make_beats(rr, cls)
    track = np.full(int(beats['t'][-1] * HZ) + HZ, SINUS)
    out, _ = B.postprocess(track, beats, HZ)
    assert classes_in(out) == [SINUS]
    out, _ = B.postprocess(track, beats, HZ, options={'runs_to_rhythm': {'VT': True}})
    assert classes_in(out) == [SINUS, VT, SINUS]
    # a slow V run (idioventricular, 50 bpm) is still not VT
    slow = make_beats([0.8] * 10 + [1.2] * 5 + [0.8] * 10, cls)
    track = np.full(int(slow['t'][-1] * HZ) + HZ, SINUS)
    out, _ = B.postprocess(track, slow, HZ, options={'runs_to_rhythm': {'VT': True}})
    assert classes_in(out) == [SINUS]


def test_long_invalid_stretch_is_joined_across_classes():
    # C/H: VT(invalid, slow V) | AFIB 1 s (invalid) | SVT (invalid, slow) over 12 s -> SINUS,
    # although every single region is shorter than the 10 s long-invalid threshold
    rr = [0.8] * 5 + [1.0] * 5 + [0.9] * 2 + [1.0] * 6 + [0.8] * 5
    cls = [N] * 5 + [V] * 5 + [N] * 2 + [S] * 6 + [N] * 5
    beats = make_beats(rr, cls)
    t = beats['t']
    track = np.full(int(t[-1] * HZ) + HZ, SINUS)
    track[int(t[5] * HZ):int(t[10] * HZ)] = VT
    track[int(t[10] * HZ):int(t[12] * HZ)] = AF
    track[int(t[12] * HZ):int(t[18] * HZ)] = SVT
    out, _ = B.postprocess(track, beats, HZ, options={'long_invalid': 10.0})
    assert classes_in(out) == [SINUS]


def test_beat_symbols_prematurity_gate_and_afib():
    t = np.concatenate([np.arange(10) * 0.8, [7.2 + 0.5], [7.7 + 0.8, 7.7 + 1.6, 7.7 + 2.4]])
    sym = np.array([N] * 10 + [S] + [S, N, N])         # beat 10 premature, beat 11 not
    r = np.full(len(t), SINUS)
    out = B.beat_symbols(r, t, sym, s_prematurity=0.0)
    assert list(out[10:12]) == [S, S]
    out = B.beat_symbols(r, t, sym, s_prematurity=0.85)
    assert list(out[10:12]) == [S, N]
    r[:] = AF
    assert (B.beat_symbols(r, t, sym, s_prematurity=0.0) != S).all()
    assert (B.beat_symbols(r, t, sym, s_in_afib_to_n=False)[10:12] == S).all()
    r[:] = SVT                                        # inside SVT the gate does not apply
    assert list(B.beat_symbols(r, t, sym, s_prematurity=0.85)[10:12]) == [S, S]


def test_tune_final_decode_uses_the_effective_class_weights(tmp_path):
    """tune.tune builds the written decode in the MAIN process: it must read the same
    train_config the grid workers scored with, not fall back to the manifest weights."""
    import json
    from ecgr.rhythm import tune
    from ecgr.rhythm import config as rc
    cfg = tmp_path / 'train_config.json'
    weights = [0.25, 0.5, 1.0, 4.0, 2.0][:rc.NUM_CLASSES]
    cfg.write_text(json.dumps({'class_weights': weights}))
    tune._CTX.clear()
    tune._init(str(cfg))
    scale = tune._af_decode(dict(gap=0, amin=3, aprob=0, alpha=0.5))['class_scale']
    for name, w in zip(rc.CLASS_NAMES, weights):
        assert scale[name] == pytest.approx(w ** -0.5)
    tune._CTX.clear()
