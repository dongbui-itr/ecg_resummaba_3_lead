"""The label contract and the window geometry - no TensorFlow, and no dataset on disk.

The 60 s contract in one line: a window is the whole strip, labels are trusted only inside
the reviewed span, and everything outside it is IGNORE rather than background.
"""
import numpy as np
import pytest
import wfdb

from ecgr import config
from ecgr import signal_ops as so
from ecgr.data.build_npy import SPAN_SLACK_SAMPLES, cut_window, process_record, window_starts
from ecgr.labels import (best_lead, core_bounds, decode_beats, ignore_outside,
                         labels_from_annotations, run_class, step_samples)

SEG = config.SEGMENT_SAMPLES        # 15000
IGN = config.IGNORE_LABEL


# --- leads -----------------------------------------------------------------------------

def test_build_leads_puts_the_annotated_lead_first(synthetic_record):
    signal, _ = synthetic_record
    for primary in range(3):
        leads = so.build_leads(signal, primary=primary)
        assert leads.shape == (len(signal), config.IN_CHANNELS)
        assert leads.dtype == np.float32
        # channel 0 of the rolled montage is the primary lead of the unrolled one
        plain = so.build_leads(signal, primary=0)
        assert np.allclose(leads[:, 0], plain[:, primary], atol=1e-4)
        # and lead_order is the map back: model channel c -> record lead lead_order[c]
        order = so.lead_order(3, primary)
        assert order[0] == primary and sorted(order) == [0, 1, 2]
        for c in range(3):
            assert np.allclose(leads[:, c], plain[:, order[c]], atol=1e-4)


def test_a_single_lead_record_keeps_its_one_real_lead_on_channel_zero(synthetic_record):
    signal, _ = synthetic_record
    plain = so.build_leads(signal, primary=0)
    for fill in ('zero', 'duplicate'):
        leads = so.build_leads(signal[:, 0], primary=0, fill_mode=fill)
        assert leads.shape[1] == config.IN_CHANNELS
        assert np.allclose(leads[:, 0], plain[:, 0], atol=1e-4)


def test_build_leads_rejects_an_impossible_primary(synthetic_record):
    signal, _ = synthetic_record
    with pytest.raises(ValueError):
        so.build_leads(signal, primary=5)


def test_normalize_window_is_per_lead(synthetic_record):
    signal, _ = synthetic_record
    window = so.normalize_window(signal[:2500])
    assert np.allclose(window.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(window.std(axis=0), 1, atol=1e-5)


def test_is_flat_looks_at_lead_zero_only(synthetic_record):
    signal, _ = synthetic_record
    window = signal[:2500].copy()
    assert not so.is_flat(window)
    window[:, 1:] = 0.0
    assert not so.is_flat(window), "a dead secondary lead must not discard the window"
    window[:, 0] = 0.0
    assert so.is_flat(window)
    assert so.is_flat(window[:0]), "an empty span carries nothing"


def test_resample_leads_hits_the_exact_length():
    x = np.random.default_rng(0).standard_normal((3600, 3))
    y = so.resample_leads(x, 360, 250)
    assert y.shape == (2500, 3)


def test_pad_to_length_repeats_the_last_sample():
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    padded = so.pad_to_length(x, 6)
    assert padded.shape == (6, 3)
    assert np.array_equal(padded[4:], np.repeat(x[-1:], 2, axis=0))
    assert so.pad_to_length(x, 4) is x or np.array_equal(so.pad_to_length(x, 4), x)


# --- inference window geometry ---------------------------------------------------------

def test_segment_starts_never_runs_past_the_signal():
    for length in (SEG, SEG + 1, 40000, 650000):
        starts = so.segment_starts(length)
        assert starts[0] == 0
        assert starts[-1] + SEG == length, "last window must end on the end"
        assert np.all(np.diff(starts) > 0)
        assert np.all(np.diff(starts) <= SEG - config.EC57_SEGMENT_OVERLAP)


def test_segment_starts_pads_only_below_one_window():
    assert list(so.segment_starts(SEG - 1)) == [0]
    assert list(so.segment_starts(10)) == [0]
    assert list(so.segment_starts(SEG)) == [0], "a 60 s strip is exactly one window"


def test_core_bounds_tile_exactly():
    for length in (2500, 7500, SEG, 40000, 650000):
        starts = so.segment_starts(max(length, SEG))
        lo, hi = core_bounds(starts, SEG, signal_length=length)
        assert lo[0] == 0 and hi[-1] == length
        assert np.array_equal(hi[:-1], lo[1:]), "gaps or overlaps between segment cores"


def test_geometry_is_read_at_call_time(monkeypatch):
    """Evaluation re-derives the window from a checkpoint (config.apply_geometry); every
    geometry default here must follow it rather than the value bound at import."""
    saved = (config.SEGMENT_SAMPLES, config.OUTPUT_STEPS)
    try:
        config.apply_geometry(2500)
        assert (config.SEGMENT_SAMPLES, config.OUTPUT_STEPS, config.STEP_SAMPLES) == (2500, 500, 5)
        assert config.EC57_SEGMENT_OVERLAP <= 2500 // 6
        assert list(so.segment_starts(2500)) == [0]
        assert so.segment_starts(10000)[-1] + 2500 == 10000
        assert len(labels_from_annotations([100], ['N'])) == 500
        assert step_samples() == 5
        segments, starts = so.segment_record(np.zeros((5000, 3), np.float32))
        assert segments.shape[1] == 2500
    finally:
        config.apply_geometry(*saved)
        config.EC57_SEGMENT_OVERLAP = 10 * config.SAMPLING_RATE
    assert config.SEGMENT_SAMPLES == SEG


# --- training window geometry: the reviewed span inside a 60 s strip --------------------

def test_a_reviewed_span_inside_a_strip_gives_the_whole_strip_as_one_window():
    # the everyday case: a 10 s review in the middle of a 60 s record
    assert window_starts(7500, 10000, 15000) == [0]
    assert window_starts(7500, 9999, 15000) == [0]            # the 2499-sample spans
    assert window_starts(0, 2499, 15000) == [0]
    assert window_starts(12500, 14999, 15000) == [0]
    assert window_starts(1148, 3647, 15000) == [0]


def test_a_short_record_is_one_window_the_caller_pads():
    assert window_starts(0, 2500, 7500) == [0]
    assert window_starts(5000, 7500, 7500) == [0]
    window, padded = cut_window(np.ones((7500, 3), np.float32), 0, SEG)
    assert window.shape == (SEG, 3) and padded == SEG - 7500


def test_a_long_record_centres_the_window_on_the_span():
    starts = window_starts(20000, 22500, 30000)
    assert len(starts) == 1 and starts[0] <= 20000 and starts[0] + SEG >= 22500
    assert 0 <= starts[0] <= 30000 - SEG
    assert window_starts(27000, 30000 - 1, 30000) == [30000 - SEG], "clipped to the record"


def test_a_span_longer_than_a_window_slides_inside_it():
    starts = window_starts(0, 45000, 45000)
    hop = config.SEGMENT_STRIDE_SECONDS * config.SAMPLING_RATE
    assert starts[0] == 0 and starts[-1] + SEG == 45000
    assert all(s + SEG <= 45000 for s in starts)
    assert all(b - a <= hop for a, b in zip(starts, starts[1:]))


def test_window_starts_refuses_a_span_with_too_little_reviewed_signal():
    assert config.MIN_REVIEWED_SAMPLES == 2500 - SPAN_SLACK_SAMPLES
    assert window_starts(7500, 9500, 15000) == []
    assert window_starts(0, config.MIN_REVIEWED_SAMPLES - 1, 15000) == []
    assert window_starts(0, config.MIN_REVIEWED_SAMPLES, 15000) == [0]


# --- labels ----------------------------------------------------------------------------

def test_label_block_spans_the_pr_interval():
    labels = labels_from_annotations([1000], ['N'])
    hit = np.flatnonzero(labels)
    step = 1000 // step_samples()
    assert hit[0] == step - config.LABEL_STEPS_BEFORE
    assert hit[-1] == step + config.LABEL_STEPS_AFTER
    assert len(labels) == config.OUTPUT_STEPS == 3000


def test_ectopic_blocks_win_over_normal_ones():
    # S(3) > V(2) > N(1): a neighbouring normal beat cannot overwrite an ectopic block
    labels = labels_from_annotations([1000, 1020], ['S', 'N'])
    assert config.CLASS_NAMES.index('S') in labels
    labels = labels_from_annotations([1000, 1020], ['N', 'S'])
    assert config.CLASS_NAMES.index('S') in labels


def test_unmapped_symbols_are_not_targets():
    assert not labels_from_annotations([1000], ['+']).any()
    assert not labels_from_annotations([1000], ['~']).any()


def test_run_class_takes_the_majority_not_the_maximum():
    assert run_class(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3])) == 1
    assert run_class(np.array([0, 0, 3, 3, 3, 1])) == 3


def test_ignore_outside_marks_the_unreviewed_steps_and_keeps_in_span_beats_whole():
    """Outside [7500, 10000) everything is IGNORE - except the block of a beat whose R peak
    is inside the span, which may poke past the boundary and stays labelled."""
    per = step_samples()
    labels = labels_from_annotations([7510, 8750, 9990, 12000, 3000], ['N', 'S', 'N', 'V', 'N'])
    out = ignore_outside(labels, 7500, 10000)
    lo, hi = 7500 // per, 10000 // per
    assert (out[lo:hi] != IGN).all(), "every step inside the span keeps its label"
    assert (out[lo:hi] == labels[lo:hi]).all()
    assert (out[:lo - config.LABEL_STEPS_BEFORE - 1] == IGN).all()
    assert (out[hi + config.LABEL_STEPS_AFTER + 2:] == IGN).all()
    # the beat at 12000 and the one at 3000 are outside: gone entirely
    assert (out[12000 // per - 10: 12000 // per + 4] == IGN).all()
    assert (out[3000 // per - 10: 3000 // per + 4] == IGN).all()
    # the beat at 7510 starts its block before the span: those steps are its, not IGNORE
    first = 7510 // per - config.LABEL_STEPS_BEFORE
    assert first < lo and (out[first:lo] == config.CLASS_NAMES.index('N')).all()
    # an empty span ignores everything
    assert (ignore_outside(labels, 5000, 5000) == IGN).all()


def test_ignore_is_a_distinct_value_from_background():
    assert IGN == 255 and IGN not in range(config.NUM_CLASSES)
    labels = np.zeros(config.OUTPUT_STEPS, np.int64)
    out = ignore_outside(labels, 0, 2500)
    assert (out[:500] == 0).all() and (out[500:] == IGN).all()


# --- one real record through the builder ----------------------------------------------

def test_process_record_labels_only_the_reviewed_span(tmp_path, monkeypatch, synthetic_record):
    """A 60 s strip with beats throughout, reviewed on [7500, 10000): the window is the whole
    strip, the steps outside the span are IGNORE, the beats inside are labelled, and the
    beats outside contribute no label at all."""
    signal, _ = synthetic_record                      # 30 s; tile it to 60 s
    strip = np.concatenate([signal, signal], axis=0)[:15000]
    peaks = np.arange(int(0.5 * 250), 15000 - 250, 200)   # 0.8 s R-R
    monkeypatch.setattr(config, 'DATA_DIR', str(tmp_path))
    folder = tmp_path / 'db' / '11' / 'ev'
    folder.mkdir(parents=True)
    wfdb.wrsamp('strip', fs=250, units=['mV'] * 3, sig_name=['CH0', 'CH1', 'CH2'],
                p_signal=strip.astype(np.float64), write_dir=str(folder))
    symbols = ['N'] * len(peaks)
    symbols[len(peaks) // 2] = 'V'
    wfdb.wrann('strip', 'atr', np.asarray(peaks), symbol=symbols, write_dir=str(folder))

    segments, labels, stats = process_record(('11', 'ev', 1, 7500, 10000), 'db')
    assert stats['errors'] == 0, stats.get('last_error')
    assert len(segments) == 1 and segments[0].shape == (SEG, config.IN_CHANNELS)
    lab = labels[0]
    per = step_samples()
    inside = slice(7500 // per, 10000 // per)
    assert (lab[inside] != IGN).all()
    assert (lab[inside] > 0).sum() > 0, "beats inside the span must be labelled"
    outside = np.ones(config.OUTPUT_STEPS, bool)
    outside[7500 // per - config.LABEL_STEPS_BEFORE: 10000 // per + config.LABEL_STEPS_AFTER + 1] = False
    assert (lab[outside] == IGN).all(), "no label may survive outside the reviewed span"
    assert stats['steps_ignored'] > stats['steps_labelled'] > 0
    assert stats['padded'] == 0
    # channel 1 was annotated: the model's channel 0 is the record's CH1
    expected = so.normalize_window(so.build_leads(strip, primary=1))
    assert np.allclose(segments[0], expected, atol=1e-5)


def test_process_record_pads_a_short_strip_and_ignores_the_padding(tmp_path, monkeypatch,
                                                                  synthetic_record):
    signal, peaks = synthetic_record                  # 30 s = 7500 samples
    monkeypatch.setattr(config, 'DATA_DIR', str(tmp_path))
    folder = tmp_path / 'db' / '12' / 'ev'
    folder.mkdir(parents=True)
    wfdb.wrsamp('strip', fs=250, units=['mV'] * 3, sig_name=['CH0', 'CH1', 'CH2'],
                p_signal=signal.astype(np.float64), write_dir=str(folder))
    wfdb.wrann('strip', 'atr', np.asarray(peaks), symbol=['N'] * len(peaks),
               write_dir=str(folder))
    segments, labels, stats = process_record(('12', 'ev', 0, 2500, 5000), 'db')
    assert stats['errors'] == 0, stats.get('last_error')
    assert len(segments) == 1 and stats['padded'] == 1
    per = step_samples()
    assert (labels[0][7500 // per:] == IGN).all(), "padding carries no label"
    assert (labels[0][2500 // per: 5000 // per] != IGN).all()
    assert np.allclose(segments[0][7500:], segments[0][7499], atol=1e-6), "edge padding"


# --- the round trip --------------------------------------------------------------------

def test_decode_beats_recovers_every_beat_exactly_once(synthetic_record):
    """labels -> oracle predictions -> decode must give back the beats it started from."""
    signal, peaks = synthetic_record
    leads = so.build_leads(signal, primary=0)
    segments, starts = so.segment_record(leads)

    preds = np.zeros((len(segments), config.OUTPUT_STEPS, config.NUM_CLASSES), np.float32)
    preds[..., 0] = 1.0
    for i, start in enumerate(starts):
        inside = peaks[(peaks >= start) & (peaks < start + SEG)] - start
        for step, value in enumerate(labels_from_annotations(inside, ['N'] * len(inside))):
            if value:
                preds[i, step] = 0.0
                preds[i, step, value] = 1.0

    positions, symbols = decode_beats(preds, segments, starts, signal_length=len(leads))
    assert len(positions) == len(peaks), "a beat was dropped or double counted"
    assert np.all(np.diff(positions) > 0), "positions must come back sorted"
    assert set(symbols) == {'N'}
    assert np.abs(positions - peaks).max() <= 2, "R peak position drifted"


def test_decode_beats_over_a_long_record_uses_every_window_once(synthetic_record):
    """A 3-minute record is several overlapping 60 s windows; the overlap rule must still
    give back every beat exactly once."""
    signal, peaks = synthetic_record
    long = np.concatenate([signal] * 6, axis=0)                   # 180 s
    long_peaks = np.concatenate([peaks + k * len(signal) for k in range(6)])
    leads = so.build_leads(long, primary=0)
    segments, starts = so.segment_record(leads)
    assert len(starts) > 1
    preds = np.zeros((len(segments), config.OUTPUT_STEPS, config.NUM_CLASSES), np.float32)
    preds[..., 0] = 1.0
    for i, start in enumerate(starts):
        inside = long_peaks[(long_peaks >= start) & (long_peaks < start + SEG)] - start
        for step, value in enumerate(labels_from_annotations(inside, ['N'] * len(inside))):
            if value:
                preds[i, step] = 0.0
                preds[i, step, value] = 1.0
    positions, _ = decode_beats(preds, segments, starts, signal_length=len(leads))
    assert len(positions) == len(long_peaks)
    assert np.abs(positions - long_peaks).max() <= 2


def test_decode_beats_honours_min_rr(synthetic_record):
    signal, _ = synthetic_record
    leads = so.build_leads(signal, primary=0)
    segments, starts = so.segment_record(leads)
    preds = np.zeros((len(segments), config.OUTPUT_STEPS, config.NUM_CLASSES), np.float32)
    preds[..., 1] = 1.0                      # every step is a beat: one run per segment
    positions, _ = decode_beats(preds, segments, starts, signal_length=len(leads))
    assert np.all(np.diff(positions) >= config.MIN_RR_INTERVAL)


def test_decode_beats_ignores_the_padding_of_a_short_record():
    leads = so.build_leads(np.random.default_rng(0).standard_normal((1000, 3)), primary=0)
    segments, starts = so.segment_record(leads)
    preds = np.zeros((len(segments), config.OUTPUT_STEPS, config.NUM_CLASSES), np.float32)
    preds[..., 1] = 1.0
    positions, _ = decode_beats(preds, segments, starts, signal_length=1000)
    assert positions.max() < 1000, "a detection came from the edge padding"


# --- output 2: the most reliable lead --------------------------------------------------

def test_best_lead_is_the_argmax_of_the_time_average_inside_the_record():
    q = np.zeros((2, config.OUTPUT_STEPS, 3))
    q[..., 0] = 0.6
    q[..., 1] = 0.9
    q[..., 2] = 0.2
    lead, means = best_lead(q)
    assert lead == 1 and np.allclose(means, [0.6, 0.9, 0.2])
    # padding of a short record is left out: lead 2 is perfect only in the padding
    q = np.zeros((1, config.OUTPUT_STEPS, 3))
    q[0, :, 0] = 0.7
    q[0, 500:, 2] = 1.0
    lead, means = best_lead(q, starts=[0], signal_length=2500)
    assert lead == 0 and means[2] == 0.0
    lead, _ = best_lead(q, starts=[0], signal_length=15000)
    assert lead == 2


# --- duplicate recordings (dataset-2 stores every strip twice) -------------------------

def test_record_files_collapses_byte_identical_copies(tmp_path, monkeypatch):
    """Every dataset-2 event folder holds the same strip twice, under two different event-id
    prefixes and byte for byte identical. Processing both emitted every dataset-2 window
    twice, which silently doubled the largest single contributor's weight in the loss."""
    from ecgr.data import build_npy

    monkeypatch.setattr(config, 'DATA_DIR', str(tmp_path))
    event = tmp_path / 'db' / '123' / 'abc'
    event.mkdir(parents=True)
    (event / 'aaa-strip.dat').write_bytes(b'\x01\x02' * 500)
    (event / 'bbb-strip.dat').write_bytes(b'\x01\x02' * 500)      # the identical copy

    files, dropped = build_npy._record_files('db', '123', 'abc')
    assert len(files) == 1 and dropped == 1

    # a genuinely different second recording is still kept - that was the old behaviour and
    # the dedupe must not take it away
    (event / 'ccc-strip.dat').write_bytes(b'\x03\x04' * 500)
    files, dropped = build_npy._record_files('db', '123', 'abc')
    assert len(files) == 2 and dropped == 1

    files, dropped = build_npy._record_files('db', '123', 'abc', dedupe=False)
    assert len(files) == 3 and dropped == 0


# --- filling the channel axis when a record has too few leads ---------------------------

def test_missing_leads_are_zero_padded_by_default(synthetic_record):
    """config.LEAD_FILL_MODE = 'zero': a record with one lead gets silence in the rest, not a
    copy. Silence is what an electrode coming off looks like and what training produces on
    purpose (AUGMENT_LEAD_DROP_PROB); a copy would be a second vote for the first lead."""
    signal, _ = synthetic_record
    assert config.LEAD_FILL_MODE == 'zero'
    leads = so.build_leads(signal[:, 0], primary=0)
    assert leads.shape[1] == config.IN_CHANNELS
    assert np.any(leads[:, 0] != 0), "the real lead must survive"
    for c in range(1, config.IN_CHANNELS):
        assert np.all(leads[:, c] == 0.0), f"channel {c} should be zero-padded"

    # two real leads of a three-lead model: only the third is padded
    two = so.build_leads(signal[:, :2], primary=0)
    assert np.any(two[:, 1] != 0) and np.all(two[:, 2] == 0.0)


def test_duplicate_fill_is_still_available(synthetic_record):
    signal, _ = synthetic_record
    leads = so.build_leads(signal[:, 0], primary=0, fill_mode='duplicate')
    for c in range(1, config.IN_CHANNELS):
        assert np.array_equal(leads[:, 0], leads[:, c])


def test_an_unknown_fill_mode_is_refused(synthetic_record):
    signal, _ = synthetic_record
    with pytest.raises(ValueError, match="fill_mode"):
        so.build_leads(signal[:, 0], primary=0, fill_mode='mirror')
