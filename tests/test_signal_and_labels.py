"""The label contract and the window geometry - no TensorFlow, no data on disk."""
import numpy as np
import pytest

from ecgr import config
from ecgr import signal_ops as so
from ecgr.data.build_npy import SPAN_SLACK_SAMPLES, window_starts
from ecgr.labels import (STEP_SAMPLES, core_bounds, decode_beats, labels_from_annotations,
                         run_class)


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


def test_a_single_lead_record_keeps_its_one_real_lead_on_channel_zero(synthetic_record):
    """Whatever the fill, the one real signal must land on channel 0 - that is the lead the
    labels, the R-peak search and the rhythm descriptor all read."""
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
    window = so.normalize_window(signal[:config.SEGMENT_SAMPLES])
    assert np.allclose(window.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(window.std(axis=0), 1, atol=1e-5)


def test_is_flat_looks_at_lead_zero_only(synthetic_record):
    signal, _ = synthetic_record
    window = signal[:config.SEGMENT_SAMPLES].copy()
    assert not so.is_flat(window)
    window[:, 1:] = 0.0
    assert not so.is_flat(window), "a dead secondary lead must not discard the window"
    window[:, 0] = 0.0
    assert so.is_flat(window)


def test_resample_leads_hits_the_exact_length():
    x = np.random.default_rng(0).standard_normal((3600, 3))
    y = so.resample_leads(x, 360, 250)
    assert y.shape == (2500, 3)


# --- window geometry -------------------------------------------------------------------

def test_segment_starts_never_runs_past_the_signal():
    for length in (2500, 2501, 4000, 15000, 650000):
        starts = so.segment_starts(length)
        assert starts[0] == 0
        assert starts[-1] + config.SEGMENT_SAMPLES == length, "last window must end on the end"
        assert np.all(np.diff(starts) > 0)


def test_segment_starts_pads_only_below_one_window():
    assert list(so.segment_starts(2499)) == [0]
    assert list(so.segment_starts(10)) == [0]


def test_core_bounds_tile_exactly():
    for length in (2500, 4000, 15000, 650000):
        starts = so.segment_starts(max(length, config.SEGMENT_SAMPLES))
        lo, hi = core_bounds(starts, config.SEGMENT_SAMPLES, signal_length=length)
        assert lo[0] == 0 and hi[-1] == length
        assert np.array_equal(hi[:-1], lo[1:]), "gaps or overlaps between segment cores"


# --- the reviewed span, i.e. the bug that cost 4% of the corpus ------------------------

def test_a_span_one_sample_short_of_ten_seconds_still_yields_a_window():
    # 114,920 of 497,716 portal records have exactly this span. The old builder dropped
    # every one of them, and 19,763 of them silently produced no window at all.
    assert window_starts(7500, 9999, 15000) == [7500]
    assert window_starts(0, 2499, 15000) == [0]
    assert window_starts(12500, 14999, 15000) == [12500]


def test_window_starts_stays_inside_the_record():
    for start, stop in ((0, 2499), (12500, 14999), (13000, 15000)):
        for s in window_starts(start, stop, 15000):
            assert 0 <= s <= 15000 - config.SEGMENT_SAMPLES


def test_window_starts_refuses_a_genuinely_short_span():
    assert window_starts(7500, 9500, 15000) == []
    assert window_starts(0, config.SEGMENT_SAMPLES - SPAN_SLACK_SAMPLES - 1, 15000) == []


def test_window_starts_covers_the_end_of_a_long_span():
    starts = window_starts(8294, 13293, 15000)
    assert starts[-1] + config.SEGMENT_SAMPLES == 13293, "the span's tail must be covered"
    assert all(s + config.SEGMENT_SAMPLES <= 13293 for s in starts)


# --- labels ----------------------------------------------------------------------------

def test_label_block_spans_the_pr_interval():
    labels = labels_from_annotations([1000], ['N'])
    hit = np.flatnonzero(labels)
    step = 1000 // STEP_SAMPLES
    assert hit[0] == step - config.LABEL_STEPS_BEFORE
    assert hit[-1] == step + config.LABEL_STEPS_AFTER


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


# --- the round trip --------------------------------------------------------------------

def test_decode_beats_recovers_every_beat_exactly_once(synthetic_record):
    """labels -> oracle predictions -> decode must give back the beats it started from."""
    signal, peaks = synthetic_record
    leads = so.build_leads(signal, primary=0)
    segments, starts = so.segment_record(leads)

    preds = np.zeros((len(segments), config.OUTPUT_STEPS, config.NUM_CLASSES), np.float32)
    preds[..., 0] = 1.0
    for i, start in enumerate(starts):
        inside = peaks[(peaks >= start) & (peaks < start + config.SEGMENT_SAMPLES)] - start
        for step, value in enumerate(labels_from_annotations(inside, ['N'] * len(inside))):
            if value:
                preds[i, step] = 0.0
                preds[i, step, value] = 1.0

    positions, symbols = decode_beats(preds, segments, starts, signal_length=len(leads))
    assert len(positions) == len(peaks), "a beat was dropped or double counted"
    assert np.all(np.diff(positions) > 0), "positions must come back sorted"
    assert set(symbols) == {'N'}
    assert np.abs(positions - peaks).max() <= 2, "R peak position drifted"


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
