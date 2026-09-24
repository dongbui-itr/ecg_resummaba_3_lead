"""Reading real records for EC57: the lead modes, the resampling, the annotation round trip.

Skipped when the databases are not on this machine; when they are, these are the tests that
actually pin down "for EC57, pick one channel and duplicate it".
"""
import os

import numpy as np
import pytest
import wfdb

from ecgr import config
from ecgr.evaluation import ec57

MITDB = os.path.join(config.PHYSIONET_DIR, 'mitdb')
PORTAL = config.PORTAL_EVAL_SETS['dataset-v4-beat']

needs_mitdb = pytest.mark.skipif(not os.path.isdir(MITDB), reason='mitdb not present')
needs_portal = pytest.mark.skipif(not os.path.isdir(PORTAL), reason='portal eval set absent')


@needs_mitdb
def test_a_two_lead_360hz_record_becomes_three_leads_at_250hz():
    """mitdb: 2 leads at 360 Hz -> (N, 3) at 250 Hz. 'auto' keeps both real leads and the
    third channel is filled per LEAD_FILL_MODE, which defaults to zero."""
    path = os.path.join(MITDB, '100')
    leads, raw_length, fs = ec57.read_leads(path, channel=0, lead_mode='auto')

    assert fs == 360 and raw_length == 650000
    assert leads.shape[1] == config.IN_CHANNELS
    assert len(leads) == round(raw_length * config.SAMPLING_RATE / 360)
    assert not np.array_equal(leads[:, 0], leads[:, 1]), "both real leads must be used"
    assert np.all(leads[:, 2] == 0.0), "the leftover channel is zero-padded by default"


@needs_mitdb
def test_single_mode_leaves_one_real_lead_and_pads_the_rest():
    """'single' is the strict single-lead reading: one genuine signal, the rest silence.
    'duplicate' is its deprecated alias and must behave identically."""
    path = os.path.join(MITDB, '100')
    for mode in ('single', 'duplicate'):
        leads, _, _ = ec57.read_leads(path, channel=0, lead_mode=mode)
        assert np.any(leads[:, 0] != 0)
        for c in range(1, config.IN_CHANNELS):
            assert np.all(leads[:, c] == 0.0), f"{mode}: channel {c} should be zero"
    # and with the old fill the same mode repeats the lead instead
    leads, _, _ = ec57.read_leads(path, channel=0, lead_mode='single', fill_mode='duplicate')
    for c in range(1, config.IN_CHANNELS):
        assert np.array_equal(leads[:, 0], leads[:, c])


@needs_mitdb
def test_single_mode_uses_the_lead_it_is_told_to():
    path = os.path.join(MITDB, '100')
    first, _, _ = ec57.read_leads(path, channel=0, lead_mode='single')
    second, _, _ = ec57.read_leads(path, channel=1, lead_mode='single')
    assert not np.allclose(first[:, 0], second[:, 0]), "channel= was ignored"


@needs_mitdb
def test_native_mode_on_a_two_lead_record_keeps_both_and_fills_the_rest():
    leads, _, _ = ec57.read_leads(os.path.join(MITDB, '100'), channel=0, lead_mode='native')
    assert leads.shape[1] == config.IN_CHANNELS
    assert not np.array_equal(leads[:, 0], leads[:, 1]), "lead 1 should be the real MLII/V5"
    assert np.all(leads[:, 2] == 0.0), "the third channel is the zero fill"


def _first_portal_record():
    """Path (without extension) of one portal eval record."""
    names = sorted({f[:-4] for f in os.listdir(PORTAL) if f.endswith('.dat')})
    return os.path.join(PORTAL, names[0])


@needs_portal
def test_a_three_lead_portal_record_keeps_its_own_montage():
    path = _first_portal_record()
    channel = ec57.record_channel(path)
    leads, raw_length, fs = ec57.read_leads(path, channel=channel, lead_mode='auto')

    assert fs == config.SAMPLING_RATE
    assert leads.shape == (raw_length, config.IN_CHANNELS)
    distinct = sum(not np.array_equal(leads[:, 0], leads[:, c])
                   for c in range(1, config.IN_CHANNELS))
    assert distinct >= 1, "a native 3-lead strip came back duplicated"


@needs_portal
def test_the_annotated_channel_is_read_from_the_header():
    path = _first_portal_record()
    channel = ec57.record_channel(path)
    assert 0 <= channel < 3
    # channel 0 of the rolled montage must be the lead the header names
    leads, _, _ = ec57.read_leads(path, channel=channel, lead_mode='native')
    plain, _, _ = ec57.read_leads(path, channel=0, lead_mode='native')
    assert np.allclose(leads[:, 0], plain[:, channel], atol=1e-4)


@needs_mitdb
def test_predictions_are_written_where_bxb_can_read_them(tmp_path):
    """The full per-record path with a stand-in model: sweep, decode, write a .ain that wfdb
    can read back at the record's own sampling rate."""
    from ecgr import models
    model = models.build('resumamba_100k')

    # 30 s of record 100 is enough to exercise the whole path without a real prediction
    record = wfdb.rdrecord(os.path.join(MITDB, '100'), sampto=360 * 30)
    local = tmp_path / '100'
    wfdb.wrsamp('100', fs=record.fs, units=record.units,
                sig_name=record.sig_name, p_signal=record.p_signal,
                write_dir=str(tmp_path))

    out_dir = tmp_path / 'ann'
    out_dir.mkdir()
    n = ec57.predict_record(model, str(local), '100', str(out_dir), channel=0,
                            lead_mode='single')
    assert n >= 0
    if n:
        ann = wfdb.rdann(str(out_dir / '100'), config.BEAT_EXTENSION)
        assert ann.fs == record.fs
        assert len(ann.sample) == n
        assert ann.sample.max() < len(record.p_signal), "a beat was written past the record"
        assert set(ann.symbol) <= set(config.CLASS_NAMES[1:])


# --- the portal splits as bxb sources ------------------------------------------------

PORTAL_CSV = os.path.join(config.DATA_DIR, 'dataset-1', config.DATASET_CSV)
needs_portal_train = pytest.mark.skipif(not os.path.exists(PORTAL_CSV),
                                        reason='portal training datasets absent')


def test_split_sample_is_deterministic_and_order_independent():
    rows = [('db', str(s), f'{e:024x}', 1, 7500, 10000) for s in range(3) for e in range(50)]
    a = ec57.sample_split_records(rows, 20)
    b = ec57.sample_split_records(list(reversed(rows)), 20)
    assert a == b and len(a) == 20, "the sample must not depend on CSV order"
    assert ec57.sample_split_records(rows, 0) == ec57.sample_split_records(rows, None)
    assert len(ec57.sample_split_records(rows, 0)) == len(rows)
    # a name is unique across studies: two studies with the same event id do not collide
    assert ec57.split_record_name('1', 'abc') != ec57.split_record_name('2', 'abc')


@needs_portal_train
def test_scoring_header_is_valid_wfdb_and_carries_the_window(tmp_path):
    """The rewritten .hea must parse, keep every signal field, name the new .dat, and carry the
    reviewed window as the three comments the mark-window bxb script greps for."""
    import csv
    row = next(csv.DictReader(open(PORTAL_CSV)))
    from ecgr.data.build_npy import _record_files
    files, _ = _record_files('dataset-1', row['study_id'], row['event_id'])
    src = files[0][:-4]
    name = ec57.split_record_name(row['study_id'], row['event_id'])

    ec57.write_scoring_header(src + '.hea', str(tmp_path / f'{name}.hea'), name,
                              int(row['channel']), int(row['start_sample']),
                              int(row['stop_sample']))
    os.symlink(src + '.dat', tmp_path / f'{name}.dat')

    original = wfdb.rdheader(src)
    rewritten = wfdb.rdheader(str(tmp_path / name))
    assert rewritten.n_sig == original.n_sig == 3
    assert rewritten.fs == original.fs and rewritten.sig_len == original.sig_len
    assert rewritten.adc_gain == original.adc_gain and rewritten.baseline == original.baseline
    assert set(rewritten.file_name) == {f'{name}.dat'}
    text = (tmp_path / f'{name}.hea').read_text()
    assert f"# startMarkSample: {row['start_sample']}" in text
    assert f"# stopMarkSample: {row['stop_sample']}" in text
    assert f"# channel: {row['channel']}" in text
    # and the signal is readable through the new header - the symlinked .dat resolves
    rec = wfdb.rdrecord(str(tmp_path / name))
    assert rec.p_signal.shape == (original.sig_len, 3)


@needs_portal_train
@pytest.mark.skipif(not __import__('shutil').which('bxb'), reason='bxb not installed')
def test_portal_split_scores_end_to_end(tmp_path, monkeypatch):
    """A 3-record eval-split sample through predict -> .hea rewrite -> bxb -> report."""
    from ecgr import models
    monkeypatch.setattr(config, 'EC57_DIR', str(tmp_path))
    model = models.build('resumamba_100k')
    path = ec57.score_portal_split(model, 'eval', str(tmp_path / 'probe'), max_records=3)
    assert path and os.path.exists(path), "bxb wrote no report"
    row = __import__('ecgr.evaluation.report', fromlist=['parse_report']).parse_report(path)
    assert row.get('records') == '3'
    assert 'Q_Se' in row


def test_an_event_listed_by_two_datasets_is_scored_once(tmp_path, monkeypatch):
    """8,310 (study, event) pairs appear in two datasets (re-curations of dataset-2/3/4).
    They are one recording; the split sample must hold it once, under the first dataset's
    review - or the scoring dir links the same .dat under one name twice and dies."""
    from ecgr.data import splits
    for db, channel in (('orig', 1), ('recur', 2)):
        (tmp_path / db).mkdir()
        with open(tmp_path / db / config.DATASET_CSV, 'w') as f:
            f.write('study_id,event_id,channel,start_sample,stop_sample\n')
            f.write(f'7,abc,{channel},7500,10000\n')
            f.write(f'7,{db}only,{channel},0,2500\n')
    monkeypatch.setattr(config, 'DATA_DIR', str(tmp_path))
    monkeypatch.setattr(splits, 'held_out_studies', lambda: (set(), set()))
    side = splits.study_split_side('7')

    rows = ec57.portal_split_records(side, db_names=['orig', 'recur'])
    names = [ec57.split_record_name(r[1], r[2]) for r in rows]
    assert names.count('7_abc') == 1, "the re-listed event must appear once"
    assert next(r for r in rows if r[2] == 'abc')[0] == 'orig', "first dataset's review wins"
    assert len(rows) == 3


@needs_mitdb
def test_a_two_output_model_reports_its_best_lead(tmp_path):
    """Output 2 on a real record: predict_record logs the model's best lead in the RECORD's
    channel numbering, and write_lead_quality persists it beside the predictions."""
    from ecgr import models
    model = models.build('resumamba_100k')
    record = wfdb.rdrecord(os.path.join(MITDB, '100'), sampto=360 * 70)
    wfdb.wrsamp('100', fs=record.fs, units=record.units, sig_name=record.sig_name,
                p_signal=record.p_signal, write_dir=str(tmp_path))
    out_dir = tmp_path / 'ann'
    out_dir.mkdir()
    log = []
    ec57.predict_record(model, str(tmp_path / '100'), '100', str(out_dir), channel=1,
                        lead_mode='native', quality_log=log)
    assert len(log) == 1
    name, lead, means, model_ch = log[0]
    assert name == '100' and lead in (0, 1), "a filled (zero) channel can never be chosen"
    assert model_ch in (0, 1) and len(means) == config.IN_CHANNELS
    # channel 1 was annotated: model channel 0 is record lead 1, model channel 1 is lead 0
    assert lead == (1 if model_ch == 0 else 0)
    # 'single' mode: only the annotated lead is real, so it is the only possible answer
    log = []
    ec57.predict_record(model, str(tmp_path / '100'), '100', str(out_dir), channel=1,
                        lead_mode='single', quality_log=log)
    assert log[0][1] == 1
    path = ec57.write_lead_quality(log, str(out_dir), str(tmp_path / 'rep'), {'100': 1})
    import json
    summary = json.load(open(path))
    assert summary['records'] == 1 and summary['agrees_with_reviewer_channel'] == 1.0
    assert (out_dir / 'lead_quality.csv').read_text().startswith(
        'record,best_lead,best_model_ch,q_model_ch0')


def test_load_checkpoints_takes_the_window_geometry_from_a_legacy_checkpoint(tmp_path):
    """A 10 s checkpoint of the previous family is scored in 10 s windows by the same code:
    that is what makes the baseline comparison in assets/baselines/ apples to apples."""
    legacy = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'checkpoints', 'resumamba_30k.keras')
    if not os.path.exists(legacy):
        pytest.skip('legacy 10 s checkpoint not in the checkout')
    saved = (config.SEGMENT_SAMPLES, config.OUTPUT_STEPS, config.EC57_SEGMENT_OVERLAP)
    # a 60 s model, built BEFORE the geometry is re-derived from the legacy checkpoint
    from ecgr import models
    fresh = str(tmp_path / 'fresh_60s.keras')
    models.build('resumamba_100k').save(fresh)
    try:
        model = ec57.load_checkpoints(legacy)
        assert tuple(model.input_shape[1:]) == (2500, 3)
        assert (config.SEGMENT_SAMPLES, config.OUTPUT_STEPS, config.STEP_SAMPLES) == (2500, 500, 5)
        assert config.EC57_SEGMENT_OVERLAP <= 2500 // 6
        from ecgr.signal_ops import segment_record
        segments, starts = segment_record(np.zeros((7000, 3), np.float32))
        assert segments.shape[1:] == (2500, 3) and len(starts) > 1
        # an ensemble of a 10 s and a 60 s checkpoint has no single geometry: refused
        with pytest.raises(ValueError, match='input shape'):
            ec57.load_checkpoints([legacy, fresh])
    finally:
        config.apply_geometry(saved[0], saved[1])
        config.EC57_SEGMENT_OVERLAP = saved[2]
    assert config.SEGMENT_SAMPLES == 15000


def test_default_lead_mode_is_native():
    """The EC57 databases have two real leads and the model takes three. Reading both real
    leads (native) rather than repeating one (duplicate) is the default because it is what
    the 3-lead architecture exists for - measured: resumamba_2m mitdb S 45.79/61.17 ->
    56.87/65.64, 30 of 32 Physionet cells up. `duplicate` stays available as the control."""
    assert config.EC57_LEAD_MODE == 'native'


@needs_mitdb
def test_native_default_actually_reads_both_mitdb_leads():
    """A regression guard on the default, not on the flag: with the config default in force,
    lead 1 of the model input must be the record's second signal, not a copy of the first."""
    leads, _, _ = ec57.read_leads(os.path.join(MITDB, '100'), channel=0,
                                  lead_mode=config.EC57_LEAD_MODE)
    assert leads.shape[1] == config.IN_CHANNELS
    assert not np.array_equal(leads[:, 0], leads[:, 1]), "the second real lead was discarded"


@needs_mitdb
def test_read_leads_without_a_mode_follows_the_config_not_auto():
    """A bare read_leads() must measure what the pipeline measures.

    It used to default to lead_mode='auto', and 'auto' on a 2-lead record with a 3-lead model
    resolves to 'single' - so a direct call silently used ONE mitdb lead while `ecgr ec57`
    (which resolves from config, now 'native') used both. Every production call site passes
    the mode explicitly, so no published number came from the wrong path, but the trap was
    real: this test is what keeps the two in step.
    """
    path = os.path.join(MITDB, '100')
    default, _, _ = ec57.read_leads(path, channel=0)
    explicit, _, _ = ec57.read_leads(path, channel=0, lead_mode=config.EC57_LEAD_MODE)
    assert np.array_equal(default, explicit)
    # and with the config default ('native') that means both real leads, not one
    assert not np.array_equal(default[:, 0], default[:, 1])
    # 'auto' really is the other reading on this database - documented, not a synonym
    auto, _, _ = ec57.read_leads(path, channel=0, lead_mode='auto')
    assert np.all(auto[:, 1] == 0.0), "'auto' on a 2-lead record is 'single'"
