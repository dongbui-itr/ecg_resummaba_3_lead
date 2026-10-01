"""Rhythm EC57: the epicmp wrapper, annotation round trip, Gross-line parsing.

Skipped when the real data is not present, same style as tests/test_ec57.py - these tests
check the annotation plumbing and the report parser without needing mitdb/afdb/rhythm_eval
on disk.
"""
import os
import tempfile

import numpy as np
import pytest
import wfdb

from ecgr import config
from ecgr.evaluation import epicmp, report
from ecgr.rhythm import config as rc
from ecgr.rhythm import ec57, wfdb_ann
from ecgr.rhythm.labels import reference_episodes

MITDB = os.path.join(config.PHYSIONET_DIR, 'mitdb')
AFDB = os.path.join(config.PHYSIONET_DIR, 'afdb')
RHYTHM_EVAL = rc.EVAL_DIR

needs_mitdb = pytest.mark.skipif(not os.path.isdir(MITDB), reason='mitdb not present')
needs_afdb = pytest.mark.skipif(not os.path.isdir(AFDB), reason='afdb not present')
needs_rhythm_eval = pytest.mark.skipif(not os.path.isdir(RHYTHM_EVAL),
                                       reason='rhythm_eval holdout not present')
needs_wfdb_tools = pytest.mark.skipif(not epicmp.have_wfdb_tools(),
                                      reason='epicmp / sumstats not on PATH')


# ---------------------------------------------------------------------------
# paced-record exclusion
# ---------------------------------------------------------------------------

def test_excluded_records_default_and_flag(tmp_path, monkeypatch):
    from ecgr.rhythm import ec57
    assert ec57.excluded_records('mitdb') == {'102', '104', '107', '217'}
    assert ec57.excluded_records('mitdb', include_excluded=True) == set()
    assert ec57.excluded_records('afdb') == set()
    (tmp_path / 'mitdb').mkdir()
    for name in ('100', '102', '104', '201'):
        (tmp_path / 'mitdb' / f"{name}.dat").touch()
    monkeypatch.setattr(config, 'PHYSIONET_DIR', str(tmp_path))
    assert ec57.physionet_records('mitdb')[1] == ['100', '201']
    assert ec57.physionet_records('mitdb', include_excluded=True)[1] == ['100', '102', '104',
                                                                          '201']


# ---------------------------------------------------------------------------
# write_episode_annotations round-trips through wfdb.rdann
# ---------------------------------------------------------------------------

def test_write_episode_annotations_round_trips():
    episodes = [
        {'rhythm': 'SINUS', 'start': 0, 'stop': 3, 'prob': 1.0},
        {'rhythm': 'AFIB', 'start': 3, 'stop': 8, 'prob': 0.9},
        {'rhythm': 'NOISE', 'start': 8, 'stop': 9, 'prob': 0.6},
        {'rhythm': 'SVT', 'start': 9, 'stop': 12, 'prob': 0.8},
    ]
    fs = 250
    with tempfile.TemporaryDirectory() as tmp:
        n = wfdb_ann.write_episode_annotations(episodes, 'rec01', tmp, 'rhi', fs)
        assert n == 3, "NOISE is skipped, the other three episodes are written"

        ann = wfdb.rdann(os.path.join(tmp, 'rec01'), 'rhi')
        assert list(ann.sample) == [0, 3 * fs, 9 * fs]
        assert list(ann.symbol) == ['+', '+', '+']
        codes = [a.split('\x00')[0] for a in ann.aux_note]
        assert codes == ['(N', '(AFIB', '(SVTA']
        assert ann.fs == fs


def test_write_episode_annotations_skips_all_noise():
    with tempfile.TemporaryDirectory() as tmp:
        n = wfdb_ann.write_episode_annotations(
            [{'rhythm': 'NOISE', 'start': 0, 'stop': 10, 'prob': 0.5}], 'rec01', tmp, 'rhi', 250)
        assert n == 0
        assert not os.path.exists(os.path.join(tmp, 'rec01.rhi'))


def test_avb2_avb3_use_explicit_non_standard_codes():
    for name, code in (('AVB2', '(AVB2'), ('AVB3', '(AVB3')):
        assert wfdb_ann.RHYTHM_AUX[name] == code


# ---------------------------------------------------------------------------
# per-class '(AFIB vs (N' files - what epicmp -A actually scores
# ---------------------------------------------------------------------------

def test_class_codes_spell_only_the_target_as_afib():
    episodes = [
        {'rhythm': 'SINUS', 'start': 0, 'stop': 2},
        {'rhythm': 'SVT', 'start': 2, 'stop': 4},
        {'rhythm': 'NOISE', 'start': 4, 'stop': 5},
        {'rhythm': 'SVT', 'start': 5, 'stop': 7},
        {'rhythm': 'AFL', 'start': 7, 'stop': 9},
        {'rhythm': 'AFIB', 'start': 9, 'stop': 12},         # past n_seconds: clipped
    ]
    svt = wfdb_ann.class_codes(episodes, 'SVT', 10).tolist()
    assert svt == ['(N', '(N', '(AFIB', '(AFIB', '(N', '(AFIB', '(AFIB', '(N', '(N', '(N']
    af = wfdb_ann.class_codes(episodes, 'AFIB', 10, keep_afl=True).tolist()
    assert af == ['(N'] * 7 + ['(AFL', '(AFL', '(AFIB']
    # flutter is only spelled in the AFIB reference
    assert '(AFL' not in wfdb_ann.class_codes(episodes, 'AFIB', 10).tolist()
    assert '(AFL' not in wfdb_ann.class_codes(episodes, 'VT', 10, keep_afl=True).tolist()


def test_write_class_annotations_collapses_and_round_trips():
    episodes = [{'rhythm': 'VT', 'start': 3, 'stop': 5}, {'rhythm': 'VT', 'start': 5, 'stop': 6},
                {'rhythm': 'SVT', 'start': 6, 'stop': 8}]
    fs = 250
    with tempfile.TemporaryDirectory() as tmp:
        for cls in rc.EC57_CLASSES:                        # wfdb-python: letters only
            assert all(e.isalpha() for e in rc.class_extensions(cls)), cls
        ref, hyp = rc.class_extensions('VT')
        n = wfdb_ann.write_class_annotations(episodes, 'VT', 'rec01', tmp, hyp, fs, 10)
        assert n == 3                                       # (N at 0, (AFIB at 3, (N at 6
        ann = wfdb.rdann(os.path.join(tmp, 'rec01'), hyp)
        assert list(ann.sample) == [0, 3 * fs, 6 * fs]
        assert [a.split('\x00')[0] for a in ann.aux_note] == ['(N', '(AFIB', '(N']

        # a record with none of the class is still a file: one (N, so it is scored
        n = wfdb_ann.write_class_annotations(episodes, 'AVB3', 'rec02', tmp, 'aavb', fs, 10)
        assert n == 1
        ann = wfdb.rdann(os.path.join(tmp, 'rec02'), 'aavb')
        assert list(ann.sample) == [0] and ann.aux_note[0].startswith('(N')


def test_atr_reference_episodes_maps_codes_strictly(tmp_path):
    fs, n = 360, 360 * 60
    sig = (0.01 * np.sin(2 * np.pi * np.arange(n) / fs)).reshape(-1, 1)
    wfdb.wrsamp('r1', fs=fs, units=['mV'], sig_name=['MLII'], p_signal=sig,
               write_dir=str(tmp_path))
    marks = [(10, '(N'), (10 * fs + 7, '(AFIB'), (20 * fs, '(AFL'), (30 * fs, '(NOD'),
             (40 * fs, '(VT'), (50 * fs, '(BII')]
    wfdb.Annotation(record_name='r1', extension='atr',
                    sample=np.array([s for s, _ in marks]),
                    symbol=['+'] * len(marks), aux_note=[c + '\x00' for _, c in marks],
                    fs=fs).wrann(write_fs=True, write_dir=str(tmp_path))
    episodes, got_fs, seconds = wfdb_ann.atr_reference_episodes(str(tmp_path / 'r1'))
    assert (got_fs, seconds) == (fs, 60)
    # float seconds, exactly where the marks are ...
    assert [(e['rhythm'], e['start'], e['stop']) for e in episodes] == [
        ('SINUS', 10 / fs, 10 + 7 / fs), ('AFIB', 10 + 7 / fs, 20), ('AFL', 20, 30),
        ('SINUS', 30, 40), ('VT', 40, 50), ('AVB2', 50, 60)]
    # ... and on the whole-second grid the per-class codes are what s // fs always gave
    codes = wfdb_ann.class_codes(episodes, 'AFIB', seconds, keep_afl=True, grid_hz=1)
    assert codes.tolist() == ['(N'] * 10 + ['(AFIB'] * 10 + ['(AFL'] * 10 + ['(N'] * 30
    fine = wfdb_ann.class_codes(episodes, 'AFIB', seconds, grid_hz=25)
    assert len(fine) == 60 * 25 and fine[249] == '(N' and fine[250] == '(AFIB'


# ---------------------------------------------------------------------------
# labels.reference_episodes - the synthesized ground truth for rc.EVAL_DIR
# ---------------------------------------------------------------------------

def test_reference_episodes_runs_of_the_same_class():
    labels = np.array([0, 0, 0, 1, 1, rc.IGNORE, 1, 2, 2])   # SINUS, AFIB(+gap), SVT
    episodes = reference_episodes(labels)
    assert [(e['rhythm'], e['start'], e['stop']) for e in episodes] == [
        ('SINUS', 0, 3), ('AFIB', 3, 7), ('SVT', 7, 9)]


def test_reference_episodes_all_ignore_is_empty():
    labels = np.full(10, rc.IGNORE)
    assert reference_episodes(labels) == []


# ---------------------------------------------------------------------------
# report.parse_gross / summarize_episodes - the epicmp Gross line
# ---------------------------------------------------------------------------

GROSS_FIXTURE = """(AF detection)
Record  TPs   FN  TPp   FP  ESe E+P DSe D+P  Ref duration  Test duration
   100    0    0    0    0    -   -   -   -         0.000          0.000
   201    3    0    2    0  100 100  77  44      5:05.964       8:52.794
________________________________________________________________________
Sum       3    0    2    0                    0:05.964       0:52.794
Gross                        98 100  96  73
Average                      96 100  94  70

Summary of results from 2 records
"""


def test_parse_gross_reads_the_four_episode_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'mitdb_rhythm_report_line.out')
        with open(path, 'w') as f:
            f.write(GROSS_FIXTURE)
        row = report.parse_gross(path)
        assert row == {'E_Se': '98', 'E_+P': '100', 'D_Se': '96', 'D_+P': '73', 'records': '2'}


def test_parse_gross_keeps_dash_for_unscorable_classes():
    text = "Gross                         -   -   -   -\nSummary of results from 1 records\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'nstdb_rhythm_report_line.out')
        with open(path, 'w') as f:
            f.write(text)
        row = report.parse_gross(path)
        assert row == {'E_Se': '-', 'E_+P': '-', 'D_Se': '-', 'D_+P': '-', 'records': '1'}


def test_summarize_episodes_one_row_per_db_and_class(tmp_path):
    for db, cls in (('mitdb', 'AFIB'), ('mitdb', 'VT'), ('rhythm_eval', 'AVB2')):
        (tmp_path / db).mkdir(exist_ok=True)
        (tmp_path / db / f'{db}_{cls}_report_line.out').write_text(GROSS_FIXTURE)
    rows = report.summarize_episodes(str(tmp_path))
    assert [(r['db'], r['class']) for r in rows] == [
        ('mitdb', 'AFIB'), ('mitdb', 'VT'), ('rhythm_eval', 'AVB2')]   # db with '_' too
    assert rows[0]['E_Se'] == '98' and rows[0]['E_F1'] == '99.0' and rows[0]['D_F1'] == '82.9'
    assert (tmp_path / 'rhythm_ec57_summary.csv').exists()
    assert report.f1_token('-', '0') == '-' and report.f1_token('0', '0') == '-'


# ---------------------------------------------------------------------------
# epicmp driver - script presence, safety checks
# ---------------------------------------------------------------------------

def test_epicmp_scripts_exist_and_are_shell_quoted():
    for name in (epicmp.SCRIPT_FULL, epicmp.SCRIPT_SHORT):
        path = os.path.join(config.WFDB_SCRIPTS_DIR, name)
        assert os.path.exists(path), path
        text = open(path).read()
        assert '"$DB_PATH"' in text or "'$DB_PATH'" in text


def test_run_epicmp_raises_on_missing_script():
    with pytest.raises(FileNotFoundError):
        epicmp.run_epicmp('nope', '/tmp', '/tmp', 'atr', 'rhi', script='does-not-exist.sh')


@needs_wfdb_tools
def test_run_epicmp_scores_a_synthetic_pair(tmp_path):
    """A minimal synthetic record with matching ref/AI rhythm annotations - full round trip
    through the real epicmp/sumstats binaries, no Physionet data required."""
    fs, n = 250, 250 * 20
    sig = (0.01 * np.sin(2 * np.pi * np.arange(n) / fs)).reshape(-1, 1)
    record_dir = tmp_path / 'src'
    record_dir.mkdir()
    wfdb.wrsamp('rec', fs=fs, units=['mV'], sig_name=['I'], p_signal=sig,
               write_dir=str(record_dir))

    episodes = [{'rhythm': 'SINUS', 'start': 0, 'stop': 10, 'prob': 1.0},
               {'rhythm': 'VT', 'start': 10, 'stop': 20, 'prob': 1.0}]
    ref, hyp = rc.class_extensions('VT')
    wfdb_ann.write_class_annotations(episodes, 'VT', 'rec', str(record_dir), ref, fs, 20)
    wfdb_ann.write_class_annotations(episodes, 'VT', 'rec', str(record_dir), hyp, fs, 20)
    # the same episodes seen through the SVT pair: nothing on either side
    ref_s, hyp_s = rc.class_extensions('SVT')
    wfdb_ann.write_class_annotations(episodes, 'SVT', 'rec', str(record_dir), ref_s, fs, 20)
    wfdb_ann.write_class_annotations(episodes, 'SVT', 'rec', str(record_dir), hyp_s, fs, 20)

    report_root = tmp_path / 'reports'
    path = epicmp.run_epicmp('synthdb', str(record_dir), str(report_root), ref, hyp,
                             script=epicmp.SCRIPT_SHORT, label='VT', quiet=True)
    assert path and path.endswith('synthdb/synthdb_VT_report_line.out')
    assert report.parse_gross(path).get('E_Se') == '100'
    path2 = epicmp.run_epicmp('synthdb', str(record_dir), str(report_root), ref_s, hyp_s,
                              script=epicmp.SCRIPT_SHORT, label='SVT', quiet=True)
    assert os.path.exists(path), "scoring a second class keeps the first class's report"
    assert report.parse_gross(path2).get('E_Se') == '-'
    rows = report.episode_rows(str(report_root))
    assert [(r['class'], r['E_F1']) for r in rows] == [('SVT', '-'), ('VT', '100.0')]


def test_afl_counts_as_af_only_with_x(tmp_path, monkeypatch):
    """Reference: AFIB 10-30 s, AFL 30-50 s. Hypothesis: AFIB 10-50 s. The AFIB called during
    flutter is a false positive under the EC57 default and not one with -x (EC57_AFL_AS_AF)."""
    fs, n = 250, 250 * 60
    sig = (0.01 * np.sin(2 * np.pi * np.arange(n) / fs)).reshape(-1, 1)
    record_dir = tmp_path / 'src'
    record_dir.mkdir()
    wfdb.wrsamp('rec', fs=fs, units=['mV'], sig_name=['I'], p_signal=sig,
               write_dir=str(record_dir))
    ref_eps = [{'rhythm': 'SINUS', 'start': 0, 'stop': 10, 'prob': 1.0},
               {'rhythm': 'AFIB', 'start': 10, 'stop': 30, 'prob': 1.0},
               {'rhythm': 'AFL', 'start': 30, 'stop': 50, 'prob': 1.0},
               {'rhythm': 'SINUS', 'start': 50, 'stop': 60, 'prob': 1.0}]
    hyp_eps = [{'rhythm': 'SINUS', 'start': 0, 'stop': 10, 'prob': 1.0},
               {'rhythm': 'AFIB', 'start': 10, 'stop': 50, 'prob': 1.0},
               {'rhythm': 'SINUS', 'start': 50, 'stop': 60, 'prob': 1.0}]
    ref, hyp = rc.class_extensions('AFIB')
    wfdb_ann.write_class_annotations(ref_eps, 'AFIB', 'rec', str(record_dir), ref, fs, 60,
                                     keep_afl=True)
    wfdb_ann.write_class_annotations(hyp_eps, 'AFIB', 'rec', str(record_dir), hyp, fs, 60)

    def duration_ppv(flag):
        monkeypatch.setattr(rc, 'EC57_AFL_AS_AF', flag)
        path = epicmp.run_epicmp('synthdb', str(record_dir), str(tmp_path / f'rep{flag}'), ref,
                                 hyp, script=epicmp.SCRIPT_SHORT, label='AFIB', quiet=True,
                                 extra_flags=ec57.epicmp_flags('AFIB'))
        return report.parse_gross(path)
    strict, grouped = duration_ppv(False), duration_ppv(True)
    assert strict['D_Se'] == grouped['D_Se'] == '100'
    assert int(strict['D_+P']) < 60 and grouped['D_+P'] == '100'
    assert ec57.epicmp_flags('VT') == ()
