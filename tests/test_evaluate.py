"""The standalone evaluation entry point: argument handling, preflight, target marking.

The scoring itself is covered by tests/test_ec57.py; what is worth pinning here is that the
script refuses to start on a missing prerequisite (the whole reason it exists is that those
failures otherwise appear as an empty report an hour later) and that the acceptance targets
are applied to the right metrics.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluate  # noqa: E402


def test_targets_cover_the_acceptance_criteria():
    """mitdb Q >= 99.95 both ways, mitdb S Se > 43 / +P > 80, v4 S Se > 88 / +P > 92."""
    assert evaluate.TARGETS['mitdb'] == {'Q_Se': 99.95, 'Q_+P': 99.95,
                                         'S_Se': 43.0, 'S_+P': 80.0}
    assert evaluate.TARGETS['dataset-v4-beat'] == {'S_Se': 88.0, 'S_+P': 92.0}
    # a database with no target must not be marked, and every target names a real metric
    assert 'nstdb' not in evaluate.TARGETS
    for db, targets in evaluate.TARGETS.items():
        assert set(targets) <= set(evaluate.METRICS)


def test_parse_args_defaults_and_ensemble():
    args = evaluate.parse_args(['--checkpoint', 'a.keras'])
    assert args.checkpoint == ['a.keras']
    assert args.lead_mode == 'native' and args.min_run == 1 and args.s_boost == 1.0
    assert not args.bxb_only and not args.skip_v4
    many = evaluate.parse_args(['--checkpoint', 'a.keras', 'b.keras', '--lead-mode', 'native',
                                '--min-run', '2'])
    assert many.checkpoint == ['a.keras', 'b.keras'] and many.min_run == 2


def test_preflight_rejects_a_missing_checkpoint(tmp_path, monkeypatch, capsys):
    from ecgr import config
    monkeypatch.setattr(config, 'PHYSIONET_DIR', str(tmp_path))
    args = evaluate.parse_args(['--checkpoint', str(tmp_path / 'nope.keras'),
                                '--dbs', '--skip-v4', '--bxb-only'])
    with pytest.raises(SystemExit) as exc:
        evaluate.preflight(args)
    assert exc.value.code == 2
    assert 'checkpoint not found' in capsys.readouterr().err


def test_preflight_names_the_missing_database(tmp_path, monkeypatch, capsys):
    from ecgr import config
    monkeypatch.setattr(config, 'PHYSIONET_DIR', str(tmp_path))
    ckpt = tmp_path / 'm.keras'
    ckpt.write_bytes(b'x')
    args = evaluate.parse_args(['--checkpoint', str(ckpt), '--dbs', 'mitdb', '--skip-v4',
                                '--bxb-only'])
    with pytest.raises(SystemExit):
        evaluate.preflight(args)
    err = capsys.readouterr().err
    assert 'mitdb' in err and 'ECGR_PHYSIONET_DIR' in err


def test_preflight_passes_when_everything_is_there(tmp_path, monkeypatch):
    from ecgr import config
    (tmp_path / 'mitdb').mkdir()
    monkeypatch.setattr(config, 'PHYSIONET_DIR', str(tmp_path))
    ckpt = tmp_path / 'm.keras'
    ckpt.write_bytes(b'x')
    args = evaluate.parse_args(['--checkpoint', str(ckpt), '--dbs', 'mitdb', '--skip-v4',
                                '--bxb-only'])
    assert evaluate.preflight(args) == ['mitdb']


def test_table_marks_targets_and_lists_what_is_below(capsys):
    rows = [
        {'db': 'mitdb', 'records': '44', 'Q_Se': '99.96', 'Q_+P': '99.90', 'V_Se': '95.0',
         'V_+P': '96.0', 'S_Se': '56.9', 'S_+P': '65.6'},
        {'db': 'nstdb', 'records': '12', 'Q_Se': '96.4', 'Q_+P': '84.0', 'V_Se': '83.0',
         'V_+P': '67.0', 'S_Se': '78.0', 'S_+P': '24.0'},
    ]
    evaluate.print_table(rows)
    out = capsys.readouterr().out
    assert '99.96*' in out, "a metric at or above its target must be starred"
    assert '99.90!' in out, "a metric below its target must be flagged"
    assert '56.90' in out and '65.60!' in out
    assert 'mitdb            Q_+P' in out and 'mitdb            S_+P' in out
    # nstdb has no targets: none of its cells may carry a mark
    nstdb_line = next(l for l in out.splitlines() if l.startswith('nstdb'))
    assert '*' not in nstdb_line and '!' not in nstdb_line


def test_table_shows_deltas_against_a_baseline(capsys):
    rows = [{'db': 'mitdb', 'records': '44', 'Q_Se': '99.96', 'Q_+P': '99.90',
             'V_Se': '95.0', 'V_+P': '96.0', 'S_Se': '56.9', 'S_+P': '65.6'}]
    baseline = {'mitdb': {'Q_Se': '99.90', 'Q_+P': '99.86', 'V_Se': '95.6', 'V_+P': '94.1',
                          'S_Se': '45.8', 'S_+P': '61.2'}}
    evaluate.print_table(rows, baseline)
    out = capsys.readouterr().out
    assert '+0.06' in out and '+11.10' in out, "deltas against the baseline must be shown"
    assert '-0.60' in out, "a metric that went down must show a negative delta"


def test_dashes_survive_a_database_without_that_class(capsys):
    """ahadb has no reference S beats; bxb writes '-', which must not become 0.00."""
    evaluate.print_table([{'db': 'ahadb', 'records': '79', 'Q_Se': '99.9', 'Q_+P': '99.8',
                           'V_Se': '89.0', 'V_+P': '98.0', 'S_Se': '-', 'S_+P': '-'}])
    out = capsys.readouterr().out
    assert '0.00' not in out
    assert out.count('-') >= 2
