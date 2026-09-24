"""The non-regression check: what counts as a drop, what is skipped, what the verdict is."""
import csv
import os

import pytest

from ecgr import config
from ecgr.evaluation import report

BASE = {'mitdb': {'db': 'mitdb', 'Q_Se': '99.90', 'Q_+P': '99.86', 'V_Se': '95.64',
                  'V_+P': '94.10', 'S_Se': '45.79', 'S_+P': '61.17'},
        'ahadb': {'db': 'ahadb', 'Q_Se': '99.93', 'Q_+P': '99.74', 'V_Se': '90.71',
                  'V_+P': '98.23', 'S_Se': '-', 'S_+P': '0.00'},
        'nstdb': {'db': 'nstdb', 'Q_Se': '96.10', 'Q_+P': '86.66', 'V_Se': '85.85',
                  'V_+P': '65.12', 'S_Se': '80.62', 'S_+P': '34.93'}}


def _write(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=report.COLUMNS, extrasaction='ignore')
        w.writeheader()
        for r in rows.values():
            w.writerow({**{'records': '1'}, **r})


def test_a_drop_beyond_tolerance_is_a_regression_and_within_it_is_not():
    new = {db: dict(r) for db, r in BASE.items()}
    new['mitdb']['S_Se'] = '45.75'              # -0.04: noise
    new['mitdb']['V_+P'] = '93.50'              # -0.60: a regression
    new['nstdb']['Q_Se'] = '97.00'              # up
    drops, compared = report.regressions(new, BASE, tolerance=0.1)
    assert [(d['db'], d['metric']) for d in drops] == [('mitdb', 'V_+P')]
    assert drops[0]['delta'] == -0.6
    assert ('nstdb', 'Q_Se') in compared and ('mitdb', 'S_Se') in compared


def test_cells_the_baseline_cannot_score_are_skipped():
    """ahadb has no S beats: bxb writes '-' for S_Se and 0.00 for S_+P. Neither is a number
    to regress against - a new model scoring S_+P '-' or 0.00 there is not a drop."""
    new = {db: dict(r) for db, r in BASE.items()}
    new['ahadb']['S_+P'] = '-'
    drops, compared = report.regressions(new, BASE)
    assert not drops
    assert ('ahadb', 'S_+P') not in compared and ('ahadb', 'S_Se') not in compared
    assert ('ahadb', 'V_Se') in compared


def test_databases_missing_on_either_side_are_ignored_and_dbs_filters():
    new = {'mitdb': dict(BASE['mitdb'])}
    new['mitdb']['Q_Se'] = '90.00'
    drops, compared = report.regressions(new, BASE)
    assert len(drops) == 1 and all(db == 'mitdb' for db, _ in compared)
    drops, _ = report.regressions(new, BASE, dbs=['nstdb'])
    assert not drops


def test_check_no_regression_writes_the_verdict(tmp_path, capsys):
    new = {db: dict(r) for db, r in BASE.items()}
    new['nstdb']['S_+P'] = '30.00'
    _write(new, tmp_path / 'new.csv')
    _write(BASE, tmp_path / 'base.csv')
    drops = report.check_no_regression(str(tmp_path / 'new.csv'), str(tmp_path / 'base.csv'),
                                       out_json=str(tmp_path / 'verdict.json'))
    assert len(drops) == 1 and drops[0]['metric'] == 'S_+P'
    out = capsys.readouterr().out
    assert 'fell below the baseline' in out and 'nstdb' in out
    import json
    verdict = json.load(open(tmp_path / 'verdict.json'))
    assert verdict['passed'] is False and verdict['tolerance_pp'] == config.REGRESSION_TOLERANCE_PP

    _write(BASE, tmp_path / 'same.csv')
    assert report.check_no_regression(str(tmp_path / 'same.csv'), str(tmp_path / 'base.csv')) == []
    assert 'no regression' in capsys.readouterr().out


def test_every_size_has_a_shipped_baseline():
    for size, ref in config.BASELINE_FOR.items():
        path = config.baseline_summary(size)
        assert path and os.path.exists(path), f"{size} -> {ref}: {path}"
        rows = report.load_summary(path)
        assert {'mitdb', 'nstdb', 'escdb', 'ahadb', 'afdb', 'dataset-v4-beat'} <= set(rows)


def test_regress_cli_exit_status(tmp_path):
    from ecgr import cli
    new = {db: dict(r) for db, r in BASE.items()}
    _write(new, tmp_path / 'new.csv')
    _write(BASE, tmp_path / 'base.csv')
    assert cli.main(['regress', '--summary', str(tmp_path / 'new.csv'),
                     '--baseline', str(tmp_path / 'base.csv')]) == 0
    new['mitdb']['Q_Se'] = '99.00'
    _write(new, tmp_path / 'new.csv')
    assert cli.main(['regress', '--summary', str(tmp_path / 'new.csv'),
                     '--baseline', str(tmp_path / 'base.csv')]) == 1
    assert cli.main(['regress', '--summary', str(tmp_path / 'nope.csv'),
                     '--baseline', str(tmp_path / 'base.csv')]) == 2
