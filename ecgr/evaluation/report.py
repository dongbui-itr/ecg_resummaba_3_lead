"""Turning bxb's text reports into one table.

bxb writes a per-database report; `summarize` aggregates the Gross lines (all beats pooled
over the database - the EC57 headline numbers) into one CSV, and `compare` prints several
models side by side.
"""
import csv
import glob
import os

from .. import config

METRICS = ['Q_Se', 'Q_+P', 'V_Se', 'V_+P', 'S_Se', 'S_+P']
COLUMNS = ['db', 'records'] + METRICS + ['total_QRS', 'total_VEB', 'total_SVEB']


def parse_report(path):
    """Gross and Total lines of one <db>_QRS_report_line.out.

    '-' is kept as written for metrics a database cannot score (S on ahadb, V and S on afdb):
    turning it into 0 would read as "0% sensitivity" instead of "no reference beats".
    """
    row = {}
    with open(path) as f:
        for line in f:
            t = line.split()
            if line.startswith('Gross') and len(t) >= 7:
                row.update(zip(METRICS, t[1:7]))
            elif line.startswith('Total QRS complexes:'):
                row['total_QRS'], row['total_VEB'], row['total_SVEB'] = t[3], t[6], t[9]
            elif line.startswith('Summary of results from'):
                row['records'] = t[4]
    return row


def summarize(ec57_dir):
    """Aggregate every report under `ec57_dir` into ec57_summary.csv; returns the rows."""
    reports = sorted(glob.glob(os.path.join(ec57_dir, '*', '*_QRS_report_line.out')))
    rows = [{'db': os.path.basename(os.path.dirname(p)), **parse_report(p)} for p in reports]
    if not rows:
        print(f"no bxb reports under {ec57_dir}")
        return []

    path = os.path.join(ec57_dir, 'ec57_summary.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nEC57 summary ({len(rows)} databases) -> {path}")
    for r in rows:
        print("  " + "  ".join(f"{m}={r.get(m, '?')}" for m in ['db'] + METRICS)
              .replace('db=', ''))
    return rows


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compare(tags, ec57_root=None):
    """Print several models' summaries side by side, per database and class.

    Adds FP/1k - false positives per 1000 reference beats of that class - because positive
    predictivity cannot be compared across databases: S prevalence runs from 0.14% (escdb) to
    10.8% (the portal set), and at low prevalence +P is set almost entirely by the false
    positive rate. FP/1k is the quantity that transfers.
    """
    root = ec57_root or config.EC57_DIR
    data = {}
    for tag in tags:
        path = os.path.join(root, tag, 'ec57_summary.csv')
        if not os.path.exists(path):
            print(f"{tag}: no ec57_summary.csv ({path})")
            continue
        with open(path) as f:
            data[tag] = {r['db']: r for r in csv.DictReader(f)}
    if not data:
        return

    dbs = sorted({db for rows in data.values() for db in rows})
    for cls, support_key in (('Q', 'total_QRS'), ('V', 'total_VEB'), ('S', 'total_SVEB')):
        print(f"\n{'=' * 78}\nclass {cls}\n{'=' * 78}")
        header = f"{'database':18s} {'support':>9s} | " + " | ".join(f"{t:>26s}" for t in data)
        print(header)
        print(f"{'':18s} {'':>9s} | " + " | ".join(
            f"{'Se':>7s} {'+P':>6s} {'F1':>6s} {'FP/1k':>5s}" for _ in data))
        for db in dbs:
            cells, support = [], None
            for tag in data:
                row = data[tag].get(db)
                n = _num(row[support_key]) if row else None
                se, pp = (_num(row[f'{cls}_Se']), _num(row[f'{cls}_+P'])) if row else (None, None)
                if not n or se is None or pp is None or se + pp == 0 or pp == 0:
                    cells.append(f"{'-':>26s}")
                    continue
                support = int(n)
                tp = se / 100 * n
                fp = tp * (100 - pp) / pp
                cells.append(f"{se:7.2f} {pp:6.2f} {2 * se * pp / (se + pp):6.2f} "
                             f"{1000 * fp / n:5.0f}")
            if support:
                print(f"{db:18s} {support:>9,} | " + " | ".join(cells))
    print("\n-  = this database has no reference beats of that class")


# ---------------------------------------------------------------------------
# Non-regression against a baseline summary
# ---------------------------------------------------------------------------

def load_summary(path):
    """{db: row} of one ec57_summary.csv."""
    with open(path) as f:
        return {row['db']: row for row in csv.DictReader(f)}


def regressions(summary, baseline, tolerance=None, metrics=METRICS, dbs=None):
    """Every (db, metric) where `summary` fell more than `tolerance` below `baseline`.

    Both are {db: row} as load_summary returns. Cells the baseline cannot score ('-', no
    reference beats of that class, or a 0.00 +P against a '-' Se) are skipped, as are
    databases missing from either side. Returns a list of dicts sorted by the size of the
    drop, largest first, plus the list of cells that were compared.
    """
    tol = config.REGRESSION_TOLERANCE_PP if tolerance is None else float(tolerance)
    drops, compared = [], []
    for db in sorted(set(summary) & set(baseline)):
        if dbs and db not in dbs:
            continue
        for metric in metrics:
            new, old = _num(summary[db].get(metric)), _num(baseline[db].get(metric))
            if new is None or old is None:
                continue
            se_key = metric.replace('+P', 'Se')
            if _num(baseline[db].get(se_key)) is None:      # the class has no reference beats
                continue
            compared.append((db, metric))
            if new < old - tol:
                drops.append({'db': db, 'metric': metric, 'new': new, 'baseline': old,
                              'delta': round(new - old, 2)})
    return sorted(drops, key=lambda d: d['delta']), compared


def print_regression(summary, baseline, tolerance=None, label='baseline', dbs=None):
    """Side-by-side table with deltas, and the verdict. Returns the list of regressions."""
    tol = config.REGRESSION_TOLERANCE_PP if tolerance is None else float(tolerance)
    drops, compared = regressions(summary, baseline, tol, dbs=dbs)
    bad = {(d['db'], d['metric']) for d in drops}
    width = max([len(db) for db in summary] + [8])
    print(f"\n{'database':{width}s} " + ' '.join(f"{m:>15s}" for m in METRICS)
          + f"      (new / {label}, delta; ! = below baseline - {tol} pp)")
    for db in sorted(summary):
        cells = []
        for metric in METRICS:
            new, old = _num(summary[db].get(metric)), _num(baseline.get(db, {}).get(metric))
            if new is None:
                cells.append(f"{'-':>15s}")
            elif old is None:
                cells.append(f"{new:6.2f}{'':>9s}")
            else:
                mark = '!' if (db, metric) in bad else ' '
                cells.append(f"{new:6.2f}{new - old:+6.2f}{mark}  ")
        print(f"{db:{width}s} " + ' '.join(cells))
    if drops:
        print(f"\n{len(drops)} of {len(compared)} cells fell below the baseline by more than "
              f"{tol} pp:")
        for d in drops:
            print(f"  {d['db']:16s} {d['metric']:5s} {d['new']:6.2f} vs {d['baseline']:6.2f} "
                  f"({d['delta']:+.2f})")
    else:
        print(f"\nno regression: every one of {len(compared)} comparable cells is within "
              f"{tol} pp of the baseline or above it")
    return drops


def check_no_regression(summary_csv, baseline_csv, tolerance=None, dbs=None, out_json=None):
    """Diff a summary against a baseline; write the verdict as JSON; return the drops."""
    import json
    summary, baseline = load_summary(summary_csv), load_summary(baseline_csv)
    drops = print_regression(summary, baseline, tolerance,
                             label=os.path.splitext(os.path.basename(baseline_csv))[0], dbs=dbs)
    if out_json:
        with open(out_json, 'w') as f:
            json.dump({'summary': summary_csv, 'baseline': baseline_csv,
                       'tolerance_pp': (config.REGRESSION_TOLERANCE_PP if tolerance is None
                                        else tolerance),
                       'passed': not drops, 'regressions': drops}, f, indent=2)
    return drops
