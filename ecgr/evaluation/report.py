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
