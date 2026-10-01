"""Turning bxb's and epicmp's text reports into one table.

bxb writes a per-database report; `summarize` aggregates the Gross lines (all beats pooled
over the database - the EC57 headline numbers) into one CSV, and `compare` prints several
models side by side. `summarize_episodes` does the same for epicmp's rhythm-episode reports
(evaluation/epicmp.py), whose Gross line carries four numbers instead of six.
"""
import csv
import glob
import os

from .. import config

METRICS = ['Q_Se', 'Q_+P', 'V_Se', 'V_+P', 'S_Se', 'S_+P']
COLUMNS = ['db', 'records'] + METRICS + ['total_QRS', 'total_VEB', 'total_SVEB']

# epicmp's Gross line: episode Se/+P, duration Se/+P of the ONE rhythm spelled '(AFIB' - so
# one report per (database, class), the class being what the annotation pair spelled that way
# (rhythm/wfdb_ann.write_class_annotations). F1 is derived from the two Gross tokens.
EPISODE_METRICS = ['E_Se', 'E_+P', 'D_Se', 'D_+P']
EPISODE_COLUMNS = ['db', 'class', 'records', 'E_Se', 'E_+P', 'E_F1', 'D_Se', 'D_+P', 'D_F1']
EPISODE_REPORT_SUFFIX = '_report_line.out'


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


def parse_gross(path, metrics=EPISODE_METRICS):
    """Gross and Summary lines of one epicmp/sumstats <db>_rhythm_report_line.out.

    Same convention as parse_report: '-' is kept as written for a class the database has no
    reference episodes of, rather than turned into a misleading 0.
    """
    row = {}
    with open(path) as f:
        for line in f:
            t = line.split()
            if line.startswith('Gross') and len(t) >= 1 + len(metrics):
                row.update(zip(metrics, t[1:1 + len(metrics)]))
            elif line.startswith('Summary of results from'):
                row['records'] = t[4]
    return row


def f1_token(se, pp):
    """'93.8'-style F1 of two Gross tokens, '-' when either is unscorable."""
    se, pp = _num(se), _num(pp)
    if se is None or pp is None or se + pp == 0:
        return '-'
    return f"{2 * se * pp / (se + pp):.1f}"


def episode_row(path):
    """(db, class) + Gross metrics + F1 of one <db>/<db>_<class>_report_line.out."""
    db = os.path.basename(os.path.dirname(path))
    name = os.path.basename(path)[:-len(EPISODE_REPORT_SUFFIX)]
    cls = name[len(db) + 1:] if name.startswith(db + '_') else name
    row = {'db': db, 'class': cls, **parse_gross(path)}
    row['E_F1'] = f1_token(row.get('E_Se'), row.get('E_+P'))
    row['D_F1'] = f1_token(row.get('D_Se'), row.get('D_+P'))
    return row


def episode_rows(ec57_dir, pattern='*' + EPISODE_REPORT_SUFFIX):
    return [episode_row(p) for p in sorted(glob.glob(os.path.join(ec57_dir, '*', pattern)))]


def print_episode_table(rows, classes=None, dbs=None, target=None):
    """The reference product's layout: class x (Duration | Episode) rows, one Se/PPV/F1
    triple per database. `target` = {(db, class): {'D': (se, pp), 'E': (se, pp)}} adds the
    reference product's numbers as an extra row per class."""
    by = {(r['db'], r['class']): r for r in rows}
    classes = classes or sorted({r['class'] for r in rows})
    dbs = dbs or sorted({r['db'] for r in rows})
    cell = lambda se, pp: f"{str(se):>5s} {str(pp):>5s} {f1_token(se, pp):>6s}"  # noqa: E731
    print(f"{'class':6s} {'':9s} | " + " | ".join(f"{db:^18s}" for db in dbs))
    print(f"{'':6s} {'':9s} | " + " | ".join(f"{'Se':>5s} {'PPV':>5s} {'F1':>6s}" for _ in dbs))
    for cls in classes:
        for kind, key in (('Duration', 'D'), ('Episode', 'E')):
            cells = []
            for db in dbs:
                r = by.get((db, cls))
                cells.append(cell(r[f'{key}_Se'], r[f'{key}_+P']) if r else f"{'-':>18s}")
            print(f"{cls:6s} {kind:9s} | " + " | ".join(cells))
            if target:
                cells = []
                for db in dbs:
                    t = (target.get((db, cls)) or {}).get(key)
                    cells.append(cell(*t) if t else f"{'':>18s}")
                print(f"{'':6s} {'  target':9s} | " + " | ".join(cells))


def write_episode_xlsx(rows, path):
    """Same rows as the CSV in one sheet - only when openpyxl is installed (the reference
    project's ec57_results_to_xlsx.py depends on it; this project does not)."""
    try:
        from openpyxl import Workbook
    except ImportError:
        return None
    wb = Workbook()
    ws = wb.active
    ws.title = 'rhythm_ec57'
    ws.append(EPISODE_COLUMNS)
    for r in rows:
        ws.append([_num(r.get(c)) if _num(r.get(c)) is not None else r.get(c, '-')
                   for c in EPISODE_COLUMNS])
    wb.save(path)
    return path


def summarize_episodes(ec57_dir, classes=None, dbs=None, target=None, quiet=False):
    """Aggregate every per-class epicmp report under `ec57_dir` into rhythm_ec57_summary.csv
    (one row per database x class) and print the class x Duration/Episode table."""
    rows = episode_rows(ec57_dir)
    if not rows:
        print(f"no epicmp reports under {ec57_dir}")
        return []

    path = os.path.join(ec57_dir, 'rhythm_ec57_summary.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    write_episode_xlsx(rows, path[:-4] + '.xlsx')
    if not quiet:
        print(f"\nrhythm EC57 summary ({len(rows)} database x class rows) -> {path}")
        print_episode_table(rows, classes=classes, dbs=dbs, target=target)
        print("-  = no reference episodes of that class in this database")
    return rows


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
