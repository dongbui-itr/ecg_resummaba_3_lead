#!/usr/bin/env python
"""Combine several ec57_summary.csv files into one comparison table.

    python compare_ec57.py

Edit the CONFIG block below and run - no command-line arguments, same reasoning as
evaluate.py: which directory got summarised is part of the record, not a shell line that
scrolls out of history.

ROOT is walked recursively for files named exactly `ec57_summary.csv` - the file
evaluate.py / `ecgr ec57` writes one of per checkpoint, e.g.

    eval_results/resumamba_30k/ec57_summary.csv
    eval_results/resumamba_2m/ec57_summary.csv

Each file's rows (one per database) are stacked into one table with a `checkpoint` column
prepended (the summary file's parent directory name), so the result can be sorted/filtered by
database to compare checkpoints against each other. The three `total_*` beat-count columns
(total_QRS, total_VEB, total_SVEB) are dropped - they are counts of a database, not a score,
and duplicating them per checkpoint gives nothing to compare.
"""
import os
import sys

import pandas as pd

# ===========================================================================
# CONFIG - edit this block
# ===========================================================================

ROOT = 'eval_results'            # directory to search recursively
OUT = 'ec57_compare.csv'         # combined CSV to write

# ===========================================================================
# End of CONFIG
# ===========================================================================

DROP_COLUMNS = ['total_QRS', 'total_VEB', 'total_SVEB']
SUMMARY_NAME = 'ec57_summary.csv'


def find_summaries(root):
    """Every SUMMARY_NAME under root, depth-first, as absolute paths."""
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if SUMMARY_NAME in filenames:
            paths.append(os.path.join(dirpath, SUMMARY_NAME))
    return sorted(paths)


def checkpoint_label(path, root):
    """The summary's parent directory name, e.g. `resumamba_2m` for the layout above.

    Falls back to the path relative to root when two summaries share a parent directory
    name (nested layouts), so rows never collide under one label.
    """
    return os.path.basename(os.path.dirname(path)) or os.path.relpath(path, root)


def load_summaries(paths, root):
    frames = []
    labels_seen = {}
    for path in paths:
        label = checkpoint_label(path, root)
        if labels_seen.get(label, path) != path:
            label = os.path.relpath(os.path.dirname(path), root)
        labels_seen[label] = path

        df = pd.read_csv(path)
        df = df.drop(columns=DROP_COLUMNS, errors='ignore')
        df.insert(0, 'checkpoint', label)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def print_comparison(table):
    """Grouped by db so every checkpoint's row for that database sits together."""
    metric_cols = [c for c in table.columns if c not in ('checkpoint', 'db', 'records')]
    width = max(table['checkpoint'].map(len).max(), len('checkpoint'))

    for db, group in table.groupby('db', sort=True):
        print(f"\n{db}")
        print('-' * len(db))
        head = f"  {'checkpoint':{width}s} {'records':>7s} " + \
               ' '.join(f"{m:>7s}" for m in metric_cols)
        print(head)
        for _, row in group.iterrows():
            cells = ' '.join(f"{str(row[m]):>7s}" for m in metric_cols)
            print(f"  {row['checkpoint']:{width}s} {str(row['records']):>7s} {cells}")


def main():
    if not os.path.isdir(ROOT):
        print(f"error: no such directory: {ROOT}", file=sys.stderr)
        return 2

    paths = find_summaries(ROOT)
    if not paths:
        print(f"error: no {SUMMARY_NAME} found under {ROOT}", file=sys.stderr)
        return 1

    table = load_summaries(paths, ROOT)
    table.to_csv(OUT, index=False)

    print(f"found {len(paths)} {SUMMARY_NAME} file(s) under {ROOT}:")
    for path in paths:
        print(f"  {path}")
    print_comparison(table)
    print(f"\ncombined table written to {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
