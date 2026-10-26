#!/usr/bin/env python
"""Rebuild the portal rhythm inventories from the records themselves.

    python scripts/make_rhythm_inventories.py [--out DIR] [--sources a b ...]

Some machines receive the portal exports as records only - no .xlsx. Nothing is lost: every
column config.SOURCES reads out of the spreadsheet is also written into the data.

    dataset-rhythm/*, dataset-sinus/*   each record's .hea carries
                                            # eventType: AVB3
                                            # channel: 1
                                            # eventStartSample: 7502
                                            # eventStopSample: 8168
    dataset-1 / -4 / -5                 dataset_info_full_all.csv carries study_id,
                                        event_id, event_type, channel and the 10 s strip
                                        as start_sample / stop_sample

One trap, and it is in the export rather than here: four of the sources misspell the key as
`eventTye` (no 'p'). Reading only `eventType` makes dataset-2/3-vt-svt-avb2-avb3 and
dataset-2/3-tachy-brady-pause look untyped, which quietly removes EVERY AVB event from the
corpus - they are the only AVB source on this machine. Both spellings are accepted.

The output columns are named to match what config.SOURCES already asks each source for, so
nothing downstream changes.
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HEADER_FIELD = re.compile(
    r'^#\s*(eventTy[p]?e|eventStartSample|eventStopSample|channel):\s*(.*)$', re.M)
# dataset-1/4/5 are described by the portal's own index rather than by their headers
FROM_INDEX = {'dataset-1', 'dataset-4', 'dataset-5'}


def header_rows(root, spec):
    """One row per event under <root>/<study>/<event>/*.hea."""
    rows = []
    if not os.path.isdir(root):
        return rows
    for study in sorted(os.scandir(root), key=lambda e: e.name):
        if not study.is_dir():
            continue
        for event in os.scandir(study.path):
            if not event.is_dir():
                continue
            head = next((f.path for f in os.scandir(event.path)
                         if f.name.endswith('.hea')), None)
            if head is None:
                continue
            with open(head, errors='ignore') as fh:
                kv = dict(HEADER_FIELD.findall(fh.read()))
            etype = kv.get('eventType') or kv.get('eventTye')
            if not etype:
                continue
            row = {spec['study']: study.name, spec['event']: event.name,
                   spec['type']: etype.strip(), 'channel': kv.get('channel', '')}
            for key, col in (('span', spec.get('span')), ('strip', spec.get('strip'))):
                if not col:
                    continue
                a, b = kv.get('eventStartSample'), kv.get('eventStopSample')
                if a is not None and b is not None:
                    row[col[0]], row[col[1]] = a, b
            rows.append(row)
    return rows


def index_rows(name, spec, index_csv):
    """dataset-1/4/5 out of the portal index, mapped onto the columns the spec reads."""
    import pandas as pd
    df = pd.read_csv(index_csv, low_memory=False,
                     dtype={'study_id': str, 'event_id': str, 'dataset': str})
    df = df[df['dataset'] == name]
    rows = []
    for r in df.to_dict('records'):
        row = {spec['study']: r['study_id'], spec['event']: r['event_id'],
               spec['type']: r['event_type'], 'channel': r.get('channel', '')}
        if spec.get('strip'):
            row[spec['strip'][0]], row[spec['strip'][1]] = r['start_sample'], r['stop_sample']
        if spec.get('span'):
            row[spec['span'][0]], row[spec['span'][1]] = r['start_sample'], r['stop_sample']
        rows.append(row)
    return rows


def main():
    from ecgr.rhythm import config as rc

    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--sources', nargs='*', default=None)
    args = ap.parse_args()
    out = args.out or rc.GENERATED_INVENTORY_DIR
    os.makedirs(out, exist_ok=True)

    names = args.sources or list(rc.SOURCES)
    total = 0
    for name in names:
        spec = rc.SOURCES[name]
        if name in FROM_INDEX:
            rows = index_rows(name, spec, rc.PORTAL_EVENT_INDEX)
            how = 'portal index'
        else:
            rows = header_rows(os.path.join(rc.DATA_ROOT, spec['root']), spec)
            how = '.hea headers'
        if not rows:
            print(f"{name:12s} no events found - source absent?")
            continue
        cols = list({k: None for row in rows for k in row})
        path = os.path.join(out, f"{name}.csv")
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        types = {}
        for r in rows:
            types[r[spec['type']]] = types.get(r[spec['type']], 0) + 1
        top = sorted(types.items(), key=lambda kv: -kv[1])[:5]
        total += len(rows)
        print(f"{name:12s} {len(rows):>7,} events from {how:13s} -> {os.path.basename(path)}")
        print(f"{'':12s}   {dict(top)}")
    print(f"\n{total:,} events written to {out}")


if __name__ == '__main__':
    main()
