#!/usr/bin/env python
"""What rhythm training data this machine actually has, and whether the holdout holds.

    python scripts/rhythm_data_audit.py [--json OUT]

Two questions, kept apart because they fail differently.

AVAILABILITY is a survey: every source config.SOURCES / PTB-XL / Challenge-2020 /
PhysioNet names, present or not, with its record count. A missing source makes a SMALLER
dataset, which is visible and recoverable.

LEAKAGE is a proof, and its failure is invisible. The check that matters is not "do the two
sets intersect" but "are the two sets even comparable": the rhythm test export here names
its folders by EVENT id (24-hex ObjectId) while training study ids are short decimals, so a
holdout taken from those folder names intersects the training studies in zero places and
reports itself clean while the test set sits in the training data. inventory now resolves
event ids to study ids and refuses when it cannot; this script re-proves the result from the
outside, against the study ids actually written next to the beat windows.

EC57 (mitdb, afdb, escdb, nstdb, ahadb, cudb) must appear in no training source at all.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def availability():
    from ecgr.rhythm import config as rc
    rows = []
    for name, spec in rc.SOURCES.items():
        root = os.path.join(rc.DATA_ROOT, spec['root'])
        inv = os.path.join(rc.DATA_ROOT, spec['inventory']) if spec.get('inventory') else None
        rows.append(dict(kind='portal', name=name, path=spec['root'],
                         records=os.path.isdir(root),
                         inventory=bool(inv and os.path.exists(inv))))
    for name, path in (('ptbxl', rc.PTBXL_DIR), ('challenge2020', rc.CHALLENGE2020_DIR)):
        rows.append(dict(kind='external', name=name, path=path,
                         records=os.path.isdir(path), inventory=os.path.isdir(path)))
    for db, spec in rc.PHYSIONET_TRAIN_DBS.items():
        n = len(glob.glob(os.path.join(spec['dir'], '*.dat')))
        rows.append(dict(kind='physionet', name=db, path=spec['dir'],
                         records=n or os.path.isdir(spec['dir']), inventory=bool(n)))
    return rows


def leakage():
    import numpy as np
    from ecgr import config as bc
    from ecgr.rhythm import config as rc, inventory

    held, test = inventory.held_out_studies()
    held, test = {int(s) for s in held}, {int(s) for s in test}

    written = {}
    for split in ('train', 'eval'):
        ids = set()
        if os.path.isdir(bc.NPY_DIR):
            for db in os.listdir(bc.NPY_DIR):
                for p in glob.glob(os.path.join(bc.NPY_DIR, db, split,
                                                f'{db}_{split}_studyids_batch_*.npy')):
                    ids |= {int(v) for v in np.load(p)}
        written[split] = ids

    report = {
        'rhythm_test_dir': rc.EVAL_DIR,
        'rhythm_test_studies': len(test),
        'held_out_studies': len(held),
        'beat_npy_dir': bc.NPY_DIR,
        'beat_train_studies': len(written['train']),
        'beat_eval_studies': len(written['eval']),
        'overlaps': {
            'beat_train_x_rhythm_test': len(written['train'] & test),
            'beat_eval_x_rhythm_test': len(written['eval'] & test),
            'beat_train_x_held_out': len(written['train'] & held),
            'beat_eval_x_held_out': len(written['eval'] & held),
            'beat_train_x_beat_eval': len(written['train'] & written['eval']),
        },
        'rhythm_test_covered_by_v4_list': len(test & held) == len(test),
    }
    # the identifier-space trap, asserted rather than assumed
    report['holdout_ids_are_study_shaped'] = all(len(str(s)) < 24 for s in test)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    rows = availability()
    print(f"{'kind':10s} {'source':14s} {'records':>9s} {'inventory':>10s}  path")
    print('-' * 88)
    for r in rows:
        rec = r['records']
        rec = ('yes' if rec is True else 'MISSING' if rec is False else f'{rec}')
        print(f"{r['kind']:10s} {r['name']:14s} {rec:>9s} "
              f"{('ok' if r['inventory'] else 'MISSING'):>10s}  {r['path'][:44]}")
    have = sum(1 for r in rows if r['records'] and r['inventory'])
    print(f"\nusable sources: {have} of {len(rows)}")

    print("\n--- leakage ---")
    rep = leakage()
    for k, v in rep['overlaps'].items():
        print(f"  {k:28s} {v:>6,}  {'ok' if v == 0 else 'LEAK'}")
    print(f"  rhythm test covered by v4 list : {rep['rhythm_test_covered_by_v4_list']}")
    print(f"  holdout ids are study-shaped   : {rep['holdout_ids_are_study_shaped']}")
    bad = [k for k, v in rep['overlaps'].items() if v] + \
          ([] if rep['holdout_ids_are_study_shaped'] else ['holdout id shape'])
    print("\nVERDICT: " + ("clean" if not bad else f"PROBLEM - {bad}"))

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'availability': rows, 'leakage': rep}, f, indent=2)
        print(f"-> {args.json}")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
