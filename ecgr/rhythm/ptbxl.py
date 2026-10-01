"""PTB-XL as a rhythm training source: one label per 10 s record, 3-of-12 lead subsets.

PTB-XL (physionet.org/content/ptb-xl, 21,799 12-lead 10 s ECGs at 500 Hz) is not one of the
EC57 databases, so training on it keeps the Physionet scoring legitimate. It brings what the
portal exports cannot: bundle-branch block, paced rhythm, pre-excitation, sinus tachy / brady
and ectopy recorded as such (the portal files them under OTHERS) - the records behind the mitdb
false positives - plus 1.5k AFIB records from another population and another device.

Each record's SCP codes become one class over the whole 10 s (config.PTBXL_CODE_TO_CLASS and
the rule in config), each patient sits in one strat_fold, and the fold decides the split
(config.PTBXL_FOLDS). The event dicts follow inventory.collect_test's contract, plus a `leads`
field (column indices into the 12-lead record) that build.read_leads honours, so build.py
processes them like any portal strip: read, resample 500 -> 250 Hz, band-pass, z-score.
"""
import ast
import hashlib
import os
from collections import Counter

import numpy as np
import pandas as pd

from . import config as rc
from .labels import class_index

LEAD_NAMES = list(rc.PTBXL_LIMB_LEADS) + list(rc.PTBXL_PRECORDIAL_LEADS)
RECORD_SAMPLES = 5000


def study_id(patient_id):
    return str(rc.PTBXL_STUDY_OFFSET + int(patient_id))


def record_label(codes):
    """The class name of a record with SCP `codes`, or None with the reason it is skipped.

    Returns (class_name, None) | (None, reason). 'plain' is a SINUS record with none of the
    hard-negative codes - kept only under config.PTBXL_PLAIN_CAP by the caller.
    """
    codes = set(codes)
    if codes & set(rc.PTBXL_SKIP_CODES):
        return None, 'ambiguous'
    classes = {rc.PTBXL_CODE_TO_CLASS[c] for c in codes if c in rc.PTBXL_CODE_TO_CLASS}
    if len(classes) > 1:
        return None, 'several_classes'
    if classes:
        return classes.pop(), None
    if codes & set(rc.PTBXL_HARD_NEGATIVE_CODES):
        return 'SINUS', None
    return None, 'plain'


def _rng(ecg_id, salt=''):
    digest = hashlib.md5(f"ptbxl:{salt}:{int(ecg_id)}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:12], 16))


def hash01(ecg_id):
    return _rng(ecg_id, 'cap').random()


def lead_subsets(ecg_id, n=None):
    """`n` sorted 3-lead subsets (column indices) of one record, fixed by a hash of ecg_id.

    The first always pairs a limb lead with a precordial lead (a montage a 3-lead Holter can
    approximate); the rest are drawn freely from the 12. No two subsets are the same.
    """
    n = rc.PTBXL_WINDOWS_PER_RECORD if n is None else n
    rng = _rng(ecg_id, 'leads')
    limb, chest = len(rc.PTBXL_LIMB_LEADS), len(LEAD_NAMES)
    first = {int(rng.integers(0, limb)), int(rng.integers(limb, chest))}
    while len(first) < rc.IN_CHANNELS:
        first.add(int(rng.integers(0, chest)))
    subsets = [tuple(sorted(first))]
    while len(subsets) < n:
        s = tuple(sorted(int(i) for i in rng.choice(chest, rc.IN_CHANNELS, replace=False)))
        if s not in subsets:
            subsets.append(s)
    return subsets


def read_database(ptbxl_dir=None):
    """The database csv as (ecg_id, patient_id, strat_fold, filename_hr, codes) rows."""
    path = os.path.join(ptbxl_dir or rc.PTBXL_DIR, 'ptbxl_database.csv')
    df = pd.read_csv(path, usecols=['ecg_id', 'patient_id', 'strat_fold', 'filename_hr',
                                    'scp_codes'])
    df['codes'] = df['scp_codes'].apply(lambda s: sorted(ast.literal_eval(s)))
    return df.drop(columns='scp_codes')


def make_events(row, cls_name, ptbxl_dir=None):
    """The PTBXL_WINDOWS_PER_RECORD event dicts of one labelled record."""
    cls = class_index(cls_name)
    record = os.path.join(ptbxl_dir or rc.PTBXL_DIR, row['filename_hr'])
    events = []
    for k, leads in enumerate(lead_subsets(row['ecg_id'])):
        events.append(dict(source=rc.PTBXL_SOURCE, study_id=study_id(row['patient_id']),
                           event_id=f"{int(row['ecg_id'])}_{k}", types=[cls_name],
                           codes=list(row['codes']), spans=[(cls, 0, RECORD_SAMPLES)],
                           strip=(0, RECORD_SAMPLES), whole=False, needs_runs=False,
                           record=record, leads=list(leads),
                           lead_names=[LEAD_NAMES[i] for i in leads]))
    return events


def collect(ptbxl_dir=None, plain_cap=None, folds=None):
    """PTB-XL events per split: ({'train': [events], 'eval': [events]}, report).

    Fold 10 is never read. Plain SINUS records (no arrhythmia, no hard-negative code) are
    kept by the `plain_cap` smallest hashes over the folds in use, so a rebuild picks the
    same ones and the 1.3M portal SINUS seconds are not flooded.
    """
    ptbxl_dir = ptbxl_dir or rc.PTBXL_DIR
    plain_cap = rc.PTBXL_PLAIN_CAP if plain_cap is None else plain_cap
    folds = folds or rc.PTBXL_FOLDS
    side = {f: split for split, fs in folds.items() for f in fs}
    df = read_database(ptbxl_dir)

    labelled, plain, skipped = [], [], Counter()
    for row in df.to_dict('records'):
        split = side.get(int(row['strat_fold']))
        if split is None:
            skipped['reserved_fold'] += 1
            continue
        cls_name, reason = record_label(row['codes'])
        if reason == 'plain':
            plain.append((hash01(row['ecg_id']), split, row))
        elif reason:
            skipped[reason] += 1
        else:
            labelled.append((split, cls_name, row))
    plain.sort(key=lambda t: t[0])
    labelled += [(split, 'SINUS', row) for _, split, row in plain[:plain_cap]]
    skipped['plain_over_cap'] = max(0, len(plain) - plain_cap)

    out = {split: [] for split in folds}
    records = {split: Counter() for split in folds}
    for split, cls_name, row in labelled:
        if not os.path.exists(os.path.join(ptbxl_dir, row['filename_hr'] + '.hea')):
            skipped['no_file'] += 1
            continue
        out[split] += make_events(row, cls_name, ptbxl_dir)
        records[split][cls_name] += 1
    report = dict(records=int(len(df)), kept={s: dict(c) for s, c in records.items()},
                  windows_per_record=rc.PTBXL_WINDOWS_PER_RECORD, plain_cap=plain_cap,
                  plain_kept=min(plain_cap, len(plain)), skipped=dict(skipped),
                  studies={s: len({ev['study_id'] for ev in evs}) for s, evs in out.items()})
    for split in folds:
        print(f"{rc.PTBXL_SOURCE:10s}: {split} {sum(records[split].values()):>6,} records "
              f"-> {len(out[split]):>7,} events {dict(records[split])}")
    print(f"{'':10s}  skipped {dict(skipped)}")
    return out, report
