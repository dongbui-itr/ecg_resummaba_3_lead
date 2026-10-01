"""PhysioNet/CinC Challenge 2020 (CPSC-2018, CPSC-2018 extra, Georgia) as a rhythm source.

12-lead 500 Hz ECGs of 6-144 s (median 10 s) with SNOMED-CT diagnoses for the whole record.
None of them is an EC57 database. The challenge's own copies of PTB-XL and INCART are NOT read
(config.CHALLENGE2020_SUBSETS): PTB-XL and incartdb are sources already, with their own splits,
and a second copy would put the same patient on both sides.

Label per record (config.CHALLENGE2020_*): AF or atrial flutter -> AFIB (AFL = AF convention),
second-degree AV block -> AVB2, complete block -> AVB3; two different classes -> skipped; the
paroxysmal ones (SVT, VT) are skipped too - a record-level label cannot say where the run is;
everything else SINUS, kept when a hard-negative code is present (sinus tachy / brady /
arrhythmia, PACs, PVCs, bundle-branch blocks, first-degree block) or under a hash cap of plain
sinus records. Each record gives up to CHALLENGE2020_WINDOWS_PER_RECORD 10 s windows, each with
its own 3-of-12 lead subset (ptbxl.lead_subsets). Split by record hash.
"""
import glob
import hashlib
import os
from collections import Counter

import wfdb

from . import config as rc
from .labels import class_index


def _hash01(text):
    return int(hashlib.sha1(text.encode()).hexdigest()[:12], 16) / float(16 ** 12)


def read_codes(hea_path):
    """(fs, n samples, set of SNOMED codes) from a challenge header, without reading signals."""
    with open(hea_path) as f:
        lines = f.read().splitlines()
    head = lines[0].split()
    codes = set()
    for line in lines:
        if line.startswith('#') and 'Dx:' in line:
            codes |= {c.strip() for c in line.split(':', 1)[1].split(',') if c.strip()}
    return int(head[2]), int(head[3]), codes


def record_label(codes):
    """(class_name, None) | (None, reason); reason 'plain' = sinus with no hard-negative code."""
    classes = {rc.CHALLENGE2020_CODE_TO_CLASS[c] for c in codes
               if c in rc.CHALLENGE2020_CODE_TO_CLASS}
    if codes & set(rc.CHALLENGE2020_SKIP_CODES):
        return None, 'paroxysmal_or_ambiguous'
    if len(classes) > 1:
        return None, 'several_classes'
    if classes:
        return classes.pop(), None
    if codes & set(rc.CHALLENGE2020_HARD_NEGATIVE_CODES):
        return 'SINUS', None
    return None, 'plain'


def make_events(path, fs, n, cls_name, key):
    from .ptbxl import lead_subsets
    w = rc.SEGMENT_SECONDS * fs
    count = rc.CHALLENGE2020_WINDOWS_PER_RECORD
    subsets = lead_subsets(int(_hash01(key) * 1e12), n=count)
    n_windows = n // w
    cls = class_index(cls_name)
    events = []
    for k in range(count):
        # consecutive 10 s windows while the record has them; a 10 s record gives the same
        # window again through a different lead subset (as PTB-XL does)
        a = (k % n_windows) * w
        spans = [] if cls == rc.SINUS else [(cls, a, a + w)]
        events.append(dict(source=rc.CHALLENGE2020_SOURCE, study_id=None, event_id=f"{key}_{k}",
                           types=[cls_name], spans=spans, strip=(a, a + w), whole=False,
                           needs_runs=False, record=path, leads=list(subsets[k])))
    return events


def collect(root=None):
    """({'train': [events], 'eval': [events]}, report)."""
    root = root or rc.CHALLENGE2020_DIR
    out, report = {'train': [], 'eval': []}, {}
    kept, skipped = {'train': Counter(), 'eval': Counter()}, Counter()
    plain = []
    ids = {}
    for subset in rc.CHALLENGE2020_SUBSETS:
        for hea in sorted(glob.glob(os.path.join(root, subset, '*', '*.hea'))):
            fs, n, codes = read_codes(hea)
            if n < rc.SEGMENT_SECONDS * fs:
                skipped['shorter_than_10s'] += 1
                continue
            key = f"{subset}/{os.path.basename(hea)[:-4]}"
            cls_name, reason = record_label(codes)
            if reason == 'plain':
                plain.append((_hash01('plain-cap:' + key), key, hea, fs, n))
                continue
            if reason:
                skipped[reason] += 1
                continue
            ids[key] = (hea, fs, n, cls_name)
    for _, key, hea, fs, n in sorted(plain)[:rc.CHALLENGE2020_PLAIN_CAP]:
        ids[key] = (hea, fs, n, 'SINUS')
    skipped['plain_over_cap'] = max(0, len(plain) - rc.CHALLENGE2020_PLAIN_CAP)
    for i, (key, (hea, fs, n, cls_name)) in enumerate(sorted(ids.items())):
        split = 'eval' if _hash01('split:' + key) < rc.CHALLENGE2020_EVAL_FRACTION else 'train'
        events = make_events(hea[:-4], fs, n, cls_name, key)
        for ev in events:
            ev['study_id'] = str(rc.CHALLENGE2020_STUDY_OFFSET + i)
        out[split] += events
        kept[split][cls_name] += 1
    for split in out:
        print(f"challenge : {split:5s} {sum(kept[split].values()):,} records -> "
              f"{len(out[split]):,} windows {dict(kept[split])}")
    print(f"            skipped {dict(skipped)}")
    report = {'kept': {s: dict(c) for s, c in kept.items()}, 'skipped': dict(skipped)}
    return out, report
