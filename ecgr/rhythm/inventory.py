"""Which events exist, which may be trained on, and which side of the split they fall.

Three inventories feed the rhythm task:

  * the export spreadsheets of every source in config.SOURCES (one row per caliper mark),
  * the rhythm test set under config.EVAL_DIR (one JSON per event, with the reviewer's
    rhythms), which is never trained on,
  * the beat task's v4 eval list, held out as well so both models share one holdout.

The holdout is enforced the same three ways the beat task enforces it (data/splits.py): the
held-out studies are dropped before the split, `verify_split` proves the three sets are
disjoint before a single window is written, and build.audit_written re-proves it afterwards
from the study ids stored next to the windows.
"""
import glob
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from ..data import splits
from . import config as rc
from .labels import class_index, merge_intervals

# A caliper position of -1 means "the whole recording" in these exports.
WHOLE = (0, 10 ** 9)


def _int_or_none(value):
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return None
    return v if v >= 0 else None


def _read_table(path):
    """The export at `path`, cached as csv next to the npy (the xlsx parse costs ~1 min)."""
    cache_dir = os.path.join(rc.NPY_DIR, 'inventory_cache')
    stamp = int(os.path.getmtime(path))
    cache = os.path.join(cache_dir, f"{os.path.basename(path)}.{stamp}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, low_memory=False)
    df = pd.read_excel(path) if path.endswith(('.xlsx', '.xls')) else pd.read_csv(path)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def _hash01(text):
    return int(hashlib.md5(str(text).encode()).hexdigest(), 16) % 10 ** 6 / 10 ** 6


def read_source(name):
    """Events of one source: {(study, event): event dict}.

    event = {source, study_id, event_id, types, spans [(cls, a, b)], strip (a, b) | None,
             whole, root}
    Several rows of the same event (one per caliper mark) collapse into one event with
    several spans. Rows whose type maps to no class are dropped.
    """
    src = rc.SOURCES[name]
    df = _read_table(os.path.join(rc.DATA_ROOT, src['inventory']))
    events = {}
    skipped = defaultdict(int)
    for row in df.to_dict('records'):
        raw_type = str(row.get(src['type'], '')).strip().upper()
        cls_name = rc.EVENT_TYPE_TO_CLASS.get(raw_type)
        if cls_name is None:
            skipped[raw_type] += 1
            continue
        study = str(row[src['study']]).strip()
        event = str(row[src['event']]).strip()
        if not study.isdigit() or not event or event == 'nan':
            skipped['<no id>'] += 1
            continue
        key = (study, event)
        ev = events.setdefault(key, dict(source=name, study_id=study, event_id=event,
                                         types=set(), spans=[], strip=None,
                                         whole=src['whole'], root=src['root'],
                                         header_span=src.get('header_span', False),
                                         xlsx_span=[]))
        ev['types'].add(raw_type)

        strip = None
        if src['strip']:
            a, b = (_int_or_none(row.get(c)) for c in src['strip'])
            if a is not None and b is not None and b > a:
                strip = (a, b)
                ev['strip'] = strip

        span = None
        if src['span']:
            a, b = (_int_or_none(row.get(c)) for c in src['span'])
            if a is not None and b is not None and b > a:
                span = (a, b)
            elif src['whole']:
                span = WHOLE                         # -1 / -1: the whole recording
        cls = class_index(cls_name)
        ev['xlsx_span'].append(span is not None and span != WHOLE)
        if span is not None:
            ev['spans'].append((cls, *span))
        elif cls_name not in rc.RUN_CLASSES and strip is not None:
            # AFIB / AVB / SINUS strip with no caliper: the reviewer named the strip's rhythm
            ev['spans'].append((cls, *strip))
        # Spans recovered from the beat labels at build time: a run-type event with no caliper,
        # and every SINUS-group strip - an ectopy strip may hold a real >= 3-beat run, which
        # would otherwise be trained as SINUS.
        ev.setdefault('needs_runs', False)
        if (cls_name in rc.RUN_CLASSES and span is None) or \
                (cls_name == 'SINUS' and ev['strip'] is not None):
            ev['needs_runs'] = True
    for ev in events.values():
        ev['types'] = sorted(ev['types'])
        ev['spans'] = sorted(set(ev['spans']))
    return events, dict(skipped)


def is_sinus_only(ev):
    return all(c == rc.SINUS for c, _, _ in ev['spans']) and not any(
        rc.EVENT_TYPE_TO_CLASS[t] in rc.RUN_CLASSES for t in ev['types'])


def outside_policy(ev):
    """'ignore' if any rhythm of the event is a persistent one (config.OUTSIDE_SPAN)."""
    names = {rc.CLASS_NAMES[c] for c, _, _ in ev['spans']} | \
            {rc.EVENT_TYPE_TO_CLASS[t] for t in ev['types']}
    return 'ignore' if any(rc.OUTSIDE_SPAN.get(n) == 'ignore' for n in names) else 'sinus'


# ---------------------------------------------------------------------------
# Holdout
# ---------------------------------------------------------------------------

def rhythm_test_studies(eval_dir=None):
    """Study ids of the rhythm test set, from its folder names AND its JSONs."""
    eval_dir = eval_dir or rc.EVAL_DIR
    if not os.path.isdir(eval_dir):
        raise FileNotFoundError(
            f"rhythm test set not found: {eval_dir}. It is the holdout - refusing to build "
            f"training data without it (set ECGR_RHYTHM_EVAL_DIR).")
    ids = {d for d in os.listdir(eval_dir)
           if d.isdigit() and os.path.isdir(os.path.join(eval_dir, d))}
    for path in glob.glob(os.path.join(eval_dir, '*', '*', '*.json')):
        try:
            with open(path) as f:
                ids.add(str(json.load(f)['studyId']).strip())
        except (OSError, KeyError, ValueError):
            continue
    if not ids:
        raise ValueError(f"no study found under {eval_dir}")
    return ids


def held_out_studies():
    """(everything that may not be trained on, the rhythm test studies)."""
    test = rhythm_test_studies()
    print(f"test holdout : {len(test)} studies from {rc.EVAL_DIR}")
    held = set(test)
    if rc.EXCLUDE_V4_EVAL_STUDIES:
        held |= splits.eval_studies()
    return held, test


def verify_split(train_ids, eval_ids, test_ids, held_ids, out_path=None):
    """Raise unless train, eval and the held-out studies are pairwise disjoint."""
    train, evaluation = set(map(str, train_ids)), set(map(str, eval_ids))
    test, held = set(map(str, test_ids)), set(map(str, held_ids))
    overlaps = {
        'train_x_eval': sorted(train & evaluation),
        'train_x_rhythm_test': sorted(train & test),
        'eval_x_rhythm_test': sorted(evaluation & test),
        'train_x_held_out': sorted(train & held),
        'eval_x_held_out': sorted(evaluation & held),
    }
    violations = {k: v for k, v in overlaps.items() if v}
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({'verified': not violations,
                       'checked_at': datetime.now().isoformat(timespec='seconds'),
                       'rhythm_test_dir': rc.EVAL_DIR,
                       'counts': {'train': len(train), 'eval': len(evaluation),
                                  'rhythm_test': len(test), 'held_out': len(held)},
                       'overlap_counts': {k: len(v) for k, v in overlaps.items()},
                       'overlap_studyids': violations,
                       'train_studyids': sorted(train, key=int),
                       'eval_studyids': sorted(evaluation, key=int)}, f, indent=2)
    if violations:
        detail = ', '.join(f"{k}: {len(v)} (e.g. {v[:5]})" for k, v in violations.items())
        raise ValueError(f"rhythm split integrity violated - {detail}. Nothing was written.")
    print("split OK     : train n eval = train n test = eval n test = 0")


def collect(sources=None, max_sinus_per_study=None):
    """Every trainable event of `sources`, deduplicated, held-out studies removed, split.

    Returns ({'train': [events], 'eval': [events]}, report dict).
    """
    sources = sources or rc.TRAIN_SOURCES
    cap = rc.MAX_SINUS_EVENTS_PER_STUDY if max_sinus_per_study is None else max_sinus_per_study
    held, test = held_out_studies()

    seen, events, report = set(), [], {'sources': {}}
    for name in sources:
        if name in (rc.PTBXL_SOURCE, rc.CHALLENGE2020_SOURCE) or \
                name in rc.PHYSIONET_TRAIN_SOURCES:
            continue
        found, skipped = read_source(name)
        kept = dropped_held = dropped_dup = 0
        for key, ev in found.items():
            if ev['study_id'] in held:
                dropped_held += 1
                continue
            if ev['event_id'] in seen:
                dropped_dup += 1
                continue
            seen.add(ev['event_id'])
            events.append(ev)
            kept += 1
        report['sources'][name] = dict(events=len(found), kept=kept, held_out=dropped_held,
                                       duplicate=dropped_dup, skipped_types=skipped)
        print(f"{name:10s}: {len(found):>7,} events, kept {kept:>7,}, held out "
              f"{dropped_held:>6,}, duplicate {dropped_dup:>6,}")

    # Cap the SINUS-only strips per study: the dedicated sinus sources first (sinus_first),
    # then by a hash of the event id, so a rebuild picks the same strips.
    by_study = defaultdict(list)
    for ev in events:
        if is_sinus_only(ev):
            by_study[ev['study_id']].append(ev)
    drop = set()
    hard_kept = 0
    for study, evs in by_study.items():
        evs = sorted(evs, key=lambda e: (not rc.SOURCES.get(e['source'], {}).get('sinus_first'),
                                         _hash01(e['event_id'])))
        hard = [e for e in evs if set(e['types']) & set(rc.HARD_SINUS_TYPES)]
        # the first `cap` of the whole list as before, then up to the hard budget on top
        keep = {e['event_id'] for e in evs[:cap]}
        extra = [e for e in hard if e['event_id'] not in keep][:rc.MAX_HARD_SINUS_EVENTS_PER_STUDY]
        keep |= {e['event_id'] for e in extra}
        hard_kept += len(extra)
        drop |= {e['event_id'] for e in evs if e['event_id'] not in keep}
    events = [ev for ev in events if ev['event_id'] not in drop]
    report['sinus_cap'] = dict(per_study=cap, dropped=len(drop), hard_extra=hard_kept)
    print(f"hard sinus   : {hard_kept:,} look-alike strips kept on top of the cap "
          f"({rc.MAX_HARD_SINUS_EVENTS_PER_STUDY} per study)")
    print(f"sinus cap    : {cap} per study, {len(drop):,} strips dropped, "
          f"{len(events):,} events left")

    studies = sorted({ev['study_id'] for ev in events}, key=int)
    train_ids, eval_ids = splits.split_studies(studies)
    # PTB-XL is split by its own folds, not by the study hash; its (offset) patient ids join
    # the check so the audit proves them disjoint as well.
    extra = {'train': [], 'eval': []}
    if rc.PTBXL_SOURCE in sources:
        from . import ptbxl
        extra, report['sources'][rc.PTBXL_SOURCE] = ptbxl.collect()
    wanted = {db: rc.PHYSIONET_TRAIN_DBS[db] for db in rc.PHYSIONET_TRAIN_SOURCES
              if db in sources}
    if wanted:
        from . import physionet_train
        more, report['sources']['physionet_train'] = physionet_train.collect(wanted)
        for split in extra:
            extra[split] += more[split]
    if rc.CHALLENGE2020_SOURCE in sources:
        from . import challenge2020
        more, report['sources'][rc.CHALLENGE2020_SOURCE] = challenge2020.collect()
        for split in extra:
            extra[split] += more[split]
    verify_split(train_ids + sorted({ev['study_id'] for ev in extra['train']}),
                 eval_ids + sorted({ev['study_id'] for ev in extra['eval']}), test, held,
                 out_path=os.path.join(rc.NPY_DIR, 'split_verification.json'))
    side = {s: 'train' for s in train_ids} | {s: 'eval' for s in eval_ids}
    out = {'train': [], 'eval': []}
    for ev in events:
        out[side[ev['study_id']]].append(ev)
    for split in out:
        out[split] += extra[split]
    report['studies'] = {'train': len(train_ids), 'eval': len(eval_ids),
                         'held_out': len(held), 'rhythm_test': len(test),
                         'ptbxl': {s: len({ev['study_id'] for ev in extra[s]}) for s in extra}}
    return out, report


def collect_test(eval_dir=None):
    """The rhythm test set as events. The reviewer's 10-50 s mark is the known region."""
    eval_dir = eval_dir or rc.EVAL_DIR
    events = []
    for path in sorted(glob.glob(os.path.join(eval_dir, '*', '*', '*.json'))):
        with open(path) as f:
            info = json.load(f)
        mark = (int(info['startMarkSample']), int(info['stopMarkSample']))
        spans = []
        for r in (info.get('ref') or {}).get('rhythms') or []:
            name = rc.EVENT_TYPE_TO_CLASS.get(str(r['type']).upper())
            if name is not None:
                spans.append((class_index(name), int(r['start']), int(r['stop'])))
        events.append(dict(source='rhythm_eval', study_id=str(info['studyId']),
                           event_id=str(info['eventId']), types=[str(info['type'])],
                           spans=sorted(spans), strip=mark, whole=False, needs_runs=False,
                           record=path[:-5], dataset=info.get('dataset')))
    return events


def event_region(ev, length):
    """(windows region, known region) of one event, clipped to a record of `length`.

    Strip sources: the region is the reviewed strip (plus any caliper reaching past it), and
    the strip is known - an unmarked second inside it is SINUS.
    Caliper sources (60 s recordings): the region is the union of the spans, and what is
    known depends on the rhythm (config.OUTSIDE_SPAN).
    """
    spans = [(c, max(0, a), min(length, b)) for c, a, b in ev['spans']]
    spans = [s for s in spans if s[2] > s[1]]
    if ev['strip'] is not None:
        a, b = max(0, ev['strip'][0]), min(length, ev['strip'][1])
        region = [(a, b)]
        known = [(a, b)]
        if ev.get('known') is not None:     # the long PhysioNet records say it per stretch
            known = [(max(x, a), min(y, b)) for x, y in ev['known'] if x < b and y > a]
    else:
        region = [(a, b) for _, a, b in spans]
        known = [(0, length)] if outside_policy(ev) == 'sinus' else []
    return spans, region, known


def window_starts(region, length, segment=rc.SEGMENT_SAMPLES, hop=None,
                  slack=rc.SPAN_SLACK_SAMPLES):
    """Window starts covering every interval of `region`, each window inside [0, length)."""
    hop = int(rc.WINDOW_HOP_SECONDS * rc.SAMPLING_RATE) if hop is None else int(hop)
    if length < segment:
        return []
    starts = set()
    for a, b in merge_intervals(region):
        span = b - a
        if span <= segment:
            # centred on the interval; a 2499-sample strip lands on its own start
            s = a - (segment - span) // 2 if span < segment - slack else a
            starts.add(int(np.clip(s, 0, length - segment)))
            continue
        s = list(range(a, b - segment + 1, hop))
        # The tail window ends exactly at the span's end. Close behind the last hop it would
        # be a near-duplicate (a 2559-sample AF span gave windows 59 samples apart), so it
        # REPLACES that window instead of joining it.
        tail = b - segment
        if tail - s[-1] >= hop // 2:
            s.append(tail)
        elif tail > s[-1]:
            s[-1] = tail if len(s) > 1 else (a + tail) // 2     # one window: centre it
        starts |= {int(np.clip(x, 0, length - segment)) for x in s}
    return sorted(starts)
