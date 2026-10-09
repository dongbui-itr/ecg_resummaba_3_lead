"""Holter report strips -> labelled rhythm windows on disk.

One event becomes one or more 10 s windows, (2500, 3) float16 over the three leads in their
native order, each paired with a (10,) uint8 vector of per-second classes (labels.py) and the
study it came from:

    <NPY_DIR>/<split>/segments_<k>.npy    (n, 2500, 3) float16, band-passed + z-scored per lead
    <NPY_DIR>/<split>/labels_<k>.npy      (n, 10) uint8, config.IGNORE where unlabelled
    <NPY_DIR>/<split>/labels_samples_<k>.npy  (n, 2500) uint8, the same spans per sample
                                          (labels.sample_labels), for the per-sample models
    <NPY_DIR>/<split>/beats_<k>.npy       (n, 2500) uint8 beat labels (labels.beat_labels):
                                          0 none, 1/2/3 N/S/V at the R sample, IGNORE where
                                          the record has no beat annotation
    <NPY_DIR>/<split>/studyids_<k>.npy    (n,) int64
    <NPY_DIR>/<split>/events_<k>.json     source / event id of every window, for tracing
    <NPY_DIR>/manifest.json               geometry, counts per class, the class weights

split is 'train' / 'eval' (the portal sources, split by study hash, plus PTB-XL split by its
folds - ptbxl.py) or 'test' (the rhythm test set, config.EVAL_DIR - built for scoring only,
never read by training).

The windows are stored CLEAN. Noise, lead permutation and the lead target are produced on the
fly by augment.py, so every epoch sees a different corruption of the same strip and the
stored data stays a faithful copy of what the reviewers looked at.
"""
import glob
import json
import os
import re
import shutil
import sys
from collections import Counter
from functools import partial
from multiprocessing import get_context

import numpy as np
import wfdb
from tqdm import tqdm

from ..signal_ops import build_leads, normalize_window, resample_leads
from . import config as rc
from . import inventory
from .labels import beat_labels, beat_runs, sample_labels, second_labels

SHARD_WINDOWS = 20000


def record_path(ev):
    """Base path (no extension) of the event's recording, or None."""
    if ev.get('record'):
        return ev['record'] if os.path.exists(ev['record'] + '.hea') else None
    root = os.path.join(rc.DATA_ROOT, ev['root'])
    heads = sorted(glob.glob(os.path.join(root, ev['study_id'], ev['event_id'], '*.hea')))
    if not heads:
        heads = sorted(glob.glob(os.path.join(root, '*', ev['study_id'], ev['event_id'],
                                              '*.hea')))
    return heads[0][:-4] if heads else None


def _is_dead(window, threshold=0.1):
    """Every lead flat: nothing to read a rhythm from."""
    return bool(np.all(np.ptp(window, axis=0) < threshold))


def read_leads(path, leads=None, chunk=None):
    """(signal (n, 3) at SAMPLING_RATE, ratio to rescale sample positions).

    `leads`: column indices to keep of a record with more signals than IN_CHANNELS (PTB-XL's
    3-of-12 subsets); None = the record's first channels, zero-filled if fewer.
    `chunk`: (sampfrom, sampto) at the record's own rate - only that part is read (the long
    PhysioNet training records); positions are then relative to sampfrom.
    """
    if chunk is None:
        record = wfdb.rdrecord(path)
    else:
        record = wfdb.rdrecord(path, sampfrom=int(chunk[0]), sampto=int(chunk[1]))
    raw = np.nan_to_num(record.p_signal)
    if leads is not None:
        raw = raw[:, list(leads)]
    ratio = rc.SAMPLING_RATE / record.fs
    if record.fs != rc.SAMPLING_RATE:
        raw = resample_leads(raw, record.fs, rc.SAMPLING_RATE)
    # primary=0 keeps the native CH order: the rhythm label belongs to no lead in particular
    leads = build_leads(raw, fs=rc.SAMPLING_RATE, in_channels=rc.IN_CHANNELS, primary=0,
                        fill_mode='zero')
    return leads, ratio


def _scale_event(ev, ratio):
    if ratio == 1.0:
        return ev
    s = lambda v: int(round(v * ratio)) if v < 10 ** 8 else v      # noqa: E731
    ev = dict(ev)
    ev['spans'] = [(c, s(a), s(b)) for c, a, b in ev['spans']]
    if ev.get('known') is not None:
        ev['known'] = [(s(a), s(b)) for a, b in ev['known']]
    if ev['strip'] is not None:
        ev['strip'] = (s(ev['strip'][0]), s(ev['strip'][1]))
    return ev


_HEADER_START = re.compile(r'eventStartSample:\s*(-?\d+)')
_HEADER_STOP = re.compile(r'eventStopSample:\s*(-?\d+)')


def header_span(path):
    """The reviewer's caliper as written in the record's .hea comments, or None.

    dataset-4's export has no caliper columns, but its headers carry one for most events
    (measured over 40 headers per class: SVT 38, VT 39, AVB2 40, AVB3 20/20, AFIB 18) - and the
    reference project read exactly these lines. -1 means no caliper.
    """
    try:
        with open(path + '.hea') as f:
            text = f.read()
    except OSError:
        return None
    a, b = _HEADER_START.search(text), _HEADER_STOP.search(text)
    if not (a and b):
        return None
    a, b = int(a.group(1)), int(b.group(1))
    return (a, b) if 0 <= a < b else None


def _apply_header_span(ev, path, stats):
    """Replace the strip-wide / beat-derived guess by the header caliper when there is one."""
    if not ev.get('header_span') or any(s for s in ev.get('xlsx_span', [])):
        return ev
    span = header_span(path)
    if span is None:
        stats['header_span_missing'] += 1
        return ev
    classes = sorted({rc.CLASS_NAMES.index(rc.EVENT_TYPE_TO_CLASS[t]) for t in ev['types']})
    sinus_only = all(c == rc.SINUS for c in classes)
    stats['header_span'] += 1
    # SINUS-group strips still get their >= 3-beat runs from the .atr; a run-type event does
    # not need them any more - the caliper says where the run is.
    return dict(ev, spans=[(c, *span) for c in classes],
                needs_runs=sinus_only and ev['strip'] is not None)


def read_beats(path, ratio, chunk=None):
    """(R samples at SAMPLING_RATE relative to the chunk, symbols) from <path>.atr, or
    (None, None) when the record has no beat annotation. `chunk` = (sampfrom, sampto) at the
    record's own rate for the long records."""
    if not os.path.exists(path + '.atr'):
        return None, None
    if chunk is None:
        ann = wfdb.rdann(path, 'atr')
    else:
        ann = wfdb.rdann(path, 'atr', sampfrom=int(chunk[0]), sampto=int(chunk[1]),
                         shift_samps=True)
    symbols = np.asarray(ann.symbol)
    keep = np.array([sym in rc.BEAT_SYMBOL_TO_CLASS or sym in rc.BEAT_IGNORE_SYMBOLS
                     for sym in symbols], dtype=bool)
    samples = np.round(np.asarray(ann.sample)[keep] * ratio).astype(np.int64)
    return samples, symbols[keep]


def process_event(ev):
    """Cut one event into (segments, labels, sample labels, beat labels, stats). Never
    raises."""
    stats = Counter()
    segments, labels, per_sample, beats = [], [], [], []
    path = record_path(ev)
    if path is None:
        stats['no_file'] += 1
        return segments, labels, per_sample, beats, stats
    try:
        leads, ratio = read_leads(path, ev.get('leads'), ev.get('chunk'))
        ev = _scale_event(_apply_header_span(ev, path, stats), ratio)
        length = len(leads)
        beat_samples, beat_symbols = read_beats(path, ratio, ev.get('chunk'))
        if beat_samples is None:
            stats['no_beat_annotation'] += 1
        if ev.get('needs_runs') and ev['strip'] is not None and beat_samples is not None:
            runs = beat_runs(beat_samples, beat_symbols, lo=ev['strip'][0], hi=ev['strip'][1])
            ev = dict(ev, spans=sorted(set(ev['spans']) | set(runs)))
            stats['beat_runs'] += len(runs)

        spans, region, known = inventory.event_region(ev, length)
        # a run-type event whose run could not be located has nothing trustworthy to say
        if any(rc.EVENT_TYPE_TO_CLASS[t] in rc.RUN_CLASSES for t in ev['types']) and \
                not any(rc.CLASS_NAMES[c] in rc.RUN_CLASSES for c, _, _ in spans):
            stats['run_not_found'] += 1
            return segments, labels, per_sample, beats, stats
        if not region:
            stats['no_region'] += 1
            return segments, labels, per_sample, beats, stats

        hop = rc.SEGMENT_SAMPLES if ev['source'] == 'rhythm_eval' else None
        for start in inventory.window_starts(region, length, hop=hop):
            window = leads[start:start + rc.SEGMENT_SAMPLES]
            if _is_dead(window):
                stats['dead'] += 1
                continue
            lab = second_labels(start, spans, known)
            if np.all(lab == rc.IGNORE):
                stats['unlabelled'] += 1
                continue
            segments.append(normalize_window(window).astype(np.float16))
            labels.append(lab)
            per_sample.append(sample_labels(start, spans, known))
            beats.append(beat_labels(start, beat_samples, beat_symbols))
        stats['events_used'] += bool(segments)
    except Exception as e:                        # one broken record must not stop a build
        stats['errors'] += 1
        stats[f"error:{type(e).__name__}"] += 1
        segments, labels, per_sample, beats = [], [], [], []
    return segments, labels, per_sample, beats, stats


def _write_split(split, events, workers):
    out_dir = os.path.join(rc.NPY_DIR, split)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    totals, seconds = Counter(), np.zeros(rc.NUM_CLASSES + 1, dtype=np.int64)
    buf_seg, buf_lab, buf_smp, buf_bt, buf_sid, buf_evt = [], [], [], [], [], []
    shard = 0

    def flush(force=False):
        nonlocal buf_seg, buf_lab, buf_smp, buf_bt, buf_sid, buf_evt, shard
        while len(buf_seg) >= SHARD_WINDOWS or (force and buf_seg):
            k = min(SHARD_WINDOWS, len(buf_seg))
            base = os.path.join(out_dir, '{}_' + f'{shard}')
            np.save(base.format('segments') + '.npy', np.stack(buf_seg[:k]))
            np.save(base.format('labels') + '.npy', np.stack(buf_lab[:k]).astype(np.uint8))
            np.save(base.format('labels_samples') + '.npy',
                    np.stack(buf_smp[:k]).astype(np.uint8))
            np.save(base.format('beats') + '.npy', np.stack(buf_bt[:k]).astype(np.uint8))
            np.save(base.format('studyids') + '.npy', np.asarray(buf_sid[:k], dtype=np.int64))
            with open(base.format('events') + '.json', 'w') as f:
                json.dump(buf_evt[:k], f)
            buf_seg, buf_lab, buf_smp, buf_bt, buf_sid, buf_evt = (
                buf_seg[k:], buf_lab[k:], buf_smp[k:], buf_bt[k:], buf_sid[k:], buf_evt[k:])
            shard += 1

    ctx = get_context('spawn')
    chunk = max(1, min(32, len(events) // (workers * 8) or 1))
    with ctx.Pool(workers) as pool:
        results = pool.imap(process_event, events, chunksize=chunk)
        for ev, (segments, labels, samples, beats, stats) in tqdm(
                zip(events, results), total=len(events), desc=f"rhythm {split}",
                smoothing=0.05, disable=not sys.stderr.isatty(), mininterval=2.0):
            totals.update(stats)
            buf_seg.extend(segments)
            buf_lab.extend(labels)
            buf_smp.extend(samples)
            buf_bt.extend(beats)
            buf_sid.extend([int(ev['study_id'])] * len(segments))
            buf_evt.extend([[ev['source'], ev['event_id']]] * len(segments))
            for lab in labels:
                seconds += np.bincount(np.where(lab == rc.IGNORE, rc.NUM_CLASSES, lab),
                                       minlength=rc.NUM_CLASSES + 1)
            totals['windows'] += len(segments)
            flush()
    flush(force=True)
    per_class = {name: int(seconds[i]) for i, name in enumerate(rc.CLASS_NAMES)}
    per_class['IGNORE'] = int(seconds[-1])
    print(f"rhythm {split}: {totals['windows']:,} windows from {totals['events_used']:,} of "
          f"{len(events):,} events | seconds {per_class}")
    print(f"  skipped: " + ', '.join(f"{k} {v:,}" for k, v in sorted(totals.items())
                                     if k not in ('windows', 'events_used')))
    if events and totals['windows'] == 0:
        raise ValueError(f"rhythm {split}: {len(events)} events produced 0 windows - "
                         f"records not found under {rc.DATA_ROOT}?")
    return {'windows': int(totals['windows']), 'events': len(events),
            'seconds': per_class, 'stats': {k: int(v) for k, v in totals.items()}}


def class_weights_from_counts(seconds):
    counts = np.array([max(1, seconds[n]) for n in rc.CLASS_NAMES], dtype=np.float64)
    w = np.sqrt(np.median(counts) / counts)
    return [float(x) for x in np.clip(w, 0.2, 5.0)]


def build(sources=None, limit=None, workers=None, splits=('train', 'eval', 'test'),
          audit=True):
    """Build the requested splits. `limit` keeps the first N events per split (smoke test)."""
    workers = workers or rc.WORKERS
    os.makedirs(rc.NPY_DIR, exist_ok=True)
    manifest_path = os.path.join(rc.NPY_DIR, 'manifest.json')
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    todo = {}
    if {'train', 'eval'} & set(splits):
        by_split, report = inventory.collect(sources)
        manifest['inventory'] = report
        todo.update({s: by_split[s] for s in ('train', 'eval') if s in splits})
    if 'test' in splits:
        todo['test'] = inventory.collect_test()

    manifest.setdefault('splits', {})
    for split, events in todo.items():
        if limit:
            events = events[:limit]
        print(f"\nrhythm {split}: {len(events):,} events, {workers} workers")
        manifest['splits'][split] = _write_split(split, events, workers)

    manifest.update({
        'segment_samples': rc.SEGMENT_SAMPLES, 'in_channels': rc.IN_CHANNELS,
        'sampling_rate': rc.SAMPLING_RATE, 'output_seconds': rc.OUTPUT_SECONDS,
        'class_names': rc.CLASS_NAMES, 'ignore_label': rc.IGNORE,
        'signal_dtype': 'float16', 'labels_dtype': 'uint8', 'sample_labels': True,
        'beat_labels': True,
        'sources': list(sources or rc.TRAIN_SOURCES), 'limit': limit,
    })
    if 'train' in manifest['splits']:
        manifest['class_weights'] = class_weights_from_counts(
            manifest['splits']['train']['seconds'])
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest -> {manifest_path}")
    if audit:
        audit_written()
    return manifest


def written_studies(split):
    ids = set()
    for path in glob.glob(os.path.join(rc.NPY_DIR, split, 'studyids_*.npy')):
        ids |= {str(int(v)) for v in np.load(path)}
    return ids


def audit_written(out_path=None):
    """Re-prove the separation from the study ids stored with the windows.

    train n eval = 0, and neither touches a study of the rhythm test folder (read again from
    disk here, not taken from the build) or of the v4 eval list.
    """
    written = {s: written_studies(s) for s in ('train', 'eval', 'test')}
    test_dir = inventory.rhythm_test_studies()
    held, _ = inventory.held_out_studies()
    overlaps = {
        'train_x_eval': sorted(written['train'] & written['eval']),
        'train_x_rhythm_test_dir': sorted(written['train'] & test_dir),
        'eval_x_rhythm_test_dir': sorted(written['eval'] & test_dir),
        'train_x_held_out': sorted(written['train'] & held),
        'eval_x_held_out': sorted(written['eval'] & held),
        'test_not_in_rhythm_test_dir': sorted(written['test'] - test_dir),
    }
    violations = {k: v for k, v in overlaps.items() if v}
    report = {'verified': not violations,
              'counts': {k: len(v) for k, v in written.items()} | {'rhythm_test_dir':
                                                                   len(test_dir)},
              'overlap_counts': {k: len(v) for k, v in overlaps.items()},
              'overlap_studyids': violations}
    out_path = out_path or os.path.join(rc.NPY_DIR, 'audit_written.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"audit        : {report['counts']} -> {out_path}")
    if violations:
        raise ValueError(f"written rhythm data violates the split: "
                         f"{ {k: len(v) for k, v in violations.items()} }")
    print("audit OK     : train n eval = 0, train/eval n rhythm_eval = 0")
    return report
