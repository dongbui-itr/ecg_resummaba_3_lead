"""Portal records -> labelled npy batches.

One reviewed record becomes a stack of 10 s windows sliding by 1 s, each with a 500-step
label vector over three leads with the annotated lead on channel 0. Only the reviewed span
[start_sample, stop_sample] is used: the rest of the strip carries annotations nobody signed
off on.

Two arithmetic faults in the previous version are fixed here, and between them they touched
a fifth of the corpus:

  * The reviewed span was converted with `start // fs * fs`, i.e. truncated to whole
    SECONDS. Both rates are 250 Hz, so the conversion was the identity apart from that
    truncation - which moved 95k windows up to 249 samples outside the span the reviewer
    certified, and shortened 19,763 records enough that they produced NO window at all.
  * 114,920 records (23.1%) have a reviewed span of exactly 2499 samples - one sample short
    of 10 s. A strict `window must fit inside the span` rule discards every one of them.
    SPAN_SLACK_SAMPLES lets a window reach a tenth of a second past the boundary, which
    recovers them; anything shorter than that is still refused and counted.

The event mix made this expensive rather than merely untidy: the dropped records were ~14%
of all SVE and VE event strips, i.e. concentrated in the two classes the model is weakest on.

A third fault is in the data rather than the arithmetic, and `_record_files` now handles it:
**every dataset-2 event folder holds the same recording twice**, under two different
event-id prefixes and byte for byte identical (200/200 sampled events). Processing every
`.dat` match therefore emitted each dataset-2 window twice - and since dataset-2 is the
largest single contributor of segments, roughly a quarter of the whole training set was an
exact duplicate of another quarter. It is not a train/eval leak (the split is per study),
but it silently doubles one dataset's weight in the loss and doubles the RAM `cache()` needs.
"""
import glob
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import get_context

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from .. import config
from ..labels import labels_from_annotations
from ..signal_ops import build_leads, is_flat, normalize_window, resample_leads
from . import splits

# How far a window may reach past the reviewed span. A tenth of a second is under half a
# QRS complex and shorter than the label block itself, so it cannot pull in an unlabelled
# beat; it exists purely to accept the off-by-one spans above.
SPAN_SLACK_SAMPLES = int(0.1 * config.SAMPLING_RATE)

STAT_KEYS = ('record_files', 'segments', 'N', 'V', 'S', 'steps_N', 'steps_V', 'steps_S',
             'skipped_flat', 'skipped_short_span', 'skipped_no_file', 'skipped_duplicate',
             'errors')


def new_stats():
    return dict.fromkeys(STAT_KEYS, 0)


def _content_key(path):
    """(size, md5) of a .dat file - its identity as a recording, not as a file name."""
    with open(path, 'rb') as f:
        return os.fstat(f.fileno()).st_size, hashlib.md5(f.read()).hexdigest()


def _record_files(db_name, study_id, event_id, dedupe=True):
    """The distinct .dat recordings of one event, plus the number of copies dropped.

    Two layouts are allowed for (some datasets nest one folder deeper), and byte-identical
    files are collapsed to one. The dedupe is by CONTENT, not by name: a folder with two
    genuinely different recordings keeps both, which is the old behaviour, while dataset-2's
    two identical copies of the same strip become one.

    Hashing costs one extra read of a 90 kB file per event, against reading and band-passing
    it - it does not show up in the build time.
    """
    pattern = os.path.join(config.DATA_DIR, db_name, f"{study_id}/{event_id}/*.dat")
    files = sorted(glob.glob(pattern))
    if not files:
        files = sorted(glob.glob(os.path.join(config.DATA_DIR, db_name, '*',
                                              f"{study_id}/{event_id}/*.dat")))
    if not dedupe or len(files) < 2:
        return files, 0

    seen, unique = set(), []
    for path in files:
        try:
            key = _content_key(path)
        except OSError:
            unique.append(path)          # unreadable here fails later, with a real message
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique, len(files) - len(unique)


def window_starts(i_start, i_end, total, segment_samples=config.SEGMENT_SAMPLES,
                  hop=None, slack=SPAN_SLACK_SAMPLES):
    """Start offsets of the windows that cover the reviewed span [i_start, i_end).

    Sliding by `hop` while the window stays inside the span, then:
      * a final window anchored at the span's END, so the last up-to-hop samples of a long
        span are not thrown away
      * for a span up to `slack` short of one window, a single window centred on the span
    Returns [] when the span is genuinely too short, which the caller counts.
    """
    hop = config.SEGMENT_STRIDE_SECONDS * config.SAMPLING_RATE if hop is None else hop
    i_start, i_end = max(0, int(i_start)), min(int(i_end), int(total))
    span = i_end - i_start
    if span < segment_samples - slack or total < segment_samples:
        return []

    if span < segment_samples:                      # the 2499-sample case
        start = i_start - (segment_samples - span) // 2
        return [int(np.clip(start, 0, total - segment_samples))]

    starts = list(range(i_start, i_end - segment_samples + 1, hop))
    tail = i_end - segment_samples
    if tail > starts[-1]:
        starts.append(tail)
    return starts


def process_record(record_info, db_name):
    """Cut one reviewed record into labelled windows.

    Returns (segments, labels, stats) - lists of (SEGMENT_SAMPLES, IN_CHANNELS) float32 and
    (OUTPUT_STEPS,) int64 arrays, plus a counter dict. Never raises: one unreadable record
    out of half a million must not take a build down, so failures are counted and reported.
    """
    study_id, event_id, channel, start_sample, stop_sample = record_info
    stats = new_stats()
    segments, labels = [], []

    paths, duplicates = _record_files(db_name, study_id, event_id)
    stats['skipped_duplicate'] += duplicates
    if not paths:
        stats['skipped_no_file'] += 1
        return segments, labels, stats

    for path in paths:
        path = path[:-4]
        try:
            record = wfdb.rdrecord(path)
            annotation = wfdb.rdann(path, 'atr')
            raw = np.nan_to_num(record.p_signal)
            ann_samples = np.asarray(annotation.sample, dtype=np.int64)
            ann_symbols = np.asarray(annotation.symbol)

            # Sample positions live at the record's own rate; rescale everything by one
            # ratio so the signal, the annotations and the reviewed span stay aligned.
            ratio = config.SAMPLING_RATE / record.fs
            if record.fs != config.SAMPLING_RATE:
                raw = resample_leads(raw, record.fs)
                ann_samples = np.round(ann_samples * ratio).astype(np.int64)

            leads = build_leads(raw, fs=config.SAMPLING_RATE, primary=int(channel))
            i_start = int(round(int(start_sample) * ratio))
            i_end = int(round(int(stop_sample) * ratio))

            starts = window_starts(i_start, i_end, len(leads))
            if not starts:
                stats['skipped_short_span'] += 1
                continue

            size = config.SEGMENT_SAMPLES
            for start in starts:
                window = leads[start:start + size]
                if is_flat(window):
                    stats['skipped_flat'] += 1
                    continue

                # Beats inside the window, a few samples clear of its edges
                idx = np.flatnonzero((ann_samples >= start + 5)
                                     & (ann_samples < start + size - 5))
                window_labels = labels_from_annotations(ann_samples[idx] - start,
                                                        ann_symbols[idx])
                segments.append(normalize_window(window).astype(np.float32))
                labels.append(window_labels)

                stats['segments'] += 1
                for name in ('N', 'V', 'S'):
                    value = config.CLASS_NAMES.index(name)
                    hit = int(np.count_nonzero(window_labels == value))
                    stats[name] += hit > 0
                    stats[f'steps_{name}'] += hit
            stats['record_files'] += 1
        except Exception as e:                      # one broken record must not stop a build
            stats['errors'] += 1
            stats['last_error'] = f"{os.path.basename(path)}: {e}"

    return segments, labels, stats


def _flush(segments, labels, study_ids, out_dir, db_name, split, batch_no):
    """Write one batch: segments, labels and the study each segment came from.

    The study id per segment is what makes the split auditable straight from the data
    (see splits.audit_written_data): without it the only way to check the separation is to
    recompute what the split *should* have been, which cannot catch a build that wrote
    something else.
    """
    base = os.path.join(out_dir, f"{db_name}_{split}")
    np.save(f"{base}_segments_batch_{batch_no}.npy", np.asarray(segments, dtype=np.float32))
    np.save(f"{base}_labels_batch_{batch_no}.npy", np.asarray(labels, dtype=np.int64))
    np.save(f"{base}_studyids_batch_{batch_no}.npy", np.asarray(study_ids, dtype=np.int64))
    return len(segments)


def _build_split(db_name, split, chosen, out_dir, workers):
    """Process `chosen` records into npy batches under `out_dir`.

    Records are handed to a process pool in order (imap, not imap_unordered) so a rebuild
    with the same inputs writes byte-identical batches, and each segment is paired with the
    study it came from as the pool results come back.
    """
    totals = new_stats()
    errors = []
    buf_seg, buf_lab, buf_sid = [], [], []
    batch_no = 0

    def flush(force=False):
        nonlocal buf_seg, buf_lab, buf_sid, batch_no
        n = config.BATCH_SEGMENTS
        while len(buf_seg) >= n or (force and buf_seg):
            k = min(n, len(buf_seg))
            _flush(buf_seg[:k], buf_lab[:k], buf_sid[:k], out_dir, db_name, split, batch_no)
            print(f"  batch {batch_no}: {k} segments")
            buf_seg, buf_lab, buf_sid = buf_seg[k:], buf_lab[k:], buf_sid[k:]
            batch_no += 1

    worker = partial(process_record, db_name=db_name)
    ctx = get_context('spawn')
    chunk = max(1, min(64, (len(chosen) // (workers * 8)) or 1))
    with ctx.Pool(workers) as pool:
        results = pool.imap(worker, chosen, chunksize=chunk)
        # disable=not a tty: the bar refreshes hundreds of times a second at this throughput,
        # and redirected to a log file every refresh becomes its own line.
        for record_info, (segments, labels, stats) in tqdm(
                zip(chosen, results), total=len(chosen), desc=f"{db_name} {split}",
                smoothing=0.05, disable=not sys.stderr.isatty(), mininterval=2.0):
            buf_seg.extend(segments)
            buf_lab.extend(labels)
            buf_sid.extend([int(record_info[0])] * len(segments))
            for key in STAT_KEYS:
                totals[key] += stats[key]
            if 'last_error' in stats and len(errors) < 5:
                errors.append(stats['last_error'])
            flush()
    flush(force=True)
    return totals, errors


def build_dataset(db_name, limit=None, workers=None):
    """Process every reviewed record of `db_name` into train/eval npy batches."""
    workers = workers or config.WORKERS
    csv_path = os.path.join(config.DATA_DIR, db_name, config.DATASET_CSV)
    df = pd.read_csv(csv_path)
    records = list(zip(df['study_id'], df['event_id'], df['channel'],
                       df['start_sample'], df['stop_sample']))

    by_study = defaultdict(list)
    for rec in records:
        by_study[rec[0]].append(rec)

    # Drop the held-out studies before anything else, so they can reach neither split
    excluded, test_ids = splits.held_out_studies()
    held_out = [sid for sid in by_study if str(sid).strip() in excluded]
    for sid in held_out:
        del by_study[sid]
    print(f"{db_name}: held out {len(held_out)} eval/test studies, {len(by_study)} left")
    if not by_study:
        raise ValueError(f"{db_name}: every study is held out, nothing to build")

    train_ids, eval_ids = splits.split_studies(list(by_study))
    print(f"{db_name}: hash split -> {len(train_ids)} train / {len(eval_ids)} eval studies")

    out_root = os.path.join(config.NPY_DIR, db_name)
    # Fail-fast: runs before the first npy is written, so a bad split never reaches disk.
    splits.verify_split_integrity(db_name, train_ids, eval_ids, test_ids=test_ids,
                                  out_path=os.path.join(out_root, 'split_verification.json'))

    summary = {}
    for split, study_ids in (("train", train_ids), ("eval", eval_ids)):
        out_dir = os.path.join(out_root, split)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        chosen = [rec for sid in study_ids for rec in by_study[sid]]
        if limit:
            chosen = chosen[:limit]
        print(f"\n{db_name} {split}: {len(chosen)} records, {workers} workers")

        totals, errors = _build_split(db_name, split, chosen, out_dir, workers)
        summary[split] = totals

        print(f"{db_name} {split}: {totals['segments']:,} segments from "
              f"{totals['record_files']:,} record files over {len(chosen):,} reviewed events "
              f"| windows containing N: {totals['N']:,} V: {totals['V']:,} S: {totals['S']:,} "
              f"| label steps N: {totals['steps_N']:,} V: {totals['steps_V']:,} "
              f"S: {totals['steps_S']:,}")
        print(f"  skipped: flat {totals['skipped_flat']:,}, span too short "
              f"{totals['skipped_short_span']:,}, no file {totals['skipped_no_file']:,}, "
              f"duplicate copies {totals['skipped_duplicate']:,}, "
              f"errors {totals['errors']:,}")
        for message in errors:
            print(f"  error e.g. {message}")

        # Records but zero segments means they were never found on disk - fail loudly
        # instead of silently writing an empty split.
        if chosen and totals['segments'] == 0:
            raise ValueError(
                f"{db_name} {split}: {len(chosen)} records produced 0 segments - records not "
                f"found? Check the layout under {os.path.join(config.DATA_DIR, db_name)}")

    with open(os.path.join(out_root, 'studyid_info.json'), 'w') as f:
        json.dump({'train_studyids': [int(s) for s in train_ids],
                   'eval_studyids': [int(s) for s in eval_ids],
                   'held_out_studyids': sorted(int(s) for s in held_out),
                   'segment_samples': config.SEGMENT_SAMPLES,
                   'in_channels': config.IN_CHANNELS,
                   'primary_lead_first': config.PRIMARY_LEAD_FIRST,
                   'output_steps': config.OUTPUT_STEPS,
                   'num_classes': config.NUM_CLASSES,
                   'span_slack_samples': SPAN_SLACK_SAMPLES,
                   'stats': summary}, f, indent=4)
    return summary


def build_all(db_names=None, limit=None, workers=None):
    grand = {}
    for db in (db_names or config.TRAIN_DATASETS):
        print(f"\n{'=' * 70}\n{db}\n{'=' * 70}")
        grand[db] = build_dataset(db, limit=limit, workers=workers)

    print(f"\n{'=' * 70}\nnpy build totals\n{'=' * 70}")
    for split in ('train', 'eval'):
        seg = sum(v[split]['segments'] for v in grand.values() if split in v)
        s = sum(v[split]['steps_S'] for v in grand.values() if split in v)
        vv = sum(v[split]['steps_V'] for v in grand.values() if split in v)
        n = sum(v[split]['steps_N'] for v in grand.values() if split in v)
        print(f"{split:6s}: {seg:>9,} segments | label steps N {n:>12,} V {vv:>11,} S {s:>11,}")
    return grand
