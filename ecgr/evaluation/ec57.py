"""Beat-level evaluation: sweep whole records, decode beats, score with bxb.

Four kinds of source are scored the same way:

  * the Physionet benchmark databases (config.EC57_DBS) - long records, annotated on
    channel 0, reference beats in .atr (.qrs for afdb)
  * the held-out portal beat-eval set (config.PORTAL_EVAL_SETS) - 60 s strips, annotated on
    whichever lead the reviewer worked on, and only inside a reviewed window
  * the portal TRAIN and EVAL splits themselves (`portal-train`, `portal-eval`): a
    deterministic sample of the reviewed events the tfrecords were cut from, scored the same
    way as the beat-eval set so the three portal numbers are directly comparable. The eval
    split is the same studies the step-level F1 is computed on, at beat level; the train
    split is data the model has SEEN, so its number is an overfitting diagnostic - the gap to
    portal-eval - and never a performance claim.

**Leads.** The model takes config.IN_CHANNELS leads with the annotated one on channel 0, and
none of the five EC57 databases has that many: they are two-lead recordings annotated on the
first signal. So ONE lead is chosen and repeated across the channel axis
(`lead_mode='duplicate'`, what `auto` picks for these databases). The portal beat-eval set is
natively 3-lead like the training data, so `auto` gives it its real montage; `--lead-mode
duplicate` forces the single-lead reading everywhere, which is the strictest test and the one
comparable with the older 1-lead checkpoints. Training reproduces the duplicated case at
config.AUGMENT_LEAD_DUPLICATE_PROB, so neither reading is out of distribution.

Predictions are written into <ec57_out>/_ann/<db>/ as real files and kept: bxb can then be
re-run with a different exclusion list or a fixed script without paying for inference again.
Scoring happens in a disposable symlink farm under <ec57_out>/_work/<db>/, so the source
databases are never touched and two models never overwrite each other.
"""
import csv
import hashlib
import json
import os
import re
import shutil

import numpy as np
import tensorflow as tf
import wfdb

from .. import config, models  # noqa: F401  - models registers the custom layers
from ..labels import decode_beats
from ..signal_ops import build_leads, resample_leads, segment_record
from . import bxb, report


# ---------------------------------------------------------------------------
# One record
# ---------------------------------------------------------------------------

def read_leads(record_path, channel=0, lead_mode='auto', in_channels=None):
    """(leads, raw_length, fs) for one record: (N, in_channels) at config.SAMPLING_RATE.

    `channel` is the 0-based signal the annotations refer to; it becomes channel 0.
    `lead_mode` decides what fills the rest:
        'duplicate' - repeat that one lead, the single-lead reading
        'native'    - the record's own leads, filled up by repetition if it has too few
        'auto'      - 'native' when the record has enough leads, 'duplicate' otherwise
    """
    n_ch = config.IN_CHANNELS if in_channels is None else int(in_channels)
    rec = wfdb.rdrecord(record_path)
    raw = np.nan_to_num(np.atleast_2d(rec.p_signal.T).T)     # (N, n_sig)
    n_sig = raw.shape[1]
    if not 0 <= channel < n_sig:
        raise ValueError(f"{os.path.basename(record_path)}: annotated channel {channel} but "
                         f"the record has {n_sig} signal(s)")

    if lead_mode == 'auto':
        lead_mode = 'native' if n_sig >= n_ch else 'duplicate'
    if lead_mode == 'duplicate':
        raw = raw[:, channel:channel + 1]
        channel = 0

    length = len(raw)
    if rec.fs != config.SAMPLING_RATE:
        raw = resample_leads(raw, rec.fs)
    leads = build_leads(raw, fs=config.SAMPLING_RATE, in_channels=n_ch, primary=channel)
    return leads, length, rec.fs


def predict_segments(model, segments, batch_size=None):
    """Forward pass over (n, T, C) segments through a compiled function cached on the model.

    `model.predict` builds a data iterator and a progress loop on every call, and that fixed
    cost dominates for the portal strips: a 60 s record is 7 windows, and predict() spends
    113 ms on it where the compiled call spends 8 - measured on the 30k size. Over the 5,227
    beat-eval records plus two 5,000-record split samples that is ~30 minutes per model of
    pure overhead. Long Physionet records are compute-bound either way.
    """
    fn = getattr(model, '_ecgr_predict', None)
    if fn is None:
        @tf.function(reduce_retracing=True)
        def fn(x):
            return model(x, training=False)
        model._ecgr_predict = fn
    size = batch_size or config.BATCH_SIZE
    out = [fn(tf.constant(segments[i:i + size])).numpy()
           for i in range(0, len(segments), size)]
    return np.concatenate(out, axis=0)


def predict_record(model, record_path, record_name, out_dir, channel=0, s_boost=1.0,
                   batch_size=None, lead_mode='auto'):
    """Predict one record and write its .<BEAT_EXTENSION> annotation into `out_dir`."""
    leads, raw_length, fs = read_leads(record_path, channel=channel, lead_mode=lead_mode)

    segments, starts = segment_record(leads)
    preds = predict_segments(model, segments, batch_size)

    # signal_length keeps the decoder inside the real signal: a record shorter than one
    # window is edge-padded, and a detection in that padding is an artefact of the padding.
    positions, symbols = decode_beats(preds, segments, starts, s_boost=s_boost,
                                      signal_length=len(leads))
    if len(positions) == 0:
        return 0

    # Back to the record's own sampling rate for bxb, clipped to the signal
    if fs != config.SAMPLING_RATE:
        positions = np.round(positions * (fs / config.SAMPLING_RATE)).astype(np.int64)
    inside = positions < raw_length
    positions, symbols = positions[inside], symbols[inside]
    if len(positions) == 0:
        return 0

    wfdb.Annotation(record_name=record_name, extension=config.BEAT_EXTENSION,
                    sample=positions, symbol=list(symbols),
                    fs=fs).wrann(write_fs=True, write_dir=out_dir)
    return len(positions)


# ---------------------------------------------------------------------------
# Record inventories and the scoring directory
# ---------------------------------------------------------------------------

_CHANNEL_COMMENT = re.compile(r'^#\s*-?\s*channel\s*:\s*(\d+)', re.IGNORECASE | re.MULTILINE)


def record_channel(record_path, default=0):
    """0-based lead the record is annotated on: '# channel: N' in the .hea, else the .json."""
    try:
        with open(record_path + '.hea', errors='replace') as f:
            m = _CHANNEL_COMMENT.search(f.read())
        if m:
            return int(m.group(1))
    except OSError:
        pass
    try:
        with open(record_path + '.json') as f:
            value = json.load(f).get('channel')
        if value is not None:
            return int(value)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return default


def list_records(src_dir, extensions=('hea', 'dat', 'atr')):
    """Names of every record in `src_dir` that has all of `extensions`."""
    names = sorted({f[:-4] for f in os.listdir(src_dir) if f.endswith('.dat')})
    complete = [n for n in names
                if all(os.path.exists(os.path.join(src_dir, f"{n}.{e}")) for e in extensions)]
    missing = len(names) - len(complete)
    if missing:
        print(f"  {missing} records lack one of {extensions}, skipped")
    return complete


def annotation_dir(ec57_out, db_name):
    return os.path.join(ec57_out, '_ann', db_name)


def build_scoring_dir(src_dir, ann_dir, work_dir, records, extensions):
    """Assemble a disposable symlink farm bxb can run in; return the records it holds.

    A record with no stored prediction is left out, so the caller controls exactly what gets
    scored without ever mutating the stored annotations or the source database.
    """
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    def link(src, dst):
        if not os.path.islink(dst) and not os.path.exists(dst):
            os.symlink(src, dst)

    scored, missing = [], []
    for name in records:
        ann = os.path.join(ann_dir, f"{name}.{config.BEAT_EXTENSION}")
        if not os.path.exists(ann):
            missing.append(name)
            continue
        for ext in extensions:
            path = os.path.join(src_dir, f"{name}.{ext}")
            if os.path.exists(path):
                link(path, os.path.join(work_dir, f"{name}.{ext}"))
        link(ann, os.path.join(work_dir, f"{name}.{config.BEAT_EXTENSION}"))
        scored.append(name)

    if missing:
        print(f"  {len(missing)} records have no stored .{config.BEAT_EXTENSION} yet "
              f"(e.g. {missing[:3]}) - run without --bxb-only to predict them")
    return scored


def _print_report(path):
    print(f"\nEC57 report: {path}")
    with open(path) as f:
        for line in f:
            if line.startswith(('Record', 'Average', 'Gross', 'Total', 'Summary')):
                print('  ' + line.rstrip())


# ---------------------------------------------------------------------------
# The portal train / eval splits as bxb sources
# ---------------------------------------------------------------------------

def split_record_name(study_id, event_id):
    """Record name in the scoring directory: unique across every dataset and study.

    The portal files are named after the capture time ("event-strip-captured-2024-11-13-..."),
    which repeats across studies, so the original basenames cannot share one flat folder.
    """
    return f"{study_id}_{event_id}"


def portal_split_records(split, db_names=None):
    """Every reviewed event of `split` ('train' | 'eval') across the training datasets.

    Rows of (db, study_id, event_id, channel, start_sample, stop_sample) from the same CSVs,
    with the held-out studies removed and the side decided by the same hash the data builder
    uses - so this is exactly the population the tfrecords of that split were cut from.

    One event, one row: 8,310 (study, event) pairs are listed by TWO datasets - the
    "AFib - v2" and "filter-vt-svt" sets are re-curations of dataset-2/3/4 events, and only
    1,555 of them agree with the original row on (channel, start, stop). The first dataset in
    `db_names` order keeps the event, i.e. the original review wins over the re-curation.
    Without this the scoring directory tries to link the same recording under one name
    twice (FileExistsError), and the sample would score a strip twice where it did fit.
    """
    from ..data import splits
    excluded, _ = splits.held_out_studies()
    rows, seen, relisted = [], set(), 0
    for db in (db_names or config.TRAIN_DATASETS):
        path = os.path.join(config.DATA_DIR, db, config.DATASET_CSV)
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                sid, eid = str(r['study_id']).strip(), str(r['event_id']).strip()
                if sid in excluded or splits.study_split_side(sid) != split:
                    continue
                if (sid, eid) in seen:
                    relisted += 1
                    continue
                seen.add((sid, eid))
                rows.append((db, sid, eid, int(r['channel']),
                             int(r['start_sample']), int(r['stop_sample'])))
    if relisted:
        print(f"  {relisted:,} events listed by a second dataset - the first listing is kept")
    return rows


def sample_split_records(rows, n):
    """The first `n` rows in md5 order of their record name; n = 0 or None means all.

    Hash order rather than CSV order or a seeded RNG: it is the same subset for every model,
    every rerun and every machine, and adding a dataset to the CSVs does not reshuffle it.
    """
    ordered = sorted(rows, key=lambda r: hashlib.md5(
        split_record_name(r[1], r[2]).encode()).hexdigest())
    return ordered[:n] if n else ordered


def write_scoring_header(src_hea, dst_hea, name, channel, start, stop):
    """Rewrite a portal .hea under a new record name, with the reviewed window as comments.

    Only the record name and the .dat file name change; every signal-line field is kept
    verbatim, so wfdb and bxb read the same gains, baselines and formats as before. The old
    comments are dropped and replaced by the three the mark-window bxb script reads - taken
    from the CSV, not from the header, because the eight datasets spell the window three
    different ways (startSample / startMarkSample / eventStartSample) and dataset-2's
    headers name a .dat that is not the one beside them.
    """
    with open(src_hea, errors='replace') as f:
        lines = [line.rstrip('\n') for line in f]
    head = lines[0].split()
    nsig = int(head[1])
    if len(head) > 3 and head[3].isdigit():
        stop = min(int(stop), int(head[3]))
    head[0] = name
    out = [' '.join(head)]
    for line in lines[1:1 + nsig]:
        tokens = line.split()
        tokens[0] = f"{name}.dat"
        out.append(' '.join(tokens))
    out += [f"# channel: {int(channel)}",
            f"# startMarkSample: {int(start)}",
            f"# stopMarkSample: {int(stop)}"]
    with open(dst_hea, 'w') as f:
        f.write('\n'.join(out) + '\n')


def build_split_scoring_dir(records, ann_dir, work_dir):
    """Symlink farm for a split sample: `records` are (name, record_path, channel, start, stop).

    The .dat and .atr are symlinked under the new name, the .hea is rewritten (it embeds the
    file name), and the stored .ain prediction is linked in. Returns the names that have one.
    """
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    scored, missing = [], 0
    for name, src, channel, start, stop in records:
        ann = os.path.join(ann_dir, f"{name}.{config.BEAT_EXTENSION}")
        if not os.path.exists(ann):
            missing += 1
            continue
        write_scoring_header(src + '.hea', os.path.join(work_dir, name + '.hea'),
                             name, channel, start, stop)
        os.symlink(src + '.dat', os.path.join(work_dir, name + '.dat'))
        os.symlink(src + '.atr', os.path.join(work_dir, name + '.atr'))
        os.symlink(ann, os.path.join(work_dir, f"{name}.{config.BEAT_EXTENSION}"))
        scored.append(name)
    if missing:
        print(f"  {missing} records have no stored .{config.BEAT_EXTENSION} yet - run "
              f"without --bxb-only to predict them")
    return scored


def score_portal_split(model, split, ec57_out, max_records=None, s_boost=1.0,
                       bxb_only=False, lead_mode=None):
    """Predict + bxb over a deterministic sample of one portal split, inside reviewed windows.

    Scored exactly like the beat-eval set (same lead handling, same mark-window script), so
    `portal-train`, `portal-eval` and `dataset-v4-beat` are three points on one scale:
    the first is data the model trained on, the second the studies its step-level F1 was
    selected on, the third a curated holdout it never saw in any form.
    """
    from ..data.build_npy import _record_files
    db_name = f"portal-{split}"
    lead_mode = lead_mode or config.EC57_LEAD_MODE
    n = config.PORTAL_SPLIT_RECORDS if max_records is None else max_records

    rows = portal_split_records(split)
    chosen = sample_split_records(rows, n)
    print(f"===== {db_name}: {len(chosen):,} of {len(rows):,} reviewed events "
          f"({lead_mode} leads)"
          f"{' - SEEN IN TRAINING, overfitting diagnostic only' if split == 'train' else ''}"
          f" =====")

    resolved, unresolved = [], 0
    for db, sid, eid, channel, start, stop in chosen:
        files, _ = _record_files(db, sid, eid)
        if not files:
            unresolved += 1
            continue
        resolved.append((split_record_name(sid, eid), files[0][:-4], channel, start, stop))
    if unresolved:
        print(f"  {unresolved} events have no record on disk, skipped")

    ann_dir = annotation_dir(ec57_out, db_name)
    os.makedirs(ann_dir, exist_ok=True)
    if bxb_only:
        print(f"  --bxb-only: reusing the stored predictions in {ann_dir}")
    else:
        empty = errors = 0
        for i, (name, src, channel, _, _) in enumerate(resolved, 1):
            try:
                empty += predict_record(model, src, name, ann_dir, channel=channel,
                                        s_boost=s_boost, lead_mode=lead_mode) == 0
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  error on {name}: {e}")
            if i % 1000 == 0 or i == len(resolved):
                print(f"  {i}/{len(resolved)} records predicted")
        if empty or errors:
            print(f"  {empty} records yielded no beats, {errors} failed")

    work_dir = os.path.join(ec57_out, '_work', db_name)
    scored = build_split_scoring_dir(resolved, ann_dir, work_dir)
    if not scored:
        print(f"{db_name}: nothing to score")
        return None
    print(f"  scoring {len(scored)}/{len(resolved)} records")

    path = bxb.run_bxb(db_name, work_dir, ec57_out, 'atr', config.BEAT_EXTENSION,
                       script=bxb.SCRIPT_MARK_WINDOW)
    if path:
        _print_report(path)
    return path


# ---------------------------------------------------------------------------
# One database
# ---------------------------------------------------------------------------

def score_physionet_db(model, db_name, ec57_out, max_records=None, s_boost=1.0,
                       bxb_only=False, exclude=None, lead_mode=None):
    src_dir = os.path.join(config.PHYSIONET_DIR, db_name)
    if not os.path.isdir(src_dir):
        print(f"database not found: {src_dir}")
        return None

    lead_mode = lead_mode or config.EC57_LEAD_MODE
    channel = config.EC57_LEAD.get(db_name, config.EC57_LEAD_DEFAULT)
    ref_ext = config.EC57_BEAT_REF_EXT.get(db_name, 'atr')
    records = sorted(f[:-4] for f in os.listdir(src_dir) if f.endswith('.dat'))

    skip = {str(r) for r in (exclude or config.EC57_EXCLUDE_RECORDS).get(db_name, [])}
    present = sorted(skip & set(records))
    if present:
        records = [r for r in records if r not in skip]
        print(f"{db_name}: excluding {len(present)} records: {', '.join(present)}")
    if max_records:
        records = records[:max_records]
    print(f"===== {db_name}: {len(records)} records, lead {channel} ({lead_mode}) =====")

    ann_dir = annotation_dir(ec57_out, db_name)
    os.makedirs(ann_dir, exist_ok=True)
    if bxb_only:
        print(f"  --bxb-only: reusing the stored predictions in {ann_dir}")
    else:
        for i, name in enumerate(records, 1):
            try:
                n = predict_record(model, os.path.join(src_dir, name), name, ann_dir,
                                   channel=channel, s_boost=s_boost, lead_mode=lead_mode)
                if n == 0:
                    print(f"  {name}: NO beats detected - no annotation written")
                elif i % 20 == 0 or i == len(records):
                    print(f"  {i}/{len(records)} records predicted (last: {name}, {n} beats)")
            except Exception as e:
                print(f"  error on {name}: {e}")

    work_dir = os.path.join(ec57_out, '_work', db_name)
    scored = build_scoring_dir(src_dir, ann_dir, work_dir, records,
                               ('hea', 'dat', ref_ext, 'atr'))
    if not scored:
        print(f"{db_name}: nothing to score")
        return None
    print(f"  scoring {len(scored)} records")

    path = bxb.run_bxb(db_name, work_dir, ec57_out, ref_ext, config.BEAT_EXTENSION,
                       script=bxb.SCRIPT_FULL)
    if path:
        _print_report(path)
    return path


def score_portal_set(model, db_name, src_dir, ec57_out, max_records=None, s_boost=1.0,
                     bxb_only=False, mark_window=True, lead_mode=None):
    """Predict + bxb over one flat portal eval folder.

    mark_window=True scores only the reviewed window of each strip, taken from its .hea
    (`# startMarkSample` / `# stopMarkSample`). That window is the part a reviewer certified;
    the rest of the 60 s strip carries annotations nobody signed off on, and scoring the
    whole record pools those in and dilutes the rates - a typical strip keeps all of its S
    beats inside the window but only a fraction of its N beats.
    """
    if not os.path.isdir(src_dir):
        print(f"portal eval set not found: {src_dir}")
        return None

    lead_mode = lead_mode or config.EC57_LEAD_MODE
    print(f"===== {db_name} ({lead_mode} leads) =====")
    records = list_records(src_dir)
    if max_records:
        records = records[:max_records]
    print(f"{len(records)} records in {src_dir}")

    ann_dir = annotation_dir(ec57_out, db_name)
    os.makedirs(ann_dir, exist_ok=True)
    if bxb_only:
        print(f"  --bxb-only: reusing the stored predictions in {ann_dir}")
    else:
        empty = 0
        for i, name in enumerate(records, 1):
            try:
                n = predict_record(model, os.path.join(src_dir, name), name, ann_dir,
                                   channel=record_channel(os.path.join(src_dir, name)),
                                   s_boost=s_boost, lead_mode=lead_mode)
                empty += (n == 0)
                if i % 500 == 0 or i == len(records):
                    print(f"  {i}/{len(records)} records predicted")
            except Exception as e:
                print(f"  error on {name}: {e}")
        if empty:
            print(f"  {empty}/{len(records)} records yielded no beats at all")

    work_dir = os.path.join(ec57_out, '_work', db_name)
    scored = build_scoring_dir(src_dir, ann_dir, work_dir, records, ('hea', 'dat', 'atr'))
    if not scored:
        print(f"{db_name}: nothing to score")
        return None
    print(f"  scoring {len(scored)}/{len(records)} records")

    # Either way -f is set explicitly: these strips are 60 s, far shorter than bxb's default
    # 5-minute learning period, which would leave an empty interval.
    path = bxb.run_bxb(db_name, work_dir, ec57_out, 'atr', config.BEAT_EXTENSION,
                       script=bxb.SCRIPT_MARK_WINDOW if mark_window else bxb.SCRIPT_SHORT)
    if path:
        _print_report(path)
    return path


# ---------------------------------------------------------------------------
# The whole evaluation
# ---------------------------------------------------------------------------

def run(checkpoint, tag, dbs=None, max_records=None, s_boost=1.0, bxb_only=False,
        skip_physionet=False, skip_portal=False, mark_window=True, lead_mode=None,
        splits=None, split_records=None):
    """Score one checkpoint over Physionet, the portal beat-eval set and the portal splits.

    `splits` names the portal splits to sample ('train', 'eval'); None takes
    config.PORTAL_SPLITS, an empty tuple scores none. `split_records` is the sample size per
    split (None = config.PORTAL_SPLIT_RECORDS, 0 = every reviewed event of the split).
    """
    from ..training.train import setup_gpus
    setup_gpus()
    ec57_out = os.path.join(config.EC57_DIR, tag)
    os.makedirs(ec57_out, exist_ok=True)
    lead_mode = lead_mode or config.EC57_LEAD_MODE
    splits = tuple(config.PORTAL_SPLITS) if splits is None else tuple(splits)

    model = None
    if not bxb_only:
        print(f"loading {checkpoint}")
        model = tf.keras.models.load_model(checkpoint, compile=False)
        expected = (config.SEGMENT_SAMPLES, config.IN_CHANNELS)
        if tuple(model.input_shape[1:]) != expected:
            raise ValueError(
                f"{checkpoint} takes {model.input_shape[1:]} but this run is configured for "
                f"{expected}. Set ECGR_IN_CHANNELS to match the checkpoint.")
        print(f"model: {model.name}, {model.count_params():,} parameters\n")

    with open(os.path.join(ec57_out, 'checkpoint.txt'), 'w') as f:
        f.write(f"{checkpoint}\n"
                f"model: {model.name if model else '(not loaded, --bxb-only)'}\n"
                f"params: {model.count_params() if model else '-'}\n"
                f"s_boost: {s_boost}\n"
                f"in_channels: {config.IN_CHANNELS}\n"
                f"lead_mode: {lead_mode}\n"
                f"portal_splits: {', '.join(splits) or '-'} "
                f"x {config.PORTAL_SPLIT_RECORDS if split_records is None else split_records}"
                f" records\n")

    if not skip_physionet:
        for db in (dbs or config.EC57_DBS):
            score_physionet_db(model, db, ec57_out, max_records=max_records,
                               s_boost=s_boost, bxb_only=bxb_only, lead_mode=lead_mode)
            print()

    if not skip_portal:
        for db_name, src_dir in sorted(config.PORTAL_EVAL_SETS.items()):
            score_portal_set(model, db_name, src_dir, ec57_out, max_records=max_records,
                             s_boost=s_boost, bxb_only=bxb_only, mark_window=mark_window,
                             lead_mode=lead_mode)
            print()

    for split in splits:
        score_portal_split(model, split, ec57_out, max_records=split_records,
                           s_boost=s_boost, bxb_only=bxb_only, lead_mode=lead_mode)
        print()

    return report.summarize(ec57_out)
