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
first signal. Two independent knobs decide what reaches the model. `lead_mode`
(config.EC57_LEAD_MODE, 'native' since 2026-09-20) says how many of the record's REAL leads
go in: 'native' both of them, 'single' only the annotated one ('duplicate' is its deprecated
alias) - the strictest test, and the reading comparable with the older 1-lead checkpoints -
and 'auto' takes 'native' only where the record ALREADY has IN_CHANNELS leads, which on all
five of these databases means 'single'. `fill_mode` (config.LEAD_FILL_MODE, 'zero') says what
occupies the channels left over: silence, or the annotated lead repeated. Training produces
both cases on purpose (config.AUGMENT_LEAD_DROP_PROB, config.AUGMENT_LEAD_DUPLICATE_PROB), so
no reading is out of distribution. The portal beat-eval set is natively 3-lead like the
training data, so it gets its real montage under 'native' and nothing is filled.

Predictions are written into <ec57_out>/_ann/<db>/ as real files and kept: bxb can then be
re-run with a different exclusion list or a fixed script without paying for inference again.
Scoring happens in a disposable symlink farm under <ec57_out>/_work/<db>/, so the source
databases are never touched and two models never overwrite each other.

**Output 2.** A two-output model also says, per record, which lead it found most readable
(labels.best_lead over its lead_quality output). That answer is written next to the
predictions as `_ann/<db>/lead_quality.csv` - record, best lead in the RECORD's own channel
numbering, and the per-lead mean quality - and summarised per source in
`<ec57_out>/<db>/lead_quality_summary.json`. On the portal beat-eval set the summary also
reports how often the model's choice coincides with the channel the reviewer worked on: a
diagnostic, not a score - the reviewer's channel is a default 83% of the time, not a
judgement of quality.

**Window length.** The sweep reads its geometry from the CHECKPOINT (config.apply_geometry):
a 10 s model is swept in 10 s windows, a 60 s model in 60 s windows, by the same code, which
is what makes the non-regression comparison between the two families apples to apples.
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
from ..labels import best_lead, decode_beats
from ..signal_ops import build_leads, lead_order, resample_leads, segment_record
from . import bxb, report


# ---------------------------------------------------------------------------
# One record
# ---------------------------------------------------------------------------

def read_leads(record_path, channel=0, lead_mode=None, in_channels=None, fill_mode=None):
    """(leads, raw_length, fs) for one record: (N, in_channels) at config.SAMPLING_RATE.

    `channel` is the 0-based signal the annotations refer to; it becomes channel 0.

    Two independent questions, one each:
      `lead_mode` - how many of the record's REAL leads to use. None takes
                    config.EC57_LEAD_MODE, so a bare call measures what the pipeline does;
                    it used to default to 'auto', which is a different reading on every EC57
                    database and made a direct call silently disagree with `ecgr ec57`.
          'native'  - all of them (config default)
          'single'  - only the annotated one ('duplicate' is a deprecated alias, from when
                      filling was always by repetition)
          'auto'    - 'native' only when the record already has IN_CHANNELS leads, else
                      'single'. Every EC57 database has two leads and the model takes three,
                      so on all five of them 'auto' means 'single' - it is NOT a synonym for
                      'native' there.
      `fill_mode` - what occupies the channels left over (default config.LEAD_FILL_MODE)
          'zero'      - silence, the default: what the model sees when an electrode is off
          'duplicate' - repeat the annotated lead
    """
    n_ch = config.IN_CHANNELS if in_channels is None else int(in_channels)
    lead_mode = lead_mode or config.EC57_LEAD_MODE
    fill = config.LEAD_FILL_MODE if fill_mode is None else fill_mode
    rec = wfdb.rdrecord(record_path)
    raw = np.nan_to_num(np.atleast_2d(rec.p_signal.T).T)     # (N, n_sig)
    n_sig = raw.shape[1]
    if not 0 <= channel < n_sig:
        raise ValueError(f"{os.path.basename(record_path)}: annotated channel {channel} but "
                         f"the record has {n_sig} signal(s)")

    if lead_mode == 'auto':
        lead_mode = 'native' if n_sig >= n_ch else 'single'
    if lead_mode in ('single', 'duplicate'):      # 'duplicate': deprecated alias
        raw = raw[:, channel:channel + 1]
        channel = 0

    length = len(raw)
    if rec.fs != config.SAMPLING_RATE:
        raw = resample_leads(raw, rec.fs)
    leads = build_leads(raw, fs=config.SAMPLING_RATE, in_channels=n_ch, primary=channel,
                        fill_mode=fill)
    return leads, length, rec.fs


class Ensemble:
    """Several trained models read as one: their softmax outputs are averaged per step.

    Averaging probabilities is the one test-time change that tends to raise sensitivity AND
    positive predictivity together - the members' independent false calls cancel while their
    shared true calls add - at the price of one forward pass per member. Every eval stage
    treats it like a model: it has a name, an input shape, a parameter count and, through
    predict_segments, a prediction.
    """

    def __init__(self, members):
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = list(members)
        self.name = 'ensemble(' + '+'.join(m.name for m in self.members) + ')'
        self.input_shape = self.members[0].input_shape

    def count_params(self):
        return sum(m.count_params() for m in self.members)

    @property
    def outputs(self):
        """Mirrors keras.Model.outputs enough for models.has_quality_output: an ensemble emits
        lead quality when every member does."""
        return self.members[0].outputs if all(models.has_quality_output(m)
                                              for m in self.members) else self.members[0].outputs[:1]

    def predict_probs(self, segments, batch_size=None):
        return self.predict_full(segments, batch_size)[0]

    def predict_full(self, segments, batch_size=None):
        results = [predict_segments_full(m, segments, batch_size) for m in self.members]
        beats = np.mean([r[0] for r in results], axis=0)
        qualities = [r[1] for r in results]
        quality = np.mean(qualities, axis=0) if all(q is not None for q in qualities) else None
        return beats, quality


def load_checkpoints(paths, adapt_geometry=True):
    """One .keras path -> that model; several -> an Ensemble of them.

    With adapt_geometry the run's window geometry is re-derived from the checkpoint's input
    shape (config.apply_geometry), so a checkpoint of the 10 s family is swept in 10 s
    windows and one of the 60 s family in 60 s windows - by the same code. Members of an
    ensemble must agree on it.
    """
    paths = [paths] if isinstance(paths, str) else list(paths)
    loaded = [tf.keras.models.load_model(p, compile=False) for p in paths]
    shapes = {tuple(m.input_shape[1:]) for m in loaded}
    if len(shapes) > 1:
        raise ValueError(f"ensemble members disagree on the input shape: {sorted(shapes)}")
    (length, channels), = shapes
    if channels != config.IN_CHANNELS:
        raise ValueError(f"{paths[0]} takes {channels} leads but this run is configured for "
                         f"{config.IN_CHANNELS} (ECGR_IN_CHANNELS)")
    if length != config.SEGMENT_SAMPLES:
        if not adapt_geometry:
            raise ValueError(f"{paths[0]} takes {length} samples but this run is configured "
                             f"for {config.SEGMENT_SAMPLES}")
        steps = models.split_outputs(loaded[0].outputs)[0].shape[1]
        config.apply_geometry(length, steps)
        print(f"geometry     : {length} samples -> {steps} steps, from the checkpoint "
              f"({length / config.SAMPLING_RATE:g} s windows, sweep overlap "
              f"{config.EC57_SEGMENT_OVERLAP / config.SAMPLING_RATE:g} s)")
    return loaded[0] if len(loaded) == 1 else Ensemble(loaded)


def predict_segments(model, segments, batch_size=None):
    """Forward pass over (n, T, C) segments through a compiled function cached on the model.

    `model.predict` builds a data iterator and a progress loop on every call, and that fixed
    cost dominates for the portal strips: a 60 s record is 7 windows, and predict() spends
    113 ms on it where the compiled call spends 8 - measured on the 30k size. Over the 5,227
    beat-eval records plus two 5,000-record split samples that is ~30 minutes per model of
    pure overhead. Long Physionet records are compute-bound either way.
    """
    return predict_segments_full(model, segments, batch_size)[0]


def predict_segments_full(model, segments, batch_size=None):
    """(beat softmax (n, steps, classes), lead quality (n, steps, leads) or None)."""
    if hasattr(model, 'predict_full'):                # an Ensemble
        return model.predict_full(segments, batch_size)
    fn = getattr(model, '_ecgr_predict', None)
    if fn is None:
        @tf.function(reduce_retracing=True)
        def fn(x):
            return model(x, training=False)
        model._ecgr_predict = fn
    size = batch_size or config.BATCH_SIZE
    beats, quality = [], []
    for i in range(0, len(segments), size):
        b, q = models.split_outputs(fn(tf.constant(segments[i:i + size])))
        beats.append(b.numpy())
        if q is not None:
            quality.append(q.numpy())
    return np.concatenate(beats, axis=0), (np.concatenate(quality, axis=0) if quality else None)


def predict_record(model, record_path, record_name, out_dir, channel=0, s_boost=1.0,
                   batch_size=None, lead_mode=None, fill_mode=None, quality_log=None):
    """Predict one record and write its .<BEAT_EXTENSION> annotation into `out_dir`.

    Returns the number of beats written. With `quality_log` (a list) and a two-output model,
    appends (record_name, best_lead_in_record_numbering, [mean quality per model channel]) -
    output 2 for this record. The model's channel 0 is the annotated lead (`channel`), so the
    argmax is mapped back through the same rotation build_leads applied; a channel the
    record does not have (the zero fill) can never win, its quality is masked out.
    """
    leads, raw_length, fs = read_leads(record_path, channel=channel, lead_mode=lead_mode,
                                       fill_mode=fill_mode)

    segments, starts = segment_record(leads)
    preds, quality = predict_segments_full(model, segments, batch_size)
    if quality is not None and quality_log is not None:
        quality_log.append(record_lead_choice(record_path, record_name, channel, lead_mode,
                                              quality, starts, len(leads)))

    # signal_length keeps the decoder inside the real signal: a record shorter than one
    # window is edge-padded, and a detection in that padding is an artefact of the padding.
    positions, symbols = decode_beats(preds, segments, starts, s_boost=s_boost,
                                      signal_length=len(leads),
                                      min_run_steps=config.DECODE_MIN_RUN_STEPS)
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


def record_lead_choice(record_path, record_name, channel, lead_mode, quality, starts,
                       signal_length):
    """(record_name, best lead in the record's numbering, per-model-channel mean quality)."""
    n_sig = wfdb.rdheader(record_path).n_sig
    lead_mode = lead_mode or config.EC57_LEAD_MODE
    if lead_mode == 'auto':
        lead_mode = 'native' if n_sig >= config.IN_CHANNELS else 'single'
    real = [channel] if lead_mode in ('single', 'duplicate') else \
        lead_order(n_sig, channel)[:config.IN_CHANNELS]
    _, means = best_lead(quality, starts, signal_length)
    masked = np.full_like(means, -1.0)
    masked[:len(real)] = means[:len(real)]         # filled channels cannot be "the best lead"
    model_channel = int(np.argmax(masked))
    return (record_name, int(real[model_channel]), [round(float(m), 4) for m in means],
            model_channel)


def write_lead_quality(quality_log, ann_dir, report_dir, reviewer_channels=None):
    """Persist output 2: lead_quality.csv beside the predictions, a summary in the report dir.

    `reviewer_channels` maps record name -> the channel the reviewer worked on (portal sets);
    when given, the summary reports how often the model's choice coincides with it.
    """
    if not quality_log:
        return None
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    n_ch = max(len(row[2]) for row in quality_log)
    # Two numberings on purpose: `best_lead` is the RECORD's channel (what a clinician or the
    # .hea means by CH0/CH1/CH2); `best_model_ch` and the q_model_ch* columns are the model's,
    # where channel 0 is always the annotated lead (signal_ops.build_leads).
    with open(os.path.join(ann_dir, 'lead_quality.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['record', 'best_lead', 'best_model_ch']
                        + [f'q_model_ch{c}' for c in range(n_ch)])
        for name, lead, means, model_ch in quality_log:
            writer.writerow([name, lead, model_ch] + list(means))
    hist = {}
    for _, lead, _, _ in quality_log:
        hist[str(lead)] = hist.get(str(lead), 0) + 1
    summary = {'records': len(quality_log), 'best_lead_histogram': dict(sorted(hist.items())),
               'mean_quality_per_model_channel':
                   [round(float(np.mean([m[c] for _, _, m, _ in quality_log if len(m) > c])), 4)
                    for c in range(n_ch)]}
    if reviewer_channels:
        pairs = [(lead, reviewer_channels.get(name)) for name, lead, _, _ in quality_log
                 if name in reviewer_channels]
        if pairs:
            summary['agrees_with_reviewer_channel'] = round(
                sum(int(a == b) for a, b in pairs) / len(pairs), 4)
            summary['reviewer_channel_histogram'] = {
                str(c): sum(1 for _, b in pairs if b == c) for c in sorted({b for _, b in pairs})}
    path = os.path.join(report_dir, 'lead_quality_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  lead quality (output 2): best-lead histogram {summary['best_lead_histogram']}"
          + (f", agrees with the reviewer's channel on "
             f"{100 * summary['agrees_with_reviewer_channel']:.1f}% of records"
             if 'agrees_with_reviewer_channel' in summary else ''))
    return path


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
                       bxb_only=False, lead_mode=None, fill_mode=None):
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
        quality_log = []
        for i, (name, src, channel, _, _) in enumerate(resolved, 1):
            try:
                empty += predict_record(model, src, name, ann_dir, channel=channel,
                                        s_boost=s_boost, lead_mode=lead_mode,
                                        fill_mode=fill_mode, quality_log=quality_log) == 0
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
    # After bxb, not before: run_bxb recreates <ec57_out>/<db_name>/ for its reports.
    if not bxb_only:
        write_lead_quality(quality_log, ann_dir, os.path.join(ec57_out, db_name),
                           {name: channel for name, _, channel, _, _ in resolved})
    if path:
        _print_report(path)
    return path


# ---------------------------------------------------------------------------
# One database
# ---------------------------------------------------------------------------

def score_physionet_db(model, db_name, ec57_out, max_records=None, s_boost=1.0,
                       bxb_only=False, exclude=None, lead_mode=None, fill_mode=None):
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
    print(f"===== {db_name}: {len(records)} records, lead {channel} "
          f"({lead_mode}, fill {fill_mode or config.LEAD_FILL_MODE}) =====")

    ann_dir = annotation_dir(ec57_out, db_name)
    os.makedirs(ann_dir, exist_ok=True)
    if bxb_only:
        print(f"  --bxb-only: reusing the stored predictions in {ann_dir}")
    else:
        quality_log = []
        for i, name in enumerate(records, 1):
            try:
                n = predict_record(model, os.path.join(src_dir, name), name, ann_dir,
                                   channel=channel, s_boost=s_boost, lead_mode=lead_mode,
                                   fill_mode=fill_mode, quality_log=quality_log)
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
    # After bxb, not before: run_bxb recreates <ec57_out>/<db_name>/ for its reports.
    if not bxb_only:
        write_lead_quality(quality_log, ann_dir, os.path.join(ec57_out, db_name))
    if path:
        _print_report(path)
    return path


def score_portal_set(model, db_name, src_dir, ec57_out, max_records=None, s_boost=1.0,
                     bxb_only=False, mark_window=True, lead_mode=None, fill_mode=None):
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
        quality_log, reviewer = [], {}
        for i, name in enumerate(records, 1):
            try:
                reviewer[name] = record_channel(os.path.join(src_dir, name))
                n = predict_record(model, os.path.join(src_dir, name), name, ann_dir,
                                   channel=reviewer[name], s_boost=s_boost,
                                   lead_mode=lead_mode, fill_mode=fill_mode,
                                   quality_log=quality_log)
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
    # After bxb, not before: run_bxb recreates <ec57_out>/<db_name>/ for its reports.
    if not bxb_only:
        write_lead_quality(quality_log, ann_dir, os.path.join(ec57_out, db_name), reviewer)
    if path:
        _print_report(path)
    return path


# ---------------------------------------------------------------------------
# The whole evaluation
# ---------------------------------------------------------------------------

def run(checkpoint, tag, dbs=None, max_records=None, s_boost=1.0, bxb_only=False,
        skip_physionet=False, skip_portal=False, mark_window=True, lead_mode=None,
        splits=None, split_records=None, fill_mode=None):
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
    fill_mode = fill_mode or config.LEAD_FILL_MODE
    splits = tuple(config.PORTAL_SPLITS) if splits is None else tuple(splits)

    model = None
    if not bxb_only:
        print(f"loading {checkpoint}")
        model = load_checkpoints(checkpoint)
        model.summary() if hasattr(model, 'summary') else None
        print(f"model: {model.name}, {model.count_params():,} parameters, "
              f"lead quality output: {'yes' if models.has_quality_output(model) else 'no'}\n")

    with open(os.path.join(ec57_out, 'checkpoint.txt'), 'w') as f:
        f.write(f"{checkpoint}\n"
                f"segment_samples: {config.SEGMENT_SAMPLES} ({config.SEGMENT_SECONDS:g} s)\n"
                f"output_steps: {config.OUTPUT_STEPS}\n"
                f"min_run_steps: {config.DECODE_MIN_RUN_STEPS}\n"
                f"min_peak_prob: {config.DECODE_MIN_PEAK_PROB}\n"
                f"model: {model.name if model else '(not loaded, --bxb-only)'}\n"
                f"params: {model.count_params() if model else '-'}\n"
                f"s_boost: {s_boost}\n"
                f"in_channels: {config.IN_CHANNELS}\n"
                f"lead_mode: {lead_mode}\n"
                f"fill_mode: {fill_mode}\n"
                f"portal_splits: {', '.join(splits) or '-'} "
                f"x {config.PORTAL_SPLIT_RECORDS if split_records is None else split_records}"
                f" records\n")

    if not skip_physionet:
        for db in (dbs or config.EC57_DBS):
            score_physionet_db(model, db, ec57_out, max_records=max_records,
                               s_boost=s_boost, bxb_only=bxb_only, lead_mode=lead_mode,
                               fill_mode=fill_mode)
            print()

    if not skip_portal:
        for db_name, src_dir in sorted(config.PORTAL_EVAL_SETS.items()):
            score_portal_set(model, db_name, src_dir, ec57_out, max_records=max_records,
                             s_boost=s_boost, bxb_only=bxb_only, mark_window=mark_window,
                             lead_mode=lead_mode, fill_mode=fill_mode)
            print()

    for split in splits:
        score_portal_split(model, split, ec57_out, max_records=split_records,
                           s_boost=s_boost, bxb_only=bxb_only, lead_mode=lead_mode,
                           fill_mode=fill_mode)
        print()

    return report.summarize(ec57_out)
