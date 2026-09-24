"""npy batches -> tfrecords, one tfrecord per npy batch.

Feature layout - raw bytes, not typed lists:

    signal  bytes  float32[SEGMENT_SAMPLES, IN_CHANNELS], row-major
    labels  bytes  uint8[OUTPUT_STEPS]        (4 classes + IGNORE_LABEL fit in a byte)

`float_list` stores every sample as a separate protobuf field. At 60 s x 3 leads that is
45,000 fields per example, and both writing and `parse_example` pay per field; one `bytes`
feature is a single memcpy each way. One example is ~183 kB.

A `dataset_manifest.json` is written next to the tfrecords with the shape, dtype and class
histogram of what was actually written. pipeline.py checks it against the live config, so a
config/data mismatch is a clear message at startup instead of an opaque shape error inside
the first training step.
"""
import glob
import json
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .. import config


def _example(signal, labels):
    """One serialized tf.train.Example. `signal` float32 (T, C), `labels` uint8 (steps,)."""
    feature = {
        'signal': tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[np.ascontiguousarray(signal, dtype=np.float32).tobytes()])),
        'labels': tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[np.ascontiguousarray(labels, dtype=np.uint8).tobytes()])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def label_histogram(labels):
    """Class-step counts over the LABELLED steps only, plus the number of ignored steps.

    IGNORE_LABEL is not a class: counting it in the histogram would make the class shares
    (and anything sized from them) reflect how much of a strip was reviewed rather than what
    the reviewed part contained.
    """
    flat = np.asarray(labels).reshape(-1)
    labelled = flat[flat != config.IGNORE_LABEL]
    counts = np.bincount(labelled.astype(np.int64), minlength=config.NUM_CLASSES)
    return counts[:config.NUM_CLASSES], int(flat.size - labelled.size)


def _convert_batch(seg_file, out_dir):
    """One npy batch -> one tfrecord. Returns (name, count, class-step histogram, ignored)."""
    segments = np.load(seg_file)
    labels = np.load(seg_file.replace("_segments_", "_labels_"))

    # Lets a 3-lead npy tree feed a 1- or 2-lead run without rebuilding it. Channel 0 is the
    # annotated lead (signal_ops.build_leads), so truncating keeps the lead the labels refer
    # to; without this the parser would fail at the first training step instead.
    want = max(1, config.IN_CHANNELS)
    if segments.ndim == 3 and segments.shape[-1] != want:
        if segments.shape[-1] < want:
            raise ValueError(
                f"{os.path.basename(seg_file)} has {segments.shape[-1]} leads but the run is "
                f"configured for {want} (ECGR_IN_CHANNELS) - rebuild the npy tree")
        print(f"  {os.path.basename(seg_file)}: {segments.shape[-1]} leads -> keeping {want}")
        segments = segments[..., :want]

    if segments.shape[1:] != (config.SEGMENT_SAMPLES, want):
        raise ValueError(f"{seg_file}: segments are {segments.shape[1:]}, "
                         f"expected {(config.SEGMENT_SAMPLES, want)}")
    if labels.shape[1:] != (config.OUTPUT_STEPS,):
        raise ValueError(f"{seg_file}: labels are {labels.shape[1:]}, "
                         f"expected {(config.OUTPUT_STEPS,)}")
    bad = (labels >= config.NUM_CLASSES) & (labels != config.IGNORE_LABEL)
    if bad.any():
        raise ValueError(f"{seg_file}: label value {labels[bad].min()} is neither a class "
                         f"(< {config.NUM_CLASSES}) nor IGNORE ({config.IGNORE_LABEL})")

    name = os.path.basename(seg_file).replace("_segments_", "_")[:-4] + ".tfrecord"
    counts, ignored = label_histogram(labels)
    with tf.io.TFRecordWriter(os.path.join(out_dir, name)) as writer:
        for i in range(len(segments)):
            writer.write(_example(segments[i], labels[i]))
    return name, len(segments), counts, ignored


def build_dataset(db_name):
    """Convert the npy batches of `db_name` into tfrecords. Returns per-split counters."""
    summary = {}
    for split in ("train", "eval"):
        npy_dir = os.path.join(config.NPY_DIR, db_name, split)
        out_dir = os.path.join(config.TFRECORD_DIR, db_name, split)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        batches = sorted(glob.glob(os.path.join(npy_dir, "*_segments_batch_*.npy")))
        if not batches:
            print(f"{db_name} {split}: no npy batches in {npy_dir}, skipping")
            continue

        total = ignored = 0
        counts = np.zeros(config.NUM_CLASSES, dtype=np.int64)
        for seg_file in tqdm(batches, desc=f"{db_name} {split}",
                             disable=not sys.stderr.isatty(), mininterval=2.0):
            _, n, hist, ign = _convert_batch(seg_file, out_dir)
            total += n
            counts += hist
            ignored += ign
        summary[split] = {'segments': total, 'files': len(batches),
                          'class_steps': counts.tolist(), 'ignored_steps': int(ignored)}
        print(f"{db_name} {split}: {total:,} segments in {len(batches)} tfrecords -> {out_dir}")
    return summary


def write_manifest(summary, merge=True):
    """Record what was written, so pipeline.py can refuse data the config does not match.

    `merge` keeps the entries of datasets this call did not touch. Rebuilding one dataset
    (`--db dataset-2`) is a normal thing to do, and without the merge the manifest would
    afterwards claim the tree holds only that one - which is worse than having no manifest,
    because `check_manifest` prints the totals it reads and they would be silently wrong.
    Only entries for datasets still present on disk are carried over.
    """
    os.makedirs(config.TFRECORD_DIR, exist_ok=True)
    path = os.path.join(config.TFRECORD_DIR, config.DATASET_MANIFEST)

    if merge:
        previous = (read_manifest() or {}).get('per_dataset', {})
        kept = {db: v for db, v in previous.items()
                if db not in summary and os.path.isdir(os.path.join(config.TFRECORD_DIR, db))}
        if kept:
            print(f"manifest: keeping {len(kept)} dataset(s) this build did not touch "
                  f"({', '.join(sorted(kept))})")
        summary = {**kept, **summary}

    totals = {}
    for split in ('train', 'eval'):
        segments = sum(v[split]['segments'] for v in summary.values() if split in v)
        steps = np.zeros(config.NUM_CLASSES, dtype=np.int64)
        ignored = 0
        for v in summary.values():
            if split in v:
                steps += np.asarray(v[split]['class_steps'], dtype=np.int64)
                ignored += int(v[split].get('ignored_steps', 0))
        share = (steps / steps.sum()).round(6).tolist() if steps.sum() else []
        totals[split] = {'segments': segments, 'class_steps': steps.tolist(),
                         'class_share': share, 'ignored_steps': ignored}

    manifest = {
        'built': datetime.now().isoformat(timespec='seconds'),
        'npy_dir': config.NPY_DIR,
        'feature_encoding': 'bytes',
        'signal_dtype': 'float32',
        'labels_dtype': 'uint8',
        'segment_samples': config.SEGMENT_SAMPLES,
        'segment_seconds': config.SEGMENT_SECONDS,
        'in_channels': config.IN_CHANNELS,
        'primary_lead_first': config.PRIMARY_LEAD_FIRST,
        'output_steps': config.OUTPUT_STEPS,
        'num_classes': config.NUM_CLASSES,
        'ignore_label': config.IGNORE_LABEL,
        'sampling_rate': config.SAMPLING_RATE,
        'class_names': config.CLASS_NAMES,
        'datasets': sorted(summary),
        'per_dataset': summary,
        'totals': totals,
    }
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=4)
    print(f"\nmanifest -> {path}")
    for split, v in totals.items():
        share = ', '.join(f"{n} {100 * s:.2f}%" for n, s in
                          zip(config.CLASS_NAMES, v['class_share'] or [0] * 4))
        labelled = int(sum(v['class_steps']))
        print(f"  {split:5s}: {v['segments']:>9,} segments | labelled steps: {share} | "
              f"ignored {v['ignored_steps']:,} of {labelled + v['ignored_steps']:,}")
    return manifest


def read_manifest():
    """The manifest already in TFRECORD_DIR, or None. Mirrors pipeline.read_manifest, which
    cannot be imported here: pipeline imports this module's output format, not the reverse."""
    path = os.path.join(config.TFRECORD_DIR, config.DATASET_MANIFEST)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def build_all(db_names=None):
    summary = {}
    for db in (db_names or config.TRAIN_DATASETS):
        print(f"\n{'=' * 70}\n{db}\n{'=' * 70}")
        result = build_dataset(db)
        if result:
            summary[db] = result
    if not summary:
        raise RuntimeError(f"no npy batches found under {config.NPY_DIR} - run "
                           f"`python -m ecgr data --step npy` first")
    return write_manifest(summary)
