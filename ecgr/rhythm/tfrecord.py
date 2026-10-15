"""Rhythm npy shards -> tfrecords, one tfrecord per npy shard (same windows, same order).

    <out>/<split>/rhythm_<k>.tfrecord     one tf.train.Example per 10 s window:
        signal         bytes  float16[SEGMENT_SAMPLES, IN_CHANNELS]  (band-passed, z-scored)
        labels         bytes  uint8[SEGMENT_SAMPLES]  per-sample rhythm class, IGNORE unlabelled
        beats          bytes  uint8[SEGMENT_SAMPLES]  0 / N / S / V at the R sample, IGNORE
        study_id       int64
        source, event_id  bytes
    <out>/tfrecord_manifest.json          geometry, class names, windows per split / shard

Raw bytes rather than typed lists for the reason build_tfrecord.py gives (one memcpy per
feature instead of 7,500 protobuf fields). `parse` is the matching tf.data decoder.
"""
import glob
import json
import os

import numpy as np
import tensorflow as tf

from . import config as rc

FEATURES = {
    'signal': tf.io.FixedLenFeature([], tf.string),
    'labels': tf.io.FixedLenFeature([], tf.string),
    'beats': tf.io.FixedLenFeature([], tf.string),
    'study_id': tf.io.FixedLenFeature([], tf.int64),
    'source': tf.io.FixedLenFeature([], tf.string),
    'event_id': tf.io.FixedLenFeature([], tf.string),
}


def _bytes(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _example(signal, labels, beats, study_id, source, event_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'signal': _bytes(np.ascontiguousarray(signal, np.float16).tobytes()),
        'labels': _bytes(np.ascontiguousarray(labels, np.uint8).tobytes()),
        'beats': _bytes(np.ascontiguousarray(beats, np.uint8).tobytes()),
        'study_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(study_id)])),
        'source': _bytes(str(source).encode()),
        'event_id': _bytes(str(event_id).encode()),
    })).SerializeToString()


def parse(record):
    """One serialized example -> dict of tensors (signal float32 (T, C), labels / beats uint8)."""
    f = tf.io.parse_single_example(record, FEATURES)
    signal = tf.reshape(tf.io.decode_raw(f['signal'], tf.float16),
                        (rc.SEGMENT_SAMPLES, rc.IN_CHANNELS))
    return dict(signal=tf.cast(signal, tf.float32),
                labels=tf.reshape(tf.io.decode_raw(f['labels'], tf.uint8), (rc.SEGMENT_SAMPLES,)),
                beats=tf.reshape(tf.io.decode_raw(f['beats'], tf.uint8), (rc.SEGMENT_SAMPLES,)),
                study_id=f['study_id'], source=f['source'], event_id=f['event_id'])


def export(npy_dir=None, out_dir=None, splits=('train', 'eval', 'test')):
    """Write every shard of `npy_dir` as a tfrecord under `out_dir` (default <npy_dir>_tfrecord)."""
    npy_dir = npy_dir or rc.NPY_DIR
    out_dir = out_dir or npy_dir.rstrip('/') + '_tfrecord'
    manifest = dict(npy_dir=npy_dir, segment_samples=rc.SEGMENT_SAMPLES,
                    in_channels=rc.IN_CHANNELS, sampling_rate=rc.SAMPLING_RATE,
                    class_names=rc.CLASS_NAMES, ignore_label=rc.IGNORE,
                    signal_dtype='float16', features=list(FEATURES), splits={})
    for split in splits:
        shards = sorted(glob.glob(os.path.join(npy_dir, split, 'segments_*.npy')),
                        key=lambda p: int(p.rsplit('_', 1)[1][:-4]))
        if not shards:
            continue
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)
        per_shard = {}
        for path in shards:
            k = path.rsplit('_', 1)[1][:-4]
            seg = np.load(path, mmap_mode='r')
            lab = np.load(path.replace('segments_', 'labels_samples_'), mmap_mode='r')
            bts = np.load(path.replace('segments_', 'beats_'), mmap_mode='r')
            sid = np.load(path.replace('segments_', 'studyids_'))
            with open(path.replace('segments_', 'events_').replace('.npy', '.json')) as f:
                evs = json.load(f)
            name = f'rhythm_{k}.tfrecord'
            with tf.io.TFRecordWriter(os.path.join(out_dir, split, name)) as w:
                for i in range(len(seg)):
                    w.write(_example(seg[i], lab[i], bts[i], sid[i], *evs[i]))
            per_shard[name] = len(seg)
            print(f"tfrecord {split}/{name}: {len(seg):,} windows", flush=True)
        manifest['splits'][split] = dict(windows=sum(per_shard.values()), shards=per_shard)
    with open(os.path.join(out_dir, 'tfrecord_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {os.path.join(out_dir, 'tfrecord_manifest.json')}")
    return manifest
