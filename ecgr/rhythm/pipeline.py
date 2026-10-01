"""tf.data over the rhythm npy shards.

The whole split is held in RAM as float16 (a 3-lead window is 15 kB, so a few hundred thousand
windows are a few GB) and batches are gathered from it by index: shuffle the INDICES, batch
them, then fetch - far cheaper than shuffling tensors, and the fetch is one fancy-indexing
call per batch.

Three modes:
    'train'  fresh augmentation every batch (augment.augment with a random seed)
    'noisy'  the same augmentation with a seed fixed per batch position: a corrupted
             validation set that is identical every epoch, so epochs are comparable
    'clean'  the stored windows untouched, every second clean
"""
import glob
import json
import os

import numpy as np
import tensorflow as tf

from . import config as rc
from .augment import augment, no_augment


def read_manifest(npy_dir=None):
    path = os.path.join(npy_dir or rc.NPY_DIR, 'manifest.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"no rhythm manifest at {path} - run "
                                f"`python -m ecgr.rhythm data` first")
    with open(path) as f:
        manifest = json.load(f)
    live = {'segment_samples': rc.SEGMENT_SAMPLES, 'in_channels': rc.IN_CHANNELS,
            'output_seconds': rc.OUTPUT_SECONDS, 'class_names': rc.CLASS_NAMES}
    bad = {k: (manifest.get(k), v) for k, v in live.items() if manifest.get(k) != v}
    if bad:
        raise ValueError(f"rhythm data in {npy_dir or rc.NPY_DIR} does not match the config: "
                         f"{bad} - rebuild it")
    return manifest


def load_arrays(split, npy_dir=None, max_windows=None, label_steps=None):
    """(segments float16 (n, 2500, 3), labels uint8 (n, label_steps), study ids (n,)).

    label_steps: rows of the model's 'rhythm' output - 10 (per second, labels_<k>.npy, the
    default), 2500 (per sample, labels_samples_<k>.npy) or any divisor of 2500 in between,
    e.g. 500 for 20 ms: the per-sample labels at the centre of each block."""
    label_steps = label_steps or rc.OUTPUT_SECONDS
    per_second = label_steps == rc.OUTPUT_SECONDS
    if not per_second and rc.SEGMENT_SAMPLES % label_steps:
        raise ValueError(f"label_steps {label_steps} does not divide {rc.SEGMENT_SAMPLES}")
    stride = rc.SEGMENT_SAMPLES // label_steps
    root = os.path.join(npy_dir or rc.NPY_DIR, split)
    label_file = 'labels_' if per_second else 'labels_samples_'
    shards = sorted(glob.glob(os.path.join(root, 'segments_*.npy')),
                    key=lambda p: int(p.rsplit('_', 1)[1][:-4]))
    if not shards:
        raise FileNotFoundError(f"no {split} shards under {root}")
    segs, labs, sids, total = [], [], [], 0
    for path in shards:
        segs.append(np.load(path))
        label_path = path.replace('segments_', label_file)
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"{label_path} missing - the per-sample labels came with "
                                    f"the build after 2026-09-28; rebuild with "
                                    f"`python -m ecgr.rhythm data`")
        lab = np.load(label_path)
        labs.append(lab if per_second or stride == 1 else
                    np.ascontiguousarray(lab[:, stride // 2::stride]))
        sids.append(np.load(path.replace('segments_', 'studyids_')))
        total += len(labs[-1])
        if max_windows and total >= max_windows:
            break
    segs, labs, sids = (np.concatenate(a) for a in (segs, labs, sids))
    if max_windows:
        segs, labs, sids = segs[:max_windows], labs[:max_windows], sids[:max_windows]
    return segs, labs, sids


def make_dataset(segments, labels, batch_size, mode='clean', shuffle=None, seed_offset=0,
                 augment_kw=None, outputs=('rhythm', 'lead')):
    """augment_kw: overrides for augment.augment (noise_prob, permute_prob, snr_range,
    wreck_prob, flip_prob, drop_prob)."""
    augment_kw = augment_kw or {}
    n = len(labels)
    shuffle = (mode == 'train') if shuffle is None else shuffle

    def fetch(idx):
        idx = np.sort(idx)
        return segments[idx], labels[idx]

    ds = tf.data.Dataset.range(n)
    if shuffle:
        ds = ds.shuffle(n, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=(mode == 'train'))

    def load(idx):
        x, y = tf.numpy_function(fetch, [idx], [tf.float16, tf.uint8])
        x.set_shape([None, rc.SEGMENT_SAMPLES, rc.IN_CHANNELS])
        y.set_shape([None, labels.shape[1]])
        return x, y

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
    if mode == 'train':
        ds = ds.map(lambda x, y: augment(
            x, y, tf.random.uniform([2], 0, 2 ** 31 - 1, dtype=tf.int64), **augment_kw),
            num_parallel_calls=tf.data.AUTOTUNE)
    elif mode == 'noisy':
        ds = ds.enumerate().map(lambda i, xy: augment(
            xy[0], xy[1], tf.stack([i + seed_offset, tf.constant(20260924, tf.int64)]),
            **augment_kw),
            num_parallel_calls=tf.data.AUTOTUNE)
    elif mode == 'clean':
        ds = ds.map(no_augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        raise ValueError(f"mode must be train / noisy / clean, got {mode!r}")
    # augment produces every target; Keras wants exactly the model's outputs
    ds = ds.map(lambda x, t: (x, {k: t[k] for k in outputs}),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_split(split, batch_size=None, mode=None, max_windows=None, npy_dir=None,
               label_steps=None, outputs=('rhythm', 'lead')):
    mode = mode or ('train' if split == 'train' else 'noisy')
    segs, labs, _ = load_arrays(split, npy_dir, max_windows, label_steps)
    print(f"{split:5s}: {len(labs):,} windows ({mode}), labels {labs.shape[1:]}")
    return make_dataset(segs, labs, batch_size or rc.BATCH_SIZE, mode=mode, outputs=outputs)
