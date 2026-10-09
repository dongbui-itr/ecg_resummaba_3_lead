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
from .labels import to_current_labels


def read_manifest(npy_dir=None):
    path = os.path.join(npy_dir or rc.NPY_DIR, 'manifest.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"no rhythm manifest at {path} - run "
                                f"`python -m ecgr.rhythm data` first")
    with open(path) as f:
        manifest = json.load(f)
    live = {'segment_samples': rc.SEGMENT_SAMPLES, 'in_channels': rc.IN_CHANNELS,
            'output_seconds': rc.OUTPUT_SECONDS, 'class_names': rc.CLASS_NAMES}
    bad = {k: (manifest.get(k), v) for k, v in live.items() if manifest.get(k) != v
           and not (k == 'class_names' and is_legacy(manifest))}
    if bad:
        raise ValueError(f"rhythm data in {npy_dir or rc.NPY_DIR} does not match the config: "
                         f"{bad} - rebuild it")
    return manifest


def is_legacy(manifest):
    """True for a build labelled in rc.LEGACY_CLASS_NAMES (AVB2 and AVB3 apart, before
    2026-10-07): its labels are mapped onto the five classes as they are loaded."""
    return manifest.get('class_names') == rc.LEGACY_CLASS_NAMES


def manifest_class_weights(manifest):
    """The build's rhythm class weights for rc.CLASS_NAMES. A legacy build's are recomputed
    from its train-second counts with AVB2 + AVB3 pooled (build.class_weights_from_counts)."""
    if not is_legacy(manifest):
        return [float(w) for w in (manifest.get('class_weights') or [1.0] * rc.NUM_CLASSES)]
    from .build import class_weights_from_counts
    old = (manifest.get('splits', {}).get('train', {}) or {}).get('seconds') or {}
    if not old:
        return [1.0] * rc.NUM_CLASSES
    seconds = {n: 0 for n in rc.CLASS_NAMES}
    for i, name in enumerate(rc.LEGACY_CLASS_NAMES):
        seconds[rc.CLASS_NAMES[rc.LEGACY_TO_CLASS[i]]] += int(old.get(name, 0))
    return class_weights_from_counts(seconds)


def load_arrays(split, npy_dir=None, max_windows=None, label_steps=None, beats=False):
    """(segments float16 (n, 2500, 3), labels uint8 (n, label_steps), study ids (n,)) - and
    with beats=True a 4th array, the (n, 2500) uint8 beat labels (beats_<k>.npy).

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
    segs, labs, sids, bts, total = [], [], [], [], 0
    for path in shards:
        segs.append(np.load(path))
        if beats:
            beat_path = path.replace('segments_', 'beats_')
            if not os.path.exists(beat_path):
                raise FileNotFoundError(f"{beat_path} missing - beat labels came with the "
                                        f"build after 2026-10-06; rebuild with "
                                        f"`python -m ecgr.rhythm data`")
            bts.append(np.load(beat_path))
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
    manifest_path = os.path.join(npy_dir or rc.NPY_DIR, 'manifest.json')
    if os.path.exists(manifest_path) and is_legacy(read_manifest(npy_dir)):
        labs = to_current_labels(labs)                  # AVB2 / AVB3 -> AVB
    if max_windows:
        segs, labs, sids = segs[:max_windows], labs[:max_windows], sids[:max_windows]
    if beats:
        bts = np.concatenate(bts)[:max_windows] if max_windows else np.concatenate(bts)
        return segs, labs, sids, bts
    return segs, labs, sids


def window_categories(labels):
    """(n, steps) rhythm labels -> (n,) category index into rc.CLASS_NAMES: the rarest
    arrhythmia (rc.STRAT_PRIORITY order) with >= rc.STRAT_MIN_SECONDS in the window, else
    SINUS. Works on per-second, 20 ms, 8 ms or per-sample labels."""
    labels = np.asarray(labels)
    steps = labels.shape[1]
    need = max(1, int(round(rc.STRAT_MIN_SECONDS * steps / rc.OUTPUT_SECONDS)))
    cat = np.full(len(labels), rc.SINUS, dtype=np.int64)
    for name in reversed(rc.STRAT_PRIORITY):           # rarest last, so it wins
        c = rc.CLASS_NAMES.index(name)
        cat[(labels == c).sum(axis=1) >= need] = c
    return cat


def stratified_plan(categories, batch_size, steps_per_epoch=None, quota=None,
                    max_repeat=None):
    """{category: windows per batch} for `categories` (window_categories): rc.STRAT_BATCH_QUOTA
    scaled to batch_size, a category absent from the data or repeated more than max_repeat
    times per epoch hands its surplus to SINUS. Returns (plan, repeat factors, steps)."""
    quota = dict(rc.STRAT_BATCH_QUOTA if quota is None else quota)
    max_repeat = rc.STRAT_MAX_REPEAT if max_repeat is None else max_repeat
    n = len(categories)
    steps = steps_per_epoch or max(1, n // batch_size)
    counts = {name: int((categories == i).sum()) for i, name in enumerate(rc.CLASS_NAMES)}
    total = float(sum(quota.values()))
    plan = {k: int(round(v / total * batch_size)) for k, v in quota.items()}
    plan['SINUS'] += batch_size - sum(plan.values())
    for name in list(plan):
        if name == 'SINUS':
            continue
        if counts.get(name, 0) == 0:
            plan['SINUS'] += plan.pop(name)
            continue
        cap = int(np.ceil(counts[name] * max_repeat / steps))
        if plan[name] > cap:
            plan['SINUS'] += plan[name] - cap
            plan[name] = cap
    if counts.get('SINUS', 0) == 0 and plan.get('SINUS', 0):      # tiny smoke-test splits
        share = plan.pop('SINUS')
        present = [k for k in plan if counts.get(k, 0) > 0]
        for k in present:
            plan[k] += share // len(present)
        plan[present[0]] += share - (share // len(present)) * len(present)
    plan = {k: v for k, v in plan.items() if v > 0}
    repeat = {k: plan[k] * steps / counts[k] for k in plan}
    return plan, repeat, steps


def stratified_batches(categories, plan, steps, seed=None):
    """Generator of `steps` index batches: plan[c] windows of each category, each category
    cycling its own shuffled permutation (reshuffled when exhausted)."""
    rng = np.random.default_rng(seed)
    pools = {}
    for name, k in plan.items():
        idx = np.flatnonzero(categories == rc.CLASS_NAMES.index(name))
        pools[name] = [rng.permutation(idx), 0, k]
    for _ in range(steps):
        parts = []
        for name, state in pools.items():
            perm, pos, k = state
            take, got = [], 0
            while got < k:
                room = min(k - got, len(perm) - pos)
                take.append(perm[pos:pos + room])
                got += room
                pos += room
                if pos >= len(perm):
                    perm, pos = rng.permutation(perm), 0
            state[0], state[1] = perm, pos
            parts.append(np.concatenate(take))
        yield rng.permutation(np.concatenate(parts)).astype(np.int64)


def make_dataset(segments, labels, batch_size, mode='clean', shuffle=None, seed_offset=0,
                 augment_kw=None, outputs=('rhythm', 'lead'), beats=None, sampler='uniform',
                 steps_per_epoch=None):
    """augment_kw: overrides for augment.augment (noise_prob, permute_prob, snr_range,
    wreck_prob, flip_prob, drop_prob). beats: (n, 2500) uint8 beat labels, needed when
    'beat' is among the outputs. sampler='stratified' (training only) draws each batch by
    window category (window_categories / stratified_plan) instead of uniformly."""
    augment_kw = augment_kw or {}
    n = len(labels)
    shuffle = (mode == 'train') if shuffle is None else shuffle
    if 'beat' in outputs and beats is None:
        raise ValueError("the model has a 'beat' output: pass the beat labels "
                         "(load_arrays(..., beats=True))")
    if beats is None:
        beats = np.zeros((n, 1), np.uint8)           # placeholder, never reaches a target

    def fetch(idx):
        idx = np.sort(idx)
        return segments[idx], labels[idx], beats[idx]

    if sampler == 'stratified' and mode == 'train':
        cats = window_categories(labels)
        plan, repeat, steps = stratified_plan(cats, batch_size, steps_per_epoch)
        print(f"sampler      : stratified, per batch {plan}; repeats/epoch "
              f"{ {k: round(v, 2) for k, v in repeat.items()} }, {steps} steps")
        ds = tf.data.Dataset.from_generator(
            lambda: stratified_batches(cats, plan, steps),
            output_signature=tf.TensorSpec([batch_size], tf.int64))
    elif sampler not in ('uniform', 'stratified'):
        raise ValueError(f"sampler must be uniform or stratified, got {sampler!r}")
    else:
        ds = tf.data.Dataset.range(n)
        if shuffle:
            ds = ds.shuffle(n, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=(mode == 'train'))

    want_beats = 'beat' in outputs

    def load(idx):
        x, y, b = tf.numpy_function(fetch, [idx], [tf.float16, tf.uint8, tf.uint8])
        x.set_shape([None, rc.SEGMENT_SAMPLES, rc.IN_CHANNELS])
        y.set_shape([None, labels.shape[1]])
        b.set_shape([None, beats.shape[1]])
        return x, y, b

    def beats_or_none(b):
        return b if want_beats else None

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
    if mode == 'train':
        ds = ds.map(lambda x, y, b: augment(
            x, y, tf.random.uniform([2], 0, 2 ** 31 - 1, dtype=tf.int64),
            beats=beats_or_none(b), **augment_kw),
            num_parallel_calls=tf.data.AUTOTUNE)
    elif mode == 'noisy':
        ds = ds.enumerate().map(lambda i, xyb: augment(
            xyb[0], xyb[1], tf.stack([i + seed_offset, tf.constant(20260924, tf.int64)]),
            beats=beats_or_none(xyb[2]), **augment_kw),
            num_parallel_calls=tf.data.AUTOTUNE)
    elif mode == 'clean':
        ds = ds.map(lambda x, y, b: no_augment(x, y, beats_or_none(b)),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        raise ValueError(f"mode must be train / noisy / clean, got {mode!r}")
    # augment produces every target; Keras wants exactly the model's outputs
    ds = ds.map(lambda x, t: (x, {k: t[k] for k in outputs}),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_split(split, batch_size=None, mode=None, max_windows=None, npy_dir=None,
               label_steps=None, outputs=('rhythm', 'lead'), sampler='uniform',
               steps_per_epoch=None, return_arrays=False):
    mode = mode or ('train' if split == 'train' else 'noisy')
    want_beats = 'beat' in outputs
    arrays = load_arrays(split, npy_dir, max_windows, label_steps, beats=want_beats)
    segs, labs = arrays[0], arrays[1]
    bts = arrays[3] if want_beats else None
    print(f"{split:5s}: {len(labs):,} windows ({mode}), labels {labs.shape[1:]}"
          + (", beat labels" if want_beats else ""))
    ds = make_dataset(segs, labs, batch_size or rc.BATCH_SIZE, mode=mode, outputs=outputs,
                      beats=bts, sampler=sampler, steps_per_epoch=steps_per_epoch)
    return (ds, arrays) if return_arrays else ds
