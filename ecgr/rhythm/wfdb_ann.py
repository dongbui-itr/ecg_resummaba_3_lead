"""Rhythm episodes <-> WFDB rhythm-change annotations, for epicmp.

epicmp compares two annotation files by walking their `+` (rhythm-change) annotations and
string-matching `aux_note`. It has no idea what a class NAME means; it only cares that the
reference and the hypothesis spell the same rhythm the same way. `RHYTHM_AUX` is this
project's spelling: the MIT-BIH codes ('(AFIB', '(SVTA', '(VT', '(N' for sinus) where one
exists, and an explicit non-standard code for the block class MIT-BIH has no code for
(AVB = second- or third-degree block) - consistent between reference and hypothesis is what scores correctly,
not standards conformance. That file (.rhi / .rhy) is for reading with rdann.

What epicmp SCORES is another matter: `epicmp -A` only compares the episodes spelled '(AFIB'
(and knows '(AFL' in the reference). So the scored files are per class - `class_codes` /
`write_class_annotations` spell the class under test '(AFIB' and everything else '(N', one
reference/hypothesis pair per class (rc.class_extensions), exactly the reference project's
dict_ext branch.
"""
import os

import numpy as np
import wfdb

from . import config as rc
from . import inventory
from .labels import reference_episodes, second_labels

RHYTHM_AUX = {
    'SINUS': '(N',
    'AFIB': '(AFIB',
    'SVT': '(SVTA',
    'VT': '(VT',
    'AVB': '(AVB',       # not a MIT-BIH rhythm code - epicmp only string-matches ref vs. AI
}


def write_episode_annotations(episodes, record_name, out_dir, extension, fs):
    """Episodes (labels.decode_episodes / reference_episodes shape) -> a WFDB annotation file.

    One `+` annotation per episode start, `aux_note` from RHYTHM_AUX. NOISE episodes are
    skipped - "no readable lead" is not a rhythm claim either side of a comparison should
    make, and epicmp has no reference class to match it against. `episodes['start']/['stop']`
    are in SECONDS (rhythm predictions and the synthesized reference both work at 1 Hz
    granularity), so the sample position is simply `second * fs` - no resampling ratio is
    needed the way beat positions need one, because a second is a second at any sampling
    rate. Returns the number of episodes written.
    """
    kept = [e for e in episodes if e['rhythm'] != 'NOISE']
    if not kept:
        return 0
    kept = sorted(kept, key=lambda e: e['start'])
    positions = np.array([int(round(e['start'] * fs)) for e in kept], dtype=np.int64)
    aux_note = [RHYTHM_AUX[e['rhythm']] + '\x00' for e in kept]
    os.makedirs(out_dir, exist_ok=True)
    wfdb.Annotation(record_name=record_name, extension=extension, sample=positions,
                    symbol=['+'] * len(kept), aux_note=aux_note,
                    fs=fs).wrann(write_fs=True, write_dir=out_dir)
    return len(kept)


def class_codes(episodes, target, n_seconds, keep_afl=False, grid_hz=None):
    """Aux_note codes, `grid_hz` per second (rc.EC57_GRID_HZ), of the ONE-class comparison
    epicmp -A performs: `target`'s steps are '(AFIB', every other step '(N' - whatever the
    other class is, NOISE or a step no episode covers included. `keep_afl` (reference side of
    the AFIB task only) leaves flutter as '(AFL', the code epicmp knows from the EC57
    reference databases. Episode bounds (seconds, int or float) are floored onto the grid, so
    grid 1 is exactly the whole-second scoring every earlier table used."""
    g = rc.EC57_GRID_HZ if grid_hz is None else grid_hz
    n = int(n_seconds) * g
    codes = np.full(n, '(N', dtype='<U8')
    for e in episodes:
        a = max(0, int(np.floor(e['start'] * g)))
        b = min(n, int(np.floor(e['stop'] * g)))
        if b <= a:
            continue
        if e['rhythm'] == target:
            codes[a:b] = '(AFIB'
        elif keep_afl and e['rhythm'] == 'AFL' and target == 'AFIB':
            codes[a:b] = '(AFL'
    return codes


def write_class_annotations(episodes, target, record_name, out_dir, extension, fs, n_seconds,
                            keep_afl=False, grid_hz=None):
    """One `+` at time 0 and at every code change of class_codes(...) - consecutive equal
    codes collapse into one annotation. Always writes a file (a record with no `target` at
    all is a single '(N'), so every record of a database is scored. Returns the number of
    `+` annotations written."""
    g = rc.EC57_GRID_HZ if grid_hz is None else grid_hz
    codes = class_codes(episodes, target, n_seconds, keep_afl=keep_afl, grid_hz=g)
    if len(codes) == 0:
        codes = np.array(['(N'])
    change = np.concatenate([[0], np.flatnonzero(codes[1:] != codes[:-1]) + 1])
    positions = np.round(change * fs / g).astype(np.int64)
    os.makedirs(out_dir, exist_ok=True)
    wfdb.Annotation(record_name=record_name, extension=extension, sample=positions,
                    symbol=['+'] * len(change), aux_note=[codes[i] + '\x00' for i in change],
                    fs=fs).wrann(write_fs=True, write_dir=out_dir)
    return len(change)


BEAT_SYMBOL = {1: 'N', 2: 'S', 3: 'V'}


def write_beat_annotations(times, classes, record_name, out_dir, extension, fs):
    """Beat hypothesis for bxb: one N / S / V annotation at each R time (seconds)."""
    times = np.asarray(times, dtype=np.float64)
    classes = np.asarray(classes, dtype=int)
    keep = np.isin(classes, list(BEAT_SYMBOL))
    samples = np.round(times[keep] * fs).astype(np.int64)
    order = np.argsort(samples, kind='stable')
    os.makedirs(out_dir, exist_ok=True)
    if len(samples) == 0:
        samples, symbols = np.array([0], dtype=np.int64), ['Q']   # bxb wants a file
    else:
        samples = samples[order]
        symbols = [BEAT_SYMBOL[int(c)] for c in classes[keep][order]]
    wfdb.Annotation(record_name=record_name, extension=extension, sample=samples,
                    symbol=symbols, fs=fs).wrann(write_fs=True, write_dir=out_dir)
    return int(keep.sum())


def atr_reference_episodes(record_path, aux_to_class=None):
    """A Physionet record's `+` rhythm marks -> episodes in seconds (decode_episodes' shape).

    Codes map through rc.PHYSIONET_AUX_TO_CLASS; '(AFL' is kept as the pseudo-class 'AFL' so
    the AFIB reference can spell it, everything else unmapped is SINUS. Each mark holds until
    the next one; the last one until the end of the record. Returns (episodes, fs, seconds).
    """
    aux_to_class = rc.PHYSIONET_AUX_TO_CLASS if aux_to_class is None else aux_to_class
    header = wfdb.rdheader(record_path)
    fs, n_seconds = header.fs, header.sig_len // header.fs
    ann = wfdb.rdann(record_path, 'atr')
    # float seconds: class_codes floors them onto its grid (grid 1 = the old s // fs)
    marks = [(s / fs, (x or '').split('\x00')[0].strip())
             for s, sym, x in zip(ann.sample, ann.symbol, ann.aux_note)
             if sym == '+' and x]
    episodes = []
    for (start, code), (stop, _) in zip(marks, marks[1:] + [(n_seconds, None)]):
        name = aux_to_class.get(code, 'AFL' if code == '(AFL' else 'SINUS')
        if stop > start:
            episodes.append({'rhythm': name, 'start': start, 'stop': stop, 'prob': 1.0})
    return episodes, fs, n_seconds


def synth_reference_labels(ev, length, fs):
    """Per-second ground-truth label vector for one rhythm_eval event, over its WHOLE record.

    `ev` is one event dict from inventory.collect_test (spans/strip already in the record's
    own sample positions - collect_test never resamples). `length`/`fs` are the record's own,
    from wfdb.rdheader - NOT rc.SAMPLING_RATE, since the rhythm_eval records are not
    guaranteed to be 250 Hz. Reuses labels.second_labels exactly as build.py does per 10 s
    training window, just over the whole record as one "window" starting at 0: the function
    is generic in `second` (here the record's own fs) and `n_seconds`, so no new labelling
    logic is introduced - only the ground truth this project already computes for training,
    read for every second of the record instead of every window.
    """
    spans, region, known = inventory.event_region(ev, length)
    n_seconds = length // fs
    return second_labels(0, spans, known, n_seconds=n_seconds, second=fs)


def write_reference_annotations(ev, record_name, out_dir, length, fs,
                                extension=rc.RHYTHM_REF_EXTENSION):
    """Synthesize + write the rhythm_eval reference annotation for one event's record."""
    labels = synth_reference_labels(ev, length, fs)
    episodes = reference_episodes(labels)
    return write_episode_annotations(episodes, record_name, out_dir, extension, fs)
