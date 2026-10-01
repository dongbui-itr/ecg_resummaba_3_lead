"""Rhythm labels: reviewer spans -> one class per second, and per-second output -> episodes.

A label here is a statement about TIME. The reviewer marks where a rhythm is (a caliper span,
a 10 s strip, or - for runs - the beats themselves), and `second_labels` turns those marks into
the (10,) vector a window is trained against:

  * a second takes an arrhythmia class when that class covers >= MIN_SECOND_COVER of it;
  * otherwise it is SINUS if it lies in a region where "no mark" is known to mean sinus;
  * otherwise it is IGNORE - nobody vouched for it, and the loss does not look at it.

One correction on top: a span shorter than half a second (an AVB3 caliper of 0.34 s exists in
dataset-rhythm/dataset-2) covers no second by half, and would silently vanish. Every span
therefore claims at least the one second it overlaps most.
"""
import numpy as np

from . import config as rc


def class_index(name):
    return rc.CLASS_NAMES.index(name)


def merge_intervals(intervals):
    """Union of [a, b) intervals, sorted and non-overlapping."""
    out = []
    for a, b in sorted((int(a), int(b)) for a, b in intervals if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def _overlap(a, b, lo, hi):
    return max(0, min(b, hi) - max(a, lo))


def beat_runs(samples, symbols, lo=None, hi=None, fs=rc.SAMPLING_RATE,
              min_beats=rc.RUN_MIN_BEATS, pad_seconds=rc.RUN_PAD_SECONDS):
    """Runs of >= min_beats consecutive S (-> SVT) or V (-> VT) beats inside [lo, hi).

    Only beats inside the reviewed range are read: the rest of the recording carries beat
    labels nobody signed off on. Returns [(class_index, start, stop)] in sample positions.
    """
    samples = np.asarray(samples, dtype=np.int64)
    symbols = np.asarray(symbols)
    keep = np.ones(len(samples), dtype=bool)
    if lo is not None:
        keep &= samples >= lo
    if hi is not None:
        keep &= samples < hi
    samples, symbols = samples[keep], symbols[keep]
    pad = int(round(pad_seconds * fs))

    runs = []
    for name in rc.RUN_CLASSES:
        hit = np.isin(symbols, rc.RUN_BEAT_SYMBOLS[name])
        # rising / falling edges of the boolean track
        edges = np.diff(np.concatenate([[0], hit.astype(np.int8), [0]]))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if stop - start >= min_beats:
                runs.append((class_index(name), int(samples[start]) - pad,
                             int(samples[stop - 1]) + pad))
    return sorted(runs, key=lambda r: r[1])


def second_labels(window_start, spans, known, n_seconds=rc.OUTPUT_SECONDS,
                  second=rc.SECOND_SAMPLES, min_cover=rc.MIN_SECOND_COVER):
    """(n_seconds,) uint8 labels of the window starting at `window_start`.

    spans: [(class_index, start, stop)] - rhythm marks, SINUS spans allowed (they count as
           known sinus). Sample positions in the record's coordinates.
    known: [(start, stop)] - where the absence of a mark means SINUS.
    """
    w0 = int(window_start)
    bounds = [(w0 + s * second, w0 + (s + 1) * second) for s in range(n_seconds)]
    cover = np.zeros((n_seconds, rc.NUM_CLASSES))
    known_cover = np.zeros(n_seconds)
    known = merge_intervals(list(known) + [(a, b) for c, a, b in spans if c == rc.SINUS])

    for s, (lo, hi) in enumerate(bounds):
        for c, a, b in spans:
            cover[s, c] += _overlap(a, b, lo, hi) / second
        known_cover[s] = sum(_overlap(a, b, lo, hi) for a, b in known) / second

    labels = np.full(n_seconds, rc.IGNORE, dtype=np.uint8)
    arrhythmia = cover.copy()
    arrhythmia[:, rc.SINUS] = 0.0
    best = np.argmax(arrhythmia, axis=1)
    for s in range(n_seconds):
        if arrhythmia[s, best[s]] >= min_cover:
            labels[s] = best[s]
        elif known_cover[s] >= min_cover:
            labels[s] = rc.SINUS

    # A short span must not vanish: it claims the second it overlaps most, unless that second
    # already belongs to another arrhythmia.
    for c, a, b in spans:
        if c == rc.SINUS:
            continue
        ov = np.array([_overlap(a, b, lo, hi) for lo, hi in bounds])
        if ov.max() <= 0 or np.any(labels == c):
            continue
        s = int(np.argmax(ov))
        if labels[s] in (rc.SINUS, rc.IGNORE):
            labels[s] = c
    return labels


def sample_labels(window_start, spans, known, n_samples=rc.SEGMENT_SAMPLES):
    """(n_samples,) uint8 labels of the window starting at `window_start`, one per sample.

    Same inputs as second_labels. A sample inside an arrhythmia span takes that class; where
    spans of two arrhythmias overlap, the one earlier in DECODE_PRIORITY wins (VT over SVT over
    AFIB ...), as in decoding. Otherwise a sample in `known` or in a SINUS span is SINUS, and
    anything else IGNORE. No coverage threshold: at this resolution a span is exactly where
    the reviewer (or the beat annotation) put it.
    """
    w0 = int(window_start)
    labels = np.full(n_samples, rc.IGNORE, dtype=np.uint8)

    def paint(a, b, value):
        lo, hi = max(0, int(a) - w0), min(n_samples, int(b) - w0)
        if hi > lo:
            labels[lo:hi] = value

    for a, b in list(known) + [(a, b) for c, a, b in spans if c == rc.SINUS]:
        paint(a, b, rc.SINUS)
    rank = {class_index(n): i for i, n in enumerate(rc.DECODE_PRIORITY)}
    arrhythmia = [(c, a, b) for c, a, b in spans if c != rc.SINUS]
    for c, a, b in sorted(arrhythmia, key=lambda s: -rank.get(s[0], len(rank))):
        paint(a, b, c)
    return labels


def spans_to_seconds_table(labels):
    """Human-readable form of a (n,) label vector, for logs and tests."""
    names = rc.CLASS_NAMES
    return ' '.join('.' if v == rc.IGNORE else names[v][:2] for v in labels)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

NOISE = len(rc.CLASS_NAMES)          # the decoded track's extra class, after the six rhythms
DECODE_NAMES = list(rc.CLASS_NAMES) + ['NOISE']


def runs(track):
    """[(start, stop)] of the maximal constant stretches of a 1-D track."""
    track = np.asarray(track)
    if len(track) == 0:
        return []
    edges = np.flatnonzero(np.diff(track)) + 1
    return list(zip(np.concatenate([[0], edges]), np.concatenate([edges, [len(track)]])))


def _smooth_column(x, steps):
    if steps is None or steps <= 1:
        return x
    kernel = np.ones(int(steps), np.float32)
    count = np.convolve(np.ones(len(x), np.float32), kernel, mode='same')
    return np.convolve(x, kernel, mode='same') / count


def smooth_probs(rhythm, seconds):
    """Moving average along time over `seconds` steps (odd; 1 = off) - one width for every
    class, or {class_name: width}: a persistent rhythm (AFIB, AV block) can then be smoothed
    over seconds while a run of a few beats (VT, SVT) keeps its edges - a 2 s VT run averaged
    over 5 s falls under the sinus around it and is lost before any minimum applies. A class
    missing from the dict (SINUS included) is left as it is. The edges average what exists
    rather than zero-padding, so the first and last steps are not pulled towards nothing."""
    if isinstance(seconds, dict):
        widths = [seconds.get(name, 1) for name in rc.CLASS_NAMES]
    else:
        widths = [seconds] * rhythm.shape[1]
    if all(w is None or w <= 1 for w in widths):
        return rhythm
    return np.stack([_smooth_column(rhythm[:, c], w) for c, w in enumerate(widths)], axis=1)


def bridge_gaps(cls, merge_gap, priority):
    """Fill the gap between two episodes of the same class, one class at a time in
    `priority` order, when the gap is at most merge_gap[class] seconds and holds only SINUS,
    NOISE or classes further down the priority list. Returns a new track."""
    cls = np.array(cls)
    rank = {class_index(name): i for i, name in enumerate(priority)}
    for name in priority:
        c = class_index(name)
        gap = merge_gap.get(name, 0)
        if gap <= 0:
            continue
        mine = [(a, b) for a, b in runs(cls) if cls[a] == c]
        for (_, b1), (a2, _) in zip(mine, mine[1:]):
            if a2 - b1 > gap:
                continue
            between = cls[b1:a2]
            if all(v == rc.SINUS or v == NOISE or rank.get(int(v), -1) > rank[c]
                   for v in between):
                cls[b1:a2] = c
    return cls


def enforce_min_duration(cls, min_seconds):
    """An episode shorter than its class's minimum takes its neighbours' class when both
    neighbours agree, otherwise SINUS. Runs are processed left to right on the live track, so
    a reassigned blip is what its right-hand neighbour then sees."""
    cls = np.array(cls)
    n = len(cls)
    for a, b in runs(cls):
        name = DECODE_NAMES[cls[a]]
        if name not in min_seconds or b - a >= min_seconds[name]:
            continue
        left = cls[a - 1] if a > 0 else None
        right = cls[b] if b < n else None
        if left is not None and right is not None and left == right and left != NOISE:
            cls[a:b] = left
        else:
            cls[a:b] = rc.SINUS
    return cls


def enforce_min_confidence(cls, probs, min_prob):
    """An episode whose mean probability of its own class (the probabilities the argmax saw)
    is below min_prob[class] goes the way of a too-short one: its neighbours' class when both
    agree, otherwise SINUS. False AF episodes are the low-confidence ones - on mitdb and on the
    held-out ltafdb records alike (README 12)."""
    if not min_prob:
        return cls
    cls = np.array(cls)
    n = len(cls)
    for a, b in runs(cls):
        c = int(cls[a])
        name = DECODE_NAMES[c]
        if name not in min_prob or c >= probs.shape[1] or \
                float(probs[a:b, c].mean()) >= min_prob[name]:
            continue
        left = cls[a - 1] if a > 0 else None
        right = cls[b] if b < n else None
        if left is not None and right is not None and left == right and left not in (NOISE, c):
            cls[a:b] = left
        else:
            cls[a:b] = rc.SINUS
    return cls


def _to_steps(seconds, step_hz, odd=False):
    steps = int(round(seconds * step_hz))
    return steps + 1 if odd and steps > 1 and steps % 2 == 0 else steps


def decode_episodes(rhythm, p_noise=None, min_seconds=None, noise_threshold=None,
                    start_second=0, smooth_seconds=None, merge_gap=None, priority=None,
                    step_hz=1, class_scale=None, min_prob=None):
    """Rhythm probabilities (T, NUM_CLASSES), `step_hz` rows per second -> rhythm episodes.

    step_hz is 1 for the per-second models and rc.SAMPLE_PROBS_HZ for the per-sample ones;
    every duration below stays in SECONDS and is converted to steps here, so the same
    rc.DECODE_* values mean the same thing on both grids. At step_hz > 1 the episodes'
    start/stop are float seconds.

    p_noise: (T,) probability that the window each second belongs to is unreadable - the
    NOISE entry of the 'lead' output, repeated over the window's seconds (predict.py averages
    it where windows overlap). Seconds above `noise_threshold` become NOISE.

    The steps, in order (defaults rc.DECODE_*, see config for the rationale): smooth the
    probabilities, argmax + NOISE, bridge short gaps between same-class episodes by class
    priority, fold episodes shorter than their class minimum into their neighbours (or
    SINUS), bridge once more. Returns dicts {rhythm, start, stop (exclusive, seconds), prob}
    with prob the mean UNSMOOTHED probability of the episode's class.
    """
    rhythm = np.asarray(rhythm, dtype=np.float32)
    min_seconds = rc.DECODE_MIN_EPISODE_SECONDS if min_seconds is None else min_seconds
    thr = rc.DECODE_NOISE_THRESHOLD if noise_threshold is None else noise_threshold
    smooth = rc.DECODE_SMOOTH_SECONDS if smooth_seconds is None else smooth_seconds
    merge_gap = rc.DECODE_MERGE_GAP_SECONDS if merge_gap is None else merge_gap
    priority = rc.DECODE_PRIORITY if priority is None else priority
    names, noise = DECODE_NAMES, NOISE
    p_noise = np.zeros(len(rhythm)) if p_noise is None else np.asarray(p_noise, np.float32)
    if len(rhythm) == 0:
        return []

    if step_hz != 1:
        smooth = ({k: _to_steps(v, step_hz, odd=True) for k, v in smooth.items()}
                  if isinstance(smooth, dict) else _to_steps(smooth, step_hz, odd=True))
        merge_gap = {k: _to_steps(v, step_hz) for k, v in merge_gap.items()}
        min_seconds = {k: _to_steps(v, step_hz) for k, v in min_seconds.items()}

    scale = rc.DECODE_CLASS_SCALE if class_scale is None else class_scale
    probs = rhythm
    if scale:                   # prior correction: undo the class weights' shift of the boundary
        probs = rhythm * np.array([scale.get(n, 1.0) for n in rc.CLASS_NAMES], np.float32)
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-9)
    cls = np.argmax(smooth_probs(probs, smooth), axis=1)
    cls = np.where(p_noise > thr, noise, cls)
    min_prob = rc.DECODE_MIN_EPISODE_PROB if min_prob is None else min_prob
    cls = bridge_gaps(cls, merge_gap, priority)
    cls = enforce_min_duration(cls, min_seconds)
    cls = enforce_min_confidence(cls, probs, min_prob)
    cls = bridge_gaps(cls, merge_gap, priority)

    at = (lambda i: int(start_second + i)) if step_hz == 1 else \
        (lambda i: float(start_second + i / step_hz))
    episodes = []
    for a, b in runs(cls):
        c = int(cls[a])
        episodes.append({
            'rhythm': names[c], 'start': at(a), 'stop': at(b),
            'prob': float(rhythm[a:b, c].mean()) if c < noise else float(p_noise[a:b].mean()),
        })
    return episodes


def reference_episodes(labels, class_names=None, ignore=None):
    """A (n,) per-second ground-truth label vector -> episodes, in decode_episodes' shape.

    Used to synthesize EC57 reference annotations for the rhythm_eval holdout, which (unlike
    the five Physionet EC57 databases) carries no real WFDB rhythm annotations - only the
    label spans build.py/labels.second_labels already turns every window into.

    An IGNORE second (nobody vouched for it) does not break a run of the same class on
    either side of it, so a one-second gap inside a known AFIB stretch stays one episode; it
    only ends a run when the class actually changes. This is a simplification - it treats
    "unknown" as "probably still whatever was just true" - but it is applied identically to
    the seconds either side of it, so it cannot manufacture a class the data never showed.
    """
    names = class_names or rc.CLASS_NAMES
    ig = rc.IGNORE if ignore is None else ignore
    episodes, cur, start = [], None, None
    for i, v in enumerate(labels):
        if v == ig:
            continue
        if v != cur:
            if cur is not None:
                episodes.append({'rhythm': names[cur], 'start': start, 'stop': i, 'prob': 1.0})
            cur, start = int(v), i
    if cur is not None:
        episodes.append({'rhythm': names[cur], 'start': start, 'stop': len(labels), 'prob': 1.0})
    return episodes
