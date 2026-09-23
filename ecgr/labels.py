"""The label contract: how beat annotations become a per-step target, and back again.

Forward (training data) and inverse (inference) live in the same file on purpose - they are
one convention seen from two sides, and the EC57 numbers only mean anything if they agree.

Both sides read **channel 0**, the lead the annotations were made on (see
signal_ops.build_leads); the other leads are evidence for the model, never for the labels.
"""
import numpy as np

from . import config

STEP_SAMPLES = config.SEGMENT_SAMPLES // config.OUTPUT_STEPS   # 5 samples = 20 ms


def labels_from_annotations(sample_positions, symbols,
                            num_steps=config.OUTPUT_STEPS,
                            step_samples=STEP_SAMPLES):
    """Per-step labels of one window: a block of steps around each annotated beat.

    The block is asymmetric - LABEL_STEPS_BEFORE reaches back over the PR interval, because
    the one feature separating a non-premature atrial ectopic from a sinus beat is its P
    wave, and a block that starts at the R peak never covers it.

    Blocks are merged with max(), so the S(3) > V(2) > N(1) ordering means a neighbouring
    normal beat can never overwrite an ectopic block where the two overlap.
    """
    labels = np.zeros(num_steps, dtype=np.int64)
    for pos, sym in zip(sample_positions, symbols):
        value = config.SYMBOL_TO_LABEL.get(sym)
        if value is None:                      # symbol outside the AAMI grouping: not a target
            continue
        step = min(int(pos) // step_samples, num_steps - 1)
        lo = max(step - config.LABEL_STEPS_BEFORE, 0)
        hi = min(step + config.LABEL_STEPS_AFTER + 1, num_steps)
        labels[lo:hi] = np.maximum(labels[lo:hi], value)
    return labels


# How a run of per-step labels becomes one beat symbol.
#
# 'max' takes the highest class index in the run, so a single stray S step turns the whole
# beat into an S. That was tolerable with a 5-step block; the PR-spanning block here is 11
# steps, which more than doubles the chance of catching one bad step, and EC57 positive
# predictivity collapsed accordingly (nstdb S_+P 70.34 -> 42.94, V_+P 94.75 -> 69.11) while
# step-level metrics improved - the tell that the fault was in the decoder, not the model.
BEAT_CLASS_RULE = 'majority'      # 'majority' | 'max'


def run_class(run_labels):
    """One class for a run of per-step labels, ignoring the background class."""
    nonzero = run_labels[run_labels > 0]
    if nonzero.size == 0:
        return int(np.max(run_labels))
    if BEAT_CLASS_RULE == 'max':
        return int(np.max(nonzero))
    return int(np.bincount(nonzero).argmax())


def core_bounds(starts, segment_length=config.SEGMENT_SAMPLES, signal_length=None):
    """Per-segment [lo, hi) sample range a segment is allowed to claim a beat in.

    Consecutive inference windows overlap, so without a rule a beat inside the overlap is
    decoded twice. The rule here is to split every overlap down the middle: segment i owns
    up to the midpoint of its overlap with i+1, and segment i+1 owns from there. The pieces
    then tile the record EXACTLY - no gaps, no double counting - for any spacing of `starts`,
    including the short final step that segment_starts adds to land on the record's end.

    This replaces relying on the previous accepted position: that assumed detections arrive
    in increasing order across segments (they do not, inside an overlap) and it kept the
    copy from the window where the beat sat closest to an EDGE, i.e. the copy with the least
    context on one side - and this model is bidirectional, so both sides matter.
    """
    starts = np.asarray(starts, dtype=np.int64)
    end = starts[-1] + segment_length if signal_length is None else int(signal_length)
    mid = (starts[:-1] + starts[1:] + segment_length) // 2        # boundary between i, i+1
    lo = np.concatenate([starts[:1], mid])
    hi = np.concatenate([mid, [end]])
    return lo, np.maximum(hi, lo)


def decode_beats(preds, segments, starts, s_boost=1.0,
                 min_rr=config.MIN_RR_INTERVAL, signal_length=None, min_run_steps=1,
                 min_peak_prob=None):
    """Per-step predictions of a whole record -> (positions, symbols), sorted by position.

    A beat is a run of consecutive steps predicted non-background. Its position is the
    largest |amplitude| inside the run on channel 0 - the R peak, which is where the
    reference annotations sit - and its symbol is run_class of the run. A run is credited to
    the segment that owns its position (see core_bounds), and detections closer than min_rr
    to the previous accepted one are then dropped as the physiological floor they are.

    min_run_steps drops runs shorter than that many steps: a one-step (20 ms) detection is
    far more often an artefact flicker than a beat, whose label block is 11 steps wide, so
    the filter costs almost no sensitivity for the positive predictivity it buys. 1 = off.

    s_boost multiplies the S probability before the argmax, i.e. slides the model along its
    own sensitivity / positive-predictivity curve without retraining. Calibrate it on portal
    data only: tuning it against mitdb would make the EC57 benchmark self-scoring, the same
    prohibition that rules out training on those databases.

    min_peak_prob drops a run whose beat probability (1 - p_None) never reaches it - the
    refusal to call a beat out of signal the model cannot read. None takes
    config.DECODE_MIN_PEAK_PROB, where the measured cost and benefit are written down; 0 is
    off. It is read from the UNBOOSTED probabilities, because p_None is what says "there is
    no beat here" and s_boost rescales a different column. The filter runs before the min_rr
    suppression, so a rejected artefact no longer shadows a real beat 180 ms behind it.
    """
    preds = np.asarray(preds)
    floor = config.DECODE_MIN_PEAK_PROB if min_peak_prob is None else float(min_peak_prob)
    beat_prob = 1.0 - preds[..., 0] if floor > 0.0 else None
    if s_boost != 1.0:
        preds = preds.copy()
        preds[..., config.CLASS_NAMES.index('S')] *= s_boost
    step_labels = np.argmax(preds, axis=-1)                # (n_segments, OUTPUT_STEPS)

    segment_length = np.asarray(segments[0]).shape[0]
    lo_bound, hi_bound = core_bounds(starts, segment_length, signal_length)

    found = []
    for seg_idx, seg_pred in enumerate(step_labels):
        hits = np.flatnonzero(seg_pred > 0)
        if hits.size == 0:
            continue

        seg = np.asarray(segments[seg_idx])
        # channel 0 is the annotated lead: the reference R peak is there
        trace = seg[:, 0] if seg.ndim > 1 else seg.squeeze()
        groups = np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1)

        for group in groups:
            if len(group) < min_run_steps:
                continue
            if beat_prob is not None and beat_prob[seg_idx, group].max() < floor:
                continue
            lo = int(group[0] * STEP_SAMPLES)
            hi = min(int((group[-1] + 1) * STEP_SAMPLES), len(trace))
            window = trace[lo:hi]
            if window.size == 0:
                continue
            position = int(starts[seg_idx] + lo + np.argmax(np.abs(window)))
            if not lo_bound[seg_idx] <= position < hi_bound[seg_idx]:
                continue                       # another segment owns this stretch
            found.append((position, run_class(seg_pred[group])))

    found.sort()
    positions, symbols = [], []
    for position, label in found:
        if positions and position - positions[-1] < min_rr:
            continue
        positions.append(position)
        symbols.append(config.CLASS_NAMES[label])

    return (np.asarray(positions, dtype=np.int64),
            np.asarray(symbols, dtype='<U4'))
