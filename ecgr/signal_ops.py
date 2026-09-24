"""Signal conditioning shared by data building and inference.

There is exactly one implementation of each step here, and both the training-data builder
and the record sweeper call it. That is the point of the module: the two used to be separate
copies, and any drift between them shows up as a model that scores well on its own eval
split and badly on bxb, which is the hardest kind of bug to see.

The channel axis carries ECG LEADS. `build_leads` is the single place that decides which
physical signal lands on which channel, and it guarantees one invariant everything
downstream relies on: **channel 0 is the annotated lead**.

Window geometry is read from config at call time (None defaults), never bound at import, so
config.apply_geometry can re-point the whole inference path at a checkpoint of another
window length.
"""
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

from . import config


def butter_bandpass_filter(data, lowcut, highcut, fs, order=config.FILTER_ORDER, axis=-1):
    """Zero-phase band-pass. `axis` is the time axis - pass 0 for an (N, leads) array."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=axis)


def z_score(x):
    std = np.std(x)
    return (x - np.mean(x)) / std if std else x - np.mean(x)


def resample_leads(signal, fs_in, fs_out=config.SAMPLING_RATE):
    """Rational resampling of an (N,) or (N, leads) array along time.

    resample_poly rather than wfdb.processing.resample_sig: it is one polyphase FIR per call
    over the whole array instead of a per-lead spline interpolation, and 360 -> 250 Hz is the
    exact ratio 25/36, so no sample position is approximated. Annotation positions are
    rescaled by the same ratio at the call site.
    """
    if fs_in == fs_out:
        return np.asarray(signal, dtype=np.float64)
    from math import gcd
    g = gcd(int(fs_out), int(fs_in))
    return resample_poly(signal, int(fs_out) // g, int(fs_in) // g, axis=0)


def lead_order(n_sig, primary):
    """Record signal index for each model channel: the annotated lead first, cyclic after.

    The inverse question - "model channel c is which lead of the record?" - is answered by
    indexing this list, which is how a best-lead answer (labels.best_lead) is mapped back to
    the record's own channel numbering.
    """
    if config.PRIMARY_LEAD_FIRST:
        return [(int(primary) + k) % int(n_sig) for k in range(int(n_sig))]
    return list(range(int(n_sig)))


def build_leads(raw, fs=config.SAMPLING_RATE, in_channels=None, primary=0, fill_mode=None):
    """Raw record signal -> (N, in_channels) float32, band-passed, annotated lead first.

    `raw` is (N,) for a single-lead record or (N, n_sig) for a montage, at `fs` Hz - already
    resampled to config.SAMPLING_RATE by the caller if it needed to be. `primary` is the
    0-based index of the lead the annotations refer to.

    Two rules, and they are the whole contract of this function:

    1. **The annotated lead becomes channel 0.** The remaining leads keep their cyclic order
       after it, so the mapping is deterministic and reversible (lead_order). Labels, the
       R-peak search in decode_beats, the flatness test and the rhythm descriptor all read
       channel 0, and they would all read a lead nobody annotated without this.
    2. **A record with too few leads is filled per `fill_mode`** (default
       config.LEAD_FILL_MODE = 'zero'): the leftover channels are silence, which is what the
       model sees whenever an electrode comes off and what training produces on purpose
       (config.AUGMENT_LEAD_DROP_PROB). 'duplicate' repeats the annotated lead instead - also
       in distribution (config.AUGMENT_LEAD_DUPLICATE_PROB), but it hands the model a second
       vote for whatever the first lead already said.
    """
    n_ch = int(config.IN_CHANNELS if in_channels is None else in_channels)
    fill = config.LEAD_FILL_MODE if fill_mode is None else fill_mode
    if n_ch < 1:
        raise ValueError(f"in_channels must be >= 1, got {n_ch}")

    x = np.asarray(raw, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n_sig = x.shape[1]
    if not 0 <= primary < n_sig:
        raise ValueError(f"primary lead {primary} outside the record's {n_sig} signal(s)")

    x = butter_bandpass_filter(x, config.FILTER_LOWCUT, config.FILTER_HIGHCUT, fs, axis=0)

    x = x[:, lead_order(n_sig, primary)][:, :n_ch]

    if x.shape[1] < n_ch:
        if fill not in ('zero', 'duplicate'):
            raise ValueError(f"fill_mode must be 'zero' or 'duplicate', got {fill!r}")
        missing = n_ch - x.shape[1]
        pad = (np.repeat(x[:, :1], missing, axis=1) if fill == 'duplicate'
               else np.zeros((len(x), missing)))
        x = np.concatenate([x, pad], axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


def normalize_window(window):
    """Per-window z-score, applied per lead.

    Per lead and not over the whole window: the three portal leads routinely differ by a
    factor of several in amplitude, and one shared scale would flatten the smallest of them
    into noise just as it would a filter band.
    """
    if not config.NORMALIZE_Z_SIGNAL:
        return window
    if window.ndim > 1:
        return np.stack([z_score(window[:, c]) for c in range(window.shape[1])], axis=-1)
    return z_score(window)


def is_flat(window, threshold=config.MIN_AMPLITUDE):
    """A window with no deflection on the annotated lead carries no usable beat.

    Lead-off and saturated stretches land here. Only channel 0 is tested: the labels come
    from that lead, so a window is unusable exactly when that lead is dead, however lively
    the other two are. Callers hand in the REVIEWED part of a window: a strip whose lead 0 is
    dead outside the span but alive inside it still carries every labelled beat.
    """
    trace = window[:, 0] if window.ndim > 1 else window
    if trace.size == 0:
        return True
    return float(np.max(trace) - np.min(trace)) < threshold


def pad_to_length(x, length):
    """Edge-pad an (N, leads) array along time to `length` samples (no-op when long enough).

    Edge padding rather than zeros: after the per-window z-score either becomes a constant
    stretch, and repeating the last sample avoids a step discontinuity the band-passed
    signal never contains. Both the builder (a 30 s strip) and the sweeper (a record shorter
    than one window) pad this way, so the model meets the same padding in both places.
    """
    x = np.asarray(x)
    if len(x) >= length:
        return x
    pad = np.repeat(x[-1:], length - len(x), axis=0)
    return np.concatenate([x, pad], axis=0)


def segment_starts(length, segment_length=None, overlap=None):
    """Window start offsets that cover [0, length) with `overlap` samples between neighbours.

    Every window lies entirely inside the signal, and the last one is pulled back to end
    exactly at `length` instead of running off the end. The old version stepped until
    `length - overlap//2`, which emitted a final window that could be over 90% repeated-last-
    sample padding - and since each window is z-scored independently, that padding came back
    as a full-amplitude flat trace the model then saw as signal.
    """
    segment_length = config.SEGMENT_SAMPLES if segment_length is None else int(segment_length)
    overlap = config.EC57_SEGMENT_OVERLAP if overlap is None else int(overlap)
    step = max(1, segment_length - overlap)
    if length <= segment_length:
        return np.zeros(1, dtype=np.int64)
    starts = np.arange(0, length - segment_length + 1, step, dtype=np.int64)
    if starts[-1] + segment_length < length:
        starts = np.append(starts, length - segment_length)
    return starts


def segment_record(signal, segment_length=None, overlap=None):
    """Cut a whole record into overlapping, normalized windows for inference.

    Returns (segments, starts) with segments (n, segment_length, leads) float32. A record
    shorter than one window - and only that case - is edge-padded; `decode_beats` is told the
    true length so nothing detected inside the padding survives.
    """
    segment_length = config.SEGMENT_SAMPLES if segment_length is None else int(segment_length)
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]

    x = pad_to_length(x, segment_length)

    starts = segment_starts(len(x), segment_length, overlap)
    index = np.arange(segment_length)[None, :] + starts[:, None]
    segments = x[index]

    out = np.empty(segments.shape, dtype=np.float32)
    for k, seg in enumerate(segments):
        out[k] = normalize_window(seg)
    return out, starts
