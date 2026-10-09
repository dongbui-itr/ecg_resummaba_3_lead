"""On-the-fly augmentation of rhythm windows, and the lead-quality target it implies.

The stored windows are clean strips. Every batch is corrupted here, in the graph, and the
corruption is what DEFINES the 'lead' target (best lead, or NOISE): noise is added with a known
amplitude per lead and per sample, so the SNR of every lead in every second is known exactly
rather than estimated by a quality index that could itself be wrong.

Per sample, in this order:

  1. per-lead gain, per-lead sign flip with FLIP_LEAD_PROB (a rhythm reads the same upside
     down), and with LEAD_DROP_PROB one lead zeroed (electrode off);
  2. baseline wander - a nuisance the model must ignore, NOT counted as noise;
  3. recording noise of random intensity on 1, 2 or 3 leads, each at its own SNR, over a
     burst of random position and length, so a window mixes clean and noisy seconds - and,
     with NOISE_WINDOW_PROB, a window wrecked on every lead (the NOISE class of 'lead');
  4. the targets, from the noise actually added: which seconds are clean (`clean_seconds`,
     sets the rhythm loss weight), which lead is best over the window (`lead_scores`) and
     in each 2 s segment (`channel_scores`, the dual U-Net's 'channel' output);
  5. a random permutation of the lead order - rhythm labels belong to time, not to a lead,
     and the lead target is permuted with the leads;
  6. the per-lead z-score again, because inference z-scores the strip as it arrives, noise
     included: without it a noisy lead would reach the model LOUDER than it ever will in use.

Everything draws from stateless RNG ops off one seed per batch, so the validation split can be
corrupted identically every epoch (a fixed benchmark) while training draws a fresh seed.
"""
import numpy as np
import tensorflow as tf
from scipy.signal import firwin

from . import config as rc

_TWO_PI = 2.0 * np.pi
# Broadband noise is shaped to the band the preprocessing passes: at inference everything
# above FILTER_HIGHCUT has been filtered away, so white noise past it would be noise the model
# never meets.
_LOWPASS = firwin(31, 30.0, fs=rc.SAMPLING_RATE).astype(np.float32)


class _Seeds:
    """A stream of stateless seeds folded out of one base seed."""

    def __init__(self, seed):
        self.seed = tf.cast(seed, tf.int64)
        self.k = 0

    def __call__(self):
        self.k += 1
        return tf.random.experimental.stateless_fold_in(self.seed, self.k)


def _uniform(seeds, shape, lo=0.0, hi=1.0):
    return tf.random.stateless_uniform(shape, seeds(), lo, hi)


def _flat_top(seeds, batch, n, lanes, span_range):
    """(batch, n, lanes) envelopes: 1 over a burst of random centre and length, 0.25 s ramps."""
    fs = float(rc.SAMPLING_RATE)
    centre = _uniform(seeds, [batch, 1, lanes], 0.0, float(n))
    half = _uniform(seeds, [batch, 1, lanes], *span_range) * fs / 2.0
    ramp = 0.25 * fs
    t = tf.range(n, dtype=tf.float32)[None, :, None]
    return tf.clip_by_value((half - tf.abs(t - centre)) / ramp + 0.5, 0.0, 1.0)


def _unit_rms(x):
    return x / (tf.sqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True)) + 1e-6)


def _noise_waveforms(seeds, batch, n, c):
    """(batch, n, c) noise of unit RMS, a random mixture of three artefact shapes.

      * broadband - Gaussian, low-passed to the 30 Hz band (EMG, amplifier)
      * tonal     - three sinusoids at 1-25 Hz (tremor, periodic interference); energy in the
                    QRS band, so the kind a detector mistakes for beats
      * motion    - 1-6 Hann transients of 80-500 ms, either sign (electrode motion; the shape
                    closest to a QRS, and the one that fakes an irregular rhythm)
    """
    fs = float(rc.SAMPLING_RATE)
    white = tf.random.stateless_normal([batch * c, n, 1], seeds())
    kernel = tf.constant(_LOWPASS[::-1, None, None])
    white = tf.nn.conv1d(white, kernel, stride=1, padding='SAME')
    white = tf.transpose(tf.reshape(white, [batch, c, n]), [0, 2, 1])

    t = tf.range(n, dtype=tf.float32) / fs
    freq = _uniform(seeds, [batch, 1, c, 3], 1.0, 25.0)
    phase = _uniform(seeds, [batch, 1, c, 3], 0.0, _TWO_PI)
    tonal = tf.reduce_sum(tf.sin(_TWO_PI * freq * t[None, :, None, None] + phase), axis=-1)

    k = 6
    centre = _uniform(seeds, [batch, 1, c, k], 0.0, float(n))
    half = _uniform(seeds, [batch, 1, c, k], 0.04, 0.25) * fs
    sign = tf.sign(_uniform(seeds, [batch, 1, c, k], -1.0, 1.0))
    active = tf.cast(_uniform(seeds, [batch, 1, c, k]) <
                     _uniform(seeds, [batch, 1, c, 1], 0.15, 1.0), tf.float32)
    x = (tf.range(n, dtype=tf.float32)[None, :, None, None] - centre) / half
    bumps = tf.where(tf.abs(x) <= 1.0, 0.5 * (1.0 + tf.cos(np.pi * x)), 0.0)
    motion = tf.reduce_sum(bumps * sign * active, axis=-1) + 1e-3 * white

    w = _uniform(seeds, [batch, 1, 1, 3])
    w = w / tf.sqrt(tf.reduce_sum(tf.square(w), axis=-1, keepdims=True))
    parts = tf.stack([_unit_rms(white), _unit_rms(tonal), _unit_rms(motion)], axis=-1)
    return _unit_rms(tf.reduce_sum(parts * w, axis=-1))


def _readable(sig_rms, noise, live, snr_db=None):
    """(batch, seconds, c) bool: is each lead readable in each second, given the noise added.

    sig_rms (batch, 1, c) - RMS of each lead's clean signal over the window
    noise   (batch, n, c) - exactly what was added (wander excluded)
    live    (batch, 1, c) - lead carries a signal at all
    """
    snr_db = rc.CLEAN_SNR_DB if snr_db is None else snr_db
    c = noise.shape[-1] or rc.IN_CHANNELS
    per_sec = tf.reshape(noise, [-1, rc.OUTPUT_SECONDS, rc.SECOND_SAMPLES, c])
    noise_rms = tf.sqrt(tf.reduce_mean(tf.square(per_sec), axis=2))            # (b, s, c)
    return tf.logical_and(sig_rms / (noise_rms + 1e-6) >= 10.0 ** (snr_db / 20.0), live)


def clean_seconds(sig_rms, noise, live, snr_db=None, min_leads=None):
    """(batch, seconds) float 1/0: >= CLEAN_MIN_LEADS leads readable (or every live lead)."""
    min_leads = rc.CLEAN_MIN_LEADS if min_leads is None else min_leads
    good = _readable(sig_rms, noise, live, snr_db)
    n_good = tf.reduce_sum(tf.cast(good, tf.int32), axis=-1)
    n_live = tf.reduce_sum(tf.cast(live, tf.int32), axis=-1)                    # (b, 1)
    need = tf.minimum(min_leads, n_live)
    return tf.cast(tf.logical_and(n_good >= need, need > 0), tf.float32)


def lead_scores(clean_x, sig_rms, noise, live):
    """(batch, c) ranking key of each lead, and (batch, c) readable seconds per lead.

    key = readable seconds x 100 + window SNR (dB, clipped to [0, LEAD_SNR_CAP_DB]) x 3
          + tanh(kurtosis / 20)
    Each term is bounded below the step of the one before it, so the order is strictly
    lexicographic (config: readable seconds, then SNR, then QRS peakedness). A flat lead
    gets -1 and can never be chosen.
    """
    n_read = tf.reduce_sum(tf.cast(_readable(sig_rms, noise, live), tf.float32), axis=1)
    noise_rms = tf.sqrt(tf.reduce_mean(tf.square(noise), axis=1))              # (b, c)
    snr = 20.0 * tf.math.log(sig_rms[:, 0, :] / (noise_rms + 1e-9) + 1e-9) / np.log(10.0)
    snr = tf.clip_by_value(snr, 0.0, rc.LEAD_SNR_CAP_DB)
    z = _zscore(clean_x)
    kurt = tf.reduce_mean(tf.pow(z, 4), axis=1)                                 # (b, c)
    key = n_read * 100.0 + snr * 3.0 + tf.tanh(kurt / 20.0)
    alive = live[:, 0, :]
    return tf.where(alive, key, -1.0), tf.where(alive, n_read, 0.0)


def channel_scores(clean_x, sig_rms, noise, live):
    """(batch, NOISE_SEGMENTS, c) ranking key of each lead in each 2 s segment - lead_scores'
    lexicographic key (readable seconds, then SNR, then QRS peakedness) measured per segment
    instead of per window. Kurtosis stays a window property (a 2 s stretch holds too few
    beats for it). A flat lead gets -1."""
    c = noise.shape[-1] or rc.IN_CHANNELS
    k, per = rc.NOISE_SEGMENTS, rc.NOISE_SEGMENT_SECONDS
    readable = tf.cast(_readable(sig_rms, noise, live), tf.float32)            # (b, 10, c)
    n_read = tf.reduce_sum(tf.reshape(readable, [-1, k, per, c]), axis=2)      # (b, 5, c)
    seg = tf.reshape(noise, [-1, k, rc.SEGMENT_SAMPLES // k, c])
    noise_rms = tf.sqrt(tf.reduce_mean(tf.square(seg), axis=2))                # (b, 5, c)
    snr = 20.0 * tf.math.log(sig_rms / (noise_rms + 1e-9) + 1e-9) / np.log(10.0)
    snr = tf.clip_by_value(snr, 0.0, rc.LEAD_SNR_CAP_DB)
    kurt = tf.reduce_mean(tf.pow(_zscore(clean_x), 4), axis=1)[:, None, :]    # (b, 1, c)
    key = n_read * 100.0 + snr * 3.0 + tf.tanh(kurt / 20.0)
    return tf.where(live, key, -1.0)


def channel_label(key, segment_clean):
    """(batch, NOISE_SEGMENTS) int: LEAD_NOISE where the segment is not CLEAN (the 'noise'
    head's rule: both seconds readable on CLEAN_MIN_LEADS leads), else 1 + the best lead."""
    best = tf.argmax(key, axis=-1, output_type=tf.int32) + 1
    return tf.where(segment_clean > 0.5, best, rc.LEAD_NOISE)


def lead_label(key, n_read):
    """(batch,) int: LEAD_NOISE, or 1 + index of the best lead."""
    best = tf.argmax(key, axis=-1, output_type=tf.int32)
    best_read = tf.gather(n_read, best, axis=1, batch_dims=1)
    ok = best_read >= float(rc.LEAD_MIN_READABLE_SECONDS)
    return tf.where(ok, best + 1, rc.LEAD_NOISE)


def _zscore(x):
    mean = tf.reduce_mean(x, axis=1, keepdims=True)
    std = tf.math.reduce_std(x, axis=1, keepdims=True)
    return tf.where(std > 1e-6, (x - mean) / tf.maximum(std, 1e-6), tf.zeros_like(x))


def beat_targets(beats, clean):
    """(batch, SEGMENT_SAMPLES) uint8 beat labels -> (batch, BEAT_STEPS, 6):
    [heat | N S V | w_heat | w_type]. heat = Gaussian (sigma rc.BEAT_HEAT_SIGMA_STEPS) around
    each R step, clipped to 1; N/S/V = the beat's one-hot inside +-rc.BEAT_TYPE_RADIUS_STEPS
    of its R, 0 elsewhere; w_heat = the step's weight (0 in IGNORE zones, NOISY_SECOND_WEIGHT
    in a noisy second, 1 otherwise); w_type = w_heat on the typed steps, 0 elsewhere.
    2500 -> BEAT_STEPS by max over each block (a beat wins over 'none')."""
    nb = len(rc.BEAT_CLASSES)
    beats = tf.cast(beats, tf.int32)
    ignore = tf.cast(beats == rc.IGNORE, tf.float32)
    cls = tf.where(beats == rc.IGNORE, 0, beats)
    block = rc.SEGMENT_SAMPLES // rc.BEAT_STEPS
    cls = tf.reduce_max(tf.reshape(cls, [-1, rc.BEAT_STEPS, block]), axis=-1)
    ignore = tf.reduce_max(tf.reshape(ignore, [-1, rc.BEAT_STEPS, block]), axis=-1)
    onehot = tf.one_hot(cls, nb)[..., 1:]                                # (b, steps, 3) spikes
    spike = tf.reduce_max(onehot, axis=-1, keepdims=True)                # (b, steps, 1)

    sigma = float(rc.BEAT_HEAT_SIGMA_STEPS)
    half = int(np.ceil(3 * sigma))
    g = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2).astype(np.float32)
    heat = tf.nn.conv1d(spike, g[:, None, None], 1, 'SAME')
    heat = tf.minimum(heat, 1.0)[..., 0]

    k = 2 * rc.BEAT_TYPE_RADIUS_STEPS + 1
    typed = tf.nn.max_pool1d(onehot, k, 1, 'SAME')                      # widen N/S/V
    is_typed = tf.reduce_max(typed, axis=-1)
    # two beats closer than the radius: keep one class per step (the max-pool can light two)
    typed = tf.one_hot(tf.argmax(typed, axis=-1), nb - 1) * is_typed[..., None]

    k_ign = 2 * rc.BEAT_TARGET_HALFWIDTH_STEPS + 1
    valid = 1.0 - tf.minimum(1.0, tf.nn.max_pool1d(ignore[..., None], k_ign, 1, 'SAME')[..., 0])
    clean = tf.repeat(clean, rc.BEAT_STEPS // rc.OUTPUT_SECONDS, axis=1)
    w_heat = valid * (clean + (1.0 - clean) * rc.NOISY_SECOND_WEIGHT)
    w_type = w_heat * is_typed
    return tf.concat([heat[..., None], typed, w_heat[..., None], w_type[..., None]], axis=-1)


def targets(labels, clean, lead, beats=None, channel_key=None):
    """{'rhythm': (batch, steps, NUM_CLASSES + 1), 'lead': (batch, NUM_LEAD_CLASSES),
        'noise': (batch, NOISE_SEGMENTS, 2), 'beat' (when beats given): (batch, BEAT_STEPS, 6),
        'channel' (when channel_key given): (batch, NOISE_SEGMENTS, NUM_LEAD_CLASSES)}
    - pipeline keeps the ones the model outputs.

    labels are per second (steps = 10), per 20 ms (500) or per sample (2500); `clean` is per
    second - noise is measured per second - and is repeated over a second's samples.
    rhythm = one-hot rhythm (all zero on an IGNORE step) | loss weight of that step,
    0 on IGNORE and NOISY_SECOND_WEIGHT on a noisy one - see config. lead = one-hot.
    """
    labels = tf.cast(labels, tf.int32)
    clean_sec = clean
    # 'noise' (the 20 ms family): a 2 s segment is CLEAN only when both its seconds are
    segment_clean = tf.reduce_min(tf.reshape(
        clean, [-1, rc.NOISE_SEGMENTS, rc.NOISE_SEGMENT_SECONDS]), axis=-1)
    noise = tf.one_hot(tf.cast(segment_clean < 0.5, tf.int32), len(rc.NOISE_CLASSES))
    steps = labels.shape[1]
    if steps is not None and steps != rc.OUTPUT_SECONDS:
        clean = tf.repeat(clean, steps // rc.OUTPUT_SECONDS, axis=1)
    valid = tf.cast(labels != rc.IGNORE, tf.float32)
    onehot = tf.one_hot(tf.where(labels == rc.IGNORE, 0, labels), rc.NUM_CLASSES) * \
        valid[..., None]
    weight = valid * (clean + (1.0 - clean) * rc.NOISY_SECOND_WEIGHT)
    out = {'rhythm': tf.concat([onehot, weight[..., None]], axis=-1),
           'lead': tf.one_hot(lead, rc.NUM_LEAD_CLASSES), 'noise': noise}
    if beats is not None:
        out['beat'] = beat_targets(beats, clean_sec)
    if channel_key is not None:
        out['channel'] = tf.one_hot(channel_label(channel_key, segment_clean),
                                    rc.NUM_LEAD_CLASSES)
    return out


def no_augment(signal, labels, beats=None):
    """Stored windows as they are: no noise, so the lead target comes from the signal alone."""
    x = tf.cast(signal, tf.float32)
    sig_rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True))
    live = sig_rms > 0.05
    noise = tf.zeros_like(x)
    key, n_read = lead_scores(x, sig_rms, noise, live)
    return x, targets(labels, clean_seconds(sig_rms, noise, live), lead_label(key, n_read),
                      beats, channel_scores(x, sig_rms, noise, live))


def augment(signal, labels, seed, noise_prob=None, permute_prob=None, snr_range=None,
            wreck_prob=None, flip_prob=None, drop_prob=None, beats=None):
    """Corrupt one batch. `seed` is a [2] int tensor; the same seed gives the same batch."""
    seeds = _Seeds(seed)
    noise_prob = rc.NOISE_PROB if noise_prob is None else noise_prob
    wreck_prob = rc.NOISE_WINDOW_PROB if wreck_prob is None else wreck_prob
    permute_prob = rc.PERMUTE_LEADS_PROB if permute_prob is None else permute_prob
    flip_prob = rc.FLIP_LEAD_PROB if flip_prob is None else flip_prob
    drop_prob = rc.LEAD_DROP_PROB if drop_prob is None else drop_prob
    snr_lo, snr_hi = (rc.NOISE_SNR_DB_MIN, rc.NOISE_SNR_DB_MAX) if snr_range is None \
        else snr_range

    x = tf.cast(signal, tf.float32)
    batch = tf.shape(x)[0]
    n, c = rc.SEGMENT_SAMPLES, rc.IN_CHANNELS

    # 1. gain, sign flip, lead drop
    x = x * _uniform(seeds, [batch, 1, c], *rc.LEAD_GAIN_RANGE)
    x = x * tf.where(_uniform(seeds, [batch, 1, c]) < flip_prob, -1.0, 1.0)
    victim = tf.one_hot(tf.random.stateless_uniform([batch], seeds(), 0, c, tf.int32), c)
    drop = tf.cast(_uniform(seeds, [batch]) < drop_prob, tf.float32)
    x = x * (1.0 - victim * drop[:, None])[:, None, :]
    sig_rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True))      # (b, 1, c)
    live = sig_rms > 0.05

    # 2. baseline wander (not noise: it leaves the rhythm readable)
    t = tf.range(n, dtype=tf.float32) / float(rc.SAMPLING_RATE)
    freq = _uniform(seeds, [batch, 1, c, 2], 0.05, 0.5)
    phase = _uniform(seeds, [batch, 1, c, 2], 0.0, _TWO_PI)
    amp = _uniform(seeds, [batch, 1, c, 2], 0.0, rc.WANDER_AMP)
    wander = tf.reduce_sum(amp * tf.sin(_TWO_PI * freq * t[None, :, None, None] + phase), -1)
    wander *= tf.cast(_uniform(seeds, [batch, 1, 1]) < rc.WANDER_PROB, tf.float32) * sig_rms

    # 3. noise of random intensity
    weights = tf.math.log(tf.constant([list(rc.NOISE_LEADS_WEIGHTS)], tf.float32))
    k = tf.random.stateless_categorical(tf.tile(weights, [batch, 1]), 1, seeds())[:, 0] + 1
    rank = tf.argsort(tf.argsort(_uniform(seeds, [batch, c]), axis=-1), axis=-1)
    hit = tf.cast(rank < tf.cast(k, tf.int32)[:, None], tf.float32)[:, None, :]  # (b, 1, c)
    on = tf.cast(_uniform(seeds, [batch, 1, 1]) < noise_prob, tf.float32)
    snr = _uniform(seeds, [batch, 1, c], snr_lo, snr_hi)
    level = sig_rms * tf.pow(10.0, -snr / 20.0)
    shared = _uniform(seeds, [batch, 1, 1]) < rc.NOISE_SHARED_BURST_PROB
    env = tf.where(shared,
                   tf.tile(_flat_top(seeds, batch, n, 1, rc.NOISE_SPAN_SECONDS), [1, 1, c]),
                   _flat_top(seeds, batch, n, c, rc.NOISE_SPAN_SECONDS))
    # whole-window wrecks: all leads, the whole window, below the readable SNR
    wreck = _uniform(seeds, [batch, 1, 1]) < wreck_prob
    wreck_hi = rc.CLEAN_SNR_DB - 3.0
    wreck_snr = _uniform(seeds, [batch, 1, c], min(snr_lo, wreck_hi), wreck_hi)
    on = tf.where(wreck, 1.0, on)
    hit = tf.where(wreck, 1.0, hit)
    env = tf.where(wreck, 1.0, env)
    level = tf.where(wreck, sig_rms * tf.pow(10.0, -wreck_snr / 20.0), level)
    noise = on * hit * level * env * _noise_waveforms(seeds, batch, n, c)

    # 4. the targets, from exactly what was added
    clean = clean_seconds(sig_rms, noise, live)
    key, n_read = lead_scores(x, sig_rms, noise, live)
    chan_key = channel_scores(x, sig_rms, noise, live)
    x = x + wander + noise

    # 5. lead order - the lead target is permuted WITH the leads
    perm = tf.argsort(_uniform(seeds, [batch, c]), axis=-1)
    keep = _uniform(seeds, [batch, 1]) >= permute_prob
    perm = tf.where(keep, tf.range(c)[None, :], perm)
    x = tf.gather(x, perm, axis=2, batch_dims=1)
    key = tf.gather(key, perm, axis=1, batch_dims=1)
    n_read = tf.gather(n_read, perm, axis=1, batch_dims=1)
    chan_key = tf.gather(chan_key, perm, axis=2, batch_dims=1)

    # 6. what inference does to any strip
    x = _zscore(x)
    x.set_shape([None, n, c])
    return x, targets(labels, clean, lead_label(key, n_read), beats, chan_key)
