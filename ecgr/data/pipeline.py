"""tf.data input pipeline over the tfrecords.

Order matters for throughput: read (parallel) -> [cache] -> shuffle -> batch -> parse ->
augment -> prefetch.

  * cache() holds the serialized records in RAM, so only the first epoch touches disk. At
    60 s the train split is ~90 GB, so it is off by default (config.CACHE_DATASET).
  * shuffle BEFORE batch, on the serialized strings: shuffling raw records is far cheaper
    than shuffling parsed tensors, which is what lets the buffer be big enough to matter.
    Records come off disk grouped by study, so a small buffer leaves a batch made of one
    patient's beats - SHUFFLE_BUFFER is sized to break that up.
  * batch BEFORE parse, so tf.io.parse_example is vectorised over the batch. Parsing one
    record at a time was what kept the GPU at ~50%.

What a supervised batch looks like:

    signal   (B, SEGMENT_SAMPLES, IN_CHANNELS)   float32
    targets  {'beat_cls':     (B, OUTPUT_STEPS, NUM_CLASSES)  one-hot; an ALL-ZERO row is a
                              step with no trusted label (config.IGNORE_LABEL) and every loss
                              and metric skips it,
              'lead_quality': (B, OUTPUT_STEPS, IN_CHANNELS)  in [0, 1], the label-free
                              readability of each lead at each step - see `corrupt`}
"""
import glob
import json
import os

import tensorflow as tf

from .. import config

# Fields of the manifest that MUST match the live config: the parser reinterprets raw bytes
# with these, so a mismatch is silent corruption rather than an error.
_MANIFEST_MUST_MATCH = ('segment_samples', 'in_channels', 'output_steps', 'num_classes',
                        'signal_dtype', 'labels_dtype')

_TWO_PI = 2.0 * 3.14159265


def read_manifest(tfrecord_dir=None):
    """The dataset manifest written by build_tfrecord, or None if there is none."""
    path = os.path.join(tfrecord_dir or config.TFRECORD_DIR, config.DATASET_MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def check_manifest(tfrecord_dir=None):
    """Refuse to read tfrecords whose geometry is not the one this run is configured for.

    Without this the mismatch surfaces as `Input to reshape is a tensor with 1250000 values
    but the requested shape has 3750000` from inside the first training step - which names
    neither the file nor the field that disagrees.
    """
    manifest = read_manifest(tfrecord_dir)
    if manifest is None:
        print(f"no {config.DATASET_MANIFEST} in {tfrecord_dir or config.TFRECORD_DIR} "
              f"- reading it as float32 x {config.IN_CHANNELS} leads on trust")
        return None
    live = {'segment_samples': config.SEGMENT_SAMPLES, 'in_channels': config.IN_CHANNELS,
            'output_steps': config.OUTPUT_STEPS, 'num_classes': config.NUM_CLASSES,
            'signal_dtype': 'float32', 'labels_dtype': 'uint8'}
    bad = {k: (manifest.get(k), live[k]) for k in _MANIFEST_MUST_MATCH
           if manifest.get(k) != live[k]}
    if bad:
        detail = ', '.join(f"{k}: data={d!r} config={c!r}" for k, (d, c) in bad.items())
        raise ValueError(
            f"the tfrecords in {tfrecord_dir or config.TFRECORD_DIR} do not match this run - "
            f"{detail}. Rebuild with `python -m ecgr data` or point ECGR_TFRECORD_DIR at the "
            f"matching tree.")
    total = manifest['totals']
    print(f"data         : {total['train']['segments']:,} train / "
          f"{total['eval']['segments']:,} eval segments, "
          f"({config.SEGMENT_SAMPLES}, {config.IN_CHANNELS}) leads, built {manifest['built']}")
    return manifest


def split_files(split, db_names=None):
    """Every tfrecord of `split` ('train' | 'eval'), over all training datasets."""
    files = []
    for db in (db_names or config.TRAIN_DATASETS):
        files += sorted(glob.glob(os.path.join(config.TFRECORD_DIR, db, split, "*.tfrecord")))
    if not files:
        raise FileNotFoundError(
            f"no {split} tfrecords under {config.TFRECORD_DIR} for {db_names or 'any dataset'} "
            f"- run `python -m ecgr data` first")
    return files


def assert_no_benchmark_data(files):
    """Refuse to train on anything derived from the EC57 scoring databases.

    mitdb / nstdb / escdb / ahadb / afdb are the independent benchmark: a single tfrecord
    built from them would silently turn every EC57 number into self-scoring. Checked on the
    resolved file paths, not only on dataset names - a dataset could be named anything yet
    point at config.PHYSIONET_DIR.
    """
    banned = {db.lower() for db in config.EC57_DBS}
    physionet = os.path.realpath(config.PHYSIONET_DIR)
    bad = [f for f in files
           if os.path.realpath(f).startswith(physionet)
           or any(f"/{b}/" in f.lower() for b in banned)]
    if bad:
        raise RuntimeError(
            f"{len(bad)} training tfrecords come from an EC57 benchmark database "
            f"(e.g. {bad[:3]}). Training on them would make every EC57 number self-scored.")


def parse_signal(proto_batch):
    """Parse only the signal of a BATCH of examples.

    A separate parser rather than parse_batch-then-drop, because the two differ in what they
    REQUIRE of the file: this one declares no `labels` feature, so the self-supervised stages
    can read a corpus that has none. Dropping the labels after parsing them still makes the
    field mandatory, which quietly turned "pretrain on unlabeled recordings" - the whole
    practical point of self-supervision - into "pretrain on labelled recordings, ignoring the
    labels". Files that do carry labels are read by this parser too; the field is simply left
    alone.
    """
    parsed = tf.io.parse_example(proto_batch, {
        'signal': tf.io.FixedLenFeature([], tf.string),
    })
    signal = tf.io.decode_raw(parsed['signal'], tf.float32)
    return tf.reshape(signal, [-1, config.SEGMENT_SAMPLES, config.IN_CHANNELS])


def parse_batch(proto_batch):
    """Parse a BATCH of examples (see module docstring for why it is batched).

    Labels come back one-hot over the classes. config.IGNORE_LABEL (255) lies outside
    [0, NUM_CLASSES), and tf.one_hot writes an all-zero row for such an index - which is
    exactly the representation the losses and metrics key on: zero mass = no label here.
    """
    parsed = tf.io.parse_example(proto_batch, {
        'signal': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.FixedLenFeature([], tf.string),
    })
    signal = tf.io.decode_raw(parsed['signal'], tf.float32)
    signal = tf.reshape(signal, [-1, config.SEGMENT_SAMPLES, config.IN_CHANNELS])
    labels = tf.io.decode_raw(parsed['labels'], tf.uint8)
    labels = tf.reshape(labels, [-1, config.OUTPUT_STEPS])
    labels = tf.one_hot(tf.cast(labels, tf.int32), depth=config.NUM_CLASSES,
                        dtype=tf.float32)
    return signal, labels


def _time_scale(signal, labels):
    """Resample the batch in time by a random factor, labels included.

    The portal records are natively 250 Hz while mitdb/ahadb arrive at 360/250 Hz and get
    resampled, which subtly rescales every wave. Training over a range of time scales stops
    the model from locking onto one sampling geometry.

    Both tensors are cropped or zero-padded back to their original length so the shapes stay
    fixed. Padded label steps come out as all-zero one-hot rows, i.e. IGNORE: nothing is
    known about signal that was never recorded, and the losses skip zero-mass rows. (The 10 s
    pipeline filled them with the background class instead; with an explicit ignore value
    that is no longer the right thing to say.)

    The crop-back targets are the config's Python ints rather than tf.shape() results, so
    the STATIC shape survives the round trip: `_lead_jitter` reads the lead count from it,
    RhythmDescriptor clamps its lag band against the sequence length, and DiagSSM1D gets a
    static FFT length.
    """
    n, c = config.SEGMENT_SAMPLES, config.IN_CHANNELS
    m, k = config.OUTPUT_STEPS, config.NUM_CLASSES

    scale = tf.random.uniform([], 0.9, 1.1)
    new_n = tf.cast(tf.round(n * scale), tf.int32)
    sig = tf.image.resize(signal[..., tf.newaxis], [new_n, c])[..., 0]
    sig = tf.image.resize_with_crop_or_pad(sig[..., tf.newaxis], n, c)[..., 0]

    new_m = tf.cast(tf.round(m * scale), tf.int32)
    lab = tf.image.resize(labels[..., tf.newaxis], [new_m, k], method='nearest')[..., 0]
    lab = tf.image.resize_with_crop_or_pad(lab[..., tf.newaxis], m, k)[..., 0]

    sig.set_shape([None, n, c])
    lab.set_shape([None, m, k])
    return sig, lab


# ---------------------------------------------------------------------------
# Corruption, and the label-free lead-quality target it yields
# ---------------------------------------------------------------------------

def _lead_jitter_masks(batch, channels):
    """(dup, drop): dup (B,1,1) bool - collapse the sample to lead 0 repeated; drop (B,1,C)
    float - 1 on the one non-primary lead zeroed for that sample, else 0."""
    dup = tf.random.uniform([batch, 1, 1]) < config.AUGMENT_LEAD_DUPLICATE_PROB
    victim = tf.random.uniform([batch], 1, channels, dtype=tf.int32)
    active = tf.random.uniform([batch]) < config.AUGMENT_LEAD_DROP_PROB
    drop = tf.one_hot(victim, channels) * tf.cast(active, tf.float32)[:, None]
    return dup, drop[:, None, :]


def _apply_lead_jitter(signal, dup, drop):
    channels = signal.shape[-1] or config.IN_CHANNELS
    signal = tf.where(dup, tf.tile(signal[..., :1], [1, 1, channels]), signal)
    return signal * (1.0 - drop)


def _lead_jitter(signal):
    """Per-sample lead manipulations: duplicate the primary lead, or drop a secondary one.

    The EC57 databases have two leads and are annotated on the first, so scoring them means
    feeding the model fewer real leads than it has channels. A model trained only on three
    genuinely different leads has never seen that input, and the mismatch lands squarely on
    the channel axis the AdaIN conditioning reads. Reproducing the case here at
    AUGMENT_LEAD_DUPLICATE_PROB is what makes the benchmark an evaluation rather than a
    domain shift.

    The drop branch is the other half of the same story: an electrode that comes off leaves
    one flat lead, which is common in real recordings and must not take the prediction with
    it. Lead 0 is never dropped - it is the lead the labels refer to.

    Used directly by the CPC stage (signal only); the supervised path goes through `corrupt`
    so the same draws can also shape the lead-quality target.
    """
    channels = signal.shape[-1] or config.IN_CHANNELS
    if channels < 2:
        return signal
    dup, drop = _lead_jitter_masks(tf.shape(signal)[0], channels)
    return _apply_lead_jitter(signal, dup, drop)


def _noise_field(signal):
    """The additive recording noise of `_noise`, as a (B, n, C) field to add to the signal.

    Three components, each switched on per SAMPLE with its own probability so about half
    the batch carries some noise, amplitudes in per-lead z-score units (QRS at ~3-8 std):

      * baseline wander - two sinusoids per lead, 0.05-0.6 Hz, up to AUGMENT_WANDER_AMP std
        (respiration, slow electrode drift; the 0.5 Hz high-pass leaves the upper part)
      * broadband noise - white Gaussian up to AUGMENT_NOISE_AMP std (EMG, amplifier)
      * a motion transient - one Hann bump of 0.2-1.0 s and 1-3 std on ONE lead, either sign
        (electrode motion; the shape that most resembles a QRS and so costs the most +P)

    Motivated by where the EC57 detection errors actually are: on mitdb, 74 of 82 missed
    beats and 99 of 121 false beats of the 2m model sit in records 203, 105, 108 and 116 -
    the noisy ones - and nstdb is the noise-stress database by construction.
    """
    batch = tf.shape(signal)[0]
    n, c, fs = config.SEGMENT_SAMPLES, config.IN_CHANNELS, float(config.SAMPLING_RATE)
    t = tf.range(n, dtype=tf.float32) / fs                                      # (n,)

    freq = tf.random.uniform([batch, 1, c, 2], 0.05, 0.6)
    phase = tf.random.uniform([batch, 1, c, 2], 0.0, _TWO_PI)
    amp = tf.random.uniform([batch, 1, c, 2], 0.0, config.AUGMENT_WANDER_AMP)
    wander = tf.reduce_sum(amp * tf.sin(_TWO_PI * freq * t[None, :, None, None] + phase),
                           axis=-1)                                             # (b, n, c)
    on = tf.cast(tf.random.uniform([batch, 1, 1]) < config.AUGMENT_WANDER_PROB, tf.float32)
    field = on * wander

    amp = tf.random.uniform([batch, 1, c], 0.0, config.AUGMENT_NOISE_AMP)
    on = tf.cast(tf.random.uniform([batch, 1, 1]) < config.AUGMENT_NOISE_PROB, tf.float32)
    field += on * amp * tf.random.normal(tf.shape(signal))

    centre = tf.random.uniform([batch, 1, 1], 0.0, float(n))
    half = tf.random.uniform([batch, 1, 1], 0.2, 1.0) * fs / 2.0
    x = (tf.range(n, dtype=tf.float32)[None, :, None] - centre) / half           # (b, n, 1)
    bump = tf.where(tf.abs(x) <= 1.0, 0.5 * (1.0 + tf.cos(3.14159265 * x)), 0.0)
    lead = tf.one_hot(tf.random.uniform([batch], 0, c, dtype=tf.int32), c)[:, None, :]
    sign = tf.sign(tf.random.uniform([batch, 1, 1], -1.0, 1.0))
    amp = tf.random.uniform([batch, 1, 1], 1.0, 3.0)
    on = tf.cast(tf.random.uniform([batch, 1, 1]) < config.AUGMENT_MOTION_PROB, tf.float32)
    return field + on * sign * amp * bump * lead


def _noise(signal):
    """Synthetic recording noise added to the signal (labels are untouched). See _noise_field."""
    if not config.AUGMENT_NOISE:
        return signal
    return signal + _noise_field(signal)


def _lead_noise_field(signal):
    """Wreck ONE lead of the sample, hard enough that the model has to read the others.

    Everything in `_noise_field` switches on per SAMPLE, so when it fires it fires on every
    lead at once. The situation a 3-lead holter actually produces - one electrode in trouble
    while the other two are clean - was therefore never in the training distribution. A FLAT
    lead the model does know (AUGMENT_LEAD_DROP_PROB), but a flat lead is trivially
    detectable: the hard skill, and the one both the missed beats and the false ones turn
    on, is telling "this lead carries no evidence" apart from "this lead says there is no
    beat". It is also the corruption output 2 learns its notion of "unreliable" from.

    Three things are drawn per sample:

      * WHICH lead - any of them, lead 0 included, each about a third of the time. Lead 0 is
        the lead the labels refer to, so its amplitude is capped at
        AUGMENT_LEAD_NOISE_PRIMARY_AMP: degraded, still readable. A secondary lead is drawn
        uniformly below the higher AUGMENT_LEAD_NOISE_AMP and so lands anywhere from mildly
        degraded to swamped, because the beats remain legible on the two leads that are left.
      * WHEN - a raised-cosine envelope over a span of AUGMENT_LEAD_NOISE_SPAN..1 of the
        window, i.e. everything from a few seconds to an electrode useless for the whole strip.
      * WHAT - a mixture of white noise and 1-25 Hz oscillation. White hiss is the easy case;
        the artefact that costs positive predictivity is the one with QRS-band energy.
    """
    batch = tf.shape(signal)[0]
    n, c, fs = config.SEGMENT_SAMPLES, config.IN_CHANNELS, float(config.SAMPLING_RATE)
    t = tf.range(n, dtype=tf.float32) / fs                                      # (n,)

    victim_idx = tf.random.uniform([batch], 0, c, dtype=tf.int32)
    victim = tf.one_hot(victim_idx, c)[:, None, :]                              # (b, 1, c)
    primary = tf.cast(tf.equal(victim_idx, 0), tf.float32)
    ceiling = (primary * config.AUGMENT_LEAD_NOISE_PRIMARY_AMP +
               (1.0 - primary) * config.AUGMENT_LEAD_NOISE_AMP)[:, None, None]  # (b, 1, 1)
    amp = tf.random.uniform([batch, 1, 1], 0.0, 1.0) * ceiling

    span = tf.random.uniform([batch, 1, 1], config.AUGMENT_LEAD_NOISE_SPAN, 1.0) * float(n)
    centre = tf.random.uniform([batch, 1, 1], 0.0, float(n))
    x = (tf.range(n, dtype=tf.float32)[None, :, None] - centre) / (span / 2.0)
    env = tf.where(tf.abs(x) <= 1.0, 0.5 * (1.0 + tf.cos(3.14159265 * x)), 0.0)  # (b, n, 1)

    freq = tf.random.uniform([batch, 1, 3], 1.0, 25.0)
    phase = tf.random.uniform([batch, 1, 3], 0.0, _TWO_PI)
    # Three random-phase sinusoids have std sqrt(3/2); normalise so `amp` means what it says
    # on both branches of the mixture.
    band = tf.reduce_sum(tf.sin(_TWO_PI * freq * t[None, :, None] + phase),
                         axis=-1, keepdims=True) / 1.2247449                    # (b, n, 1)
    mix = tf.random.uniform([batch, 1, 1])
    noise = mix * tf.random.normal([batch, n, 1]) + (1.0 - mix) * band

    on = tf.cast(tf.random.uniform([batch, 1, 1]) < config.AUGMENT_LEAD_NOISE_PROB,
                 tf.float32)
    return on * amp * env * noise * victim


def _lead_noise(signal):
    """One lead wrecked per sample, added to the signal. See _lead_noise_field."""
    if not config.AUGMENT_NOISE or config.AUGMENT_LEAD_NOISE_PROB <= 0.0:
        return signal
    return signal + _lead_noise_field(signal)


def _pool_steps(x, size=None, stride=None, padding='VALID'):
    """Average-pool an (B, n, C) field along time."""
    return tf.nn.avg_pool1d(x, ksize=size, strides=stride, padding=padding)


def flat_mask(signal):
    """(B, steps, C) float: 1 where a lead is locally flat (lead-off), else 0.

    Local standard deviation over QUALITY_FLAT_WINDOW_STEPS, evaluated at step resolution.
    A lead that never moved is exactly constant after the per-lead z-score; a lead that died
    for part of the strip is constant there and lively elsewhere, and this catches the dead
    part only.
    """
    per_step = config.STEP_SAMPLES
    window = config.QUALITY_FLAT_WINDOW_STEPS * per_step
    mean = _pool_steps(signal, window, per_step, 'SAME')
    mean_sq = _pool_steps(tf.square(signal), window, per_step, 'SAME')
    std = tf.sqrt(tf.maximum(mean_sq - tf.square(mean), 0.0))
    std = std[:, :config.OUTPUT_STEPS]
    return tf.cast(std < config.QUALITY_FLAT_STD, tf.float32)


def quality_from_corruption(added, original, dup=None, drop=None):
    """The lead-quality target: (B, steps, C) in [0, 1], from what the pipeline did to each lead.

        q = sigmoid((QUALITY_NOISE_HALF - a) / QUALITY_NOISE_SCALE) / sigmoid(HALF / SCALE)

    `a` is the local RMS of the injected corruption `added` in per-lead z-score units,
    pooled to the label grid and smoothed over QUALITY_SMOOTH_STEPS. The division makes an
    untouched lead score exactly 1 - the same value clean_quality_target gives the eval
    split - so the two targets agree on what "readable" means. A lead that was flat in the
    ORIGINAL signal is 0 wherever it is flat; a dropped lead is 0 everywhere; a sample
    collapsed to lead 0 repeated inherits lead 0's quality on every channel (the copies are
    exactly as readable as their source). Nothing here reads a human label.
    """
    per_step = config.STEP_SAMPLES
    power = _pool_steps(tf.square(added), per_step, per_step, 'VALID')        # (b, steps, c)
    power = _pool_steps(power, config.QUALITY_SMOOTH_STEPS, 1, 'SAME')
    a = tf.sqrt(power + 1e-12)
    q = tf.sigmoid((config.QUALITY_NOISE_HALF - a) / config.QUALITY_NOISE_SCALE)
    q = q / tf.sigmoid(config.QUALITY_NOISE_HALF / config.QUALITY_NOISE_SCALE)
    q = q * (1.0 - flat_mask(original))
    if dup is not None:
        channels = q.shape[-1] or config.IN_CHANNELS
        q = tf.where(dup, tf.tile(q[..., :1], [1, 1, channels]), q)
    if drop is not None:
        q = q * (1.0 - drop)
    return q


def clean_quality_target(signal):
    """Quality target for an UNCORRUPTED batch: 1 wherever the lead is not flat."""
    return 1.0 - flat_mask(signal)


def corrupt(signal):
    """Noise, one wrecked lead, per-lead gain, then lead duplication/drop; returns the corrupted
    batch and the lead-quality target that describes what was done to it.

    Order matters at the end: the gain is applied BEFORE the lead jitter, not after. Applied
    after, it multiplies each lead of an already-duplicated sample by a different factor, so
    the "one lead three times" case the jitter exists to produce never actually reaches the
    model - the three leads come out proportional rather than equal. The quality target is
    unaffected by the gain: a lead scaled by 1.2 is exactly as readable as before.
    """
    batch = tf.shape(signal)[0]
    channels = signal.shape[-1] or config.IN_CHANNELS
    added = tf.zeros_like(signal)
    if config.AUGMENT_NOISE:
        added += _noise_field(signal)
        if config.AUGMENT_LEAD_NOISE_PROB > 0.0:
            added += _lead_noise_field(signal)
    # Per LEAD, not per sample: the leads of one record already differ in gain by a factor of
    # several, and a single shared factor cannot teach that.
    gain = tf.random.uniform([batch, 1, channels], 0.8, 1.25)
    corrupted = (signal + added) * gain
    if channels >= 2:
        dup, drop = _lead_jitter_masks(batch, channels)
        corrupted = _apply_lead_jitter(corrupted, dup, drop)
    else:
        dup = drop = None
    quality = quality_from_corruption(added, signal, dup, drop)
    quality.set_shape([None, config.OUTPUT_STEPS, channels])
    return corrupted, quality


def augment(signal, labels):
    """The training map: time-scale, corruption, and the two targets the model is trained on."""
    signal, labels = _time_scale(signal, labels)
    corrupted, quality = corrupt(signal)
    return corrupted, {'beat_cls': labels, 'lead_quality': quality}


def with_targets(signal, labels):
    """The evaluation map: the signal as recorded, and a quality target that only knows about
    flat leads (nothing was injected, so nothing else is known)."""
    return signal, {'beat_cls': labels, 'lead_quality': clean_quality_target(signal)}


def make_dataset(files, batch_size, training=False, cache=None, signal_only=False,
                 lead_jitter=False, targets=True):
    """The dataset over `files`. `targets=False` yields the raw (signal, one-hot labels) pair
    without the augmentation or the quality target - the parse round-trip, for tests."""
    cache = config.CACHE_DATASET if cache is None else cache
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(config.SHUFFLE_BUFFER, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if signal_only:
        # The self-supervised stages never see a label: parse_signal does not even declare
        # the field, so they run on a corpus that has none as readily as on these tfrecords.
        # Time-scale augmentation is off for them - it would teach the encoder that a strip
        # and its rescaled self are different recordings, the opposite of what a contrastive
        # objective is for. Lead jitter is left to the caller (`lead_jitter=True`): the CPC
        # stage wants it, while the masked-reconstruction stage masks leads itself and would
        # otherwise corrupt the same input twice.
        ds = ds.map(parse_signal, num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=not training)
        if lead_jitter and training and config.AUGMENT:
            ds = ds.map(_lead_jitter, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.map(parse_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not training)
    if targets:
        if training and config.AUGMENT:
            ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(with_targets, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_split(split, batch_size=None, db_names=None, signal_only=False, verbose=True,
               lead_jitter=False):
    files = split_files(split, db_names)
    assert_no_benchmark_data(files)
    if verbose:
        print(f"{split:5s}: {len(files)} tfrecords{' (signal only)' if signal_only else ''}")
    return make_dataset(files, batch_size or config.BATCH_SIZE,
                        training=(split == 'train'), signal_only=signal_only,
                        lead_jitter=lead_jitter)
