"""tf.data input pipeline over the tfrecords.

Order matters for throughput: read (parallel) -> cache -> shuffle -> batch -> parse -> augment
-> prefetch.

  * cache() holds the serialized records in RAM, so only the first epoch touches disk.
  * shuffle BEFORE batch, on the serialized strings: shuffling raw records is far cheaper
    than shuffling parsed tensors, which is what lets the buffer be big enough to matter.
    Records come off disk grouped by study, so a small buffer leaves a batch made of one
    patient's beats - SHUFFLE_BUFFER is sized to break that up.
  * batch BEFORE parse, so tf.io.parse_example is vectorised over the batch. Parsing one
    record at a time was what kept the GPU at ~50%.
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
    """Parse a BATCH of examples (see module docstring for why it is batched)."""
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
    fixed - and the padded label steps are then filled with the background class. Leaving
    them as all-zero one-hot rows (as this did) is not the same thing: they carry no class,
    but `reduce_mean` in the loss still divides by them, so up to 10% of a batch's steps
    silently diluted every gradient.

    The crop-back targets are the config's Python ints rather than tf.shape() results, so
    the STATIC shape survives the round trip. It did not before, and a time axis of None
    propagates: `_lead_jitter` below could not read the lead count, RhythmDescriptor could
    not clamp its lag band against the sequence length, and DiagSSM1D had to compute its FFT
    length at run time.
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

    empty = tf.reduce_sum(lab, axis=-1, keepdims=True) < 0.5
    background = tf.one_hot(0, k, dtype=lab.dtype)
    lab = tf.where(empty, background, lab)

    sig.set_shape([None, n, c])
    lab.set_shape([None, m, k])
    return sig, lab


def _lead_jitter(signal):
    """Per-sample lead manipulations: duplicate the primary lead, or drop a secondary one.

    The EC57 databases have two leads and are annotated on the first, so scoring them means
    feeding the model ONE lead repeated across the channel axis. A model trained only on
    three genuinely different leads has never seen that input, and the mismatch lands
    squarely on the channel axis the AdaIN conditioning reads. Reproducing the case here at
    AUGMENT_LEAD_DUPLICATE_PROB is what makes the benchmark an evaluation rather than a
    domain shift.

    The drop branch is the other half of the same story: an electrode that comes off leaves
    one flat lead, which is common in real recordings and must not take the prediction with
    it. Lead 0 is never dropped - it is the lead the labels refer to.
    """
    batch = tf.shape(signal)[0]
    channels = signal.shape[-1] or config.IN_CHANNELS
    if channels < 2:
        return signal

    dup = tf.random.uniform([batch, 1, 1]) < config.AUGMENT_LEAD_DUPLICATE_PROB
    signal = tf.where(dup, tf.tile(signal[..., :1], [1, 1, channels]), signal)

    # One randomly chosen non-primary lead, zeroed on the selected samples.
    victim = tf.random.uniform([batch], 1, channels, dtype=tf.int32)
    drop = tf.random.uniform([batch]) < config.AUGMENT_LEAD_DROP_PROB
    mask = 1.0 - tf.cast(tf.one_hot(victim, channels) *
                         tf.cast(drop, tf.float32)[:, None], signal.dtype)
    return signal * mask[:, None, :]


def _noise(signal):
    """Synthetic recording noise, per sample and per lead, in z-score units.

    Motivated by where the EC57 detection errors actually are: on mitdb, 74 of 82 missed
    beats and 99 of 121 false beats of the 2m model sit in records 203, 105, 108 and 116 -
    the noisy ones - and nstdb is the noise-stress database by construction. The training
    windows are clean by comparison (reviewed portal strips), so the model had never learned
    to hold a beat through artefact. Three components, each switched on per sample with its
    own probability so about half the batch carries some noise:

      * baseline wander - two sinusoids per lead, 0.05-0.6 Hz, up to AUGMENT_WANDER_AMP std
        (respiration, slow electrode drift; the 0.5 Hz high-pass leaves the upper part)
      * broadband noise - white Gaussian up to AUGMENT_NOISE_AMP std (EMG, amplifier)
      * a motion transient - one Hann bump of 0.2-1.0 s and 1-3 std on ONE lead, either sign
        (electrode motion; the shape that most resembles a QRS and so costs the most +P)

    Labels are untouched: the beats are where they were. The signal is already z-scored per
    lead, so the amplitudes are relative to a lead whose QRS peaks at ~3-8 std.
    """
    if not config.AUGMENT_NOISE:
        return signal
    batch = tf.shape(signal)[0]
    n, c, fs = config.SEGMENT_SAMPLES, config.IN_CHANNELS, float(config.SAMPLING_RATE)
    t = tf.range(n, dtype=tf.float32) / fs                                      # (n,)

    freq = tf.random.uniform([batch, 1, c, 2], 0.05, 0.6)
    phase = tf.random.uniform([batch, 1, c, 2], 0.0, 2.0 * 3.14159265)
    amp = tf.random.uniform([batch, 1, c, 2], 0.0, config.AUGMENT_WANDER_AMP)
    wander = tf.reduce_sum(amp * tf.sin(2.0 * 3.14159265 * freq * t[None, :, None, None]
                                        + phase), axis=-1)                      # (b, n, c)
    on = tf.cast(tf.random.uniform([batch, 1, 1]) < config.AUGMENT_WANDER_PROB, tf.float32)
    signal = signal + on * wander

    amp = tf.random.uniform([batch, 1, c], 0.0, config.AUGMENT_NOISE_AMP)
    on = tf.cast(tf.random.uniform([batch, 1, 1]) < config.AUGMENT_NOISE_PROB, tf.float32)
    signal = signal + on * amp * tf.random.normal(tf.shape(signal))

    centre = tf.random.uniform([batch, 1, 1], 0.0, float(n))
    half = tf.random.uniform([batch, 1, 1], 0.2, 1.0) * fs / 2.0
    x = (tf.range(n, dtype=tf.float32)[None, :, None] - centre) / half           # (b, n, 1)
    bump = tf.where(tf.abs(x) <= 1.0, 0.5 * (1.0 + tf.cos(3.14159265 * x)), 0.0)
    lead = tf.one_hot(tf.random.uniform([batch], 0, c, dtype=tf.int32), c)[:, None, :]
    sign = tf.sign(tf.random.uniform([batch, 1, 1], -1.0, 1.0))
    amp = tf.random.uniform([batch, 1, 1], 1.0, 3.0)
    on = tf.cast(tf.random.uniform([batch, 1, 1]) < config.AUGMENT_MOTION_PROB, tf.float32)
    return signal + on * sign * amp * bump * lead


def _lead_noise(signal):
    """Wreck ONE lead of the sample, hard enough that the model has to read the others.

    Everything in `_noise` switches on per SAMPLE, so when it fires it fires on every lead at
    once. The situation a 3-lead holter actually produces - one electrode in trouble while the
    other two are clean - was therefore never in the training distribution. A FLAT lead the
    model does know (AUGMENT_LEAD_DROP_PROB), but a flat lead is trivially detectable: the
    hard skill, and the one both the missed beats and the false ones turn on, is telling
    "this lead carries no evidence" apart from "this lead says there is no beat".

    Three things are drawn per sample:

      * WHICH lead - any of them, lead 0 included, each about a third of the time. Lead 0 is
        the lead the labels refer to, so its amplitude is capped at
        AUGMENT_LEAD_NOISE_PRIMARY_AMP: degraded, still readable. A secondary lead is drawn
        uniformly below the higher AUGMENT_LEAD_NOISE_AMP and so lands anywhere from mildly
        degraded to swamped (measured peak deviation: median 2.59, p90 4.93, max 7.93, against
        a QRS at 3-8), because the beats remain legible on the two leads that are left.
        Destroying lead 0 outright while keeping its labels would instead teach the model to
        invent beats out of artefact - the exact failure the noisy databases' positive
        predictivity is already losing to.
      * WHEN - a raised-cosine envelope over a span of AUGMENT_LEAD_NOISE_SPAN..1 of the
        window, i.e. everything from a short burst to an electrode useless for the whole strip.
      * WHAT - a mixture of white noise and 1-25 Hz oscillation. White hiss is the easy case;
        the artefact that costs positive predictivity is the one with QRS-band energy, because
        that is the one a detector mistakes for a beat.

    Labels are untouched, and most steps of any window are background, so this supplies the
    one pairing the existing noise cannot: long stretches of artefact that carry NO beat, on a
    lead the other two contradict. "Artefact is not a beat" is learnable from that and from
    very little else.

    Called before _lead_jitter, like everything else per-lead: the duplicate branch there has
    to come out with three EXACTLY equal leads (see augment), which it does because it copies
    lead 0 over whatever this did to the others.
    """
    if not config.AUGMENT_NOISE or config.AUGMENT_LEAD_NOISE_PROB <= 0.0:
        return signal
    batch = tf.shape(signal)[0]
    n, c, fs = config.SEGMENT_SAMPLES, config.IN_CHANNELS, float(config.SAMPLING_RATE)
    two_pi = 2.0 * 3.14159265
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
    phase = tf.random.uniform([batch, 1, 3], 0.0, two_pi)
    # Three random-phase sinusoids have std sqrt(3/2); normalise so `amp` means what it says
    # on both branches of the mixture.
    band = tf.reduce_sum(tf.sin(two_pi * freq * t[None, :, None] + phase),
                         axis=-1, keepdims=True) / 1.2247449                    # (b, n, 1)
    mix = tf.random.uniform([batch, 1, 1])
    noise = mix * tf.random.normal([batch, n, 1]) + (1.0 - mix) * band

    on = tf.cast(tf.random.uniform([batch, 1, 1]) < config.AUGMENT_LEAD_NOISE_PROB,
                 tf.float32)
    return signal + on * amp * env * noise * victim


def augment(signal, labels):
    """Time-scale, noise, per-lead amplitude jitter and lead manipulations, on the GPU batch.

    Order matters at the end: the gain is applied BEFORE the lead jitter, not after. Applied
    after, it multiplies each lead of an already-duplicated sample by a different factor, so
    the "one lead three times" case the jitter exists to produce never actually reaches the
    model - the three leads come out proportional rather than equal. EC57 feeds three exactly
    equal leads, so that is what training has to show it.
    """
    signal, labels = _time_scale(signal, labels)
    signal = _noise(signal)
    signal = _lead_noise(signal)
    # Per LEAD, not per sample: the leads of one record already differ in gain by a factor of
    # several, and a single shared factor cannot teach that.
    gain = tf.random.uniform([tf.shape(signal)[0], 1, signal.shape[-1] or
                              config.IN_CHANNELS], 0.8, 1.25)
    return _lead_jitter(signal * gain), labels


def make_dataset(files, batch_size, training=False, cache=None, signal_only=False,
                 lead_jitter=False):
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
    if training and config.AUGMENT:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
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
