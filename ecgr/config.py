"""Every path and hyper-parameter of the project, in one place.

Nothing here is read at import time by anything that matters except through this module, so
a run is fully described by this file plus the environment variables it reads. The env
overrides exist so a queued job cannot be changed under its feet by an edit to this file:
pin them in the launcher (run_pipeline.sh does), never edit mid-run.

    ECGR_DATA_DIR       where the portal datasets live (records + dataset_info_full.csv)
    ECGR_PHYSIONET_DIR  where mitdb/nstdb/... live
    ECGR_WORK_DIR       where npy, tfrecord, checkpoints and reports are written
    ECGR_RUN_TAG        name of this run's output folder (default: today, yymmdd_60s)
    ECGR_IN_CHANNELS    number of ECG LEADS fed to the model (default 3)
    ECGR_SEGMENT_SECONDS window length in seconds (default 60)
    ECGR_CPC_RUN        reuse another run's CPC-pretrained context encoders
    ECGR_SSL_RUN        reuse another run's SSL-pretrained backbones
    ECGR_WORKERS        processes used by the npy/tfrecord builders (default: cpus/2)
    ECGR_BATCH_SIZE     training batch size (default 32 at 60 s)
    ECGR_CACHE_DATASET  1 = hold the whole split in RAM (default 0 at 60 s: ~90 GB)

The 60 s contract (2026-09-23)
------------------------------
The model reads one whole portal strip - 60 s, 3 leads, 250 Hz - and emits two things:

    output 1  beat_cls      (3000, 4) softmax per 20 ms step: None / N / V / S
    output 2  lead_quality  (3000, 3) sigmoid per step and per LEAD: how readable that lead is;
                            argmax of its time-average is "the most reliable channel"

Only the reviewed span of a strip carries labels a human signed off on. Steps outside it
are written as IGNORE_LABEL and contribute nothing to the beat loss or to the step metric;
the rest of the 60 s is still signal the model sees (and the label-free stages learn from).
The lead-quality target is derived from the signal and from the corruption the augmentation
itself injected - no human ever labelled a lead as good or bad - so output 2 is label-free.
"""
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("ECGR_DATA_DIR", "/mnt/md0/Dong_data/portal_data/")
PHYSIONET_DIR = os.environ.get("ECGR_PHYSIONET_DIR", "/mnt/md0/Dong_data/physionet/")
WORK_DIR = os.environ.get("ECGR_WORK_DIR", os.path.join(DATA_DIR, "train"))

# RUN_TAG names the run folder. It defaults to today's date, which silently changes at
# midnight: a training started yesterday writes to <yesterday> while the eval that follows
# it this morning would look in <today> and find nothing. Pin it in the launcher.
RUN_TAG = os.environ.get("ECGR_RUN_TAG", datetime.today().strftime("%y%m%d") + "_60s")

# ---------------------------------------------------------------------------
# Signal / segmentation
# ---------------------------------------------------------------------------
SAMPLING_RATE = 250                                   # Hz, everything is resampled to this
SEGMENT_SECONDS = int(os.environ.get("ECGR_SEGMENT_SECONDS", 60))
SEGMENT_SAMPLES = SEGMENT_SECONDS * SAMPLING_RATE     # 15000
# Hop between consecutive TRAINING windows when a reviewed span is longer than one window.
# A portal strip is exactly one window long, so this only matters for the rare long span.
SEGMENT_STRIDE_SECONDS = 30

FILTER_LOWCUT, FILTER_HIGHCUT, FILTER_ORDER = 0.5, 30.0, 3

# ---------------------------------------------------------------------------
# Leads (the channel axis)
# ---------------------------------------------------------------------------
# The channel axis carries ECG LEADS, not filter bands. Every portal record is natively
# 3-lead at 250 Hz (CH0/CH1/CH2), so the model sees the whole montage instead of the single
# lead a reviewer happened to work on: a P wave that is invisible on one lead is usually
# visible on another, and that is precisely the evidence that separates a non-premature
# atrial ectopic (mitdb 232, R-R ratio 0.99) from a sinus beat.
IN_CHANNELS = int(os.environ.get("ECGR_IN_CHANNELS", 3))

# The label stream refers to ONE lead - whichever the reviewer annotated - so that lead is
# rolled to index 0 in every window. Everything downstream can then assume channel 0 is the
# reference lead: labels_from_annotations, the R-peak position in decode_beats, the flatness
# test and the rhythm descriptor all read it.
PRIMARY_LEAD_FIRST = True

# How the channel axis is filled when a record has fewer leads than IN_CHANNELS. This is a
# separate question from WHICH real leads to read (EC57_LEAD_MODE below): one says how many
# genuine signals go in, the other what occupies the leftover channels.
#
#   'zero'      - leftover channels are silence. THE DEFAULT. A flat channel is what the
#                 model already sees whenever an electrode comes off, and training produces
#                 it on purpose (AUGMENT_LEAD_DROP_PROB), so "this lead does not exist" is
#                 said in the vocabulary the model was taught. It also cannot be mistaken
#                 for evidence: a duplicated lead is a second vote for whatever the first
#                 lead says, which is exactly what a multi-lead model should not be handed.
#   'duplicate' - leftover channels repeat the annotated lead. Kept because it is the other
#                 case training covers (AUGMENT_LEAD_DUPLICATE_PROB) and because the EC57
#                 tables in README section 8 up to 2026-09-20 were produced with it.
LEAD_FILL_MODE = os.environ.get("ECGR_LEAD_FILL", "zero")        # 'zero' | 'duplicate'

NORMALIZE_Z_SIGNAL = True     # per-window z-score, per lead
MIN_AMPLITUDE = 0.1           # mV; a reviewed span flatter than this on lead 0 is dropped

# ---------------------------------------------------------------------------
# Label grid
# ---------------------------------------------------------------------------
# OUTPUT_STEPS must divide SEGMENT_SAMPLES exactly - the model pools down to this grid.
# 15000/3000 = 5 samples per step = 20 ms per label step, the same grid as the 10 s model.
STEP_SAMPLES = 5
OUTPUT_STEPS = SEGMENT_SAMPLES // STEP_SAMPLES
NUM_CLASSES = 4
CLASS_NAMES = ['None', 'N', 'V', 'S']         # index == label value

# Steps with NO trustworthy label: outside the reviewed span of a strip, or in the padding
# of a record shorter than one window. Stored as this value in the uint8 label stream; the
# pipeline turns it into an all-zero one-hot row, and every loss and metric skips rows whose
# mass is zero. 255 so it can never collide with a class index.
IGNORE_LABEL = 255

# AAMI grouping of the WFDB beat symbols actually present in the portal data.
BEAT_MAP = {'N': ['N', 'R', 'L'], 'S': ['S', 'A', 'J'], 'V': ['V', 'E']}
SYMBOL_TO_LABEL = {sym: CLASS_NAMES.index(cls)
                   for cls, syms in BEAT_MAP.items() for sym in syms}

# Label block around each beat, in output steps. Asymmetric: the P wave sits ~120-200 ms
# BEFORE the R peak, i.e. 6-10 steps at 20 ms/step, so the N/S decision must span it.
LABEL_STEPS_BEFORE = 8
LABEL_STEPS_AFTER = 2

MIN_RR_INTERVAL = int(0.18 * SAMPLING_RATE)   # shortest allowed gap between two detections
# Shortest run of non-background steps decoded as a beat (labels.decode_beats). 1 = every run.
DECODE_MIN_RUN_STEPS = int(os.environ.get("ECGR_MIN_RUN_STEPS", 1))

# Peak beat probability (1 - p_None) a decoded run must reach somewhere along it to be
# emitted at all - the "refuse to call a beat out of unreadable signal" knob. 0 = off, which
# is what every number in README section 8 was measured with.
#
# This is the mechanism that works for noisy signal, and it is NOT the obvious one. Measured
# on nstdb with resumamba_2m, pooled over the 12 scored records (21,462 reference beats), at
# the operating point that holds sensitivity at the 80% floor:
#
#     suppress by                         best +P at Se >= 80
#     ----------------------------------  -------------------
#     peak beat probability (this knob)          99.47
#     mean beat probability over the run         98.43
#     signal quality (band-passed kurtosis)      95.77
#     decoded run length (DECODE_MIN_RUN_STEPS)  92.69
#
# CALIBRATE ON PORTAL DATA ONLY, exactly as for the S boost: the threshold that produces a
# given number on nstdb was chosen against nstdb, and a benchmark tuned against itself is not
# a benchmark. The figures above are a feasibility measurement, not a setting to ship.
DECODE_MIN_PEAK_PROB = float(os.environ.get("ECGR_DECODE_MIN_PEAK_PROB", 0.0))

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
# The five primary portal datasets and nothing else. "dataset-3-filter-vt-svt-avb2-avb3",
# "dataset 2_3_4 - AFib - v2" and "dataset_ivcd" are re-curations of events already in
# dataset-2/3/4 (README section 4a) and are left out of training since 2026-09-23: the
# training population is exactly what these five CSVs list, held-out studies removed.
TRAIN_DATASETS = ["dataset-1", "dataset-2", "dataset-3", "dataset-4", "dataset-5"]
DATASET_CSV = "dataset_info_full.csv"
PORTAL_FS = 250               # native sampling rate of the portal records

# Shortest reviewed span (samples) a record must have to enter the training data. The 60 s
# window no longer constrains it - a window may extend beyond the span, the steps outside
# are IGNORE_LABEL - so this is a floor on how much LABELLED signal a record contributes.
# 10 s minus the slack that admits the 2499-sample spans (README 4a): the same population
# the 10 s pipeline trained on, so the two are comparable event for event.
MIN_REVIEWED_SAMPLES = 10 * SAMPLING_RATE - int(0.1 * SAMPLING_RATE)     # 2475

# Held out from training entirely, and scored at the end by bxb like a Physionet database.
TEST_DATASETS = ["dataset-eval"]
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Ships with the project: the v4 eval study list is part of the train/test contract, not a
# machine-local path, so a checkout is enough to rebuild the same holdout.
EVAL_STUDIES_JSON = os.environ.get(
    "ECGR_EVAL_STUDIES_JSON", os.path.join(_PKG_ROOT, "assets", "list_studies_eval_v4.json"))
EXCLUDE_EVAL_STUDIES = True
TRAIN_FRACTION = 0.8          # study-level, decided by a hash of the study id

PORTAL_EVAL_SETS = {
    "dataset-v4-beat": os.path.join(DATA_DIR, "dataset-eval", "v4", "beat-eval-dataset"),
}

# The portal train and eval SPLITS are scored by bxb too (evaluation/ec57.score_portal_split),
# on a deterministic hash-ordered sample of their reviewed events - the same sample for every
# model. 5000 is the scale of the beat-eval set itself (5,227 records) and costs ~3 minutes
# per split per model; 0 = all of them.
# 'portal-train' is data the model has seen: read it as an overfitting diagnostic - the gap
# to 'portal-eval' - never as a performance number.
PORTAL_SPLITS = ('train', 'eval')
PORTAL_SPLIT_RECORDS = int(os.environ.get("ECGR_PORTAL_SPLIT_RECORDS", 5000))

# ---------------------------------------------------------------------------
# EC57 benchmark databases
# ---------------------------------------------------------------------------
EC57_DBS = ['mitdb', 'nstdb', 'escdb', 'ahadb', 'afdb']
# afdb keeps its beats in .qrs (.atr holds only AFIB rhythm markers); everything else in .atr
EC57_BEAT_REF_EXT = {'afdb': 'qrs'}
# AAMI EC57 leaves the MIT-BIH paced recordings out of the beat scoring
EC57_EXCLUDE_RECORDS = {'mitdb': ['102', '104', '107', '217']}
BEAT_EXTENSION = 'ain'        # extension of the AI annotations handed to bxb
# Overlap between consecutive inference windows when sweeping a whole record. Each window
# is z-scored on its own and the model is bidirectional, so a beat near a window edge has
# one-sided context; labels.core_bounds credits every beat to the window where it sits
# furthest from an edge, and 10 s of overlap keeps that at >= 5 s on either side.
EC57_SEGMENT_OVERLAP = 10 * SAMPLING_RATE

# Which lead of an EC57 record is the annotated one, 0-based. Every one of these databases
# is annotated on its first signal. Per-database overrides go here.
EC57_LEAD = {}
EC57_LEAD_DEFAULT = 0
# EC57_LEAD_MODE says how many of the record's REAL leads to use; LEAD_FILL_MODE above says
# what occupies the channels left over.
# 'native'    : as many real leads as the record has, filled up if it has fewer. THE DEFAULT.
#               Every EC57 database has two real leads, and reading both is what the 3-lead
#               model was built for: measured on resumamba_2m, switching mitdb from one lead
#               repeated to MLII+V5 moves S from 45.79/61.17 to 56.87/65.64 and lifts 30 of
#               32 Physionet cells, because record 232's non-premature APCs are only visible
#               as a P wave on V5.
# 'single'    : only the annotated lead is real; the rest are filled per LEAD_FILL_MODE.
#               The strict single-lead reading, and the control for the finding above.
#               'duplicate' is accepted as a deprecated alias.
# 'auto'      : 'native' only where the record ALREADY has IN_CHANNELS leads, else 'single'.
#               Every EC57 database has two leads and the model takes three, so on all five
#               'auto' means 'single' - it is not a synonym for 'native' there.
EC57_LEAD_MODE = os.environ.get("ECGR_EC57_LEAD_MODE", "native")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# A 60 s window is six times the activation memory of the 10 s one, so the batch is a
# quarter of what it was (128 -> 32) and the learning rate follows the usual sqrt scaling
# from the 1e-3 that batch 64 was tuned at. Measured peak VRAM at batch 32, (15000, 3):
# see README section 5 / `python -m ecgr models`.
BATCH_SIZE = int(os.environ.get("ECGR_BATCH_SIZE", 32))
EPOCHS = 30
LEARNING_RATE = 7e-4
PATIENCE = 8

# Default loss for this family. Poly2 is what the paper specifies; it also makes val_loss
# useless as a stopping signal, which is why MONITOR below is the F1 (see training/losses.py).
LOSS = 'poly2'
POLY2_EPS = (0.3, -0.5)         # paper sec. 3.6, found by grid search with a flat optimum
# The step-level weighted F1 of the BEAT output. Keras prefixes a metric with the output it
# belongs to once a model has two outputs, hence 'beat_cls_'; training/train.monitor_key
# resolves the name for a single-output (legacy) model.
MONITOR = 'val_beat_cls_weighted_f1'

# First epoch (1-indexed) allowed to write a checkpoint. Earlier epochs are still measured
# and logged, they just cannot be kept as "best". The 10 s family used 10 (its 1M size
# swung between 0.8267 and 0.8519 over its first nine epochs before settling), but at 60 s
# an epoch is 1-3 GPU-hours and a process can be killed from outside: on 2026-09-24 the 3m
# training received a SIGTERM at epoch 7 and, with nothing saved before epoch 8, lost ~20 h.
# Saving from epoch 3 caps that loss at one epoch; `ecgr select` (bxb on portal-eval) is
# what picks among the saved epochs, so an early, noisy epoch costs nothing but disk.
CKPT_START_EPOCH = 3

# --- output 2: per-lead signal quality (models/resumamba.py, data/pipeline.corrupt) -------
# The head reads the backbone features and emits, per step and per lead, the probability
# that the lead is readable there. Its TARGET is built inside the input pipeline from two
# label-free sources: (a) the corruption the augmentation itself added to each lead - the
# pipeline knows exactly which lead it wrecked, where, and by how much - and (b) a flatness
# test on the original signal (a lead-off is unreadable whether or not we touched it).
#
#     q(lead, step) = sigmoid((QUALITY_NOISE_HALF - a) / QUALITY_NOISE_SCALE), scaled so q(0) = 1
#
# with `a` the local RMS of the injected corruption in per-lead z-score units, smoothed over
# QUALITY_SMOOTH_STEPS. QRS peaks sit at ~3-8 in those units: a = 0.25 (the broadband cap)
# is still q = 0.95, a = 0.5 is 0.87, a = 1.0 is 0.52, a = 2.5 (a swamped secondary lead)
# is q ~0.007. A duplicated lead inherits lead 0's quality, a dropped or flat lead is 0.
QUALITY_LOSS_WEIGHT = 0.25       # weight of the quality BCE against the beat loss
QUALITY_NOISE_HALF = 1.0
QUALITY_NOISE_SCALE = 0.3
QUALITY_SMOOTH_STEPS = 50        # 1 s: the scale at which "this stretch is unreadable" holds
# Flatness of the ORIGINAL lead, judged on a 2 s local window: a lead-off is exactly constant
# after the z-score, while the quietest T-P stretch of a live lead at 30 bpm still moves by
# a few hundredths, so the threshold sits well under that.
QUALITY_FLAT_WINDOW_STEPS = 100  # 2 s
QUALITY_FLAT_STD = 0.02          # local std (z units) below which a lead is flat

# --- self-supervised stage 1: the backbone (training/ssl.py) ---------------------------
# Masked-lead + masked-span reconstruction over the same unlabeled tfrecords. It pretrains
# the stem and both paths of the dual-path backbone, which is where nearly every parameter
# lives, and the masked-LEAD half of the objective is what teaches the model to fall back on
# a single lead - the situation the EC57 stage puts it in.
SSL_EPOCHS = 8
SSL_STEPS_PER_EPOCH = 600
SSL_LEARNING_RATE = 1e-3
SSL_MASK_RATIO = 0.35           # fraction of output steps masked per sample
SSL_MASK_SPAN_STEPS = 12        # mask in spans of ~240 ms, not isolated steps
SSL_LEAD_MASK_PROB = 0.5        # chance a sample additionally loses one whole lead
SSL_RUN = os.environ.get("ECGR_SSL_RUN")
FREEZE_BACKBONE_EPOCHS = 0      # >0 = warm up the head with the SSL backbone frozen

# --- stage 4: the temporal refinement head (models/refine.py, training/refine.py) --------
# A small bidirectional SSM re-reads the base model's predicted beat train and corrects the
# N/V/S split per step; p_None is preserved exactly and the head starts as the identity.
# Trained on the frozen best base checkpoint; the epoch is then chosen on portal-eval under a
# no-regression rule (every Q/V/S Se and +P >= base - REFINE_TOLERANCE_PP).
REFINE_EPOCHS = 6
REFINE_LEARNING_RATE = 5e-4
REFINE_WIDTH = 48
REFINE_BLOCKS = 2
REFINE_STATE_DIM = 8
REFINE_KERNEL_LEN = 512          # steps each way = 10.2 s: 10-20 R-R intervals
REFINE_TOLERANCE_PP = 0.1        # percentage points of bxb noise tolerated on 5,000 records
# Loss weights for the HEAD. Not CLASS_WEIGHTS: those fight the None/beat imbalance, which
# the head never sees (p_None is fixed), and their 2.5x on S made the head a threshold shift
# (measured: every epoch raised S_Se and lowered S_+P on portal-eval). Neutral among the
# beat classes lets it find the F1-optimal N/S boundary instead of the S-heavy one.
REFINE_CLASS_WEIGHTS = [0.3, 1.0, 1.0, 1.0]
# 'beats' redistributes among N/V/S; 's_only' moves mass between N and S only and leaves p_V
# as well as p_None untouched - see models/refine.BeatClassRefine for why that mode exists.
REFINE_MODE = 's_only'

# --- self-supervised stage 2: the context encoder (training/cpc.py) --------------------
# CPC/InfoNCE, architecture-only, so a later run can reuse an earlier run's encoders instead
# of paying for them again: set ECGR_CPC_RUN to that run's tag. At 60 s the strip is cut
# into 59 windows of 2 s at 50% overlap - the paper's own M ~ 59 for its 60 s calibration.
CPC_EPOCHS = 5
CPC_STEPS_PER_EPOCH = 400
CPC_RUN = os.environ.get("ECGR_CPC_RUN")

# Loss weight per class. S is the rarest per step and the weakest metric, so it gets the
# most; None dominates every window and is damped. Raising the S weight trades +P for Se,
# so it is deliberately modest (2.5x N) rather than inverse-frequency.
CLASS_WEIGHTS = [0.3, 1.0, 2.0, 2.5]

BATCH_SEGMENTS = 2000         # segments per npy batch file (~360 MB at 60 s x 3 leads)

# Records come off disk grouped by study, so a buffer that is too small leaves a batch made
# of one patient's beats. It shuffles the SERIALIZED records, which at 60 s x 3 leads are
# ~183 kB each, so this buffer is ~750 MB of RAM - the reason it is not simply enormous.
SHUFFLE_BUFFER = 4096

# cache() holds the whole split's serialized records in RAM, so only the first epoch touches
# disk. At 60 s the train split is ~90 GB serialized, so caching is OFF by default and every
# epoch streams from disk (a few GB/s off the RAID keeps the GPU fed). Turn it on with
# ECGR_CACHE_DATASET=1 on a machine where one copy per training job fits.
CACHE_DATASET = os.environ.get("ECGR_CACHE_DATASET", "0") not in ("0", "", "false", "False")
AUGMENT = True                # see data/pipeline.augment

# Probability that a training window is collapsed to ONE lead repeated across the channel
# axis. Without it the model only ever sees three genuinely different leads, while the EC57
# stage hands it the same lead three times - a distribution it would never have met.
AUGMENT_LEAD_DUPLICATE_PROB = 0.25
AUGMENT_LEAD_DROP_PROB = 0.15     # zero out one non-primary lead (electrode fell off)

# Synthetic recording noise (data/pipeline._noise). Off in run 260917_3lead; on from
# 260918_3lead_noise. Amplitudes are in per-lead z-score units (QRS peaks at ~3-8).
AUGMENT_NOISE = os.environ.get("ECGR_AUGMENT_NOISE", "1") not in ("0", "", "false", "False")
AUGMENT_WANDER_PROB, AUGMENT_WANDER_AMP = 0.5, 0.6
AUGMENT_NOISE_PROB, AUGMENT_NOISE_AMP = 0.5, 0.25
AUGMENT_MOTION_PROB = float(os.environ.get("ECGR_AUGMENT_MOTION_PROB", 0.2))

# ONE lead wrecked per sample (data/pipeline._lead_noise). The three components above switch
# on per SAMPLE, so when they fire they fire on every lead at once: the case a 3-lead holter
# produces constantly - one electrode in trouble while the other two are clean - was the one
# distribution training never showed. A flat lead the model does know
# (AUGMENT_LEAD_DROP_PROB), but a flat lead is trivially detectable; a lead full of artefact
# is not, and that is where both the missed beats and the false ones come from. It is also
# the corruption the lead-quality target (output 2) is built from.
AUGMENT_LEAD_NOISE_PROB = float(os.environ.get("ECGR_AUGMENT_LEAD_NOISE_PROB", 0.35))
# Ceiling for a SECONDARY lead, drawn uniformly below it, so the lead lands anywhere from
# mildly degraded to swamped: measured over 2,048 samples, the peak deviation on a corrupted
# secondary lead is median 2.59, p90 4.93, max 7.93 - i.e. the top of the range is at or above
# the QRS itself. The other two leads still carry the beats, so the labels stay honest and
# the lesson is "read the other leads" rather than "invent one".
AUGMENT_LEAD_NOISE_AMP = 2.5
# Lead 0 is the lead the labels refer to. It is corrupted too - nstdb is exactly that case -
# but capped below the QRS scale, so it degrades rather than disappears. Destroying it while
# keeping its labels would teach the model to invent beats out of artefact, which is the
# opposite of what the noisy-database positive predictivity needs.
AUGMENT_LEAD_NOISE_PRIMARY_AMP = 0.8
# Shortest burst as a fraction of the window: at 60 s, 0.05-1.0 = 3 s up to the whole strip.
AUGMENT_LEAD_NOISE_SPAN = 0.05

# Save every epoch from CKPT_START_EPOCH on (<ckpt>/epochs/epoch_NN.keras), not only the
# step-F1 improvements: the checkpoint that scores best at beat level is chosen afterwards by
# bxb on portal-eval, and the README's own warning is that step F1 does not pick it.
SAVE_EVERY_EPOCH = True

WORKERS = int(os.environ.get("ECGR_WORKERS", max(1, (os.cpu_count() or 8) // 2)))

# ---------------------------------------------------------------------------
# Non-regression baselines
# ---------------------------------------------------------------------------
# The 10 s / 3-lead family's EC57 + beat-eval summaries (README section 8b, run
# 260917_3lead), versioned under assets/ so `ecgr regress` and evaluate.py can diff a new
# checkpoint against them without the (git-ignored) eval_results tree. A 60 s size is held
# to the 10 s size it replaces; the two new large sizes are held to the best 10 s model.
BASELINES_DIR = os.path.join(_PKG_ROOT, "assets", "baselines", "10s_3lead")
BASELINE_FOR = {'resumamba_5m': 'resumamba_2m', 'resumamba_3m': 'resumamba_2m',
                'resumamba_1m': 'resumamba_1m', 'resumamba_100k': 'resumamba_100k'}
# bxb noise on 5,000 strips / 44 mitdb records: a cell may fall by this much and still count
# as "not decreased".
REGRESSION_TOLERANCE_PP = 0.1


def baseline_summary(model_name):
    """Path of the 10 s baseline ec57_summary.csv a model is held to, or None."""
    ref = BASELINE_FOR.get(model_name, model_name)
    path = os.path.join(BASELINES_DIR, f"{ref}.csv")
    return path if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Derived output layout - one folder per kind of output, one subfolder per model
# ---------------------------------------------------------------------------
NPY_DIR = os.environ.get(
    "ECGR_NPY_DIR",
    os.path.join(DATA_DIR, f"npy_{OUTPUT_STEPS}_{NUM_CLASSES}_{SAMPLING_RATE}_"
                           f"{SEGMENT_SECONDS}_{IN_CHANNELS}lead"))
# The window length is part of the tree name: a 60 s tree and a 10 s tree must never be
# mistaken for each other (check_manifest would refuse, but a name says it first).
TFRECORD_DIR = os.environ.get(
    "ECGR_TFRECORD_DIR", os.path.join(WORK_DIR, f"tfrecord_{SEGMENT_SECONDS}s_{IN_CHANNELS}lead"))
DATASET_MANIFEST = "dataset_manifest.json"   # written next to the tfrecords, checked on load

RUN_DIR = os.path.join(WORK_DIR, RUN_TAG)
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
EVAL_DIR = os.path.join(RUN_DIR, "eval")
EC57_DIR = os.path.join(RUN_DIR, "ec57")
LOGS_DIR = os.path.join(RUN_DIR, "logs")

WFDB_SCRIPTS_DIR = os.path.join(_PKG_ROOT, "scripts")


def apply_geometry(segment_samples, output_steps=None):
    """Re-derive the window geometry at run time, for scoring a checkpoint of another length.

    Evaluation reads SEGMENT_SAMPLES / OUTPUT_STEPS through this module at call time, so a
    10 s checkpoint can be scored by the very same code path as a 60 s one - which is what
    makes the non-regression comparison apples to apples. Training data geometry (the
    tfrecord tree) is NOT re-pointed: that is a build, not a run-time choice.
    """
    global SEGMENT_SAMPLES, SEGMENT_SECONDS, OUTPUT_STEPS, STEP_SAMPLES, EC57_SEGMENT_OVERLAP
    segment_samples = int(segment_samples)
    output_steps = int(output_steps or segment_samples // STEP_SAMPLES)
    if segment_samples % output_steps:
        raise ValueError(f"{output_steps} steps do not divide {segment_samples} samples")
    SEGMENT_SAMPLES = segment_samples
    SEGMENT_SECONDS = segment_samples / SAMPLING_RATE
    OUTPUT_STEPS = output_steps
    STEP_SAMPLES = segment_samples // output_steps
    # keep the overlap under one window: the 10 s model swept with 1 s of overlap
    EC57_SEGMENT_OVERLAP = min(EC57_SEGMENT_OVERLAP, max(SAMPLING_RATE, segment_samples // 6))
    return SEGMENT_SAMPLES, OUTPUT_STEPS


def ensure_run_dirs():
    """Create this run's output folders. Called by the stages that write, not at import."""
    for d in (CHECKPOINT_DIR, EVAL_DIR, EC57_DIR, LOGS_DIR):
        os.makedirs(d, exist_ok=True)


def _shared_weights(keras_name, filename, other_run):
    """This run's copy of `filename`, or the run `other_run` names if this run has none.

    Both self-supervised stages depend only on the architecture, never on labels, so an
    earlier run's weights are a legitimate starting point for a later one.
    """
    here = os.path.join(CHECKPOINT_DIR, keras_name, filename)
    if os.path.exists(here) or not other_run:
        return here
    return os.path.join(WORK_DIR, other_run, "checkpoints", keras_name, filename)


def cpc_weights_dir(keras_name):
    """Where this run's CPC context encoder for `keras_name` lives (or ECGR_CPC_RUN's)."""
    return _shared_weights(keras_name, "cpc_context.weights.h5", CPC_RUN)


def ssl_weights_dir(keras_name):
    """Where this run's SSL-pretrained backbone for `keras_name` lives (or ECGR_SSL_RUN's)."""
    return _shared_weights(keras_name, "ssl_backbone.weights.h5", SSL_RUN)


def describe():
    """One-screen summary of what this run is configured to do."""
    return "\n".join([
        f"run tag      : {RUN_TAG}",
        f"portal data  : {DATA_DIR}",
        f"physionet    : {PHYSIONET_DIR}",
        f"npy          : {NPY_DIR}",
        f"tfrecord     : {TFRECORD_DIR}",
        f"run dir      : {RUN_DIR}",
        f"datasets     : {', '.join(TRAIN_DATASETS)}",
        f"input        : ({SEGMENT_SAMPLES}, {IN_CHANNELS}) "
        f"= {SEGMENT_SECONDS:g} s @ {SAMPLING_RATE} Hz, {IN_CHANNELS} leads "
        f"(lead 0 = the annotated one)",
        f"output 1     : beat_cls ({OUTPUT_STEPS}, {NUM_CLASSES}) "
        f"= {1000 * STEP_SAMPLES / SAMPLING_RATE:.0f} ms per step, {CLASS_NAMES}; "
        f"steps outside the reviewed span = ignore ({IGNORE_LABEL})",
        f"output 2     : lead_quality ({OUTPUT_STEPS}, {IN_CHANNELS}) per-lead readability, "
        f"label-free target, loss weight {QUALITY_LOSS_WEIGHT}",
        f"loss/monitor : {LOSS} / {MONITOR}, checkpoints from epoch {CKPT_START_EPOCH}",
        f"batch / lr   : {BATCH_SIZE} / {LEARNING_RATE}, cache {'on' if CACHE_DATASET else 'off'}",
        f"ssl reuse    : {SSL_RUN or '(pretrain in this run)'}",
        f"cpc reuse    : {CPC_RUN or '(pretrain in this run)'}",
        f"ec57 leads   : {EC57_LEAD_MODE} (fill: {LEAD_FILL_MODE}), "
        f"sweep overlap {EC57_SEGMENT_OVERLAP / SAMPLING_RATE:g} s",
        f"baselines    : {BASELINES_DIR}",
    ])
