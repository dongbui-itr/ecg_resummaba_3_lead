"""Every path and hyper-parameter of the project, in one place.

Nothing here is read at import time by anything that matters except through this module, so
a run is fully described by this file plus the environment variables it reads. The env
overrides exist so a queued job cannot be changed under its feet by an edit to this file:
pin them in the launcher (run_pipeline.sh does), never edit mid-run.

    ECGR_DATA_DIR       where the portal datasets live (records + dataset_info_full.csv)
    ECGR_PHYSIONET_DIR  where mitdb/nstdb/... live
    ECGR_WORK_DIR       where npy, tfrecord, checkpoints and reports are written
    ECGR_RUN_TAG        name of this run's output folder (default: today, yymmdd)
    ECGR_IN_CHANNELS    number of ECG LEADS fed to the model (default 3)
    ECGR_CPC_RUN        reuse another run's CPC-pretrained context encoders
    ECGR_SSL_RUN        reuse another run's SSL-pretrained backbones
    ECGR_WORKERS        processes used by the npy/tfrecord builders (default: cpus/2)
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
RUN_TAG = os.environ.get("ECGR_RUN_TAG", datetime.today().strftime("%y%m%d") + "_ecgr")

# ---------------------------------------------------------------------------
# Signal / segmentation
# ---------------------------------------------------------------------------
SAMPLING_RATE = 250                                   # Hz, everything is resampled to this
SEGMENT_SECONDS = 10
SEGMENT_SAMPLES = SEGMENT_SECONDS * SAMPLING_RATE     # 2500
SEGMENT_STRIDE_SECONDS = 1                            # window hop when cutting training data

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
MIN_AMPLITUDE = 0.1           # mV; flatter windows carry no beat and are dropped

# ---------------------------------------------------------------------------
# Label grid
# ---------------------------------------------------------------------------
# OUTPUT_STEPS must divide SEGMENT_SAMPLES exactly - the model pools down to this grid.
# 2500/500 = 5 samples per step = 20 ms per label step.
OUTPUT_STEPS = 500
NUM_CLASSES = 4
CLASS_NAMES = ['None', 'N', 'V', 'S']         # index == label value

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

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
TRAIN_DATASETS = ["dataset-1", "dataset-2", "dataset-3", "dataset-4", "dataset-5",
                  "dataset-3-filter-vt-svt-avb2-avb3", "dataset 2_3_4 - AFib - v2",
                  "dataset_ivcd"]
DATASET_CSV = "dataset_info_full.csv"
PORTAL_FS = 250               # native sampling rate of the portal records

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
# per split per model; the full splits are 365,787 / 91,342 events (0 = all of them).
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
EC57_SEGMENT_OVERLAP = 1 * SAMPLING_RATE   # overlap when sweeping a whole record

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
#               as a P wave on V5. It also removes a fragility rather than hiding one: the
#               N/S boundary on sinus tachycardia (record 213) is knife-edge with MLII alone
#               - any noise fine-tune flips it - and stable once V5 is there.
# 'single'    : only the annotated lead is real; the rest are filled per LEAD_FILL_MODE.
#               The strict single-lead reading, and the control for the finding above.
#               'duplicate' is accepted as a deprecated alias - it named the mode back when
#               filling was always by repetition.
# 'auto'      : 'native' only where the record ALREADY has IN_CHANNELS leads, else 'single'.
#               Every EC57 database has two leads and the model takes three, so on all five
#               'auto' means 'single' - it is not a synonym for 'native' there. It only
#               coincides with 'native' on the 3-lead portal records.
EC57_LEAD_MODE = os.environ.get("ECGR_EC57_LEAD_MODE", "native")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1.4e-3          # config default 1e-3 x sqrt(128/64), the usual batch scaling
PATIENCE = 8

# Default loss for this family. Poly2 is what the paper specifies; it also makes val_loss
# useless as a stopping signal, which is why MONITOR below is the F1 (see training/losses.py).
LOSS = 'poly2'
POLY2_EPS = (0.3, -0.5)         # paper sec. 3.6, found by grid search with a flat optimum
MONITOR = 'val_weighted_f1'

# First epoch (1-indexed) allowed to write a checkpoint. Earlier epochs are still measured
# and logged, they just cannot be kept as "best". Measured on this family: the 1M size swings
# between 0.8267 and 0.8519 over its first nine epochs and then settles into 0.8467-0.8523
# for the next fourteen, so a peak picked from epoch 4 records the noise.
CKPT_START_EPOCH = 10

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
REFINE_KERNEL_LEN = 256          # steps each way = 5.1 s, 6-10 R-R intervals
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
# of paying for them again: set ECGR_CPC_RUN to that run's tag.
CPC_EPOCHS = 5
CPC_STEPS_PER_EPOCH = 400
CPC_RUN = os.environ.get("ECGR_CPC_RUN")

# Loss weight per class. S is the rarest per step and the weakest metric, so it gets the
# most; None dominates every window and is damped. Raising the S weight trades +P for Se,
# so it is deliberately modest (2.5x N) rather than inverse-frequency.
CLASS_WEIGHTS = [0.3, 1.0, 2.0, 2.5]

BATCH_SEGMENTS = 10000        # segments per npy batch file

# Records come off disk grouped by study, so a buffer that is too small leaves a batch made
# of one patient's beats. It shuffles the SERIALIZED records, which at 3 leads are ~30.5 kB
# each, so this buffer is ~500 MB of RAM - the reason it is not simply enormous.
SHUFFLE_BUFFER = 16384

# cache() holds the whole split's serialized records in RAM, so only the first epoch touches
# disk. Sizing it matters more at 3 leads than it did at 1: the train split is ~30 kB per
# segment, i.e. tens of GB. Two models training side by side each hold their own copy. Set
# CACHE_DATASET = False on a machine where that does not fit - it costs I/O per epoch, not
# correctness.
CACHE_DATASET = True
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

# Save every epoch from CKPT_START_EPOCH on (<ckpt>/epochs/epoch_NN.keras), not only the
# step-F1 improvements: the checkpoint that scores best at beat level is chosen afterwards by
# bxb on portal-eval, and the README's own warning is that step F1 does not pick it.
SAVE_EVERY_EPOCH = True

WORKERS = int(os.environ.get("ECGR_WORKERS", max(1, (os.cpu_count() or 8) // 2)))

# ---------------------------------------------------------------------------
# Derived output layout - one folder per kind of output, one subfolder per model
# ---------------------------------------------------------------------------
NPY_DIR = os.environ.get(
    "ECGR_NPY_DIR",
    os.path.join(DATA_DIR, f"npy_{OUTPUT_STEPS}_{NUM_CLASSES}_{SAMPLING_RATE}_"
                           f"{SEGMENT_SECONDS}_{IN_CHANNELS}lead"))
TFRECORD_DIR = os.environ.get(
    "ECGR_TFRECORD_DIR", os.path.join(WORK_DIR, f"tfrecord_{IN_CHANNELS}lead"))
DATASET_MANIFEST = "dataset_manifest.json"   # written next to the tfrecords, checked on load

RUN_DIR = os.path.join(WORK_DIR, RUN_TAG)
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
EVAL_DIR = os.path.join(RUN_DIR, "eval")
EC57_DIR = os.path.join(RUN_DIR, "ec57")
LOGS_DIR = os.path.join(RUN_DIR, "logs")

WFDB_SCRIPTS_DIR = os.path.join(_PKG_ROOT, "scripts")


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
        f"input        : ({SEGMENT_SAMPLES}, {IN_CHANNELS}) "
        f"= {SEGMENT_SECONDS} s @ {SAMPLING_RATE} Hz, {IN_CHANNELS} leads "
        f"(lead 0 = the annotated one)",
        f"output       : ({OUTPUT_STEPS}, {NUM_CLASSES}) "
        f"= {1000 * SEGMENT_SECONDS / OUTPUT_STEPS:.0f} ms per step, {CLASS_NAMES}",
        f"loss/monitor : {LOSS} / {MONITOR}, checkpoints from epoch {CKPT_START_EPOCH}",
        f"ssl reuse    : {SSL_RUN or '(pretrain in this run)'}",
        f"cpc reuse    : {CPC_RUN or '(pretrain in this run)'}",
        f"ec57 leads   : {EC57_LEAD_MODE} (fill: {LEAD_FILL_MODE})",
    ])
