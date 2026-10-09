"""Every path and hyper-parameter of the RHYTHM task, in one place.

The beat task's config (ecgr/config.py) still owns what both tasks share - the sampling rate,
the band-pass, the per-lead z-score - and this module reads it from there instead of keeping a
second copy, so the two can never preprocess a strip differently.

    ECGR_RHYTHM_DATA_ROOT   where the Holter report strips live (dataset-1, dataset-rhythm, ...)
    ECGR_RHYTHM_EVAL_DIR    the rhythm test set - EVERY study under it is held out of train+eval
    ECGR_RHYTHM_WORK_DIR    where npy, checkpoints and reports are written
    ECGR_RHYTHM_RUN_TAG     name of this run's output folder (default: today, yymmdd_rhythm)
"""
import os
from datetime import datetime

from .. import config as base

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("ECGR_RHYTHM_DATA_ROOT",
                           "/media/MegaDataSet/DATA_4TINYML/Holter_report_strip")
EVAL_DIR = os.environ.get("ECGR_RHYTHM_EVAL_DIR",
                          os.path.join(DATA_ROOT, "dataset-eval", "rhythm_eval"))
WORK_DIR = os.environ.get("ECGR_RHYTHM_WORK_DIR", "/media/Project/ECG/Model_Dong/ecgr_rhythm")
RUN_TAG = os.environ.get("ECGR_RHYTHM_RUN_TAG",
                         datetime.today().strftime("%y%m%d") + "_rhythm")

# ---------------------------------------------------------------------------
# Signal contract - shared with the beat task
# ---------------------------------------------------------------------------
SAMPLING_RATE = base.SAMPLING_RATE                   # 250 Hz
SEGMENT_SECONDS = base.SEGMENT_SECONDS               # 10 s
SEGMENT_SAMPLES = base.SEGMENT_SAMPLES               # 2500
IN_CHANNELS = 3                                      # always the full 3-lead montage

# One label per second: (10,) per window.
OUTPUT_SECONDS = SEGMENT_SECONDS
SECOND_SAMPLES = SAMPLING_RATE

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# Five rhythm classes, index == label value (2026-10-07, user decision): SINUS, AFIB (atrial
# flutter included), SVT, VT and AVB = second- OR third-degree AV block. The reference
# project's six (itr-ai-sensor_annotation-ae_ecg_classification data/config_dataset.EVENT_CLASS)
# kept AVB2 and AVB3 apart; LEGACY_CLASS_NAMES is that order, which the npy shards built
# before this date and the 6-output checkpoints still use - pipeline.load_arrays and
# labels.to_current_classes map them onto the five (AVB2, AVB3 -> AVB).
CLASS_NAMES = ['SINUS', 'AFIB', 'SVT', 'VT', 'AVB']
NUM_CLASSES = len(CLASS_NAMES)
LEGACY_CLASS_NAMES = ['SINUS', 'AFIB', 'SVT', 'VT', 'AVB2', 'AVB3']
LEGACY_TO_CLASS = [0, 1, 2, 3, 4, 4]          # legacy label index -> CLASS_NAMES index
SINUS = CLASS_NAMES.index('SINUS')
IGNORE = 255                    # stored label of a second nobody vouched for (loss weight 0)

# Two outputs:
#   'rhythm' (OUTPUT_SECONDS, NUM_CLASSES)  softmax per second
#   'lead'   (NUM_LEAD_CLASSES,)            softmax for the whole 10 s window: the lead with
#                                           the best signal, or NOISE when none is readable.
#                                           Index 1/2/3 = input channel 0/1/2.
LEAD_CLASSES = ['NOISE', 'CH1', 'CH2', 'CH3']
NUM_LEAD_CLASSES = len(LEAD_CLASSES)
LEAD_NOISE = 0
# The 20 ms family (rhythm_unet500_*) replaces 'lead' by
#   'noise'  (NOISE_SEGMENTS, 2)            softmax per 2 s segment: CLEAN / NOISE.
#            A segment is NOISE when either of its seconds is unreadable - fewer than
#            CLEAN_MIN_LEADS leads at >= CLEAN_SNR_DB (augment.clean_seconds, the definition
#            that already sets the rhythm loss weight) - so CLEAN means the whole 2 s can be read.
# and emits 'rhythm' at 20 ms: (500, NUM_CLASSES), the label of each 5-sample block's centre.
# Beat decoder (the 'b' families, rhythm_unet1250b_*): 'beat' (BEAT_STEPS, 4) softmax per 8 ms
# step - none / N / S / V (AAMI: L, R, B, e, j, n -> N; A, a, J, S -> S; V, E -> V). Labels
# come from the record's .atr (portal strips, ltafdb, nsrdb, svdb, incartdb); PTB-XL and
# Challenge 2020 have none and are IGNORE for this output. A beat labels the steps within
# BEAT_TARGET_HALFWIDTH_STEPS of its R sample; fusion / paced / unclassifiable beats (F, Q, /, f,
# !) and their BEAT_IGNORE_RADIUS_SECONDS neighbourhood are IGNORE. The rhythm decoder is
# conditioned on this output (stop-gradient): where the beats are and what they are is the
# evidence RR-regularity and ">= 3 V beats = VT" rest on.
BEAT_CLASSES = ['none', 'N', 'S', 'V']
BEAT_STEPS = SEGMENT_SAMPLES // 2
BEAT_SYMBOL_TO_CLASS = {'N': 1, 'L': 1, 'R': 1, 'B': 1, 'e': 1, 'j': 1, 'n': 1,
                        'A': 2, 'a': 2, 'J': 2, 'S': 2, 'V': 3, 'E': 3}
BEAT_IGNORE_SYMBOLS = ('F', 'Q', '/', 'f', '!', 'r', '?')
BEAT_TARGET_HALFWIDTH_STEPS = 1                   # (kept for the old one-hot tests / docs)
BEAT_IGNORE_RADIUS_SECONDS = 0.1
# 2026-10-06: the beat target separates WHERE from WHAT. Channel 0 is a Gaussian heatmap of
# the R positions (sigma BEAT_HEAT_SIGMA_STEPS), trained with a BCE on p(beat) = 1 - p(none);
# channels 1-3 are the N/S/V one-hot inside +-BEAT_TYPE_RADIUS_STEPS of each beat, trained
# with a CE on the renormalised N/S/V probabilities on those steps only. The old target (one
# hot none/N/S/V 3 steps wide) made the loss and val_beat_f1 fight over 8 ms alignment while
# bxb tolerates 150 ms: the picked beats were already 100 % Se / +P when the step F1 said 0.55.
BEAT_HEAT_SIGMA_STEPS = 2.5                       # 20 ms at 125 steps / s
BEAT_TYPE_RADIUS_STEPS = 5                        # 40 ms
BEAT_HEAT_POS_WEIGHT = 3.0                        # positives are ~6 % of the steps
BEAT_TYPE_LOSS_WEIGHT = 1.0
BEAT_LOSS_WEIGHT = 1.0
BEAT_CLASS_WEIGHTS = (0.15, 1.0, 4.0, 3.0)          # none (unused), N, S, V
BEAT_MATCH_TOLERANCE_SECONDS = 0.15               # bxb's window, for the beat-level metric
BEAT_MATCH_WINDOWS = 2048                         # eval windows the per-epoch beat matcher sees

NOISE_CLASSES = ['CLEAN', 'NOISE']
NOISE_SEGMENTS = 5
NOISE_SEGMENT_SECONDS = 2

# Portal event type -> class. Two groups differ from the reference on purpose:
#
#   * The ectopy types (SINGLE_*, *_COUPLET, *_BIGEMINY, ...) were not loaded there at all. They
#     are sinus rhythm with ectopic beats, i.e. exactly the hard negatives SVT and VT have to be
#     told apart from, so they are SINUS here - and the >= 3-beat runs the reviewer annotated
#     inside them are recovered from the beat labels (labels.beat_runs) rather than lost.
#   * SVE_RUN / VE_RUN map to SVT / VT as in the reference.
#
# Anything not listed (OTHERS, ARTIFACT, AVB1, ...) is skipped: a strip whose rhythm nobody
# named cannot be labelled.
EVENT_TYPE_TO_CLASS = {
    'SINUS': 'SINUS', 'TACHY': 'SINUS', 'BRADY': 'SINUS', 'PAUSE': 'SINUS',
    'SINGLE_SVE': 'SINUS', 'SVE_COUPLET': 'SINUS', 'SVE_BIGEMINY': 'SINUS',
    'SVE_TRIGEMINAL': 'SINUS', 'SVE_QUADRIGEMINY': 'SINUS',
    'SINGLE_VE': 'SINUS', 'VE_COUPLET': 'SINUS', 'VE_BIGEMINY': 'SINUS',
    'VE_TRIGEMINAL': 'SINUS', 'VE_QUADRIGEMINY': 'SINUS',
    'AFIB': 'AFIB',
    'SVT': 'SVT', 'SVE_RUN': 'SVT',
    'VT': 'VT', 'VE_RUN': 'VT',
    'AVB2': 'AVB',
    'AVB3': 'AVB',
}

# Classes whose extent is recovered from the beat annotations when the inventory gives no
# caliper span (dataset-1 / dataset-4): a run of >= RUN_MIN_BEATS consecutive S beats is SVT, of
# V beats VT. Measured on dataset-1 SVT strips: the run typically covers half the strip
# ("SSSSSSSSSSSSNNNNNNN"), so labelling the whole 10 s as SVT would be wrong for half of it.
RUN_CLASSES = ('SVT', 'VT')
RUN_MIN_BEATS = 3
RUN_BEAT_SYMBOLS = {'SVT': ('S', 'A', 'J'), 'VT': ('V', 'E')}
RUN_PAD_SECONDS = 0.15          # the run starts at the onset of its first QRS, not at its R peak

# What an unmarked second means, per class of the event that produced the strip, for sources
# whose caliper marks an event inside a 60 s recording:
#   'sinus'  - the reference's rule: everything outside the caliper is SINUS. Right for the
#              paroxysmal rhythms (an SVT or VT run is an episode inside a sinus strip).
#   'ignore' - not scored. Right for the persistent ones: AF or an AV block rarely stops where
#              the reviewer's caliper happens to end, so "outside = SINUS" is a label that is
#              wrong precisely where the model is least sure.
OUTSIDE_SPAN = {'SINUS': 'sinus', 'SVT': 'sinus', 'VT': 'sinus',
                'AFIB': 'ignore', 'AVB': 'ignore'}

# A second takes a rhythm class when that class's spans cover at least this fraction of it.
MIN_SECOND_COVER = 0.5

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
# Order is priority: the same event id appears in several exports (the AFib set is drawn from
# dataset-2/3/4; dataset-rhythm/dataset-2 from dataset-2), and the first source that has it wins.
# The caliper sources come first because they carry the most precise spans.
#   root         folder under DATA_ROOT with <study>/<event>/*.hea
#   inventory    the export listing its events, relative to DATA_ROOT
#   study/event  column names; type: the event type column
#   span         (start, stop) columns of the reviewer's caliper, or None
#   strip        (start, stop) columns of the 10 s strip the reviewer looked at, or None
#   whole        True: the source is a 60 s caliper recording (OUTSIDE_SPAN applies)
#   sinus_first  optional: its SINUS strips are kept first under MAX_SINUS_EVENTS_PER_STUDY
#   header_span  optional: the export has no caliper, read eventStartSample/eventStopSample
#                from the record's .hea instead (build.header_span) - as the reference did
SOURCES = {
    'rhythm-2': dict(root='dataset-rhythm/dataset-2-vt-svt-avb2-avb3',
                     inventory='dataset-rhythm/dataset-2-filter-vt-svt-avb2-avb3.xlsx',
                     study='Study ID', event='ID', type='Identified Event Type',
                     span=('eventStartSample', 'eventStopSample'), strip=None, whole=True),
    'rhythm-3': dict(root='dataset-rhythm/dataset-3-vt-svt-avb2-avb3',
                     inventory='dataset-rhythm/dataset-3-filter-vt-svt-avb2-avb3.xlsx',
                     study='Study ID', event='ID', type='Identified Event Type',
                     span=('eventStartSample', 'eventStopSample'), strip=None, whole=True),
    # dataset-AFib/'dataset 2_3_4 - AFib - v2' is the same 7,962 events again - not listed.
    'afib-2': dict(root='dataset-rhythm/dataset-afib-2',
                   inventory='dataset-rhythm/dataset-afib-2.xlsx',
                   study='Study ID', event='ID', type='eventType',
                   span=('eventStartSample', 'eventStopSample'), strip=None, whole=True),
    # dataset-sinus: the dedicated sinus exports - TACHY / BRADY / PAUSE, all SINUS. They are
    # the hard negatives of the rhythm classes (a sinus tachycardia against SVT, a brady or a
    # pause against an AV block), so their strips are kept first under the per-study SINUS cap.
    # dataset-2/3 are 60 s caliper recordings like rhythm-2/3; dataset-4 are 10 s strips whose
    # export gives only the strip (eventStartSample/eventStopSample = 7500/10000), handled like
    # dataset-1/4 - including recovering >= 3-beat S/V runs from the .atr.
    'sinus-2': dict(root='dataset-sinus/dataset-2-tachy-brady-pause',
                    inventory='dataset-sinus/dataset-2-tachy-brady-pause.xlsx',
                    study='Study ID', event='ID', type='eventType',
                    span=('eventStartSample', 'eventStopSample'), strip=None, whole=True,
                    sinus_first=True),
    'sinus-3': dict(root='dataset-sinus/dataset-3-tachy-brady-pause',
                    inventory='dataset-sinus/dataset-3-tachy-brady-pause.xlsx',
                    study='Study ID', event='ID', type='eventType',
                    span=('eventStartSample', 'eventStopSample'), strip=None, whole=True,
                    sinus_first=True),
    'sinus-4p': dict(root='dataset-sinus/dataset-4-tachy-brady-pause',
                     inventory='dataset-sinus/dataset-4-tachy-brady-pause.xlsx',
                     study='Study ID', event='ID', type='eventType',
                     span=None, strip=('eventStartSample', 'eventStopSample'), whole=False,
                     sinus_first=True),
    'sinus-4': dict(root='dataset-sinus/dataset-4-tachy-brady',
                    inventory='dataset-sinus/dataset-4-tachy-brady.xlsx',
                    study='Study ID', event='ID', type='eventType',
                    span=None, strip=('eventStartSample', 'eventStopSample'), whole=False,
                    sinus_first=True),
    'dataset-5': dict(root='dataset-5', inventory='dataset-5.xlsx',
                      study='studyFid', event='id', type='eventType',
                      span=('eventStartSample', 'eventStopSample'),
                      strip=('startSample', 'stopSample'), whole=False),
    'dataset-1': dict(root='dataset-1', inventory='dataset-1.xlsx',
                      study='Study ID', event='ID', type='eventType',
                      span=None, strip=('startSample', 'stopSample'), whole=False),
    # dataset-4's export has no caliper columns, its headers do (dataset-1's have none).
    'dataset-4': dict(root='dataset-4', inventory='dataset-4.xlsx',
                      study='studyId', event='id', type='eventType',
                      span=None, strip=('startSample', 'stopSample'), whole=False,
                      header_span=True),
}
# PTB-XL (rhythm/ptbxl.py): 12-lead 10 s resting ECGs, one SCP-coded diagnosis per record.
# Not an EC57 database, so it is a legitimate training source - and the only one with the
# hard negatives the Physionet false positives come from (bundle-branch block, paced rhythm,
# sinus tachy/brady, ectopy): the portal exports file those under OTHERS. Label per record:
#   an arrhythmia code -> its class (PTBXL_CODE_TO_CLASS; two different classes = skipped)
#   AFLT               -> AFIB (atrial flutter is scored as AF, see EC57_AFL_AS_AF)
#   SVARR              -> skipped (names no class)
#   otherwise          -> SINUS, kept only with a hard-negative code (PTBXL_HARD_NEGATIVE_CODES)
#                         or under a hash-chosen cap of plain records (PTBXL_PLAIN_CAP)
# Split by patient through strat_fold: 1-8 train, 9 eval, 10 never read (reserve). Patient ids
# are offset by PTBXL_STUDY_OFFSET so they cannot collide with a portal study id in the
# studyids_*.npy the audit reads. Each record gives PTBXL_WINDOWS_PER_RECORD windows over
# different 3-of-12 lead subsets (ptbxl.lead_subsets).
PTBXL_DIR = os.environ.get("ECGR_PTBXL_DIR", "/media/MegaDataSet/ECG/physionet_org/ptb-xl/1.0.3")
PTBXL_SOURCE = 'ptbxl'
PTBXL_CODE_TO_CLASS = {'AFIB': 'AFIB', 'AFLT': 'AFIB', 'SVTAC': 'SVT', 'PSVT': 'SVT',
                       '2AVB': 'AVB', '3AVB': 'AVB'}
PTBXL_SKIP_CODES = ('SVARR',)
PTBXL_HARD_NEGATIVE_CODES = ('PACE', 'CLBBB', 'CRBBB', 'IVCD', 'ILBBB', 'IRBBB', 'WPW', 'STACH',
                             'SBRAD', 'SARRH', 'PVC', 'PAC', 'BIGU', 'TRIGU', '1AVB')
PTBXL_PLAIN_CAP = int(os.environ.get("ECGR_RHYTHM_PTBXL_PLAIN_CAP", 3000))
PTBXL_FOLDS = {'train': (1, 2, 3, 4, 5, 6, 7, 8), 'eval': (9,)}
PTBXL_STUDY_OFFSET = 10 ** 9
PTBXL_WINDOWS_PER_RECORD = 2
PTBXL_LIMB_LEADS = ('I', 'II', 'III', 'AVR', 'AVL', 'AVF')
PTBXL_PRECORDIAL_LEADS = ('V1', 'V2', 'V3', 'V4', 'V5', 'V6')

# Long PhysioNet recordings OUTSIDE EC57 (rhythm/physionet_train.py) - allowed for training by
# project rule (2026-09-29); the EC57 databases (mitdb, afdb, escdb, nstdb, ahadb, cudb) and
# rhythm_eval never are (physionet_train refuses them). Each record is cut into 10 s windows
# read straight from the long file (a chunk with a filter margin, never the whole 24 h),
# labelled from its own .atr:
#   ltafdb   rhythm marks: (AFIB/(AFL -> AFIB, (SVTA -> SVT, (VT -> VT; (N, (SBR, (AB, (B, (T
#            -> SINUS (sinus brady, atrial/ventricular bigeminy, trigeminy: hard negatives);
#            any other code (IVR, ...) -> IGNORE
#   nsrdb    no rhythm marks, sinus throughout; runs of >= RUN_MIN_BEATS S / V beats -> SVT / VT
#   incartdb as nsrdb, plus (AFIB -> AFIB and (PREX (pre-excitation, sinus) -> SINUS; other
#            codes (WPWAF) -> IGNORE; 12-lead: one 3-lead subset per window (ptbxl.lead_subsets)
#   svdb     no rhythm marks and not known to be sinus throughout: ONLY its beat runs are
#            labelled (SVT / VT), every other second IGNORE - a source of real SVT runs
# Windows are chosen per record by category (config caps below) so the rare, informative ones
# (VT/SVT episodes, AF with ventricular ectopy, AF on/offsets, sinus tachy / brady / ectopy)
# are all kept and the hours of plain AF / sinus are subsampled. Split by record (incartdb by
# patient) with a hash: PHYSIONET_TRAIN_EVAL_FRACTION goes to 'eval'.
PHYSIONET_TRAIN_DIR = os.environ.get(
    "ECGR_PHYSIONET_TRAIN_DIR", "/media/MegaDataSet/University/private_projects/physionet_data")
PHYSIONET_TRAIN_DBS = {
    'ltafdb': dict(dir=os.path.join(PHYSIONET_TRAIN_DIR, 'ltafdb'), labels='rhythm'),
    'nsrdb': dict(dir=os.path.join(PHYSIONET_TRAIN_DIR, 'nsrdb'), labels='sinus'),
    'svdb': dict(dir=os.path.join(PHYSIONET_TRAIN_DIR, 'svdb'), labels='runs'),
    'incartdb': dict(dir="/media/MegaDataSet/ECG/physionet_org/incartdb/1.0.0",
                     labels='sinus', twelve_lead=True),
}
PHYSIONET_TRAIN_CODE_TO_CLASS = {'(AFIB': 'AFIB', '(AFL': 'AFIB', '(SVTA': 'SVT', '(VT': 'VT',
                                 '(BII': 'AVB', '(B3': 'AVB'}
PHYSIONET_TRAIN_SINUS_CODES = ('(N', '(SBR', '(AB', '(B', '(T', '(PREX', '(SAB', '(BI')
PHYSIONET_TRAIN_EVAL_FRACTION = 0.15
PHYSIONET_TRAIN_STUDY_OFFSET = 2 * 10 ** 9
# windows per record and category (a window's category: the first of this order that fits)
# (2026-09-30: AF_ectopy / AF_edge / sinus_brady raised and sinus_pac added after the mitdb AF
# errors were traced to AF with PVCs (221/219/203) missed and to sinus brady with frequent
# PACs (232: 52 of 108 false AF episodes) called AF.)
PHYSIONET_TRAIN_CAPS = {
    'VT': 80, 'SVT': 80, 'AF_edge': 150, 'AF_ectopy': 400, 'AF': 150,
    'sinus_pac': 250, 'sinus_tachy': 80, 'sinus_brady': 150, 'sinus_ectopy': 80, 'sinus': 60,
}
PHYSIONET_TRAIN_SOURCES = list(PHYSIONET_TRAIN_DBS)

# PhysioNet/CinC Challenge 2020 (rhythm/challenge2020.py): CPSC-2018 (+ extra) and Georgia
# 12-lead 500 Hz records with whole-record SNOMED diagnoses - ~1.9k AF, ~240 AFL, thousands of
# sinus tachy / brady / PAC / PVC / BBB / 1st-degree block records from other devices and
# populations than the portal. Its PTB-XL and INCART copies are left out (already sources).
CHALLENGE2020_DIR = os.environ.get(
    "ECGR_CHALLENGE2020_DIR", "/media/MegaDataSet/ECG/physionet_org/challenge-2020/1.0.2/training")
CHALLENGE2020_SOURCE = 'challenge2020'
CHALLENGE2020_SUBSETS = ('cpsc_2018', 'cpsc_2018_extra', 'georgia')
CHALLENGE2020_CODE_TO_CLASS = {'164889003': 'AFIB', '164890007': 'AFIB',     # AF, AFL
                               '195042002': 'AVB', '54016002': 'AVB',        # 2nd deg., Mobitz I
                               '28189009': 'AVB', '27885002': 'AVB'}         # Mobitz II, complete
# paroxysmal classes a record label cannot place, and rhythms of no class of ours
CHALLENGE2020_SKIP_CODES = ('426761007', '713422000', '67198005', '164895002', '111288001',
                            '164896001', '10370003', '251170000', '233917008')
CHALLENGE2020_HARD_NEGATIVE_CODES = (
    '427084000', '426177001', '426627000', '427393009',          # sinus tachy, brady, SA
    '284470004', '63593006', '427172004', '17338001', '164884008',  # PAC, SVPB, PVC, VPB, VEB
    '164909002', '59118001', '713427006', '713426002', '445118002',  # LBBB, RBBB, CRBBB, IRBBB
    '270492004', '164947007', '74390002')                        # 1st-deg. block, long PR, WPW
CHALLENGE2020_PLAIN_CAP = int(os.environ.get("ECGR_RHYTHM_CHALLENGE_PLAIN_CAP", 1500))
CHALLENGE2020_WINDOWS_PER_RECORD = 2
CHALLENGE2020_EVAL_FRACTION = 0.15
CHALLENGE2020_STUDY_OFFSET = 3 * 10 ** 9
EC57_DATABASES = ('mitdb', 'afdb', 'escdb', 'nstdb', 'ahadb', 'cudb')

TRAIN_SOURCES = list(SOURCES) + [PTBXL_SOURCE] + PHYSIONET_TRAIN_SOURCES + [CHALLENGE2020_SOURCE]
# Validation for decoding / calibration choices WITHOUT touching EC57 or rhythm_eval: the
# eval-side records (physionet_train.eval_records) of these databases, whole and continuous,
# scored by the same epicmp pipeline (`ec57 --dbs ltafdb nsrdb incartdb --skip-rhythm-eval`).
# ltafdb (24 h, rhythm marks) carries the real prevalence of AF / SVTA / VT runs; nsrdb and
# incartdb have no reference episodes of these classes and only count false positives.
VALIDATION_CLASS_DBS = {'ltafdb': ['AFIB', 'SVT', 'VT'],
                        'nsrdb': ['AFIB', 'SVT', 'VT', 'AVB'],
                        'incartdb': ['AFIB', 'SVT', 'VT', 'AVB']}
VALIDATION_TARGET_CELLS = [('ltafdb', 'AFIB'), ('ltafdb', 'SVT'), ('ltafdb', 'VT')]

# The SINUS group outnumbers every arrhythmia by two orders of magnitude in dataset-1/4
# (~200k strips against ~2k AFIB). Kept per study, chosen by a hash of the event id so a rebuild
# picks the same strips: enough variety of patients without drowning the rare classes.
MAX_SINUS_EVENTS_PER_STUDY = int(os.environ.get("ECGR_RHYTHM_MAX_SINUS_PER_STUDY", 3))
# Sinus strips whose type is a look-alike of an arrhythmia - supraventricular / ventricular
# ectopy patterns, bradycardia, pauses - get their own budget on top of the one above: they are
# the hard negatives behind the mitdb false AF episodes (232, 200, 215) and the cap threw ~40k
# of them away.
HARD_SINUS_TYPES = ('SVE_BIGEMINY', 'SVE_TRIGEMINAL', 'SVE_QUADRIGEMINY', 'SVE_COUPLET',
                    'SINGLE_SVE', 'VE_BIGEMINY', 'VE_TRIGEMINAL', 'VE_COUPLET', 'BRADY', 'PAUSE')
MAX_HARD_SINUS_EVENTS_PER_STUDY = int(os.environ.get("ECGR_RHYTHM_MAX_HARD_SINUS_PER_STUDY", 8))

# Holdout. The rhythm test set is read from the folder itself (EVAL_DIR/<study id>/...), not
# from a list: 110 of its 1,966 studies are missing from assets/list_studies_eval_v4.json, so
# the list alone would leak them. The v4 list is excluded as well, so the beat and rhythm
# models share one holdout.
EXCLUDE_V4_EVAL_STUDIES = True
TRAIN_FRACTION = base.TRAIN_FRACTION     # study-level, by hash (data/splits.study_split_side)

# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
# Hop between training windows over one event's region. Every event is at least one window.
WINDOW_HOP_SECONDS = 5
# How far a strip window may reach past a 2499-sample reviewed span (see build_npy).
SPAN_SLACK_SAMPLES = int(0.1 * SAMPLING_RATE)

# ---------------------------------------------------------------------------
# Augmentation (rhythm/augment.py)
# ---------------------------------------------------------------------------
AUGMENT = True
# Lead order is shuffled per sample: rhythm labels belong to TIME, not to a lead, so any
# permutation of the three leads carries the same labels - and a model that has seen them all
# does not care which electrode a given device wires to CH1.
PERMUTE_LEADS_PROB = 1.0
LEAD_GAIN_RANGE = (0.8, 1.25)
# A rhythm is the same upside down: per lead, the sign is flipped with FLIP_LEAD_PROB, so a
# device that wires an electrode pair the other way round - or a lead subset the portal never
# recorded (PTB-XL's V1, aVR) - is nothing new. SNR and kurtosis are even in the sign, so the
# lead / NOISE target does not move.
FLIP_LEAD_PROB = 0.3
# One lead flat (electrode off) - counts as a lost lead. 0.2: every Physionet EC57 record has
# two real leads and a zero-filled third, and that input has to be common in training.
LEAD_DROP_PROB = 0.2

# Recording noise of random intensity. Per sample: on with NOISE_PROB; hits 1, 2 or 3 leads
# with NOISE_LEADS_WEIGHTS; each noisy lead gets its own target SNR drawn uniformly in
# [NOISE_SNR_DB_MIN, NOISE_SNR_DB_MAX], over a flat-topped burst of NOISE_SPAN_SECONDS.
# Measured over 4 x 1,024 windows: 17.6-19.5% of the seconds come out noisy, few enough that
# most of each batch still teaches rhythm.
NOISE_PROB = float(os.environ.get("ECGR_RHYTHM_NOISE_PROB", 0.7))
NOISE_LEADS_WEIGHTS = (0.2, 0.35, 0.45)
NOISE_SNR_DB_MIN, NOISE_SNR_DB_MAX = -12.0, 18.0
NOISE_SPAN_SECONDS = (2.0, 16.0)             # > 10 s = most or all of the window
NOISE_SHARED_BURST_PROB = 0.7                # same burst on every noisy lead (body motion)
WANDER_PROB, WANDER_AMP = 0.3, 0.5           # baseline wander: nuisance, NOT counted as noise
# Whole-window wrecks: every lead noisy over the whole window, SNR below the readable line.
# Without them random noise makes only 5.7% of training windows NOISE (it has to hit all
# three leads badly for >= 3 s at once), too few for the 'lead' output to learn the class.
# With them, measured on a train sample: NOISE 14.6%, CH1 28.9%, CH2 27.7%, CH3 28.7%, and
# 22.7% of labelled seconds noisy.
NOISE_WINDOW_PROB = 0.12

# Readability. A lead is readable in a second when its SNR there (clean-window RMS over
# that second's noise RMS) is >= CLEAN_SNR_DB. A second is clean - full rhythm loss weight,
# and scored by the rhythm metrics - when at least
# CLEAN_MIN_LEADS leads are readable (or all live leads, if fewer are alive - a flat lead is
# never readable). 2 of 3 = one bad electrode is NOT noise: the rhythm is still readable from
# the other two, and that is the case the model has to learn to ride through.
CLEAN_SNR_DB = 6.0
CLEAN_MIN_LEADS = 2

# The 'lead' target, per window. Leads are ranked by, in this order:
#   1. how many of the 10 seconds are readable on that lead (SNR >= CLEAN_SNR_DB)
#   2. its SNR over the window, clipped to [0, LEAD_SNR_CAP_DB] - so two untouched leads tie
#   3. its QRS peakedness (kurtosis of the z-scored lead, the classic kSQI) - which decides
#      between leads the augmentation left clean, from the signal itself
# A flat lead is never chosen. The window is NOISE when even the best lead has fewer than
# LEAD_MIN_READABLE_SECONDS readable seconds.
#
# Why not the reviewer's `channel` column: it is the lead the reviewer ANNOTATED, and in
# dataset-1/4/5 it is CH2 for 83-89% of events - a default, not a quality judgement. Trained
# on, it would teach "answer CH2", which the lead permutation then makes plain wrong.
LEAD_MIN_READABLE_SECONDS = 8
LEAD_SNR_CAP_DB = 30.0

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
# Stratified batches (pipeline.make_dataset(sampler='stratified'), 2026-10-06). Uniform
# sampling gives a 64-window batch 0.7 VT and 0.3 AVB3 windows on average (train seconds:
# SINUS 77 %, AFIB 15.6 %, SVT 4.5 %, VT / AVB2 1 %, AVB3 0.5 %): most batches carry no
# gradient for the rare classes and their +P swings between epochs and seeds. A window's
# category is the rarest arrhythmia with >= STRAT_MIN_SECONDS in it (priority order below),
# else SINUS; each batch takes STRAT_BATCH_QUOTA[category] windows (scaled to the batch size),
# every category cycling its own shuffled permutation. A category that would be repeated more
# than STRAT_MAX_REPEAT times per epoch gives the surplus quota to SINUS. The class weights are
# divided by sqrt(oversampling ratio) (train.class_weights) so the prior is not corrected
# twice, and the decode prior scale follows the EFFECTIVE weights (train_config.json).
SAMPLER = os.environ.get("ECGR_RHYTHM_SAMPLER", "stratified")      # 'uniform' | 'stratified'
STRAT_PRIORITY = ['VT', 'AVB', 'SVT', 'AFIB']
STRAT_MIN_SECONDS = 2.0
STRAT_BATCH_QUOTA = {'SINUS': 24, 'AFIB': 12, 'SVT': 10, 'VT': 7, 'AVB': 11}   # AVB = 6 + 5
STRAT_MAX_REPEAT = 4.0
# Learning-rate schedule: 'plateau' (ReduceLROnPlateau, the runs before 2026-10-06) or
# 'cosine' (1 warm-up epoch, cosine to LR_FLOOR_FRACTION x lr, deterministic).
LR_SCHEDULE = os.environ.get("ECGR_RHYTHM_LR_SCHEDULE", "cosine")
LR_WARMUP_EPOCHS = 1
LR_FLOOR_FRACTION = 0.02
WEIGHT_DECAY = 1e-4
PATIENCE = 8
MONITOR = 'val_rhythm_f1'        # macro F1 over the six classes, per second
# First epoch (1-indexed) allowed to write a checkpoint - best_model.keras and epochs/*.keras.
# 6 = epochs 1-5 are measured and logged but never saved, and EarlyStopping only starts
# counting after them, so an early, still-noisy peak cannot become "best".
CKPT_START_EPOCH = int(os.environ.get("ECGR_RHYTHM_CKPT_START_EPOCH", 6))

# Backbone resolution: 2500 samples -> 250 steps (40 ms). The label grid is 1 s, so 20 ms would
# only cost compute; 40 ms still resolves a P wave and an R-R interval to the millisecond scale
# rhythm classification needs.
BACKBONE_STEPS = 250

# Rhythm loss on a NOISY second is down-weighted, not dropped: the rhythm underneath is known,
# but asking the model to state it with full confidence out of pure artefact teaches it to
# invent one. The lead output is what says "this window cannot be read" (NOISE).
NOISY_SECOND_WEIGHT = 0.25
LEAD_LOSS_WEIGHT = 0.5
NOISE_LOSS_WEIGHT = 0.5          # 'noise' output of the 20 ms family
CHANNEL_LOSS_WEIGHT = 0.5        # 'channel' output of the dual U-Net (NOISE / CH1..3 per 2 s)
# Per-class weights for the rhythm CE. None = computed from the train split's label counts
# ((median / count) ** 0.5, clipped to [0.2, 5]) and stored in the run's manifest.
CLASS_WEIGHTS = None
LABEL_SMOOTHING = 0.05

# Decoding (rhythm/labels.decode_episodes), applied in this order:
#   1. moving average of each class's probabilities over DECODE_SMOOTH_SECONDS[class] (odd;
#      1 = off; SINUS untouched) - or one width for all when it is a number
#   2. argmax; a second whose window has p(NOISE) > DECODE_NOISE_THRESHOLD is NOISE
#   3. gap bridging, one class at a time in DECODE_PRIORITY order: two episodes of a class at
#      most DECODE_MERGE_GAP_SECONDS apart, with nothing but SINUS / NOISE / lower-priority
#      classes in between, become one episode. A higher-priority class is never overwritten.
#   4. an episode shorter than DECODE_MIN_EPISODE_SECONDS takes the class of its neighbours
#      when both agree (a 2 s SVT blip inside AFIB is AFIB), otherwise SINUS
#   5. bridging again - step 4 can leave two same-class episodes touching
# Values from `ecgr.rhythm ec57 --sweep` on mitdb/afdb with rhythm_eval as the guard (README
# 12, 2026-09-29, UNet500 probabilities):
#   * smoothing per class - 5 s for the persistent rhythms, 1 s for the runs: mitdb's reference
#     VT episodes have a median length of 1.8 s (51 of 60 under 3 s), SVTA 2.4 s; a 2 s run
#     averaged over 5 s falls under the sinus around it and is gone before any minimum applies.
#   * minimums in BEATS, not seconds: "3 beats" of VT at 150-200 bpm is ~1 s. The earlier
#     3 s / 7 s rule removed 51/60 VT, 16/26 SVTA and 25/83 AFIB reference episodes of mitdb
#     by construction (VT episode Se was 16-18 % with duration Se 60 %). AVB2 goes UP to 6 s:
#     every (BII episode is >= 7 s, the false AVB2 calls around pauses are short.
#   * no gap bridging: it lowered the objective for every smoothing/minimum pair (52.7 vs
#     55.2) and gave nothing on rhythm_eval.
#   Physionet objective 51.4 -> 55.2 (mitdb VT episode F1 20.6 -> 47.2, AFIB episode 68.5 ->
#   71.9); rhythm_eval episode F1 SVT +3.4, VT +9.5, AVB3 +1.5, AFIB -0.6.
# The noise gate never fires on Physionet (p_noise > 0.5 on ~0.01 % of seconds) and is kept
# for real Holter data. Earlier defaults: smoothing 5 s for all, gaps {5,2,2,3,3}, minimums
# {AFIB 7, SVT 3, VT 3, AVB2 2, AVB3 2} (ec57.USER_MIN_SECONDS / USER_MERGE_GAP, still in the
# sweep as the 'user' preset); before that {3,1,1,2,3} with no smoothing or bridging.
# CURRENT DEFAULTS (2026-09-30) - chosen on the VALIDATION records only (`ec57 --sweep
# --sweep-on validation`: held-out ltafdb 24 h records + nsrdb + incartdb, none of them EC57),
# for checkpoint 300926_rhythm_u500_phys: no smoothing (smoothing the AFIB column spreads AF
# over the 1-2 s VT runs inside it), the pre-sweep minimums, no bridging, prior correction
# alpha = 0.5. Validation objective 41.3 (previous defaults) -> 48.3; ltafdb VT F1 29.1/27.3
# -> 46.0/48.0. The validation set holds no AV block: AVB2/AVB3 simply keep this row's values.
# The 2026-09-29 row above (smoothing 5 s for AFIB/AVB, minimums {4, 1.5, 1, 6, 3}) had been
# picked on mitdb/afdb themselves - i.e. on the test set; kept in ec57.MIN_SETS / SMOOTH_SETS.
DECODE_SMOOTH_SECONDS = 1
DECODE_NOISE_THRESHOLD = 0.5     # p(NOISE) of the 'lead'/'noise' output above which = noise
DECODE_PRIORITY = ['VT', 'SVT', 'AFIB', 'AVB']
DECODE_MERGE_GAP_SECONDS = {'AFIB': 3, 'SVT': 0, 'VT': 0, 'AVB': 0}
DECODE_MIN_EPISODE_SECONDS = {'AFIB': 3, 'SVT': 1, 'VT': 1, 'AVB': 2}     # AVB2's old value
# Per-class multipliers on the probabilities before smoothing/argmax, renormalised (prior
# correction). The rhythm CE is class-weighted (manifest class_weights, SINUS 0.22 .. AVB3 2.5),
# which multiplies a class's posterior odds by w_c / w_SINUS - up to x11 for the rare classes -
# and so over-calls them on data where they are 20x rarer than in training (mitdb VT 0.3 %).
# w_c ** -alpha undoes it (alpha = 1 exactly). These are alpha = 0.5 for the class weights of
# the 2026-10-01 build (+ challenge-2020; ec57.prior_scale(0.5)) - RECOMPUTE after a rebuild
# changes the weights.
DECODE_CLASS_SCALE = {'SINUS': 2.24, 'AFIB': 1.75, 'SVT': 1.49, 'VT': 1.21, 'AVB': 1.22}
# ^ 2026-10-07: AVB takes AVB2's value; a 5-class model must re-derive the whole row from its
# own effective class weights (ec57.prior_scale) on the validation sweep.
# ^ 2026-10-06: alpha 0.5 of the EFFECTIVE class weights of rhythm_unet1250b_1mw
# (061026_rhythm_u1250b_v2, stratified sampler lowered the rare-class weights). For the
# uniform-sampler checkpoints (rhythm_unet1250_1m) the old values were
# {'SINUS': 2.24, 'AFIB': 1.54, 'SVT': 1.13, 'VT': 0.78, 'AVB2': 0.77, 'AVB3': 0.64}.
# Minimum mean probability (of the episode's own class, after the prior correction) for an
# episode to survive; below it the episode is folded like a too-short one. {} = off.
# AFIB floor with a 5 s AFIB gap bridge: chosen on the validation records with a false-episode
# budget of 0.4 per non-AF hour (the rate mitdb's target PPV of 92 % allows: ~7 false episodes
# in 18 h). 0.8 for plain inference (validation AF episode Se 71 -> 64 %, false AF episodes 2.5
# -> 0.24 per hour); 0.7 with the 5 s hop + TTA below, whose averaged probabilities are
# smoother. False AF episodes are the low-confidence ones on mitdb and ltafdb alike.
# Re-checked for checkpoint 011026_rhythm_u500_c2020: 0.6 would be inside the budget by a hair
# (0.40 / h) for +3.3 points of episode Se at 4.6x the false episodes (78 vs 17) - 0.7 kept at
# the knee of that trade, decided on the validation table alone.
# rhythm_unet1250_1m (011026_rhythm_u1250, 8 ms output): the validation table picks AFIB gap 3 s
# and floor 0.6 - AF episode Se 71.5 %, duration Se 92.9 %, 0.15 false AF episodes per non-AF
# hour, well inside the 0.4 budget.
DECODE_MIN_EPISODE_PROB = {'AFIB': 0.6}

# Beat-level rhythm post-processing (rhythm/beats.py) - the production pipeline's second stage
# (docs/rhythm-post-process-analysis.md of the Bioflux library) re-done on this model's own
# beat decoder, with its listed defects fixed: minimum beat counts enforced (A), neighbour
# windows closed on both sides (B), gaps tested on the rhythm not the valid mask (C),
# non-matching beats inside VT/SVT take the surrounding rhythm instead of SINUS (D), an
# explicit merge priority (E), an absolute long-invalid threshold instead of 1/4 record (H),
# VT / SVT need a fast rate so slow idioventricular runs are not VT (J). Applied when the stored
# npz carries beats (models with a 'beat' output) and DECODE_BEAT_POSTPROCESS is on.
DECODE_BEAT_POSTPROCESS = os.environ.get("ECGR_RHYTHM_BEAT_PP", "1") != "0"
BEAT_PICK_THRESHOLD = 0.5            # p(beat) = 1 - p(none) a local maximum must reach
BEAT_REFRACTORY_SECONDS = 0.2        # two beats cannot be closer than this (300 bpm)
BEAT_PP_CRITERIA = {
    # duration = (n - 1) R-R intervals in seconds, as the production code counts it
    'AFIB': dict(duration=3.0),
    'VT': dict(num_beat=3, run=3, min_hr=100.0),          # run = consecutive V beats
    'SVT': dict(num_beat=3, run=3, min_hr=100.0, min_frac=0.3, onset_ratio=1.25),
    'AVB': dict(duration=2.5, max_hr=60.0),          # AVB2 and AVB3 had the same rule
    'SINUS': dict(duration=8.0),     # shorter sinus islands get merged (validation grid 2026-10-06: 8 > 5 > 3)
}
BEAT_PP_LONG_INVALID_SECONDS = 10.0  # an invalid stretch at least this long becomes SINUS
BEAT_PP_FAST_RR_SECONDS = 0.6        # an N beat inside SVT may stay if its R-R is this short
BEAT_PP_AFIB_SVT_MERGE = False       # the production AFib/SVT window arbitration (off: tuned on validation)
BEAT_PP_SVT_RATIO = 2.0              # ... SVT keeps the window when SVT time >= ratio x AF time
BEAT_PP_PRIORITY = ['AFIB', 'SINUS', 'NOISE']   # merge target among the non-spec classes
# A run of V (S) beats that meets the VT (SVT) criteria on its own becomes VT (SVT) even where
# the step track said nothing - the production VES_RUN / SVES_RUN beat events, which its EC57
# export counts as VT (SVES_RUN is commented out there). Chosen on validation (tune.py).
BEAT_PP_RUNS_TO_RHYTHM = {'VT': False, 'SVT': False}
# Beat symbols for bxb: S inside AFIB -> N (the reference databases label no S in AF), and an
# S beat outside SVT is kept only when its R-R is shorter than this fraction of the median of
# the preceding intervals (0 = no prematurity gate). Chosen on validation (tune.py).
BEAT_PP_S_IN_AFIB_TO_N = True
BEAT_PP_S_PREMATURITY = 0.0
RHYTHM_BEAT_AI_EXTENSION = 'bti'     # hypothesis beat annotation (N/S/V at R) for bxb

# Whole-record inference (predict.predict_signal): window hop (10 = back-to-back windows; 5 =
# every step voted by two windows) and test-time variants averaged ('id', 'flip', 'swap').
# The model input stays one 10 s window either way. Env overrides for the validation sweep.
# 5 s + all three variants, chosen on the validation records (2026-10-01, checkpoint
# 300926_rhythm_u500_af): AF episode Se 62 -> 67 %, duration Se 92 %, false AF episodes 0.28
# -> 0.10 per non-AF hour (with DECODE_MIN_EPISODE_PROB AFIB 0.7, re-picked for it); ltafdb VT
# duration F1 -4. Costs 6x the inference of the plain setting.
PREDICT_HOP_SECONDS = float(os.environ.get("ECGR_RHYTHM_PREDICT_HOP", 5))
PREDICT_TTA = tuple(os.environ.get("ECGR_RHYTHM_PREDICT_TTA", "id,flip,swap").split(','))
# Where windows overlap, each window's rows are weighted by their position before averaging:
# weight = floor + (1 - floor) * sin(pi * (row + 0.5) / rows) (a Hann-like taper, 1 at the
# centre). 0 = off (plain mean, every table before 2026-10-08). XAI on the dual U-Net
# (2026-10-08): per-second macro F1 92.9 at the window centre vs 86 at either edge (SVT/VT
# -13..-20 pts), so a second near a window edge should defer to a window where it is central.
# Combine with a smaller ECGR_RHYTHM_PREDICT_HOP so every second has a central window.
PREDICT_TAPER_FLOOR = float(os.environ.get("ECGR_RHYTHM_PREDICT_TAPER", 0))

# Per-sample models ('rhythm' output (SEGMENT_SAMPLES, NUM_CLASSES), the UNet family): whole-
# record probabilities are kept, and decoded, at SAMPLE_PROBS_HZ steps per second - the mean of
# every 10 samples. 40 ms is finer than any rhythm boundary a reviewer marks, and a 10 h afdb
# record stays ~11 MB of float16 instead of ~110 MB at 250 Hz. Per-second models stay at 1 Hz.
SAMPLE_PROBS_HZ = 25
# Grid of the per-class EC57 annotation files, reference AND hypothesis (wfdb_ann.class_codes).
# 1 = whole seconds, which is what every table so far was scored on; raise it (e.g. 25) to let
# sub-second episode boundaries count - then re-score the per-second models on the same grid.
EC57_GRID_HZ = int(os.environ.get("ECGR_RHYTHM_EC57_GRID_HZ", 1))

# ---------------------------------------------------------------------------
# EC57 rhythm-episode evaluation (rhythm/ec57.py, evaluation/epicmp.py)
# ---------------------------------------------------------------------------
# `epicmp -A` scores ONE rhythm: the episodes spelled '(AFIB' (plus '(AFL' in the reference,
# which it knows). So every class is scored on its own pair of annotation files in which
# that class is spelled '(AFIB' and everything else '(N' - the reference project's dict_ext
# branch. Extensions are letters only (wfdb-python rejects '_' and digits, hence the roman
# numerals): reference r<class>, hypothesis a<class>, e.g. 201.rafib / 201.aafib,
# 231.ravbii / 231.aavbii.
RHYTHM_REF_EXTENSION = 'rhy'     # rhythm_eval: all classes in one file (human-readable)
RHYTHM_AI_EXTENSION = 'rhi'      # prediction, all classes in one file (human-readable)
RHYTHM_PROBS_EXTENSION = 'npz'   # raw per-second probabilities, so decoding can be re-run
_EXTENSION_STEM = {'AVB2': 'avbii', 'AVB3': 'avbiii'}    # legacy names; 'AVB' -> ravb / aavb


def class_extensions(name):
    stem = _EXTENSION_STEM.get(name, name.lower())
    return f"r{stem}", f"a{stem}"


# Physionet .atr rhythm codes -> class. Strict on purpose: AFIB is '(AFIB' only ('(AFL' stays
# flutter, epicmp handles it in the reference), SVT is '(SVTA' only ('(NOD', '(J', '(PREX'
# are not SVT), VT is '(VT' only ('(VFL', '(IVR' are not), AVB2 is '(BII', AVB3 is escdb's
# '(B3' (mitdb has no third-degree block). Any other code, '(N' included, is SINUS.
PHYSIONET_AUX_TO_CLASS = {'(AFIB': 'AFIB', '(SVTA': 'SVT', '(VT': 'VT', '(BII': 'AVB',
                          '(B3': 'AVB'}

# Which (class, database) pairs are scored by default. mitdb/afdb for AFIB and mitdb for
# SVT/VT/AVB2 are the reference product's table; escdb is a free extra (191 (VT, 22 (SVTA
# episodes). nstdb/ahadb carry no scorable rhythm and are off by default (--dbs adds them).
# The rhythm_eval holdout is scored on every class. A database with zero reference episodes
# of a class reports '-' for Se, not an error.
# AVB is scored against '(BII' + '(B3' together; on mitdb (no third-degree block) that is
# exactly the old AVB2 cell, so the product's AVB2 target still applies.
EC57_CLASSES = ['AFIB', 'SVT', 'VT', 'AVB']
EC57_CLASS_DBS = {'AFIB': ['mitdb', 'afdb'], 'SVT': ['mitdb', 'escdb'],
                  'VT': ['mitdb', 'escdb'], 'AVB': ['mitdb']}
EC57_DEFAULT_DBS = ['mitdb', 'afdb', 'escdb']
# Records left out of the rhythm scoring by default (`ec57 --include-paced` keeps them): the
# four paced mitdb records, which EC57 itself excludes from beat scoring. Their rhythm is
# paced throughout, no class of ours, and every second called AFIB / VT on them is a false
# positive with no true positive to be had.
EC57_EXCLUDE_RECORDS = {'mitdb': ['102', '104', '107', '217']}
# Atrial flutter counts as AF (project convention, 2026-09-29: AF/AFL are reported together).
# Training: PTB-XL AFLT is labelled AFIB. Scoring: the AFIB epicmp pass runs with -x, which
# excludes the reference's (AFL from the AFIB +P comparison (EC38:1998) - an AFIB call during
# flutter is not a false positive. Reference AFL is still not required for Se. False = the
# EC57 / EC38:2007 default (no exclusion: AFIB during flutter is a false positive).
EC57_AFL_AS_AF = os.environ.get("ECGR_RHYTHM_EC57_AFL_AS_AF", "1") != "0"
# The cells the decoding sweep optimises (mean F1 over Duration + Episode of each).
EC57_TARGET_CELLS = [('mitdb', 'AFIB'), ('afdb', 'AFIB'), ('mitdb', 'SVT'), ('mitdb', 'VT'),
                     ('mitdb', 'AVB')]

WORKERS = base.WORKERS

# ---------------------------------------------------------------------------
# Derived layout
# ---------------------------------------------------------------------------
NPY_DIR = os.environ.get("ECGR_RHYTHM_NPY_DIR",
                         os.path.join(WORK_DIR, f"npy_{SAMPLING_RATE}hz_{SEGMENT_SECONDS}s_"
                                                f"{IN_CHANNELS}lead_{OUTPUT_SECONDS}sec"))
RUN_DIR = os.path.join(WORK_DIR, RUN_TAG)
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
REPORT_DIR = os.path.join(RUN_DIR, "eval")
LOGS_DIR = os.path.join(RUN_DIR, "logs")
EC57_DIR = os.path.join(RUN_DIR, "ec57")


def describe():
    return "\n".join([
        f"run tag      : {RUN_TAG}",
        f"data root    : {DATA_ROOT}",
        f"test holdout : {EVAL_DIR}",
        f"npy          : {NPY_DIR}",
        f"run dir      : {RUN_DIR}",
        f"input        : ({SEGMENT_SAMPLES}, {IN_CHANNELS}) = {SEGMENT_SECONDS} s @ "
        f"{SAMPLING_RATE} Hz, 3 leads",
        f"output       : rhythm ({OUTPUT_SECONDS}, {NUM_CLASSES}) = per second "
        f"{CLASS_NAMES} softmax",
        f"               lead ({NUM_LEAD_CLASSES},) = per window {LEAD_CLASSES} softmax",
        f"sources      : {TRAIN_SOURCES} (PTB-XL: {PTBXL_DIR}, folds "
        f"{PTBXL_FOLDS}, plain cap {PTBXL_PLAIN_CAP})",
        f"augment      : permute p={PERMUTE_LEADS_PROB}, flip p={FLIP_LEAD_PROB}, "
        f"drop p={LEAD_DROP_PROB}, noise p={NOISE_PROB} "
        f"SNR {NOISE_SNR_DB_MIN:g}..{NOISE_SNR_DB_MAX:g} dB; a lead is readable at >= "
        f"{CLEAN_SNR_DB:g} dB, NOISE = best lead < {LEAD_MIN_READABLE_SECONDS}/10 s readable",
        f"monitor      : {MONITOR}, checkpoints saved from epoch {CKPT_START_EPOCH} "
        f"(epochs 1-{CKPT_START_EPOCH - 1} measured only)",
    ])
