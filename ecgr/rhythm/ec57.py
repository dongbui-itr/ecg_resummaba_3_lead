"""Rhythm-level evaluation: sweep records, store probabilities, decode, score with epicmp.

The rhythm task's counterpart of ecgr/evaluation/ec57.py, scored against real WFDB rhythm
annotations instead of the model's own stored npy/tfrecord test split (evaluate.py).

Two kinds of source:

  * the Physionet EC57 databases - mitdb, afdb and escdb ship real rhythm-change marks in
    their .atr aux_notes ((AFIB, (SVTA, (VT, (BII, (B3, ...); nstdb/ahadb do not and are off
    by default (rc.EC57_DEFAULT_DBS, rc.EC57_CLASS_DBS say which class is scored where).
  * the rhythm task's own holdout (rc.EVAL_DIR) - NOT WFDB-annotated for rhythm. Its ground
    truth is the label spans build.py/labels already turn every window's reviewer marks into,
    so the reference annotations are synthesized (wfdb_ann.write_reference_annotations)
    instead of read from a shipped .atr.

Three stages, each re-runnable on its own:

  1. inference  - every record's per-second probabilities go to <ec57_out>/_ann/<db>/<name>.npz
                  (rhythm (T, 6) and p_noise (T,) in float16, fs, sig_len). This is the only
                  expensive stage and it runs once per checkpoint.
  2. decoding   - labels.decode_episodes on the npz (rc.DECODE_*), written as the readable
                  all-class .rhi plus one '(AFIB vs (N' hypothesis file PER CLASS
                  (wfdb_ann.write_class_annotations), next to the per-class reference files
                  built the same way from the .atr / the synthesized labels. --decode-only
                  redoes this stage and the next without the model.
  3. epicmp     - one `epicmp -A` pass per (database, class) in a disposable symlink farm
                  <ec57_out>/_work/<db>/<class>/, report <db>/<db>_<class>_report_line.out.
                  `epicmp -A` scores nothing but the rhythm spelled '(AFIB', which is why the
                  files are per class - one mixed-code file scored AFIB only.

--sweep repeats stages 2-3 over a grid of decoding parameters (sweep_decoding).
"""
import itertools
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import wfdb

# Detect IDE environment
_is_pycharm = 'PYCHARM_HOSTED' in os.environ or 'PYCHARM_MATPLOTLIB_INTERACTIVE' in os.environ

if _is_pycharm:
    # PyCharm: adjust import paths
    import sys
    from pathlib import Path
    _project_root = Path(__file__).parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from ecgr import config as base_config
    from ecgr.evaluation import epicmp, report
    from ecgr.rhythm import config as rc
    from ecgr.rhythm import inventory, wfdb_ann
    from ecgr.rhythm.build import read_leads
    from ecgr.rhythm.labels import (decode_episodes, decode_track, episodes_from_track,
                                    to_current_classes, to_current_weights)
    from ecgr.rhythm import beats as beats_pp
    from ecgr.rhythm.predict import predict_signal, probs_step_hz
else:
    # VS Code & others: keep current relative imports
    from .. import config as base_config
    from ..evaluation import epicmp, report
    from . import config as rc
    from . import inventory, wfdb_ann
    from .build import read_leads
    from .labels import (decode_episodes, decode_track, episodes_from_track,
                         to_current_classes, to_current_weights)
    from . import beats as beats_pp
    from .predict import predict_signal, probs_step_hz

# The reference product's table ("rhythm 3.0.6"), (Se, PPV), printed under our numbers.
TARGET_TABLE = {
    ('mitdb', 'AFIB'): {'D': (98, 90), 'E': (80, 92)},
    ('afdb', 'AFIB'): {'D': (96, 99), 'E': (81, 93)},
    ('mitdb', 'SVT'): {'D': (61, 24), 'E': (62, 19)},
    ('mitdb', 'VT'): {'D': (75, 57), 'E': (82, 58)},
    ('mitdb', 'AVB'): {'D': (98, 87), 'E': (100, 100)},      # = the AVB2 cell: mitdb has no (B3
}

RHYTHM_EVAL = 'rhythm_eval'


def annotation_dir(ec57_out, db_name):
    return os.path.join(ec57_out, '_ann', db_name)


def classes_for_db(db_name, classes=None):
    """The classes scored on `db_name`: rc.EC57_CLASS_DBS, every class for rhythm_eval and
    for a database no class lists (nstdb/ahadb via --dbs: Se comes out '-')."""
    classes = classes or rc.EC57_CLASSES
    if db_name == RHYTHM_EVAL:
        return list(classes)
    if db_name in rc.VALIDATION_CLASS_DBS:
        return [c for c in classes if c in rc.VALIDATION_CLASS_DBS[db_name]]
    return [c for c in classes if db_name in rc.EC57_CLASS_DBS.get(c, [])] or list(classes)


# ---------------------------------------------------------------------------
# Stage 1: inference -> npz
# ---------------------------------------------------------------------------

def probs_path(ann_dir, name):
    return os.path.join(ann_dir, f"{name}.{rc.RHYTHM_PROBS_EXTENSION}")


def save_probs(ann_dir, name, rhythm, p_noise, fs, sig_len, step_hz=1, beats=None):
    """beats (beats.pick_beats dict) are stored alongside when the model has a beat output;
    load_probs keeps its 5-tuple, load_beats reads them back."""
    os.makedirs(ann_dir, exist_ok=True)
    extra = {}
    if beats is not None:
        extra = dict(beat_t=np.asarray(beats['t'], np.float32),
                     beat_cls=np.asarray(beats['cls'], np.int8),
                     beat_conf=np.asarray(beats['conf'], np.float16),
                     beat_probs=np.asarray(beats['probs'], np.float16))
    np.savez(probs_path(ann_dir, name), rhythm=np.asarray(rhythm, np.float16),
             p_noise=np.asarray(p_noise, np.float16), fs=int(fs), sig_len=int(sig_len),
             step_hz=int(step_hz), **extra)


def load_beats(ann_dir, name):
    """The stored beats {t, cls, conf, probs} of a record, or None (no npz / no beat output)."""
    path = probs_path(ann_dir, name)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        if 'beat_t' not in z.files:
            return None
        return dict(t=z['beat_t'].astype(np.float64), cls=z['beat_cls'].astype(int),
                    conf=z['beat_conf'].astype(np.float32),
                    probs=z['beat_probs'].astype(np.float32))


def load_probs(ann_dir, name):
    """(rhythm (T, NUM_CLASSES) float32, p_noise (T,) float32, fs, sig_len, step_hz), or None
    if not stored. step_hz = rows per second; npz files written before it existed are per
    second; the 6-column ones of the AVB2/AVB3 models are merged to the five classes."""
    path = probs_path(ann_dir, name)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return (to_current_classes(z['rhythm'].astype(np.float32)),
                z['p_noise'].astype(np.float32),
                int(z['fs']), int(z['sig_len']),
                int(z['step_hz']) if 'step_hz' in z.files else 1)


def predict_record_probs(model, path, batch_size=None):
    """One WFDB record -> (rhythm, p_noise, fs, sig_len, step_hz, beats); fs is the record's
    own, beats None for a model without a beat output."""
    header = wfdb.rdheader(path)
    leads, _ratio = read_leads(path)
    rhythm, p_noise, _windows, beats = predict_signal(model, leads, batch_size=batch_size or 64,
                                                      with_beats=True)
    return rhythm, p_noise, header.fs, header.sig_len, probs_step_hz(model), beats


def predict_record_episodes(model, path, batch_size=None):
    """One WFDB record -> (episodes, fs, length), decoded with the rc.DECODE_* defaults."""
    rhythm, p_noise, fs, sig_len, step_hz, beats = predict_record_probs(
        model, path, batch_size=batch_size)
    return decode_record(rhythm, p_noise, step_hz, beats)[0], fs, sig_len


def decode_record(rhythm, p_noise, step_hz, beats=None, decode=None):
    """Stage 2 of one record: labels.decode_track -> (beat stage, when beats are stored and
    rc.DECODE_BEAT_POSTPROCESS / decode['beat_pp'] allow) -> episodes. Returns (episodes,
    beat symbols or None). `decode` = labels.decode_episodes keyword arguments plus the
    optional 'beat_pp' (bool) and 'beat_criteria' (rc.BEAT_PP_CRITERIA-shaped dict)."""
    decode = dict(decode or {})
    beat_pp = decode.pop('beat_pp', rc.DECODE_BEAT_POSTPROCESS)
    criteria = decode.pop('beat_criteria', None)
    options = decode.pop('beat_options', None)      # beats.default_options-shaped dict
    if len(rhythm) == 0:
        return [], None
    cls, _probs = decode_track(rhythm, p_noise, step_hz=step_hz, **decode)
    symbols = None
    if beats is not None and beat_pp:
        cls, symbols = beats_pp.postprocess(cls, beats, step_hz, criteria, options)
    elif beats is not None:
        symbols = np.asarray(beats['cls'], dtype=int)
    return episodes_from_track(cls, rhythm, p_noise, step_hz), symbols


# ---------------------------------------------------------------------------
# Stage 2: decoding -> per-class annotation files
# ---------------------------------------------------------------------------

def write_hypothesis(ann_dir, name, classes, decode=None):
    """Decode the stored probabilities of one record and write its all-class .rhi plus one
    hypothesis file per class. `decode` = keyword arguments of labels.decode_episodes.
    Returns the episodes, or None when the record has no npz yet."""
    stored = load_probs(ann_dir, name)
    if stored is None:
        return None
    rhythm, p_noise, fs, sig_len, step_hz = stored
    beats = load_beats(ann_dir, name)
    episodes, symbols = decode_record(rhythm, p_noise, step_hz, beats, decode)
    wfdb_ann.write_episode_annotations(episodes, name, ann_dir, rc.RHYTHM_AI_EXTENSION, fs)
    if beats is not None:
        wfdb_ann.write_beat_annotations(beats['t'], symbols, name, ann_dir,
                                        rc.RHYTHM_BEAT_AI_EXTENSION, fs)
    for cls in classes:
        _ref, hyp = rc.class_extensions(cls)
        wfdb_ann.write_class_annotations(episodes, cls, name, ann_dir, hyp, fs, sig_len // fs)
    return episodes


def write_physionet_reference(src_dir, ann_dir, name, classes):
    """Per-class reference files of one Physionet record from its .atr rhythm marks. The
    AFIB reference keeps '(AFL' (EC57 default, no -x): epicmp knows the code."""
    episodes, fs, n_seconds = wfdb_ann.atr_reference_episodes(os.path.join(src_dir, name))
    for cls in classes:
        ref, _hyp = rc.class_extensions(cls)
        wfdb_ann.write_class_annotations(episodes, cls, name, ann_dir, ref, fs, n_seconds,
                                         keep_afl=True)
    return episodes


def write_eval_reference(ev, ann_dir, name, length, fs, classes):
    """Per-class reference files of one rhythm_eval event, plus the readable .rhy."""
    from .labels import reference_episodes
    labels = wfdb_ann.synth_reference_labels(ev, length, fs)
    episodes = reference_episodes(labels)
    wfdb_ann.write_episode_annotations(episodes, name, ann_dir, rc.RHYTHM_REF_EXTENSION, fs)
    for cls in classes:
        ref, _hyp = rc.class_extensions(cls)
        wfdb_ann.write_class_annotations(episodes, cls, name, ann_dir, ref, fs, length // fs)
    return episodes


# ---------------------------------------------------------------------------
# Stage 3: epicmp, one pass per class
# ---------------------------------------------------------------------------

def _link(src, dst):
    if not os.path.islink(dst) and not os.path.exists(dst):
        os.symlink(src, dst)


def build_class_scoring_dir(work_dir, ann_dir, records, cls, link_record):
    """Disposable symlink farm for ONE class: the records (via `link_record(name, work_dir)`,
    which puts the .hea/.dat in place) and that class's reference/hypothesis pair from
    ann_dir. Returns the record names that had both files."""
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    ref, hyp = rc.class_extensions(cls)
    scored = []
    for name in records:
        r = os.path.join(ann_dir, f"{name}.{ref}")
        h = os.path.join(ann_dir, f"{name}.{hyp}")
        if not (os.path.exists(r) and os.path.exists(h)):
            continue
        link_record(name, work_dir)
        _link(r, os.path.join(work_dir, f"{name}.{ref}"))
        _link(h, os.path.join(work_dir, f"{name}.{hyp}"))
        scored.append(name)
    return scored


def epicmp_flags(cls):
    """-x on the AFIB pass when flutter counts as AF (rc.EC57_AFL_AS_AF): the reference's
    (AFL is left out of the AFIB +P comparison. The other passes spell AFL as '(N' anyway."""
    return ('-x',) if cls == 'AFIB' and rc.EC57_AFL_AS_AF else ()


def score_classes(db_name, ec57_out, records, classes, link_record, script, quiet=False):
    """Run epicmp once per class of `db_name`; the classes run in parallel (one work dir
    each, one report each). Returns {class: report path or None}."""
    ann_dir = annotation_dir(ec57_out, db_name)
    os.makedirs(os.path.join(ec57_out, db_name), exist_ok=True)
    jobs = {}
    for cls in classes:
        work_dir = os.path.join(ec57_out, '_work', db_name, cls)
        scored = build_class_scoring_dir(work_dir, ann_dir, records, cls, link_record)
        if not scored:
            print(f"  {db_name}/{cls}: nothing to score")
            continue
        jobs[cls] = work_dir
    if not quiet and jobs:
        print(f"  scoring {len(records)} records x {sorted(jobs)}")

    def one(cls):
        ref, hyp = rc.class_extensions(cls)
        return epicmp.run_epicmp(db_name, jobs[cls], ec57_out, ref, hyp, script=script,
                                 label=cls, quiet=True, extra_flags=epicmp_flags(cls))

    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        paths = dict(zip(jobs, pool.map(one, jobs)))
    return paths


def score_beats(db_name, ec57_out, records, link_record, ref_dir, script=None, quiet=False):
    """AAMI beat scoring (bxb + sumstats, QRS / VEB / SVEB Se and +P) of the stored beat
    hypotheses (.RHYTHM_BEAT_AI_EXTENSION) against the records' beat reference: .atr, or the
    database's own extension from config.EC57_BEAT_REF_EXT - afdb keeps its beats in .qrs
    (its .atr holds only rhythm marks, against which every beat would be a false positive).
    Only for the PhysioNet databases (rhythm_eval has no beat reference). Returns the QRS
    report dir or None."""
    from ..evaluation import bxb
    ref_ext = base_config.EC57_BEAT_REF_EXT.get(db_name, 'atr')
    ann_dir = annotation_dir(ec57_out, db_name)
    work_dir = os.path.join(ec57_out, '_work', db_name, 'beats')
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    scored = []
    for name in records:
        h = os.path.join(ann_dir, f"{name}.{rc.RHYTHM_BEAT_AI_EXTENSION}")
        if not os.path.exists(h):
            continue
        ref = os.path.join(ref_dir, f"{name}.{ref_ext}")
        if not os.path.exists(ref):
            continue
        link_record(name, work_dir)
        _link(ref, os.path.join(work_dir, f"{name}.{ref_ext}"))
        _link(h, os.path.join(work_dir, f"{name}.{rc.RHYTHM_BEAT_AI_EXTENSION}"))
        scored.append(name)
    if not scored:
        return None
    if not quiet:
        print(f"  bxb beats: {len(scored)} records (reference .{ref_ext})")
    report_root = os.path.join(ec57_out, 'beats')
    bxb.run_bxb(db_name, work_dir, report_root, ref_ext, rc.RHYTHM_BEAT_AI_EXTENSION,
                script=script or bxb.SCRIPT_FULL)
    line = os.path.join(report_root, db_name, f"{db_name}_QRS_report_line.out")
    if os.path.exists(line) and not quiet:
        from ..evaluation.report import parse_report
        row = parse_report(line)
        print(f"  beats {db_name}: " + ' '.join(f"{k}={v}" for k, v in row.items()))
    return os.path.join(report_root, db_name)


def _print_db_rows(ec57_out, db_name):
    rows = [r for r in report.episode_rows(ec57_out) if r['db'] == db_name]
    for r in rows:
        print(f"  {db_name:12s} {r['class']:5s} records={r.get('records', '?'):>5s}  "
              + "  ".join(f"{m}={r.get(m, '?')}" for m in report.EPISODE_COLUMNS[3:]))


# ---------------------------------------------------------------------------
# Physionet EC57 databases
# ---------------------------------------------------------------------------

def excluded_records(db_name, include_excluded=False):
    """The records rc.EC57_EXCLUDE_RECORDS leaves out of `db_name` (none with the flag)."""
    return set() if include_excluded else set(rc.EC57_EXCLUDE_RECORDS.get(db_name, []))


_VALIDATION_RECORDS = {}


def physionet_records(db_name, max_records=None, include_excluded=False):
    """(source dir, record names). A training database outside EC57 (rc.PHYSIONET_TRAIN_DBS)
    gives only its EVAL-side records - the validation set; its train records are never
    scored here."""
    if db_name in rc.PHYSIONET_TRAIN_DBS:
        from .physionet_train import eval_records
        if not _VALIDATION_RECORDS:
            _VALIDATION_RECORDS.update(eval_records())
        records = _VALIDATION_RECORDS.get(db_name, [])
        src_dir = rc.PHYSIONET_TRAIN_DBS[db_name]['dir']
        return src_dir, records[:max_records] if max_records else records
    src_dir = os.path.join(base_config.PHYSIONET_DIR, db_name)
    if not os.path.isdir(src_dir):
        return src_dir, []
    skip = excluded_records(db_name, include_excluded)
    # a record without a reference annotation (nstdb's noise-only bw / em / ma) cannot be scored
    records = sorted(f[:-4] for f in os.listdir(src_dir)
                     if f.endswith('.dat') and f[:-4] not in skip
                     and os.path.exists(os.path.join(src_dir, f[:-4] + '.atr')))
    return src_dir, records[:max_records] if max_records else records


def physionet_linker(src_dir):
    def link_record(name, work_dir):
        for ext in ('hea', 'dat'):
            path = os.path.join(src_dir, f"{name}.{ext}")
            if os.path.exists(path):
                _link(path, os.path.join(work_dir, f"{name}.{ext}"))
    return link_record


def decode_physionet_db(db_name, ec57_out, records=None, classes=None, decode=None,
                        references=True, quiet=False, include_excluded=False):
    """Stage 2 for one database: hypothesis files for every stored record (and the per-class
    references from .atr). Returns the records that have stored probabilities."""
    src_dir, all_records = physionet_records(db_name, include_excluded=include_excluded)
    records = records if records is not None else all_records
    ann_dir = annotation_dir(ec57_out, db_name)
    classes = classes_for_db(db_name, classes)
    ready = []
    for name in records:
        if write_hypothesis(ann_dir, name, classes, decode) is None:
            continue
        if references:
            write_physionet_reference(src_dir, ann_dir, name, classes)
        ready.append(name)
    if not quiet and len(ready) < len(records):
        print(f"  {len(records) - len(ready)} records have no stored .npz yet - run without "
              f"--decode-only to predict them")
    return ready


def score_physionet_rhythm_db(model, db_name, ec57_out, max_records=None, decode_only=False,
                              classes=None, decode=None, include_excluded=False):
    src_dir, records = physionet_records(db_name, max_records, include_excluded)
    if not records:
        print(f"database not found or empty: {src_dir}")
        return None
    print(f"===== rhythm {db_name}: {len(records)} records, classes "
          f"{classes_for_db(db_name, classes)} =====")
    skip = sorted(excluded_records(db_name, include_excluded))
    if skip:
        print(f"  excluded (rc.EC57_EXCLUDE_RECORDS, --include-paced keeps them): {skip}")

    ann_dir = annotation_dir(ec57_out, db_name)
    os.makedirs(ann_dir, exist_ok=True)
    if decode_only:
        print(f"  --decode-only: reusing the stored probabilities in {ann_dir}")
    else:
        for i, name in enumerate(records, 1):
            try:
                *probs, fs, sig_len, step_hz, beats = predict_record_probs(
                    model, os.path.join(src_dir, name))
                save_probs(ann_dir, name, *probs, fs, sig_len, step_hz, beats)
                if i % 20 == 0 or i == len(records):
                    print(f"  {i}/{len(records)} records predicted (last: {name}, "
                          f"{sig_len // fs} s)")
            except Exception as e:
                print(f"  error on {name}: {e}")

    ready = decode_physionet_db(db_name, ec57_out, records, classes, decode,
                                include_excluded=include_excluded)
    if not ready:
        print(f"{db_name}: nothing to score")
        return None
    paths = score_classes(db_name, ec57_out, ready, classes_for_db(db_name, classes),
                          physionet_linker(src_dir), epicmp.SCRIPT_FULL)
    _print_db_rows(ec57_out, db_name)
    try:
        score_beats(db_name, ec57_out, ready, physionet_linker(src_dir), src_dir)
    except Exception as e:                       # beat scoring never blocks the rhythm report
        print(f"  beat scoring skipped: {e}")
    return paths


# ---------------------------------------------------------------------------
# The rhythm task's own holdout (rc.EVAL_DIR) - synthesized reference
# ---------------------------------------------------------------------------

def eval_record_name(ev):
    """Flat, collision-free name: rhythm_eval events are nested study/event folders whose
    record files are not uniquely named on their own (see evaluation.ec57.split_record_name,
    same problem for the portal beat splits)."""
    return f"{ev['study_id']}_{ev['event_id']}"


def _write_scoring_header(src_hea, dst_hea, name):
    """Rewrite one rhythm_eval .hea under its flat scoring name - only the record name and
    the .dat file name change, every signal-line field is kept verbatim (same approach as
    evaluation.ec57.write_scoring_header, without the mark-window comments this task does not
    use)."""
    with open(src_hea, errors='replace') as f:
        lines = [line.rstrip('\n') for line in f]
    head = lines[0].split()
    nsig = int(head[1])
    head[0] = name
    out = [' '.join(head)]
    for line in lines[1:1 + nsig]:
        tokens = line.split()
        tokens[0] = f"{name}.dat"
        out.append(' '.join(tokens))
    out += lines[1 + nsig:]
    with open(dst_hea, 'w') as f:
        f.write('\n'.join(out) + '\n')


def eval_events(max_records=None):
    events = [ev for ev in inventory.collect_test()
              if ev.get('record') and os.path.exists(ev['record'] + '.hea')]
    return events[:max_records] if max_records else events


def eval_linker(events):
    paths = {eval_record_name(ev): ev['record'] for ev in events}

    def link_record(name, work_dir):
        path = paths[name]
        _write_scoring_header(path + '.hea', os.path.join(work_dir, name + '.hea'), name)
        if os.path.exists(path + '.dat'):
            _link(path + '.dat', os.path.join(work_dir, name + '.dat'))
    return link_record


def decode_rhythm_eval(events, ec57_out, classes=None, decode=None, references=True,
                       quiet=False):
    """Stage 2 for the holdout. Returns the flat names that have stored probabilities."""
    ann_dir = annotation_dir(ec57_out, RHYTHM_EVAL)
    classes = classes_for_db(RHYTHM_EVAL, classes)
    ready = []
    for ev in events:
        name = eval_record_name(ev)
        stored = load_probs(ann_dir, name)
        if stored is None:
            continue
        _r, _p, fs, sig_len, _step = stored
        write_hypothesis(ann_dir, name, classes, decode)
        if references:
            write_eval_reference(ev, ann_dir, name, sig_len, fs, classes)
        ready.append(name)
    if not quiet and len(ready) < len(events):
        print(f"  {len(events) - len(ready)} events have no stored .npz yet")
    return ready


def score_rhythm_eval(model, ec57_out, max_records=None, decode_only=False, classes=None,
                      decode=None):
    events = eval_events(max_records)
    print(f"===== rhythm {RHYTHM_EVAL}: {len(events)} events, classes "
          f"{classes_for_db(RHYTHM_EVAL, classes)} =====")

    ann_dir = annotation_dir(ec57_out, RHYTHM_EVAL)
    os.makedirs(ann_dir, exist_ok=True)
    if decode_only:
        print(f"  --decode-only: reusing the stored probabilities in {ann_dir}")
    else:
        errors = 0
        for i, ev in enumerate(events, 1):
            name = eval_record_name(ev)
            try:
                *probs, fs, sig_len, step_hz, beats = predict_record_probs(model, ev['record'])
                save_probs(ann_dir, name, *probs, fs, sig_len, step_hz, beats)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  error on {name}: {e}")
            if i % 500 == 0 or i == len(events):
                print(f"  {i}/{len(events)} events predicted")
        if errors:
            print(f"  {errors} events failed")

    ready = decode_rhythm_eval(events, ec57_out, classes, decode)
    if not ready:
        print(f"{RHYTHM_EVAL}: nothing to score")
        return None
    # 10-60 s strips - far shorter than epicmp's default 5-minute start, hence SCRIPT_SHORT.
    paths = score_classes(RHYTHM_EVAL, ec57_out, ready, classes_for_db(RHYTHM_EVAL, classes),
                          eval_linker(events), epicmp.SCRIPT_SHORT)
    _print_db_rows(ec57_out, RHYTHM_EVAL)
    return paths


# ---------------------------------------------------------------------------
# Decoding sweep - stages 2-3 over a parameter grid, no inference
# ---------------------------------------------------------------------------

# The pre-sweep decode, kept as the sweep's reference row.
BASELINE_DECODE = dict(smooth_seconds=1, noise_threshold=0.5,
                       merge_gap={c: 0 for c in rc.CLASS_NAMES[1:]},
                       min_seconds={'AFIB': 3, 'SVT': 1, 'VT': 1, 'AVB': 2})
USER_MIN_SECONDS = {'AFIB': 7, 'SVT': 3, 'VT': 3, 'AVB': 2}
USER_MERGE_GAP = {'AFIB': 5, 'SVT': 2, 'VT': 2, 'AVB': 3}
# Minimum-duration presets. 'user' is the reviewed rule with "3 beats" read as 3 s; 'beats'
# reads 3 beats at tachycardia rates as ~1 s and lifts AVB2 to 6 s: mitdb's reference VT
# episodes have a median length of 1.8 s (51 of 60 under 3 s), its SVTA 2.4 s, while every
# (BII episode is >= 7 s and the false AVB2 ones are short. 'baseline' is the pre-sweep decode.
MIN_SETS = {'user': USER_MIN_SECONDS,
            'beats': {'AFIB': 4, 'SVT': 1.5, 'VT': 1, 'AVB': 6},
            'baseline': BASELINE_DECODE['min_seconds']}
# Smoothing presets: one width for every class, or 'split' - persistent rhythms over 5 s,
# runs of beats 1 s (labels.smooth_probs with a dict).
SMOOTH_SETS = {1: 1, 3: 3, 5: 5,
               'split': {'AFIB': 5, 'AVB': 5, 'SVT': 1, 'VT': 1}}
# The noise threshold is out of the grid: the gate fires on ~0.01 % of Physionet seconds.
# prior = alpha of the prior correction (rc.DECODE_CLASS_SCALE = class_weight ** -alpha).
DEFAULT_GRID = dict(smooth=[1, 3, 5, 'split'], noise_threshold=[0.5],
                    gap_scale=[0, 1], min_set=['user', 'beats', 'baseline'], prior=[0])
# First validation sweep (2026-09-30, smooth {5, split} x gap {0, 1} x min {user, beats} x
# prior): every point lost to the no-smoothing baseline - smoothing the AFIB column over 5 s
# spreads AF over the 1-2 s VT runs inside it (ltafdb VT episode F1 43.0 -> 30.8).
VALIDATION_GRID = dict(smooth=[1, 'split'], noise_threshold=[0.5], gap_scale=[0],
                       min_set=['baseline', 'beats'], prior=[0, 0.5, 1])


def prior_scale(alpha, train_config=None):
    """{class: class_weight ** -alpha}; {} for alpha 0. The weights are the EFFECTIVE ones
    the model was trained with: `train_config` (a train_config.json path, default the
    ECGR_RHYTHM_TRAIN_CONFIG environment variable - the stratified sampler lowers the rare
    classes' weights), else the training manifest's."""
    if not alpha:
        return {}
    train_config = train_config or os.environ.get('ECGR_RHYTHM_TRAIN_CONFIG')
    weights = None
    if train_config and os.path.exists(train_config):
        with open(train_config) as f:
            weights = json.load(f).get('class_weights')
    if weights is None:
        from .pipeline import manifest_class_weights, read_manifest
        weights = manifest_class_weights(read_manifest())
    weights = to_current_weights(weights)
    return {n: float(w) ** -float(alpha) for n, w in zip(rc.CLASS_NAMES, weights)}


def grid_decode(smooth, noise_threshold, gap_scale, min_set, prior=0):
    """One grid point -> decode_episodes keyword arguments."""
    return dict(smooth_seconds=SMOOTH_SETS[smooth], noise_threshold=float(noise_threshold),
                merge_gap={c: int(round(g * gap_scale)) for c, g in USER_MERGE_GAP.items()},
                min_seconds=dict(MIN_SETS[min_set]), class_scale=prior_scale(prior))


def decode_label(decode):
    gaps = ','.join(str(decode['merge_gap'][c]) for c in rc.CLASS_NAMES[1:])
    mins = ','.join(str(decode['min_seconds'][c]) for c in rc.CLASS_NAMES[1:])
    smooth = decode['smooth_seconds']
    if isinstance(smooth, dict):
        smooth = '/'.join(str(smooth.get(c, 1)) for c in rc.CLASS_NAMES[1:])
    scale = decode.get('class_scale') or {}
    prior = (' scale=' + '/'.join(f"{scale.get(c, 1):.2f}" for c in rc.CLASS_NAMES)) \
        if scale else ''
    return (f"smooth={smooth} thr={decode['noise_threshold']}{prior} "
            f"gap=[{gaps}] min=[{mins}]")


def score_decode(ec57_out, decode, dbs, classes=None, events=None, include_excluded=False):
    """Re-decode + re-score the given databases with one decoding; per-class references are
    left as they are (written by the full run). Returns {(db, class): row}."""
    for db in dbs:
        if db == RHYTHM_EVAL:
            events = eval_events() if events is None else events
            ready = decode_rhythm_eval(events, ec57_out, classes, decode, references=False,
                                       quiet=True)
            if ready:
                score_classes(RHYTHM_EVAL, ec57_out, ready, classes_for_db(RHYTHM_EVAL, classes),
                              eval_linker(events), epicmp.SCRIPT_SHORT, quiet=True)
        else:
            src_dir, _ = physionet_records(db)
            ready = decode_physionet_db(db, ec57_out, None, classes, decode, references=False,
                                        quiet=True, include_excluded=include_excluded)
            if ready:
                score_classes(db, ec57_out, ready, classes_for_db(db, classes),
                              physionet_linker(src_dir), epicmp.SCRIPT_FULL, quiet=True)
    return {(r['db'], r['class']): r for r in report.episode_rows(ec57_out) if r['db'] in dbs}


def objective(rows, cells=None):
    """Mean of E_F1 and D_F1 over the target cells that are scorable."""
    values = []
    for key in (cells or rc.EC57_TARGET_CELLS):
        r = rows.get(key)
        for m in ('E_F1', 'D_F1'):
            v = report._num(r.get(m)) if r else None
            if v is not None:
                values.append(v)
    return float(np.mean(values)) if values else float('nan')


def sweep_decoding(ec57_out, grid=None, dbs=None, classes=None, check_db=RHYTHM_EVAL,
                   check_top=2, final_dbs=None, include_excluded=False, cells=None):
    """Stages 2-3 for every point of `grid` on `dbs` (default: the databases of the target
    cells), one row per point in <ec57_out>/decode_sweep.csv; then the baseline and the
    `check_top` best points re-scored on `check_db` as a sanity check. Leaves the annotation
    files of `dbs` + `final_dbs` + `check_db` decoded with the best point and returns its
    decode kwargs."""
    import csv
    grid = grid or DEFAULT_GRID
    cells_target = cells or rc.EC57_TARGET_CELLS
    dbs = dbs or sorted({db for db, _ in cells_target})
    points = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    print(f"===== decoding sweep: {len(points)} points x {dbs} =====")

    cells = [(db, c) for db in dbs for c in classes_for_db(db, classes)]
    columns = list(grid) + ['objective'] + [f"{db}_{c}_{m}" for db, c in cells
                                            for m in ('E_F1', 'D_F1')]
    results = []
    for i, point in enumerate([None] + points):
        decode = BASELINE_DECODE if point is None else grid_decode(**point)
        rows = score_decode(ec57_out, decode, dbs, classes, include_excluded=include_excluded)
        obj = objective(rows, cells_target)
        record = {**(point or {k: 'baseline' for k in grid}), 'objective': f"{obj:.2f}"}
        for db, c in cells:
            r = rows.get((db, c), {})
            record[f"{db}_{c}_E_F1"] = r.get('E_F1', '-')
            record[f"{db}_{c}_D_F1"] = r.get('D_F1', '-')
        results.append((obj, decode, record))
        print(f"  [{i:3d}/{len(points)}] {obj:6.2f}  {decode_label(decode)}")

    path = os.path.join(ec57_out, 'decode_sweep.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(r for _, _, r in results)
    print(f"sweep -> {path}")

    ranked = sorted(results[1:], key=lambda r: -r[0] if r[0] == r[0] else float('inf'))
    best = ranked[0][1] if ranked else BASELINE_DECODE
    print("\nbest points:")
    for obj, decode, _ in ranked[:5]:
        print(f"  {obj:6.2f}  {decode_label(decode)}")
    print(f"  {results[0][0]:6.2f}  {decode_label(BASELINE_DECODE)}  (baseline)")

    if check_db and check_top:
        checks = [('baseline', BASELINE_DECODE)] + \
                 [(f"top{i + 1}", d) for i, (_, d, _) in enumerate(ranked[:check_top])]
        print(f"\nsanity check on {check_db}:")
        for tag, decode in checks:
            rows = score_decode(ec57_out, decode, [check_db], classes,
                                include_excluded=include_excluded)
            cells_txt = "  ".join(f"{c}: E={rows[(check_db, c)].get('E_F1', '-')} "
                                  f"D={rows[(check_db, c)].get('D_F1', '-')}"
                                  for c in classes_for_db(check_db, classes)
                                  if (check_db, c) in rows)
            print(f"  {tag:8s} {decode_label(decode)}\n           {cells_txt}")

    # Leave every database in the best decoding, so the summary and the files agree.
    every = list(dict.fromkeys(dbs + list(final_dbs or []) + ([check_db] if check_db else [])))
    score_decode(ec57_out, best, every, classes, include_excluded=include_excluded)
    print(f"\nbest decode: {decode_label(best)}")
    return best


# ---------------------------------------------------------------------------
# The whole evaluation
# ---------------------------------------------------------------------------

def run(checkpoint, tag, dbs=None, max_records=None, decode_only=False,
        skip_physionet=False, skip_rhythm_eval=False, classes=None, sweep=False,
        include_excluded=False, sweep_on='ec57', decode=None):
    """Score one rhythm checkpoint over the Physionet EC57 databases and rc.EVAL_DIR.
    `decode` = decode_record keyword overrides (e.g. {'beat_pp': False})."""
    if _is_pycharm:
        from ecgr.training.train import setup_gpus
    else:
        from ..training.train import setup_gpus
    setup_gpus()
    ec57_out = os.path.join(rc.EC57_DIR, tag)
    os.makedirs(ec57_out, exist_ok=True)
    dbs = dbs or rc.EC57_DEFAULT_DBS

    model = None
    if not decode_only:
        if not checkpoint:
            raise ValueError("--checkpoint is required unless --decode-only / --sweep")
        print(f"loading {checkpoint}")
        model = tf.keras.models.load_model(checkpoint, compile=False)
        model.summary()

    if not skip_physionet:
        for db in dbs:
            score_physionet_rhythm_db(model, db, ec57_out, max_records=max_records,
                                      decode_only=decode_only, classes=classes,
                                      decode=decode, include_excluded=include_excluded)
            print()

    if not skip_rhythm_eval:
        score_rhythm_eval(model, ec57_out, max_records=max_records, decode_only=decode_only,
                          classes=classes, decode=decode)
        print()

    if sweep:
        cells = rc.VALIDATION_TARGET_CELLS if sweep_on == 'validation' else rc.EC57_TARGET_CELLS
        target_dbs = {d for d, _ in cells}
        sweep_decoding(ec57_out, cells=cells,
                       grid=VALIDATION_GRID if sweep_on == 'validation' else None,
                       dbs=[db for db in dbs if db in target_dbs] or list(dbs),
                       classes=classes, check_db=None if skip_rhythm_eval else RHYTHM_EVAL,
                       final_dbs=[] if skip_physionet else list(dbs),
                       include_excluded=include_excluded)

    order = list(dbs) + ([] if skip_rhythm_eval else [RHYTHM_EVAL])
    return report.summarize_episodes(ec57_out, classes=classes or rc.EC57_CLASSES,
                                     dbs=[d for d in order if os.path.isdir(
                                         os.path.join(ec57_out, d))],
                                     target=TARGET_TABLE)

if __name__ == "__main__":
    import argparse

    # IDE-specific defaults
    if _is_pycharm:
        default_checkpoint = "/media/Project/ECG/Model_Dong/ecgr_rhythm/280926_rhythm_unet_sample/checkpoints/unetmamba_rhythm_1m/best_model.keras"
        default_tag = "pp_unet_s_debug"
        default_max_records = None  # PyCharm: limit for quick testing
        default_dbs = ['mitdb']  # PyCharm: single database for quick testing
        default_skip_physionet = False
        default_skip_rhythm_eval = False
        default_decode_only = False
        default_sweep = False
        print(f"Running in PyCharm - using debug defaults (max_records={default_max_records}, dbs={default_dbs})")
    else:
        # VS Code & others
        default_checkpoint = "/media/Project/ECG/Model_Dong/ecgr_rhythm/280926_rhythm_unet_sample/checkpoints/unetmamba_rhythm_1m/best_model.keras"
        default_tag = "pp_unet_s"
        default_max_records = None
        default_dbs = rc.EC57_DEFAULT_DBS
        default_skip_physionet = False
        default_skip_rhythm_eval = False
        default_decode_only = False
        default_sweep = False
        print("Running in VS Code or other environment - using standard defaults")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=default_checkpoint,
                        help="the rhythm model checkpoint to load")
    parser.add_argument('--tag', default=default_tag,
                        help="subdir of rc.EC57_DIR to write results")
    parser.add_argument('--dbs', nargs='+', default=default_dbs,
                        help="Physionet EC57 databases to score")
    parser.add_argument('--max-records', type=int, default=default_max_records,
                        help="limit the number of records per db")
    parser.add_argument('--decode-only', action='store_true', default=default_decode_only,
                        help="skip inference, re-decode and re-score the stored .npz")
    parser.add_argument('--skip-physionet', action='store_true', default=default_skip_physionet,
                        help="skip the Physionet EC57 databases")
    parser.add_argument('--skip-rhythm-eval', action='store_true', default=default_skip_rhythm_eval,
                        help="skip the rhythm task's own holdout (rc.EVAL_DIR)")
    parser.add_argument('--classes', nargs='+', choices=rc.CLASS_NAMES[1:],
                        default=rc.EC57_CLASSES,
                        help="subset of classes to score")
    parser.add_argument('--sweep', action='store_true', default=default_sweep,
                        help="run a decoding sweep over a grid of parameters")
    parser.add_argument('--include-excluded', action='store_true', default=False,
                        help="include the paced/otherwise excluded records in scoring")

    args = parser.parse_args()
    run(args.checkpoint, args.tag, args.dbs, args.max_records, args.decode_only,
        args.skip_physionet, args.skip_rhythm_eval, args.classes, args.sweep,
        args.include_excluded)
