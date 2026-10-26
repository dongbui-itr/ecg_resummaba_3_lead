#!/usr/bin/env python3
"""Re-tune the decoding for a SMALL window overlap so it does not score below the 5 s overlap.

    python3 tune_overlap.py          # edit the CONFIG block first

Why: with PREDICT_OVERLAP_SECONDS = 1 most steps are predicted by ONE window instead of the
average of two. On EC57 / rhythm_eval the sensitivity stays, the positive predictivity drops:
more short low-confidence episodes and more borderline V beats (README / memory 2026-10-12).
The default decode was chosen at overlap 5 s; this script re-chooses the knobs that act on
exactly those errors, on the VALIDATION records only (never on EC57 / rhythm_eval):

    smooth_seconds   per-class moving average of the probabilities (labels.smooth_probs) -
                     the part of the two-window average a single window lost
    min_prob         per-class confidence floor of an episode (labels.enforce_min_confidence)
    v_margin         a V beat must beat max(p(N), p(S)) by this much (beats.relabel_v)

Steps (each reuses what is already on disk):
    1. inference     the validation records (ltafdb / nsrdb / incartdb / svdb eval side) at
                     BOTH overlaps -> <WORK_DIR>/<RUN_TAG>/ec57/<VAL_TAG_PREFIX>_ov<x>/_ann
    2. reference     the default decode at overlap REFERENCE_OVERLAP = the bar to reach
    3. grid          every GRID point on top of the default decode at overlap TARGET_OVERLAP
    4. choice        the best point that reaches the bar on EVERY validation cell (ltafdb AFIB /
                     SVT / VT episode + duration F1, beat V / S F1) within TOLERANCE and keeps
                     the AF false-episode rate within budget -> OUT_DECODE (a decode json for
                     evaluate.py DECODE)

AVB has no reference episode on the validation records, so it cannot be tuned here and keeps
the default decode.
"""
import itertools
import json
import multiprocessing
import os
import sys
import time

# ===========================================================================
# CONFIG - edit this block
# ===========================================================================

CHECKPOINT = 'checkpoints/dualunet_rhythm_1m_e26/best_model.keras'   # relative to the project root
WORK_DIR = 'eval_results'                            # git-ignored
RUN_TAG = 'dualunet_rhythm_1m_e26'
VAL_TAG_PREFIX = 'val'                               # -> val_ov5, val_ov1 under ec57/
PHYSIONET_DIR = '/media/Project/ECG/PhysionetData/'
VALIDATION_DBS = ['ltafdb', 'nsrdb', 'incartdb', 'svdb']

REFERENCE_OVERLAP = 0.5          # the setting whose scores must not drop
TARGET_OVERLAP = 1.0             # the setting being tuned
PREDICT_TTA = 'id,flip,swap'

# Grid on top of the default decode (config DECODE_* / BEAT_PP_*). None = keep the default.
GRID = dict(
    smooth=[None, {'AFIB': 3}, {'AFIB': 3, 'SVT': 3}, {'AFIB': 5}, 3],
    af_prob=[None, 0.7],                     # AFIB floor (default 0.6)
    svt_prob=[None, 0.5, 0.6],
    vt_prob=[None, 0.5, 0.6],
    v_margin=[0.0, 0.1, 0.2, 0.3],
    s_prem=[0.0],                            # S prematurity gate (beats.beat_symbols); 0 = off
)
TOLERANCE = 0.3                  # F1 points a cell may sit below the reference
AF_FP_BUDGET = 0.4               # false AF episodes per non-AF hour (or the reference's, if higher)
WORKERS = 12
GPU_MEMORY_MB = 2000             # cap: training jobs share the card; None = growth
GPU = None
OUT_DECODE = 'eval_results/tuned_decode_overlap1.json'   # relative to the project root
MAX_RECORDS = None               # records per validation db; None = all

# Debug run (TUNE_OVERLAP_DEBUG=1, see .vscode/launch.json): a few records, a 4-point grid and
# no worker pool, so a breakpoint in tune.score_decode / ec57.decode_record stops in this
# process. Its npz and decode go to their own tag / file and never mix with the full run's.
DEBUG = os.environ.get('TUNE_OVERLAP_DEBUG') == '1'
if DEBUG:
    MAX_RECORDS = int(os.environ.get('TUNE_OVERLAP_DEBUG_RECORDS', 2))
    WORKERS = 0
    VAL_TAG_PREFIX = 'dbg'
    GRID = dict(smooth=[None, {'AFIB': 3}], af_prob=[None], svt_prob=[None], vt_prob=[None],
                v_margin=[0.0, 0.2], s_prem=[0.0])
    OUT_DECODE = 'eval_results/tuned_decode_overlap1_debug.json'

# ===========================================================================
# End of CONFIG
# ===========================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def apply_environment():
    if GPU is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ['ECGR_PHYSIONET_DIR'] = PHYSIONET_DIR
    os.environ['ECGR_RHYTHM_WORK_DIR'] = resolve(WORK_DIR)
    os.environ['ECGR_RHYTHM_RUN_TAG'] = RUN_TAG
    os.environ['ECGR_RHYTHM_PREDICT_TTA'] = PREDICT_TTA


def val_tag(overlap):
    return f"{VAL_TAG_PREFIX}_ov{overlap:g}"


# --- step 1: inference -----------------------------------------------------------------------

def run_inference(overlap):
    """Validation inference at `overlap` unless every record already has its npz."""
    from ecgr.rhythm import config as rc
    from ecgr.rhythm import ec57
    tag = val_tag(overlap)
    out = os.path.join(rc.EC57_DIR, tag)
    todo = []
    for db in VALIDATION_DBS:
        _src, names = ec57.physionet_records(db, max_records=MAX_RECORDS)
        ann = ec57.annotation_dir(out, db)
        todo += [(db, n) for n in names if ec57.load_probs(ann, n) is None]
    if not todo:
        print(f"[{tag}] every validation record already has its npz - inference skipped")
        return out
    print(f"[{tag}] inference at overlap {overlap:g} s: {len(todo)} records to predict")
    rc.PREDICT_OVERLAP_SECONDS = float(overlap)          # read at call time by predict.py
    t0 = time.time()
    ec57.run(resolve(CHECKPOINT), tag, dbs=VALIDATION_DBS, max_records=MAX_RECORDS,
             skip_rhythm_eval=True)
    print(f"[{tag}] done in {(time.time() - t0) / 60:.0f} min")
    return out


# --- steps 2-4: scoring --------------------------------------------------------------------

def point_decode(pt):
    """A GRID point -> the full decode dict (defaults + the point's overrides)."""
    from ecgr.rhythm import config as rc
    d = {}
    if pt['smooth'] is not None:
        d['smooth_seconds'] = pt['smooth']
    mp = dict(rc.DECODE_MIN_EPISODE_PROB)
    for name, key in (('AFIB', 'af_prob'), ('SVT', 'svt_prob'), ('VT', 'vt_prob')):
        if pt[key] is not None:
            mp[name] = pt[key]
    if mp != rc.DECODE_MIN_EPISODE_PROB:
        d['min_prob'] = mp
    if pt.get('v_margin'):
        d['v_margin'] = pt['v_margin']
    if pt.get('s_prem'):
        d['beat_options'] = {'s_prematurity': pt['s_prem']}
    return d


def f1(t):
    se = t[0] / max(t[1], 1)
    pp = t[2] / max(t[3], 1)
    return 100 * 2 * se * pp / max(se + pp, 1e-9)


def cells(rows, beat_tot):
    """{cell name: F1} - the cells the choice must not lose on."""
    from ecgr.rhythm import config as rc
    out = {}
    for db, c in rc.VALIDATION_TARGET_CELLS:
        r = rows.get((db, c))
        if r and r['n']:
            out[f"{db} {c} E_F1"] = r['E_F1']
            out[f"{db} {c} D_F1"] = r['D_F1']
    out['beats V_F1'] = f1(beat_tot['V'])
    out['beats S_F1'] = f1(beat_tot['S'])
    out['beats QRS_F1'] = f1(beat_tot['QRS'])
    return out


def score_all(pool, ann_root, labelled_decodes):
    from ecgr.rhythm import tune
    recs = tune.validation_records(ann_root, tune.RHYTHM_DBS)
    beat_recs = tune.validation_records(ann_root, tune.BEAT_DBS)
    jobs = [(label, d, recs, beat_recs, ann_root) for label, d in labelled_decodes]
    return {label: (cells(rows, bt), fph)
            for label, rows, fph, bt in pool.map(tune.score_decode, jobs, chunksize=1)}


class SerialPool:
    """WORKERS = 0: Pool.map in this process (debugging - breakpoints in the scorer work)."""

    def map(self, fn, jobs, chunksize=None):
        return [fn(j) for j in jobs]

    def close(self):
        pass

    def join(self):
        pass


def fmt(c):
    return '  '.join(f"{k.replace(' ', ':')} {v:5.1f}" for k, v in c.items())


def main():
    apply_environment()
    if GPU_MEMORY_MB:
        from evaluate import limit_gpu
        limit_gpu(GPU_MEMORY_MB)
    from ecgr.rhythm import config as rc

    ref_root = os.path.join(run_inference(REFERENCE_OVERLAP), '_ann')
    tgt_root = os.path.join(run_inference(TARGET_OVERLAP), '_ann')

    if WORKERS:
        # numpy-only workers, no GPU (spawn: the parent holds a CUDA context)
        saved = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        pool = multiprocessing.get_context('spawn').Pool(WORKERS)
        if saved is None:
            del os.environ['CUDA_VISIBLE_DEVICES']
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = saved
    else:
        pool = SerialPool()
    try:
        ref_cells, ref_fph = score_all(pool, ref_root, [('ref', {})])['ref']
        base_cells, base_fph = score_all(pool, tgt_root, [('base', {})])['base']
        print(f"\nREFERENCE overlap {REFERENCE_OVERLAP:g} s, default decode  (AF FP/h {ref_fph:.2f})")
        print("   " + fmt(ref_cells))
        print(f"TARGET    overlap {TARGET_OVERLAP:g} s, default decode  (AF FP/h {base_fph:.2f})")
        print("   " + fmt(base_cells))

        points = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
        print(f"\ngrid: {len(points)} points x the validation records at overlap {TARGET_OVERLAP:g} s")
        t0 = time.time()
        res = score_all(pool, tgt_root, [(i, point_decode(p)) for i, p in enumerate(points)])
        print(f"grid scored in {(time.time() - t0) / 60:.1f} min")
    finally:
        pool.close()
        pool.join()

    budget = max(AF_FP_BUDGET, ref_fph)
    table = []
    for i, p in enumerate(points):
        c, fph = res[i]
        deficit = max(ref_cells[k] - c[k] for k in ref_cells)
        table.append((i, p, c, fph, deficit, sum(c.values())))
    ok = [t for t in table if t[4] <= TOLERANCE and t[3] <= budget]
    print(f"{len(ok)} of {len(table)} points reach the reference on every cell "
          f"(tolerance {TOLERANCE}, AF FP/h <= {budget:.2f})")
    pool_pick = ok or [t for t in table if t[3] <= budget] or table
    best = (max(pool_pick, key=lambda t: t[5]) if ok
            else min(pool_pick, key=lambda t: (t[4], -t[5])))
    print("\nTOP 10 by summed F1 among the passing points" if ok
          else "\nNO point reaches the reference everywhere - smallest worst-cell deficit:")
    for t in sorted(pool_pick, key=(lambda t: -t[5]) if ok else (lambda t: (t[4], -t[5])))[:10]:
        print(f"   #{t[0]:3d} sum {t[5]:6.1f} worst-deficit {t[4]:5.2f} FP/h {t[3]:.2f}  {t[1]}")

    i, p, c, fph, deficit, _ = best
    print(f"\nCHOSEN #{i}: {p}   AF FP/h {fph:.2f}, worst cell vs reference {-deficit:+.2f}")
    print(f"{'cell':22s}{'ov%g ref' % REFERENCE_OVERLAP:>10s}{'ov%g def' % TARGET_OVERLAP:>10s}"
          f"{'ov%g tuned' % TARGET_OVERLAP:>11s}")
    for k in ref_cells:
        print(f"{k:22s}{ref_cells[k]:10.1f}{base_cells[k]:10.1f}{c[k]:11.1f}")

    decode = point_decode(p)
    decode['_chosen'] = dict(point=p, overlap=TARGET_OVERLAP, reference_overlap=REFERENCE_OVERLAP,
                             reaches_reference=bool(ok), worst_deficit=round(deficit, 2),
                             af_fp_per_hour=round(fph, 3), cells=c, reference_cells=ref_cells)
    path = resolve(OUT_DECODE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(decode, f, indent=2, default=float)
    print(f"\ndecode -> {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
