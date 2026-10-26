"""ec57_confusion.py <key> : EC57 confusion matrices of a scored checkpoint (summary_111026/ec57/<key>)
on the same span the EC57 tools score (bxb / epicmp default: from 5 min to the record end):
  rhythm : per second, reference = .atr rhythm marks ((AFL -> AFIB, unmapped -> SINUS),
           hypothesis = the official decode of the stored npz (= the files epicmp scored)
  beat   : reference .atr (afdb: .qrs) vs hypothesis .bti, 150 ms matching, N/S/V + missed / false;
           mitdb paced records excluded (EC57)
-> <key>/ec57_confusion.json"""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, '/media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead')
import wfdb
from ecgr import config as base
from ecgr.rhythm import config as rc, ec57, wfdb_ann
key = sys.argv[1]
OUT = f'/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/ec57/{key}'
PH = os.environ.get('ECGR_PHYSIONET_DIR', '/media/Project/ECG/PhysionetData/')
C = rc.CLASS_NAMES; K = len(C); START = 300; TOL = rc.BEAT_MATCH_TOLERANCE_SECONDS; M = {'N': 0, 'S': 1, 'V': 2}
res = {}
for db in ['mitdb', 'nstdb', 'ahadb', 'escdb', 'afdb']:
    ann = f'{OUT}/_ann/{db}'
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{ann}/*.npz'))
    R = np.zeros((K, K), np.int64); B = np.zeros((4, 4), np.int64); nb = 0
    for n in names:
        src = os.path.join(PH, db, n)
        st = ec57.load_probs(ann, n)
        if st is None or not os.path.exists(src + '.hea'): continue
        r, p, fs, sig_len, hz = st
        eps, _ = ec57.decode_record(r, p, hz, ec57.load_beats(ann, n))
        n_s = sig_len // fs
        hyp = np.zeros(n_s, int)
        for e in eps:
            if e['rhythm'] in C: hyp[int(round(e['start'])):int(round(e['stop']))] = C.index(e['rhythm'])
        if os.path.exists(src + '.atr'):
            reps, _fs, _n = wfdb_ann.atr_reference_episodes(src)
            ref = np.zeros(n_s, int)
            for e in reps:
                c = 'AFIB' if e['rhythm'] == 'AFL' else e['rhythm']
                ref[int(np.floor(e['start'])):int(np.ceil(e['stop']))] = C.index(c) if c in C else 0
            np.add.at(R, (ref[START:], hyp[START:]), 1)
        # beats
        if n in base.EC57_EXCLUDE_RECORDS.get(db, []): continue
        rext = base.EC57_BEAT_REF_EXT.get(db, 'atr'); hb = f'{ann}/{n}.{rc.RHYTHM_BEAT_AI_EXTENSION}'
        if not (os.path.exists(f'{src}.{rext}') and os.path.exists(hb)): continue
        ra = wfdb.rdann(src, rext); ha = wfdb.rdann(hb[:-4], rc.RHYTHM_BEAT_AI_EXTENSION)
        a0 = START * fs
        rs = np.array([(s, rc.BEAT_SYMBOL_TO_CLASS[y] - 1) for s, y in zip(ra.sample, ra.symbol) if y in rc.BEAT_SYMBOL_TO_CLASS and s >= a0]).reshape(-1, 2)
        hs = np.array([(s, M[y]) for s, y in zip(ha.sample, ha.symbol) if y in M and s >= a0]).reshape(-1, 2)
        nb += 1
        # greedy in time order (sorted, two pointers): nearest unused hypothesis within tol
        used = np.zeros(len(hs), bool); j = 0; tol = TOL * fs
        for s, c in rs:
            while j < len(hs) and hs[j, 0] < s - tol: j += 1
            best, bj = tol + 1, -1; k = j
            while k < len(hs) and hs[k, 0] <= s + tol:
                if not used[k] and abs(hs[k, 0] - s) < best: best, bj = abs(hs[k, 0] - s), k
                k += 1
            if bj < 0: B[c, 3] += 1
            else: used[bj] = True; B[c, hs[bj, 1]] += 1
        for k in np.flatnonzero(~used): B[3, hs[k, 1]] += 1
    res[db] = dict(records=len(names), beat_records=nb, rhythm_seconds=R.tolist(), beat=B.tolist(),
                   beat_ref_ext=base.EC57_BEAT_REF_EXT.get(db, 'atr'))
    print(db, len(names), 'records; rhythm s', int(R.sum()), '; beats', int(B[:3].sum()), flush=True)
json.dump(res, open(f'{OUT}/ec57_confusion.json', 'w'), indent=1)
print('EC57_CM_DONE', flush=True)
