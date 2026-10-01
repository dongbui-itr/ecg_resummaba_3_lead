#!/usr/bin/env python3
"""
ec57_record_audit.py — per-record audit of the five ANSI/AAMI EC57 databases
(MIT-BIH Arrhythmia, MIT-BIH Noise Stress Test, European ST-T, AHA, MIT-BIH AF)
for single-lead beat detection + N/S/V classification and AFIB detection.

For every record it reports:
  * AAMI class counts (N, S, V, F, Q) inside the EC57 evaluation window (default: after 5 min)
  * RR statistics, prematurity index of S / N beats and how well RR alone separates S from N (AUC)
  * runs (S couplets / SVTA, V couplets / VT), longest pause
  * rhythm episodes from aux_note (AFIB, AFL, SVTA, VT, NOD, BII, PREX, ...) and AF burden
  * morphology on the chosen lead: QRS width, R / P / T amplitude, P-wave consistency,
    template correlation S vs N and V vs N (QRS and P windows)
  * noise: baseline-wander and high-frequency noise level per 10 s window, estimated SNR
  * difficulty flags that explain where a model is likely to fail

Usage
  pip install wfdb numpy scipy pandas openpyxl
  # download from PhysioNet into ./data and audit (AHA is not on PhysioNet: give its local folder)
  python ec57_record_audit.py --download --data ./data --aha /path/to/ahadb --out ec57_audit.xlsx
  # or, if the databases are already on disk as WFDB files:
  python ec57_record_audit.py --data ./data --aha /path/to/ahadb --out ec57_audit.xlsx
  # restrict to some databases / records
  python ec57_record_audit.py --data ./data --db mitdb nstdb --records 232 209 118e00

Folder layout expected under --data:  <data>/mitdb, <data>/nstdb, <data>/edb, <data>/afdb
(the names used by wfdb.dl_database). AHA: any folder with WFDB-format .hea/.dat/.atr
(e.g. converted with the ahaconv tool of the AHA DVD).
"""
from __future__ import annotations
import argparse, os, re, sys, math, warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import signal as sps

try:
    import wfdb
except ImportError:  # pragma: no cover
    sys.exit("pip install wfdb")

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
AAMI = {  # ANSI/AAMI EC57 mapping of MIT-BIH beat symbols
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N', 'B': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V', 'r': 'V',
    'F': 'F',
    '/': 'Q', 'P': 'Q', 'f': 'Q', 'Q': 'Q', 'n': 'N',
}
BEAT_SYMBOLS = set(AAMI)
DBS = {
    'mitdb': dict(name='MIT-BIH Arrhythmia', ann='atr', start_s=300, paced={'102', '104', '107', '217'}),
    'nstdb': dict(name='MIT-BIH Noise Stress Test', ann='atr', start_s=300, paced=set()),
    'edb':   dict(name='European ST-T', ann='atr', start_s=300, paced=set()),
    'ahadb': dict(name='AHA', ann='atr', start_s=300, paced=set()),
    'afdb':  dict(name='MIT-BIH Atrial Fibrillation', ann='atr', beat_ann=('qrsc', 'qrs'), start_s=0, paced=set()),
}
DS1 = {'101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205',
       '207', '208', '209', '215', '220', '223', '230'}
DS2 = {'100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219',
       '221', '222', '228', '231', '232', '233', '234'}
LEAD_PREF = ['MLII', 'ML II', 'II', 'D2', 'MLIII', 'III', 'V5', 'V4', 'ECG1', 'ECG']


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def pick_lead(sig_names):
    names = [s.strip().upper() for s in sig_names]
    for p in LEAD_PREF:
        if p.upper() in names:
            return names.index(p.upper())
    return 0


def bandpass(x, fs, lo=0.5, hi=40.0, order=3):
    hi = min(hi, 0.45 * fs)
    b, a = sps.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return sps.filtfilt(b, a, x)


def auc_mannwhitney(pos, neg):
    """P(score_pos < score_neg) — S beats should have SMALLER prematurity index."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return np.nan
    if len(neg) > 20000:
        neg = np.random.default_rng(0).choice(neg, 20000, replace=False)
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().values
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return 1.0 - u / (len(pos) * len(neg))


def runs(labels, target):
    """lengths of consecutive runs of a class."""
    out, n = [], 0
    for l in labels:
        if l == target:
            n += 1
        else:
            if n:
                out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def rhythm_segments(ann, fs, n_samples):
    """(rhythm, start, end) from aux_note '(XXX' annotations."""
    segs, cur, cur_s = [], None, 0
    for s, a in zip(ann.sample, ann.aux_note or [''] * len(ann.sample)):
        a = (a or '').strip().rstrip('\x00')
        if a.startswith('('):
            if cur is not None:
                segs.append((cur, cur_s, s))
            cur, cur_s = a[1:], s
    if cur is not None:
        segs.append((cur, cur_s, n_samples))
    return segs


def templates(x, fs, peaks, max_beats=1500):
    pre, post = int(0.30 * fs), int(0.45 * fs)
    peaks = [p for p in peaks if p - pre >= 0 and p + post < len(x)]
    if len(peaks) > max_beats:
        peaks = list(np.random.default_rng(1).choice(peaks, max_beats, replace=False))
    if not peaks:
        return None
    M = np.stack([x[p - pre:p + post] for p in peaks])
    M = M - np.median(M[:, :int(0.05 * fs)], axis=1, keepdims=True)  # baseline at -300..-250 ms
    return M, pre


def morph_features(M, pre, fs):
    """features from a stack of aligned beats (rows)."""
    tpl = np.median(M, axis=0)
    t = (np.arange(len(tpl)) - pre) / fs
    qrs_win = (t >= -0.08) & (t <= 0.10)
    p_win = (t >= -0.25) & (t <= -0.09)
    t_win = (t >= 0.15) & (t <= 0.42)
    pr_base = np.median(tpl[(t >= -0.30) & (t <= -0.26)])
    r_amp = np.max(np.abs(tpl[qrs_win] - pr_base))
    # QRS width: span where |derivative| > 15 % of its max around R
    d = np.abs(np.gradient(tpl))
    dq = d.copy(); dq[~((t >= -0.12) & (t <= 0.14))] = 0
    thr = 0.15 * dq.max() if dq.max() > 0 else 0
    idx = np.where(dq > thr)[0]
    qrs_ms = (idx[-1] - idx[0]) / fs * 1000 if len(idx) > 1 else np.nan
    p_amp = np.max(np.abs(tpl[p_win] - pr_base)) if p_win.any() else np.nan
    t_amp = np.max(np.abs(tpl[t_win] - pr_base)) if t_win.any() else np.nan
    # P consistency: mean correlation of each beat's P window with the template P window
    if p_win.sum() > 4 and len(M) > 5:
        P = M[:, p_win]; Pt = tpl[p_win]
        Pc = P - P.mean(1, keepdims=True); Ptc = Pt - Pt.mean()
        den = np.linalg.norm(Pc, axis=1) * np.linalg.norm(Ptc) + 1e-12
        p_cons = float(np.median(Pc @ Ptc / den))
    else:
        p_cons = np.nan
    return dict(tpl=tpl, t=t, r_amp=r_amp, qrs_ms=qrs_ms, p_amp=p_amp, t_amp=t_amp,
                p_rel=p_amp / (r_amp + 1e-12), t_rel=t_amp / (r_amp + 1e-12), p_cons=p_cons,
                qrs_win=qrs_win, p_win=p_win)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def noise_stats(x_raw, fs, r_amp, win_s=10):
    n = int(win_s * fs)
    if len(x_raw) < n or not np.isfinite(r_amp) or r_amp <= 0:
        return dict(snr_db_median=np.nan, pct_win_snr_lt10=np.nan, bw_rel_median=np.nan, hf_rel_median=np.nan,
                    pct_flat=np.nan)
    b_lp, a_lp = sps.butter(2, 0.7 / (fs / 2), 'low')
    bw = sps.filtfilt(b_lp, a_lp, x_raw)
    hi = min(40.0, 0.45 * fs)
    b_hp, a_hp = sps.butter(3, hi / (fs / 2), 'high')
    hf = sps.filtfilt(b_hp, a_hp, x_raw)
    k = len(x_raw) // n
    bw_w = np.array([np.ptp(bw[i * n:(i + 1) * n]) for i in range(k)]) / r_amp
    hf_w = np.array([np.std(hf[i * n:(i + 1) * n]) for i in range(k)])
    snr = 20 * np.log10(r_amp / (4 * hf_w + 1e-12) + 1e-12)  # R amplitude vs ~peak-to-peak of HF noise
    flat = np.array([np.ptp(x_raw[i * n:(i + 1) * n]) < 0.02 * r_amp for i in range(k)])
    return dict(snr_db_median=float(np.median(snr)), pct_win_snr_lt10=float(100 * np.mean(snr < 10)),
                bw_rel_median=float(np.median(bw_w)), hf_rel_median=float(np.median(hf_w / r_amp)),
                pct_flat=float(100 * flat.mean()))


# --------------------------------------------------------------------------------------
# One record
# --------------------------------------------------------------------------------------
def audit_record(db, rec, folder, cfg):
    path = os.path.join(folder, rec)
    row = dict(db=db, record=rec)
    try:
        hdr = wfdb.rdheader(path)
    except Exception as e:
        row['error'] = f'header: {e}'; return row, None
    fs = hdr.fs; row['fs'] = fs; row['dur_min'] = round(hdr.sig_len / fs / 60, 1)
    row['leads'] = ','.join(hdr.sig_name or [])
    have_sig = os.path.exists(path + '.dat') or any(os.path.exists(path + e) for e in ('.16a', '.mat'))
    # annotations
    try:
        ann = wfdb.rdann(path, cfg['ann'])
    except Exception as e:
        ann = None; row['note'] = f'no {cfg["ann"]} ({e.__class__.__name__})'
    beat_ann = ann
    if 'beat_ann' in cfg:  # AFDB: beats in .qrsc (corrected) or .qrs (unaudited)
        beat_ann = None
        for ext in cfg['beat_ann']:
            try:
                beat_ann = wfdb.rdann(path, ext); row['beat_ann'] = ext; break
            except Exception:
                continue
    start = int(cfg['start_s'] * fs)
    n_samp = hdr.sig_len

    # rhythm
    rh = rhythm_segments(ann, fs, n_samp) if ann is not None else []
    dur = defaultdict(float); n_ep = Counter()
    for r, s, e in rh:
        s2, e2 = max(s, start), e
        if e2 > s2:
            dur[r] += (e2 - s2) / fs; n_ep[r] += 1
    eval_s = max(1.0, (n_samp - start) / fs)
    for r in ('AFIB', 'AFL', 'SVTA', 'VT', 'VFL', 'NOD', 'J', 'BII', 'PREX', 'B', 'T', 'SBR', 'P', 'AB', 'IVR'):
        if dur.get(r):
            row[f'rhythm_{r}_min'] = round(dur[r] / 60, 2); row[f'rhythm_{r}_episodes'] = n_ep[r]
    row['AF_burden_pct'] = round(100 * (dur.get('AFIB', 0) + dur.get('AFL', 0)) / eval_s, 2)
    row['rhythms'] = ','.join(sorted(dur))

    if beat_ann is None:
        return row, None
    sym = np.array(beat_ann.symbol); smp = np.array(beat_ann.sample)
    is_beat = np.array([s in BEAT_SYMBOLS for s in sym])
    sym, smp = sym[is_beat], smp[is_beat]
    if not len(smp):
        row['note'] = (row.get('note', '') + ' no beats').strip(); return row, None
    cls = np.array([AAMI[s] for s in sym])
    inwin = smp >= start
    row['n_beats'] = int(inwin.sum())
    for c in 'NSVFQ':
        row[f'n_{c}'] = int(((cls == c) & inwin).sum())
    for k, v in Counter(sym[inwin]).most_common():
        row[f'sym_{k}'] = v
    row['S_pct'] = round(100 * row['n_S'] / max(1, row['n_beats']), 2)
    row['V_pct'] = round(100 * row['n_V'] / max(1, row['n_beats']), 2)
    if db == 'mitdb':
        row['split'] = 'DS1' if rec in DS1 else 'DS2' if rec in DS2 else ('paced' if rec in cfg['paced'] else 'other')
    row['paced'] = rec in cfg['paced'] or row.get('n_Q', 0) > 0.3 * row['n_beats']

    # RR
    rr = np.diff(smp) / fs
    rr_pre = np.r_[np.nan, rr]; rr_post = np.r_[rr, np.nan]
    loc = pd.Series(rr_pre).shift(1).rolling(10, min_periods=3).median().values  # median of previous 10 RR
    pi = rr_pre / loc
    comp = rr_post / rr_pre
    row['HR_mean'] = round(60 / np.nanmean(rr), 1)
    row['RR_cv'] = round(float(np.nanstd(rr) / np.nanmean(rr)), 3)
    row['RR_max_s'] = round(float(np.nanmax(rr)), 2)
    row['n_pause_gt2s'] = int((rr > 2.0).sum())
    mN, mS, mV = (cls == 'N') & inwin, (cls == 'S') & inwin, (cls == 'V') & inwin
    row['PI_N_p5'] = round(float(np.nanpercentile(pi[mN], 5)), 3) if mN.sum() > 10 else np.nan
    row['PI_N_pct_lt0.85'] = round(100 * float(np.nanmean(pi[mN] < 0.85)), 2) if mN.sum() > 10 else np.nan
    if mS.sum():
        row['PI_S_median'] = round(float(np.nanmedian(pi[mS])), 3)
        row['PI_S_pct_lt0.85'] = round(100 * float(np.nanmean(pi[mS] < 0.85)), 1)
        row['comp_S_median'] = round(float(np.nanmedian(comp[mS])), 3)
        row['AUC_RR_S_vs_N'] = round(auc_mannwhitney(pi[mS], pi[mN]), 3)
        # S beats inside AF/AFL are unusual; N beats inside AF have irregular RR -> FP risk
    if rh:
        af_mask = np.zeros(n_samp + 1, bool)
        for r, s, e in rh:
            if r in ('AFIB', 'AFL'):
                af_mask[s:e] = True
        in_af = af_mask[np.clip(smp, 0, n_samp)]
        row['n_N_in_AF'] = int((mN & in_af).sum())
        row['PI_N_in_AF_pct_lt0.85'] = round(100 * float(np.nanmean(pi[mN & in_af] < 0.85)), 1) if (mN & in_af).sum() > 10 else np.nan
    lab = cls[inwin]
    sr, vr = runs(lab, 'S'), runs(lab, 'V')
    row['S_runs_ge2'] = sum(1 for r in sr if r >= 2); row['S_runs_ge3'] = sum(1 for r in sr if r >= 3)
    row['S_in_runs_pct'] = round(100 * sum(r for r in sr if r >= 2) / max(1, sum(sr)), 1) if sr else 0
    row['V_couplets'] = sum(1 for r in vr if r == 2); row['V_runs_ge3'] = sum(1 for r in vr if r >= 3)
    row['S_majority_over_N'] = bool(row['n_S'] > row['n_N'])

    tpl_out = None
    if have_sig:
        try:
            ch = pick_lead(hdr.sig_name); row['lead_used'] = hdr.sig_name[ch]
            sig = wfdb.rdrecord(path, channels=[ch]).p_signal[:, 0]
            sig = np.nan_to_num(sig)
            x = bandpass(sig, fs)
            feats = {}
            for c in 'NSV':
                m = (cls == c) & inwin
                if m.sum() >= 5:
                    T = templates(x, fs, smp[m])
                    if T is not None:
                        feats[c] = morph_features(*T, fs)
            if 'N' in feats:
                fN = feats['N']
                for k in ('qrs_ms', 'r_amp', 'p_rel', 't_rel', 'p_cons'):
                    row[f'N_{k}'] = round(float(fN[k]), 3)
                row.update(noise_stats(sig[start:], fs, fN['r_amp']))
            for c in ('S', 'V'):
                if c in feats:
                    fc = feats[c]
                    row[f'{c}_qrs_ms'] = round(float(fc['qrs_ms']), 1)
                    row[f'{c}_p_rel'] = round(float(fc['p_rel']), 3)
                    if 'N' in feats:
                        row[f'corrQRS_{c}_vs_N'] = round(corr(fc['tpl'][fc['qrs_win']], feats['N']['tpl'][feats['N']['qrs_win']]), 3)
                        row[f'corrP_{c}_vs_N'] = round(corr(fc['tpl'][fc['p_win']], feats['N']['tpl'][feats['N']['p_win']]), 3)
            tpl_out = {c: (f['t'], f['tpl']) for c, f in feats.items()}
        except Exception as e:
            row['note'] = (row.get('note', '') + f' signal: {e}').strip()
    else:
        row['note'] = (row.get('note', '') + ' no signal file').strip()
    row['flags'] = ';'.join(flags(row))
    return row, tpl_out


def flags(r):
    f = []
    g = lambda k, d=np.nan: r.get(k, d)
    if r.get('paced'): f.append('PACED(exclude)')
    if g('lead_used', 'MLII') not in ('MLII', 'ML II', 'II'): f.append(f"LEAD={g('lead_used','?')}")
    if g('AF_burden_pct', 0) > 5: f.append('AF/AFL')
    if r.get('S_majority_over_N'): f.append('S>N(prototype N unreliable)')
    if g('n_S', 0) >= 20 and g('PI_S_median', 0) > 0.85: f.append('S_not_premature')
    if g('n_S', 0) >= 20 and g('S_in_runs_pct', 0) > 30: f.append('S_in_runs(SVTA)')
    if g('corrQRS_S_vs_N', 0) > 0.95 and g('corrP_S_vs_N', 1) > 0.8 and g('n_S', 0) >= 20: f.append('S≈N_morph+P')
    if g('N_p_cons', 1) < 0.5: f.append('P_weak/absent')
    if g('PI_N_pct_lt0.85', 0) > 2: f.append('N_irregular_RR(FP_S_risk)')
    if g('RR_max_s', 0) > 3: f.append(f"pause>{g('RR_max_s')}s")
    if g('pct_win_snr_lt10', 0) > 10: f.append('noisy')
    if g('N_qrs_ms', 0) > 120: f.append('wide_QRS_baseline(BBB)')
    if g('corrQRS_V_vs_N', 0) > 0.9 and g('n_V', 0) >= 20: f.append('V≈N_morph')
    if g('V_runs_ge3', 0) > 0: f.append('VT_runs')
    if g('N_t_rel', 0) > 0.8: f.append('tall_T(FP_QRS_risk)')
    return f


# --------------------------------------------------------------------------------------
# Database loop
# --------------------------------------------------------------------------------------
def list_records(folder):
    rf = os.path.join(folder, 'RECORDS')
    if os.path.exists(rf):
        return [l.strip() for l in open(rf) if l.strip()]
    return sorted({os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith('.hea')})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='/media/Project_3/ECG/PhysionetData/', help='parent folder containing mitdb/ nstdb/ edb/ afdb/')
    ap.add_argument('--aha', default=None, help='folder with AHA records in WFDB format')
    ap.add_argument('--db', nargs='*', default=['mitdb', 'nstdb', 'edb', 'ahadb', 'afdb'])
    ap.add_argument('--records', nargs='*', default=None)
    ap.add_argument('--download', action='store_true', help='download PhysioNet databases into --data first')
    ap.add_argument('--start-s', type=float, default=None, help='override EC57 evaluation start (s)')
    ap.add_argument('--out', default='ec57_audit.xlsx')
    ap.add_argument('--plots', action='store_true', help='save per-record template plots (PNG) next to --out')
    a = ap.parse_args()

    rows, tpls = [], {}
    for db in a.db:
        cfg = dict(DBS[db])
        if a.start_s is not None:
            cfg['start_s'] = a.start_s
        folder = a.aha if db == 'ahadb' else os.path.join(a.data, db)
        if db == 'ahadb' and not folder:
            print('[skip] ahadb: pass --aha <folder> (AHA DB is not distributed on PhysioNet)'); continue
        if a.download and db != 'ahadb':
            os.makedirs(folder, exist_ok=True)
            print(f'[download] {db} -> {folder}')
            wfdb.dl_database(db, folder)
        if not os.path.isdir(folder):
            print(f'[skip] {db}: {folder} not found'); continue
        recs = list_records(folder)
        if a.records:
            recs = [r for r in recs if r in a.records]
        for rec in recs:
            print(f'  {db}/{rec}', end=' ', flush=True)
            row, tp = audit_record(db, rec, folder, cfg)
            rows.append(row)
            if tp:
                tpls[(db, rec)] = tp
            print(row.get('flags', row.get('error', row.get('note', ''))))

    if not rows:
        sys.exit('nothing audited')
    df = pd.DataFrame(rows)
    front = ['db', 'record', 'split', 'fs', 'dur_min', 'lead_used', 'n_beats', 'n_N', 'n_S', 'n_V', 'n_F', 'n_Q',
             'S_pct', 'V_pct', 'rhythms', 'AF_burden_pct', 'HR_mean', 'RR_cv', 'RR_max_s', 'PI_S_median',
             'PI_S_pct_lt0.85', 'AUC_RR_S_vs_N', 'PI_N_pct_lt0.85', 'S_in_runs_pct', 'N_qrs_ms', 'N_p_rel',
             'N_p_cons', 'N_t_rel', 'corrQRS_S_vs_N', 'corrP_S_vs_N', 'corrQRS_V_vs_N', 'snr_db_median',
             'pct_win_snr_lt10', 'flags']
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    df = df[cols]

    # database summaries
    summ = []
    for db, g in df.groupby('db'):
        g2 = g[~g.get('paced', pd.Series(False, index=g.index)).fillna(False).astype(bool)]
        d = dict(db=db, records=len(g), records_nonpaced=len(g2))
        for c in 'NSVFQ':
            if f'n_{c}' in g2:
                d[f'n_{c}'] = int(g2[f'n_{c}'].fillna(0).sum())
        if 'n_S' in g2 and d.get('n_S'):
            top = g2.dropna(subset=['n_S']).sort_values('n_S', ascending=False).head(3)
            d['S_top3'] = ', '.join(f"{r}:{int(n)} ({100*n/d['n_S']:.0f}%)" for r, n in zip(top.record, top.n_S))
        if 'n_V' in g2 and d.get('n_V'):
            top = g2.dropna(subset=['n_V']).sort_values('n_V', ascending=False).head(3)
            d['V_top3'] = ', '.join(f"{r}:{int(n)}" for r, n in zip(top.record, top.n_V))
        if 'AF_burden_pct' in g2:
            d['records_with_AF'] = int((g2.AF_burden_pct > 0).sum())
        # FP budget for S at target Se=P+=0.85 (and V at 0.96)
        if d.get('n_S'):
            nonS = d.get('n_N', 0) + d.get('n_V', 0) + d.get('n_F', 0)
            tp = 0.85 * d['n_S']; fp = tp * 0.15 / 0.85
            d['S_FP_budget@85/85'] = int(fp); d['S_FP_rate_needed_%'] = round(100 * fp / max(1, nonS), 3)
        if d.get('n_V'):
            d['V_FP_budget@96/96'] = int(0.96 * d['n_V'] * 0.04 / 0.96)
        summ.append(d)
    sdf = pd.DataFrame(summ)

    if 'mitdb' in df.db.values:
        m = df[df.db == 'mitdb']
        split = m.groupby('split')[[c for c in ('n_N', 'n_S', 'n_V', 'n_F', 'n_Q') if c in m]].sum().reset_index()
    else:
        split = pd.DataFrame()

    legend = pd.DataFrame([
        ('n_*', 'AAMI class counts inside the EC57 window (after start_s; 300 s for MIT-BIH/NST/ESC/AHA, 0 for AFDB)'),
        ('PI', 'prematurity index = RR_pre / median(previous 10 RR); S beats < 0.85 are "premature"'),
        ('AUC_RR_S_vs_N', 'how well the prematurity index alone separates S from N in this record (1 = perfect, 0.5 = useless)'),
        ('PI_N_pct_lt0.85', '% of N beats that look premature by RR alone -> false-positive S risk'),
        ('S_in_runs_pct', '% of S beats inside runs of >= 2 S (SVTA): only the first beat of a run is premature'),
        ('N_qrs_ms / S_qrs_ms', 'QRS width of the median beat (derivative threshold, approximate)'),
        ('N_p_rel / N_t_rel', 'P / T amplitude relative to R amplitude on the chosen lead'),
        ('N_p_cons', 'median correlation of each beat\'s P window with the template (< 0.5: P weak, absent or AF)'),
        ('corrQRS_S_vs_N, corrP_S_vs_N', 'template correlation between S and N beats in the QRS / P window (> 0.95: morphology gives no help)'),
        ('snr_db_median', 'R amplitude vs 4 x std of > 40 Hz noise, per 10 s window'),
        ('flags', 'difficulty flags explaining expected failure modes'),
    ], columns=['column', 'meaning'])

    with pd.ExcelWriter(a.out, engine='openpyxl') as xw:
        sdf.to_excel(xw, sheet_name='summary', index=False)
        if len(split):
            split.to_excel(xw, sheet_name='mitdb_DS1_DS2', index=False)
        for db, g in df.groupby('db'):
            g.dropna(axis=1, how='all').to_excel(xw, sheet_name=db[:31], index=False)
        legend.to_excel(xw, sheet_name='legend', index=False)
    print(f'\nwrote {a.out}  ({len(df)} records)')

    if a.plots and tpls:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        pdir = os.path.splitext(a.out)[0] + '_templates'; os.makedirs(pdir, exist_ok=True)
        for (db, rec), tp in tpls.items():
            plt.figure(figsize=(5, 3))
            for c, (t, y) in tp.items():
                plt.plot(t * 1000, y, label=c)
            plt.axvspan(-250, -90, alpha=.08); plt.xlabel('ms from R'); plt.title(f'{db}/{rec}'); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(pdir, f'{db}_{rec}.png'), dpi=110); plt.close()
        print(f'templates in {pdir}/')


if __name__ == '__main__':
    main()
