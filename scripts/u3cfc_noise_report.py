#!/usr/bin/env python
"""How well a checkpoint's 'noise' head finds unreadable ECG on nstdb, against the TRUE SNR.

nstdb's noisy records (118e*, 119e*) are mitdb 118 / 119 with noise added - so nstdb minus
mitdb IS the noise, sample by sample, and the SNR of every 2 s segment the model judges can be
measured exactly instead of read off the record name. Checked 2026-10-20: the two files differ
by a DC offset that is constant inside each 2-minute block (5.12, 5.34, 4.90 mV ...), and by
nothing at all in the clean blocks once that offset is removed; noise is on in the 2-minute
blocks starting at minute 5, 9, 13, ... (5 min clean, then 2 on / 2 off).

SNR is nstdb's definition: signal = (QRS peak-to-peak)^2 / 8, the QRS p-p being the median
over the record's beats (mitdb reference annotations, +-60 ms around R) of each lead; noise =
the segment's noise RMS. A segment's SNR is the WORSE of the two leads - what the training
label calls readable (both live leads at >= the threshold; the third input lead is zero).

Prints, per true-SNR bin, the share of segments the head flags (p(NOISE) >= threshold), and
writes one CSV row per (record, window, segment).

    python scripts/u3cfc_noise_report.py --checkpoint best_model.keras [--threshold 0.5]
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import wfdb  # noqa: E402

from ecgr import config  # noqa: E402
from ecgr.evaluation import ec57  # noqa: E402
from ecgr.rhythm import model as rhythm_models  # noqa: E402,F401  registers custom layers
from ecgr.signal_ops import segment_record  # noqa: E402

RECORDS = [f'{r}e{s}' for r in ('118', '119') for s in ('24', '18', '12', '06', '00', '_6')]
BINS = [(-99, 0), (0, 3), (3, 6), (6, 9), (9, 12), (12, 18), (18, 99)]
BLOCK_MIN = 2.0
FIRST_NOISE_MIN = 5.0
SEG_SECONDS = 2


def qrs_pp(clean, ann, fs):
    """(n_leads,) median QRS peak-to-peak of the clean mitdb signal, +-60 ms around each R."""
    half = int(0.06 * fs)
    beats = [s for s, sym in zip(ann.sample, ann.symbol) if sym in 'NLRBAaJSVrFejnE/fQ?']
    pp = [np.ptp(clean[max(0, s - half):s + half], axis=0) for s in beats]
    return np.median(np.asarray(pp), axis=0)


def noise_signal(noisy, clean, fs):
    """nstdb - mitdb with the per-2-minute-block DC offset removed."""
    d = noisy - clean
    out = np.empty_like(d)
    edges = [0] + [int((FIRST_NOISE_MIN + BLOCK_MIN * k) * 60 * fs)
                   for k in range(int(30 / BLOCK_MIN) + 2)]
    edges = sorted({min(e, len(d)) for e in edges} | {len(d)})
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            out[a:b] = d[a:b] - d[a:b].mean(axis=0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--out-csv', default=None)
    a = p.parse_args()

    model = tf.keras.models.load_model(a.checkpoint, compile=False)
    if 'noise' not in model.output_names:
        raise SystemExit(f"{a.checkpoint} has no 'noise' output")
    fn = tf.function(lambda x: model(x, training=False)['noise'][..., 1])
    nst, mit = (os.path.join(config.PHYSIONET_DIR, d) for d in ('nstdb', 'mitdb'))

    rows = []
    for name in RECORDS:
        base = name[:3]
        noisy = wfdb.rdrecord(os.path.join(nst, name))
        clean = wfdb.rdrecord(os.path.join(mit, base))
        fs = noisy.fs
        n = min(noisy.sig_len, clean.sig_len)
        noise = noise_signal(noisy.p_signal[:n], clean.p_signal[:n], fs)
        ann = wfdb.rdann(os.path.join(mit, base), 'atr')
        s_ref = qrs_pp(clean.p_signal[:n], ann, fs) / (2 * np.sqrt(2))       # (2,)

        leads, _raw_len, _ = ec57.read_leads(os.path.join(nst, name), channel=0)
        segs, starts = segment_record(leads)
        pn = np.concatenate([np.asarray(fn(tf.constant(segs[i:i + 64])))
                             for i in range(0, len(segs), 64)])               # (W, 5)
        step = config.SAMPLING_RATE * SEG_SECONDS
        for w, st in enumerate(starts):
            for j in range(pn.shape[1]):
                a0 = int(round((st + j * step) * fs / config.SAMPLING_RATE))
                a1 = int(round((st + (j + 1) * step) * fs / config.SAMPLING_RATE))
                if a1 > n:
                    continue
                rms = np.sqrt((noise[a0:a1] ** 2).mean(axis=0)) + 1e-9
                snr = float(np.min(20 * np.log10(s_ref / rms)))
                rows.append(dict(record=name, start_s=round(a0 / fs, 2), snr_db=round(snr, 2),
                                 noisy_block=bool(rms.max() > 1e-3),
                                 p_noise=round(float(pn[w, j]), 4)))
        print(f"  {name}: {len(starts)} windows, QRS p-p {np.round(2*np.sqrt(2)*s_ref, 2)} mV")

    flag = np.array([r['p_noise'] >= a.threshold for r in rows])
    snr = np.array([r['snr_db'] for r in rows])
    noisy = np.array([r['noisy_block'] for r in rows])
    print(f"\np(NOISE) >= {a.threshold}: share of 2 s segments flagged, by TRUE nstdb SNR "
          f"(worse lead)")
    print(f"{'true SNR':>12} {'segments':>9} {'flagged':>8}")
    print(f"{'no noise':>12} {int((~noisy).sum()):>9} {flag[~noisy].mean():>8.1%}")
    for lo, hi in BINS:
        sel = noisy & (snr >= lo) & (snr < hi)
        if sel.any():
            lab = f"{lo}..{hi} dB" if -99 < lo and hi < 99 else (f"< {hi} dB" if lo == -99
                                                                  else f">= {lo} dB")
            print(f"{lab:>12} {int(sel.sum()):>9} {flag[sel].mean():>8.1%}")
    # Per record at its NOMINAL nstdb SNR - the scale the brief is written in. The bins above
    # are this script's measurement, which reads a constant ~7.2 dB below nominal: nst measured
    # N once, on the first 300 s of the raw em record after removing each second's mean
    # (physionet nst man page), not on the noise it later added. Both scales are exact; they
    # differ by that per-record constant.
    recs = np.array([r['record'] for r in rows])
    print(f"\nper record, noisy blocks only (nominal SNR = record name):")
    print(f"{'record':>8} {'nominal':>8} {'flagged':>8}   clean blocks flagged")
    for name in RECORDS:
        sel = recs == name
        nominal = int(name[-2:].replace('_', '-'))
        print(f"{name:>8} {nominal:>6} dB {flag[sel & noisy].mean():>8.1%}   "
              f"{flag[sel & ~noisy].mean():.1%}")
    low = np.isin(recs, [r for r in RECORDS if int(r[-2:].replace('_', '-')) <= 6]) & noisy
    rest = ~low
    print(f"\nnoisy blocks of the <= 6 dB records flagged (want ~100%): {flag[low].mean():.1%} "
          f"of {low.sum()}")
    print(f"everything else flagged (want ~0%): {flag[rest].mean():.1%} of {rest.sum()}")

    out = a.out_csv or os.path.join(os.path.dirname(a.checkpoint), 'nstdb_noise_segments.csv')
    with open(out, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"-> {out}")


if __name__ == '__main__':
    main()
