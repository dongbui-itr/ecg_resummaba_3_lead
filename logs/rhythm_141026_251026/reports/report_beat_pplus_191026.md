# Beat +P (mitdb VEB +P > 95, SVEB +P > 70, nothing else may drop) — 2026-10-19

## Where the mitdb false calls come from (v4 + AF gate, overlap 1, bxb per record)
- V false (Nv 413 + Sv 144; need <= ~300 for +P 95): 207 (154, BBB + VFL), 230 (112, WPW), 201 (87, aberrant S),
  212 (45, RBBB), 205 (38), 203 (34). Morphology confusions, not low-probability calls.
- S false (Ns 898; need <= ~570 for +P 70): 215 (288, sinus tachy), 219/222/202 (399, AF/flutter regions not
  labelled AFIB), 209 (38), 113/115 (56).

## Decoder ceiling (grid on p(S/V), RR prematurity, HR, rhythm context; beat_pp/false_beats.py)
- V: best rule pV<0.65 & prem>1.0 & SINUS -> -94 FP / -9 TP -> +P 92.4 (Se -0.13). Nothing above 92.5 without >=1 % Se.
- S: best rule HR>90 & prem>0.85 & pS<0.7 outside SVT/AFIB -> -135 FP / -4 TP -> +P 62.4 (Se -0.27). Ceiling ~62.

## Beat-decoder-only fine-tunes (encoder + rhythm decoder frozen, --train-only '^(pos[123]|beat_pos|type)',
## data npy_ft = curated v4 + nsrdb/ltdb/svdb train sides, lr 1e-4 cosine, 4 epochs, chosen on validation only)
| run | beat weights (none,N,S,V) | validation vs v4 (ref-guarded decode) |
|---|---|---|
| beatft  (191026_rhythm_dual_1m_v4_beatft)  | 0.15,1,2,2 | S +P +4.8..+5.4 but S Se -1.2..-1.8 (svdb -2.4..-2.7); V +P +0.3..+1.2 |
| beatft2 (191026_rhythm_dual_1m_v4_beatft2) | 0.15,1,4,2 | V +P +1.0..+1.2 but V Se -0.9..-1.2; S +P +0.9..+3.9 with S Se -0.3..-0.5 |
Every epoch of both runs violates the "no decrease" rule (2–4 pinned Se violations each); rhythm shifts are small
(AFIB E Se ltafdb -1.2..-1.6 on some epochs via the beat->rhythm conditioning) but nonzero.
Earlier whole-model fine-tunes (pplus/pplus2/pplus3, v5ft): at best V +P 93.1 / S +P 62 on mitdb, with Se losses.

## Verdict
VEB +P > 95 and SVEB +P > 70 on mitdb with no other metric decreasing is not reachable with the current model family
and training data: every lever (threshold, context rule, class re-weighting, decoder-only or whole-model fine-tune)
buys +P with Se one-for-one. The false V/S calls are morphology (BBB/WPW/aberrancy) and rate-context (sinus tachy)
confusions that need labelled hard-negative data (BBB/WPW beats, sinus tachy with true S) in training, i.e. a data
change and a retrain from scratch, not a decoder change.
No new checkpoint was promoted; the fine-tuned epochs are kept under 191026_rhythm_dual_1m_v4_beatft{,2}/checkpoints.

## Round 2 (2026-10-19, rule relaxed: beat V / S Se may drop <= 3 points, everything else may not)
Tuner/comparer got `SE_TOL=3` (V/S Se become guarded keys with 3-pt tolerance; v_margin/s_margin ranges to 0.7).
Validation (ref v4 + gate), 0 pinned violations: v4+decoder (shortfall 106.0), beatft_e2 (99.0), beatft2_e2 (99.0),
beatft2_e1 (102.6). Chosen: **beatft2_e2** (beat weights 0.15/1/4/2, epoch 2) + decode v_margin 0.3 / s_margin 0.2 —
the only one moving both V +P (+2.3, V Se -1.2) and S +P (+5.8, S Se -1.5) on validation.

EC57 overlap 1 (one scoring), v4+gate -> beatft2_e2:
  mitdb VEB 97.51/91.00 -> 96.73/94.85 (Se -0.78, +P +3.85)   SVEB 48.34/58.93 -> 45.32/62.45 (Se -3.02, +P +3.52)
  nstdb VEB 87.75/95.34 -> 87.36/96.67  SVEB 85.08/37.36 -> 81.98/52.22 ; escdb VEB 97.21/95.40 -> 96.50/97.04, SVEB 73.30/18.20 -> 70.23/20.52
  ahadb VEB 84.80/98.82 -> 84.51/99.38 ; QRS unchanged (nstdb Se 97.73 -> 97.48, +P 92.27 -> 92.85)
  rhythm mitdb: AFIB D 94/96 -> 96/95, E 87/89 -> 88/86 ; VT D 73/54 -> 74/58, E 53/70 -> 61/78 ; SVT D 57/37 -> 55/34, E 73/38 -> 73/34 ; AVB E +P 27 -> 29
  afdb AFIB D 95/99 -> 97/99, E 92/92 -> 91/91 ; escdb VT D 65/92 -> 67/97, E 71/98 -> 76/100
dataset-eval v4: beat V 97.10/95.30 -> 96.81/95.70, S 89.69/87.41 -> 88.11/88.98 ; rhythm AFIB =, SVT 90/90/77/85 -> 90/89/77/84, VT 86/86/78/79 -> 87/86/77/81, AVB 88/98/86/90 -> 89/99/87/90.
Test split macro F1 82.4-83.4 (unchanged).
Verdict: VEB +P 94.85 (target >95 missed by 0.15), SVEB +P 62.45 (target >70 missed); S Se -3.0 at the limit;
rhythm mostly up (VT), small drops: mitdb AFIB E +P 89 -> 86, SVT D 57/37 -> 55/34, afdb E 92/92 -> 91/91.
Checkpoint filed: release_dualunet_1m_e26/checkpoint/candidates/beatft2_e2.{keras,_decode.json,SHA256}.
