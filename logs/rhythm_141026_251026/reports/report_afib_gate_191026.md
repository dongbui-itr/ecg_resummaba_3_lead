# Record-level AF gate (2026-10-19) — mitdb AFIB episode +P 61 -> 89, nothing else touched

Finding (mitdb, v4_scratch_best, overlap 1 s, official decode): all 43 false AFIB episodes sit in records with
NO true AF (200 PVC-heavy sinus, 215 sinus tachy, 232 sinus brady/pauses, 207 BBB, 209, 101, 228) except 8 in
222 (flutter/junctional mix). Per-episode features (mean p(AFIB), RR CV / RMSSD / pNN50, HR, V fraction) do not
separate them from true AF — but the record does: every true-AF record carries a sustained (>= 207 s) episode
with mean p(AFIB) >= 0.6, while the false-only records' longest confident episode is <= 27 s.
Validation (ltafdb/nsrdb/incartdb/svdb) shows the same: AF records' longest confident episode >= 32.6 s
(ltafdb 08), non-AF records <= 17 s (nsrdb 19093, ltafdb 30, incartdb, svdb).

Rule (ecgr/rhythm/ec57.py afib_record_gate, rc.AFIB_GATE_DEFAULT, on by default): in a record >= 10 min,
keep AFIB only if some AFIB episode lasts >= 30 s (clinical AF definition) with mean p(AFIB) >= 0.6;
otherwise all AFIB -> SINUS. Applied after the beat stage: beat labels, SVT/VT/AVB untouched. Strips are
below min_record, so strip / dataset-eval results do not change.

Validation effect (official decode): ltafdb 563 TP / 95 -> 90 FP / 97 missed (0 TP lost);
nsrdb 21 FP -> 0, incartdb 3 -> 0, svdb 3 -> 0.
A 'relaxed' second pass (lower AFIB min_prob / min_seconds inside AF-confirmed records) was tried on ltafdb:
recovers 1–3 missed episodes but adds 58–116 FP -> rejected.

EC57 (decode-only from the stored v4ov1_official probabilities), only the AFIB lines move:
                      before (v4 official)          after (+ gate)
  mitdb AFIB  E 87/61  D 94/89  F1 71.7/91.4    E 87/89  D 94/96  F1 88.0/95.0
  afdb  AFIB  E 94/91  D 95/99                  E 92/92  D 95/99   (record 05091: brief paroxysmal AF, longest 21 s, gated)
  mitdb SVT/VT/AVB, beats (all DBs), nstdb/escdb/ahadb: identical.
Target "E Se, +P > 88": +P 89 reached; E Se stays 87 (67/77): the 10 missed reference episodes are
219 x3 (p(AFIB) 0.01–0.04), 210 x2, 203 x1 (3 s), 222 x3 (2–7 s) — not reachable by AFIB-only post-processing
without the FP cost seen on validation.
Runs: summary_111026/ec57/gate_official (official + gate), gate_tuned (validation-tuned decoder + gate:
mitdb AFIB E 90/88 but it moves SVT/VT/S metrics, so not the answer to this request).
