# P+ tuning of v4_scratch_best — final scoring (2026-10-16, window overlap 1 s, TTA id/flip/swap, taper 0)

Rule set: tune on validation only (ltafdb/nsrdb/incartdb/svdb Holter side + strip_val2); EC57 and dataset-eval v4 scored ONCE at the end.
Candidates (fixed before scoring):
- v4ov1_official : v4_scratch_best + official decode (reference)
- v4ov1_tuned    : v4_scratch_best + decoder tuned on validation with "no Se decrease" guard (tune_pplus_v4ov1_tight) — 0 pinned violations
- pplus3_e1      : v4_scratch_best fine-tuned 1 epoch (lr 1e-4 cosine, beat class weights 0.15/1/2/2, rhythm weights unchanged) + ref-guarded decoder (tune_pplus_pplus3_e1ref) — best P+ gain on validation, but S Se −1.35 (svdb −2.5) and strip AFIB E Se −0.5 vs reference
Rejected on validation: pplus (flat CW, 8 ep) e2/e4 — 5–12 Se violations; pplus2 (milder CW) e1–e4 — 5–10 violations; pplus3 e2–e4 — 3–4 violations incl. VT/V Se.

BEAT (Se / +P)        5.4.1           v4ov1_official      v4ov1_tuned        pplus3_e1
QRS  mitdb    99.57/ 99.98      99.95/ 99.95    99.95/ 99.95    99.96/ 99.95 
QRS  nstdb    80.47/ 99.77      97.73/ 92.27    97.73/ 92.27    97.46/ 92.88 
QRS  ahadb    97.90/ 85.29      99.86/ 99.66    99.86/ 99.66    99.87/ 99.66 
QRS  escdb    99.48/ 99.96      99.99/ 99.97    99.99/ 99.97    99.99/ 99.97 
V    mitdb    93.91/ 94.68      97.51/ 91.00    97.51/ 91.00    97.14/ 92.84   floor +P>95
V    nstdb    81.59/ 95.83      87.75/ 95.34    87.75/ 95.34    88.15/ 95.82 
V    ahadb    85.01/ 84.07      84.80/ 98.82    84.80/ 98.81    85.00/ 99.05 
V    escdb    93.84/ 96.70      97.21/ 95.40    97.21/ 95.40    97.35/ 95.97 
S    mitdb    58.49/ 69.22      48.34/ 58.93    48.78/ 59.09    48.60/ 61.96   floor +P>72
S    nstdb    57.29/ 88.95      85.08/ 37.36    85.08/ 37.36    81.40/ 52.37 
S    escdb    49.55/ 34.00      73.30/ 18.20    73.30/ 18.04    67.44/ 22.34   floor +P>40

RHYTHM (D Se/+P  E Se/+P)   5.4.1                 v4ov1_official        v4ov1_tuned          pplus3_e1
mitdb  AFIB    98/ 90  80/ 92       94/ 89  87/ 61       95/ 88  90/ 62       95/ 90  88/ 60   
afdb   AFIB    96/ 99  81/ 93       95/ 99  94/ 91       96/ 99  92/ 91       97/ 99  93/ 91   
mitdb  SVT     61/ 24  62/ 19       57/ 37  73/ 38       57/ 34  73/ 34       51/ 29  69/ 31   
mitdb  VT      75/ 57  82/ 58       73/ 54  53/ 70       70/ 55  59/ 67       72/ 55  61/ 71   
mitdb  AVB     98/ 87 100/100      100/ 83 100/ 27       99/ 85 100/ 31      100/ 83 100/ 29   

escdb VT (not in 5.4.1): v4ov1_official: D 65/92 E 71/98  v4ov1_tuned: D 63/93 E 78/99  pplus3_e1: D 64/94 E 79/99

DATASET-EVAL v4 (beat bxb mark window Se/+P; rhythm epicmp E_Se/E_+P/D_Se/D_+P)
v4ov1_official   QRS 99.78/99.70  V 97.10/95.30  S 89.69/87.41  | AFIB 95/83/92/92  SVT 90/90/77/85  VT 86/86/78/79  AVB 88/98/86/90
v4ov1_tuned      QRS 99.78/99.70  V 97.10/95.30  S 89.41/87.41  | AFIB 95/84/92/92  SVT 90/89/78/84  VT 87/86/77/82  AVB 89/99/87/90
pplus3_e1        QRS 99.77/99.71  V 97.10/95.28  S 88.78/89.02  | AFIB 95/85/93/92  SVT 90/88/76/85  VT 87/87/77/83  AVB 89/99/88/90

Test split macro F1 (6 seeds/variants): v4 official 82.4–83.4 (same for tuned decode); pplus3_e1 82.3–83.1.

Verdict vs the requested floors (VEB +P mitdb >95, SVEB +P mitdb >72 / escdb >40, AFIB mitdb D +P >90 & E Se/+P >90, SVT/VT mitdb all >70, AVB E +P >60): none reached by any candidate.
Largest gaps are structural (SVEB Se/+P on mitdb 48/62 vs 58/69 in 5.4.1; AFIB E +P 60–62 vs 92; AVB E +P ~30) and are not closable by decoder knobs or light fine-tunes without lowering Se elsewhere.
