# tune_pplus pplus2_e4: validation only (Holter eval side + strip_val2), overlap as stored

score -131.54  target shortfall 108.7  guard violation 2.3

## knobs changed from official

- scale_AFIB: 1.0 -> 0.7
- scale_SVT: 1.0 -> 1.25
- scale_VT: 1.0 -> 1.25
- prob_AVB: 0 -> 0.5
- gap_SVT: 0 -> 2
- smooth_SVT: 1 -> 0.5
- trim_VT: None -> 0.3
- svt_frac: 0.3 -> 0.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20
- avb_maxhr: 60.0 -> 65
- s_afib_n: True -> False

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 98.91 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 54.56 | 72 |
| B|all|S|Se | 91.33 | 90.73 | 60 |
| B|all|V|PP | 92.84 | 92.79 | 95 |
| B|all|V|Se | 77.74 | 78.12 | 94 |
| B|incartdb|QRS|PP | 99.83 | 99.83 |  |
| B|incartdb|QRS|Se | 99.91 | 99.90 |  |
| B|incartdb|V|PP | 99.54 | 99.57 |  |
| B|incartdb|V|Se | 91.44 | 91.22 |  |
| B|ltaf+inc|S|PP | 56.48 | 56.58 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 90.92 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.50 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 56.98 | 57.07 |  |
| B|ltafdb|S|Se | 91.46 | 90.91 |  |
| B|ltafdb|V|PP | 92.03 | 92.06 |  |
| B|ltafdb|V|Se | 75.16 | 75.63 |  |
| B|nsrdb|QRS|PP | 93.14 | 92.70 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 55.52 | 72 |
| B|svdb|S|Se | 90.19 | 88.98 |  |
| B|svdb|V|PP | 93.25 | 93.14 |  |
| B|svdb|V|Se | 93.55 | 93.68 |  |
| FPH|AFIB | 0.58 | 0.26 |  |
| FPH|AVB | 0.03 | 0.01 |  |
| FPH|SVT | 0.82 | 0.71 |  |
| FPH|VT | 0.15 | 0.14 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.48 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 95.08 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 89.71 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 75.29 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 4.70 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 21.16 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 10.81 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 24.04 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 56.07 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 76.14 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 87.65 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 68.18 | 70 |
| S|AFIB|D_PP | 93.35 | 93.56 | 90 |
| S|AFIB|D_Se | 89.52 | 89.49 |  |
| S|AFIB|E_PP | 84.80 | 85.35 | 92 |
| S|AFIB|E_Se | 91.84 | 91.14 |  |
| S|AVB|D_PP | 87.90 | 88.48 |  |
| S|AVB|D_Se | 85.65 | 85.25 |  |
| S|AVB|E_PP | 97.45 | 97.91 |  |
| S|AVB|E_Se | 87.42 | 86.98 |  |
| S|SVT|D_PP | 75.54 | 76.31 | 70 |
| S|SVT|D_Se | 78.87 | 78.07 |  |
| S|SVT|E_PP | 85.33 | 85.34 |  |
| S|SVT|E_Se | 80.58 | 80.02 | 70 |
| S|VT|D_PP | 77.85 | 82.32 | 70 |
| S|VT|D_Se | 83.20 | 82.57 |  |
| S|VT|E_PP | 89.18 | 89.10 |  |
| S|VT|E_Se | 77.68 | 76.08 | 82 |