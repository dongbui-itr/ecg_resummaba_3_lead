# tune_pplus rA_e36: validation only (Holter eval side + strip_val2), overlap as stored

score -164.46  target shortfall 164.5  guard violation 0.0

## knobs changed from official

- scale_VT: 1.0 -> 1.25
- min_AFIB: 3 -> 5
- prob_SVT: 0 -> 0.5
- gap_SVT: 0 -> 2
- gap_VT: 0 -> 1
- smooth_VT: 1 -> 0.5
- smooth_AVB: 1 -> 5
- trim_SVT: None -> 0.3
- trim_VT: None -> 0.3
- svt_frac: 0.3 -> 0.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20
- avb_dur: 2.5 -> 4
- v_margin: 0 -> 0.1

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 97.45 | 97.45 | 99.8 |
| B|all|QRS|Se | 99.89 | 99.89 |  |
| B|all|S|PP | 34.59 | 34.86 | 72 |
| B|all|S|Se | 76.69 | 77.96 |  |
| B|all|V|PP | 84.54 | 86.92 | 95 |
| B|all|V|Se | 78.74 | 77.07 |  |
| B|incartdb|QRS|PP | 99.68 | 99.68 |  |
| B|incartdb|QRS|Se | 99.53 | 99.53 |  |
| B|incartdb|V|PP | 99.43 | 99.42 |  |
| B|incartdb|V|Se | 78.98 | 77.31 |  |
| B|ltaf+inc|S|PP | 48.58 | 48.77 | 40 |
| B|ltaf+inc|S|Se | 75.78 | 77.15 |  |
| B|ltafdb|QRS|PP | 99.17 | 99.17 |  |
| B|ltafdb|QRS|Se | 99.88 | 99.88 |  |
| B|ltafdb|S|PP | 49.03 | 49.25 |  |
| B|ltafdb|S|Se | 75.77 | 77.14 |  |
| B|ltafdb|V|PP | 83.21 | 85.46 |  |
| B|ltafdb|V|Se | 78.11 | 76.40 |  |
| B|nsrdb|QRS|PP | 81.67 | 81.67 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.64 | 54.63 | 72 |
| B|svdb|S|Se | 85.29 | 85.65 |  |
| B|svdb|V|PP | 92.80 | 93.37 |  |
| B|svdb|V|Se | 89.84 | 89.00 |  |
| FPH|AFIB | 1.57 | 1.16 |  |
| FPH|AVB | 0.06 | 0.03 |  |
| FPH|SVT | 2.07 | 0.81 |  |
| FPH|VT | 0.15 | 0.16 |  |
| H|ltafdb|AFIB|D_PP | 99.66 | 99.61 | 90 |
| H|ltafdb|AFIB|D_Se | 94.47 | 94.52 | 98 |
| H|ltafdb|AFIB|E_PP | 93.12 | 93.78 | 92 |
| H|ltafdb|AFIB|E_Se | 69.41 | 71.76 | 90 |
| H|ltafdb|SVT|D_PP | 1.56 | 3.04 | 70 |
| H|ltafdb|SVT|D_Se | 13.28 | 13.28 | 70 |
| H|ltafdb|SVT|E_PP | 2.77 | 6.28 | 70 |
| H|ltafdb|SVT|E_Se | 12.50 | 12.50 | 70 |
| H|ltafdb|VT|D_PP | 58.82 | 62.10 | 70 |
| H|ltafdb|VT|D_Se | 39.77 | 43.75 | 70 |
| H|ltafdb|VT|E_PP | 85.71 | 88.10 | 70 |
| H|ltafdb|VT|E_Se | 30.00 | 36.36 | 70 |
| S|AFIB|D_PP | 96.59 | 96.40 | 90 |
| S|AFIB|D_Se | 83.43 | 84.56 |  |
| S|AFIB|E_PP | 91.17 | 91.02 | 92 |
| S|AFIB|E_Se | 86.42 | 86.42 |  |
| S|AVB|D_PP | 85.79 | 85.70 |  |
| S|AVB|D_Se | 88.55 | 88.83 |  |
| S|AVB|E_PP | 95.79 | 95.90 |  |
| S|AVB|E_Se | 89.59 | 89.48 |  |
| S|SVT|D_PP | 72.94 | 75.08 | 70 |
| S|SVT|D_Se | 82.23 | 81.33 |  |
| S|SVT|E_PP | 80.93 | 83.15 |  |
| S|SVT|E_Se | 76.79 | 76.02 | 70 |
| S|VT|D_PP | 73.79 | 75.60 | 70 |
| S|VT|D_Se | 81.95 | 81.41 |  |
| S|VT|E_PP | 86.34 | 85.82 |  |
| S|VT|E_Se | 73.58 | 74.72 | 82 |