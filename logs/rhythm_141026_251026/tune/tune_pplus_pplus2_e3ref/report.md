# tune_pplus pplus2_e3: validation only (Holter eval side + strip_val2), overlap as stored

score -124.34  target shortfall 107.2  guard violation 1.7

## knobs changed from official

- scale_AFIB: 1.0 -> 0.7
- scale_SVT: 1.0 -> 1.25
- scale_VT: 1.0 -> 1.25
- prob_AVB: 0 -> 0.6
- gap_SVT: 0 -> 2
- gap_VT: 0 -> 1
- smooth_SVT: 1 -> 0.5
- smooth_VT: 1 -> 0.5
- smooth_AVB: 1 -> 5
- trim_VT: None -> 0.1
- svt_frac: 0.3 -> 0.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20
- avb_maxhr: 60.0 -> 65
- s_afib_n: True -> False

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 99.01 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 54.86 | 72 |
| B|all|S|Se | 91.33 | 90.82 | 60 |
| B|all|V|PP | 92.84 | 92.79 | 95 |
| B|all|V|Se | 77.74 | 78.29 | 94 |
| B|incartdb|QRS|PP | 99.83 | 99.84 |  |
| B|incartdb|QRS|Se | 99.91 | 99.90 |  |
| B|incartdb|V|PP | 99.54 | 99.57 |  |
| B|incartdb|V|Se | 91.44 | 91.13 |  |
| B|ltaf+inc|S|PP | 56.48 | 57.17 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 91.01 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.53 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 56.98 | 57.68 |  |
| B|ltafdb|S|Se | 91.46 | 91.00 |  |
| B|ltafdb|V|PP | 92.03 | 92.08 |  |
| B|ltafdb|V|Se | 75.16 | 75.85 |  |
| B|nsrdb|QRS|PP | 93.14 | 93.52 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 55.68 | 72 |
| B|svdb|S|Se | 90.19 | 89.07 |  |
| B|svdb|V|PP | 93.25 | 93.13 |  |
| B|svdb|V|Se | 93.55 | 93.62 |  |
| FPH|AFIB | 0.58 | 0.28 |  |
| FPH|AVB | 0.03 | 0.01 |  |
| FPH|SVT | 0.82 | 0.68 |  |
| FPH|VT | 0.15 | 0.13 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.47 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 95.10 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 87.82 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 76.00 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 4.90 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 20.33 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 11.17 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 23.08 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 59.47 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 76.70 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 92.31 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 69.09 | 70 |
| S|AFIB|D_PP | 93.35 | 93.33 | 90 |
| S|AFIB|D_Se | 89.52 | 89.70 |  |
| S|AFIB|E_PP | 84.80 | 84.72 | 92 |
| S|AFIB|E_Se | 91.84 | 91.27 |  |
| S|AVB|D_PP | 87.90 | 88.45 |  |
| S|AVB|D_Se | 85.65 | 85.85 |  |
| S|AVB|E_PP | 97.45 | 98.53 |  |
| S|AVB|E_Se | 87.42 | 87.53 |  |
| S|SVT|D_PP | 75.54 | 76.48 | 70 |
| S|SVT|D_Se | 78.87 | 77.56 |  |
| S|SVT|E_PP | 85.33 | 85.68 |  |
| S|SVT|E_Se | 80.58 | 79.80 | 70 |
| S|VT|D_PP | 77.85 | 86.27 | 70 |
| S|VT|D_Se | 83.20 | 82.31 |  |
| S|VT|E_PP | 89.18 | 90.08 |  |
| S|VT|E_Se | 77.68 | 77.22 | 82 |