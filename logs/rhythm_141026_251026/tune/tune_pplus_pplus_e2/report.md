# tune_pplus pplus_e2: validation only (Holter eval side + strip_val2), overlap as stored

score -104.88  target shortfall 104.6  guard violation 0.2

## knobs changed from official

- scale_AFIB: 1.0 -> 0.85
- scale_VT: 1.0 -> 1.25
- prob_SVT: 0 -> 0.5
- prob_VT: 0 -> 0.5
- gap_AFIB: 3 -> 5
- gap_SVT: 0 -> 2
- gap_VT: 0 -> 1
- smooth_SVT: 1 -> 0.5
- smooth_VT: 1 -> 0.5
- trim_SVT: None -> 0.3
- trim_VT: None -> 0.1
- svt_frac: 0.3 -> 0.5
- long_inv: 10.0 -> 20

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 99.06 | 99.06 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 58.48 | 58.49 | 72 |
| B|all|S|Se | 89.91 | 89.94 | 60 |
| B|all|V|PP | 92.77 | 92.77 | 95 |
| B|all|V|Se | 78.15 | 78.15 | 94 |
| B|incartdb|QRS|PP | 99.83 | 99.83 |  |
| B|incartdb|QRS|Se | 99.91 | 99.91 |  |
| B|incartdb|V|PP | 99.55 | 99.55 |  |
| B|incartdb|V|Se | 93.86 | 93.86 |  |
| B|ltaf+inc|S|PP | 60.17 | 60.19 | 40 |
| B|ltaf+inc|S|Se | 90.15 | 90.18 |  |
| B|ltafdb|QRS|PP | 99.53 | 99.53 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 60.63 | 60.64 |  |
| B|ltafdb|S|Se | 90.15 | 90.17 |  |
| B|ltafdb|V|PP | 92.06 | 92.06 |  |
| B|ltafdb|V|Se | 75.31 | 75.31 |  |
| B|nsrdb|QRS|PP | 94.05 | 94.05 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 56.00 | 56.00 | 72 |
| B|svdb|S|Se | 87.72 | 87.72 |  |
| B|svdb|V|PP | 92.98 | 92.98 |  |
| B|svdb|V|Se | 93.95 | 93.95 |  |
| FPH|AFIB | 0.32 | 0.28 |  |
| FPH|AVB | 0.01 | 0.01 |  |
| FPH|SVT | 0.56 | 0.59 |  |
| FPH|VT | 0.10 | 0.11 |  |
| H|ltafdb|AFIB|D_PP | 99.65 | 99.67 | 90 |
| H|ltafdb|AFIB|D_Se | 94.92 | 94.91 | 98 |
| H|ltafdb|AFIB|E_PP | 90.26 | 90.73 | 92 |
| H|ltafdb|AFIB|E_Se | 76.24 | 75.29 | 90 |
| H|ltafdb|SVT|D_PP | 5.21 | 6.32 | 70 |
| H|ltafdb|SVT|D_Se | 17.43 | 19.92 | 70 |
| H|ltafdb|SVT|E_PP | 11.73 | 13.84 | 70 |
| H|ltafdb|SVT|E_Se | 18.27 | 22.12 | 70 |
| H|ltafdb|VT|D_PP | 56.54 | 59.52 | 70 |
| H|ltafdb|VT|D_Se | 68.75 | 71.02 | 70 |
| H|ltafdb|VT|E_PP | 90.91 | 91.55 | 70 |
| H|ltafdb|VT|E_Se | 58.18 | 62.73 | 70 |
| S|AFIB|D_PP | 93.89 | 93.92 | 90 |
| S|AFIB|D_Se | 88.63 | 88.24 |  |
| S|AFIB|E_PP | 86.43 | 86.33 | 92 |
| S|AFIB|E_Se | 90.25 | 89.87 |  |
| S|AVB|D_PP | 89.50 | 89.50 |  |
| S|AVB|D_Se | 79.67 | 79.74 |  |
| S|AVB|E_PP | 97.18 | 97.18 |  |
| S|AVB|E_Se | 82.10 | 82.10 |  |
| S|SVT|D_PP | 78.33 | 79.83 | 70 |
| S|SVT|D_Se | 75.29 | 75.28 |  |
| S|SVT|E_PP | 87.44 | 88.05 |  |
| S|SVT|E_Se | 75.40 | 76.18 | 70 |
| S|VT|D_PP | 78.87 | 85.26 | 70 |
| S|VT|D_Se | 83.32 | 82.64 |  |
| S|VT|E_PP | 87.97 | 88.77 |  |
| S|VT|E_Se | 74.26 | 74.72 | 82 |