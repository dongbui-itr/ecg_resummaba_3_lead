# tune_pplus rA_e33: validation only (Holter eval side + strip_val2), overlap as stored

score -165.42  target shortfall 165.4  guard violation 0.0

## knobs changed from official

- scale_AFIB: 1.0 -> 0.85
- min_AVB: 2 -> 3
- prob_SVT: 0 -> 0.5
- prob_AVB: 0 -> 0.5
- gap_SVT: 0 -> 2
- smooth_SVT: 1 -> 0.5
- smooth_AVB: 1 -> 5
- trim_SVT: None -> 0.3
- trim_VT: None -> 0.3
- svt_frac: 0.3 -> 0.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 97.55 | 97.55 | 99.8 |
| B|all|QRS|Se | 99.89 | 99.89 |  |
| B|all|S|PP | 33.59 | 33.31 | 72 |
| B|all|S|Se | 75.75 | 75.69 |  |
| B|all|V|PP | 83.38 | 83.38 | 95 |
| B|all|V|Se | 79.95 | 79.95 |  |
| B|incartdb|QRS|PP | 99.69 | 99.69 |  |
| B|incartdb|QRS|Se | 99.50 | 99.50 |  |
| B|incartdb|V|PP | 99.34 | 99.34 |  |
| B|incartdb|V|Se | 80.61 | 80.61 |  |
| B|ltaf+inc|S|PP | 49.19 | 49.36 | 40 |
| B|ltaf+inc|S|Se | 74.81 | 74.74 |  |
| B|ltafdb|QRS|PP | 99.21 | 99.21 |  |
| B|ltafdb|QRS|Se | 99.88 | 99.88 |  |
| B|ltafdb|S|PP | 49.58 | 49.76 |  |
| B|ltafdb|S|Se | 74.81 | 74.73 |  |
| B|ltafdb|V|PP | 82.22 | 82.22 |  |
| B|ltafdb|V|Se | 79.30 | 79.30 |  |
| B|nsrdb|QRS|PP | 82.14 | 82.14 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.84 | 54.85 | 72 |
| B|svdb|S|Se | 84.62 | 84.71 |  |
| B|svdb|V|PP | 92.42 | 92.42 |  |
| B|svdb|V|Se | 90.49 | 90.49 |  |
| FPH|AFIB | 1.32 | 1.00 |  |
| FPH|AVB | 0.05 | 0.04 |  |
| FPH|SVT | 1.91 | 0.85 |  |
| FPH|VT | 0.14 | 0.15 |  |
| H|ltafdb|AFIB|D_PP | 99.66 | 99.64 | 90 |
| H|ltafdb|AFIB|D_Se | 94.51 | 94.55 | 98 |
| H|ltafdb|AFIB|E_PP | 93.00 | 94.07 | 92 |
| H|ltafdb|AFIB|E_Se | 69.65 | 71.53 | 90 |
| H|ltafdb|SVT|D_PP | 1.67 | 3.21 | 70 |
| H|ltafdb|SVT|D_Se | 12.03 | 13.69 | 70 |
| H|ltafdb|SVT|E_PP | 3.06 | 6.34 | 70 |
| H|ltafdb|SVT|E_Se | 11.54 | 13.46 | 70 |
| H|ltafdb|VT|D_PP | 63.64 | 63.43 | 70 |
| H|ltafdb|VT|D_Se | 43.75 | 48.30 | 70 |
| H|ltafdb|VT|E_PP | 91.89 | 93.02 | 70 |
| H|ltafdb|VT|E_Se | 33.64 | 39.09 | 70 |
| S|AFIB|D_PP | 96.52 | 96.47 | 90 |
| S|AFIB|D_Se | 83.74 | 84.13 |  |
| S|AFIB|E_PP | 91.00 | 91.06 | 92 |
| S|AFIB|E_Se | 86.30 | 86.04 |  |
| S|AVB|D_PP | 85.72 | 85.46 |  |
| S|AVB|D_Se | 88.52 | 88.98 |  |
| S|AVB|E_PP | 95.68 | 95.68 |  |
| S|AVB|E_Se | 89.59 | 89.59 |  |
| S|SVT|D_PP | 73.44 | 75.33 | 70 |
| S|SVT|D_Se | 81.83 | 81.24 |  |
| S|SVT|E_PP | 82.14 | 83.40 |  |
| S|SVT|E_Se | 76.68 | 76.63 | 70 |
| S|VT|D_PP | 72.69 | 74.27 | 70 |
| S|VT|D_Se | 83.18 | 83.12 |  |
| S|VT|E_PP | 84.63 | 84.00 |  |
| S|VT|E_Se | 74.49 | 74.49 | 82 |