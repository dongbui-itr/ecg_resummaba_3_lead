# tune_pplus rA_best: validation only (Holter eval side + strip_val2), overlap as stored

score -171.23  target shortfall 171.3  guard violation 0.0

## knobs changed from official

- scale_SVT: 1.0 -> 0.8
- min_AFIB: 3 -> 5
- prob_VT: 0 -> 0.5
- gap_SVT: 0 -> 2
- smooth_VT: 1 -> 0.5
- smooth_AVB: 1 -> 3
- trim_VT: None -> 0.1
- svt_frac: 0.3 -> 0.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 97.54 | 97.54 | 99.8 |
| B|all|QRS|Se | 99.88 | 99.88 |  |
| B|all|S|PP | 39.56 | 40.99 | 72 |
| B|all|S|Se | 76.01 | 75.82 |  |
| B|all|V|PP | 85.49 | 85.49 | 95 |
| B|all|V|Se | 76.90 | 76.90 |  |
| B|incartdb|QRS|PP | 99.69 | 99.69 |  |
| B|incartdb|QRS|Se | 99.56 | 99.56 |  |
| B|incartdb|V|PP | 99.53 | 99.53 |  |
| B|incartdb|V|Se | 77.62 | 77.62 |  |
| B|ltaf+inc|S|PP | 53.04 | 53.34 | 40 |
| B|ltaf+inc|S|Se | 75.40 | 75.20 |  |
| B|ltafdb|QRS|PP | 99.21 | 99.21 |  |
| B|ltafdb|QRS|Se | 99.88 | 99.88 |  |
| B|ltafdb|S|PP | 53.60 | 53.90 |  |
| B|ltafdb|S|Se | 75.40 | 75.20 |  |
| B|ltafdb|V|PP | 84.49 | 84.49 |  |
| B|ltafdb|V|Se | 76.16 | 76.16 |  |
| B|nsrdb|QRS|PP | 82.07 | 82.07 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.34 | 54.26 | 72 |
| B|svdb|S|Se | 81.69 | 81.60 |  |
| B|svdb|V|PP | 93.18 | 93.18 |  |
| B|svdb|V|Se | 89.00 | 89.00 |  |
| FPH|AFIB | 1.90 | 1.57 |  |
| FPH|AVB | 0.02 | 0.01 |  |
| FPH|SVT | 1.30 | 0.91 |  |
| FPH|VT | 0.13 | 0.13 |  |
| H|ltafdb|AFIB|D_PP | 99.49 | 99.42 | 90 |
| H|ltafdb|AFIB|D_Se | 94.73 | 94.77 | 98 |
| H|ltafdb|AFIB|E_PP | 86.50 | 86.65 | 92 |
| H|ltafdb|AFIB|E_Se | 72.00 | 74.12 | 90 |
| H|ltafdb|SVT|D_PP | 2.87 | 3.88 | 70 |
| H|ltafdb|SVT|D_Se | 12.03 | 12.45 | 70 |
| H|ltafdb|SVT|E_PP | 5.43 | 7.64 | 70 |
| H|ltafdb|SVT|E_Se | 11.54 | 12.50 | 70 |
| H|ltafdb|VT|D_PP | 51.28 | 69.41 | 70 |
| H|ltafdb|VT|D_Se | 34.09 | 33.52 | 70 |
| H|ltafdb|VT|E_PP | 86.21 | 100.00 | 70 |
| H|ltafdb|VT|E_Se | 25.45 | 25.45 | 70 |
| S|AFIB|D_PP | 95.50 | 95.14 | 90 |
| S|AFIB|D_Se | 85.21 | 86.04 |  |
| S|AFIB|E_PP | 89.06 | 88.36 | 92 |
| S|AFIB|E_Se | 88.02 | 88.08 |  |
| S|AVB|D_PP | 86.98 | 86.82 |  |
| S|AVB|D_Se | 87.00 | 87.60 |  |
| S|AVB|E_PP | 97.00 | 96.99 |  |
| S|AVB|E_Se | 87.85 | 87.74 |  |
| S|SVT|D_PP | 74.56 | 74.74 | 70 |
| S|SVT|D_Se | 80.39 | 79.48 |  |
| S|SVT|E_PP | 82.86 | 83.30 |  |
| S|SVT|E_Se | 75.46 | 73.51 | 70 |
| S|VT|D_PP | 74.02 | 80.00 | 70 |
| S|VT|D_Se | 81.73 | 80.88 |  |
| S|VT|E_PP | 86.02 | 88.41 |  |
| S|VT|E_Se | 71.75 | 72.21 | 82 |