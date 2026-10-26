# tune_pplus rA_e30: validation only (Holter eval side + strip_val2), overlap as stored

score -164.98  target shortfall 165.0  guard violation 0.0

## knobs changed from official

- scale_AFIB: 1.0 -> 0.85
- scale_SVT: 1.0 -> 0.6
- scale_VT: 1.0 -> 1.25
- min_AFIB: 3 -> 5
- min_AVB: 2 -> 5
- prob_VT: 0 -> 0.5
- gap_SVT: 0 -> 2
- smooth_SVT: 1 -> 0.5
- smooth_VT: 1 -> 0.5
- smooth_AVB: 1 -> 5
- trim_VT: None -> 0.3
- svt_frac: 0.3 -> 0.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20
- v_margin: 0 -> 0.1

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 97.57 | 97.57 | 99.8 |
| B|all|QRS|Se | 99.88 | 99.88 |  |
| B|all|S|PP | 34.43 | 34.68 | 72 |
| B|all|S|Se | 75.42 | 76.75 |  |
| B|all|V|PP | 83.67 | 86.39 | 95 |
| B|all|V|Se | 79.16 | 77.57 |  |
| B|incartdb|QRS|PP | 99.70 | 99.70 |  |
| B|incartdb|QRS|Se | 99.50 | 99.50 |  |
| B|incartdb|V|PP | 99.40 | 99.53 |  |
| B|incartdb|V|Se | 78.81 | 77.12 |  |
| B|ltaf+inc|S|PP | 49.80 | 49.97 | 40 |
| B|ltaf+inc|S|Se | 74.45 | 75.90 |  |
| B|ltafdb|QRS|PP | 99.17 | 99.17 |  |
| B|ltafdb|QRS|Se | 99.88 | 99.88 |  |
| B|ltafdb|S|PP | 50.21 | 50.41 |  |
| B|ltafdb|S|Se | 74.44 | 75.90 |  |
| B|ltafdb|V|PP | 82.70 | 85.08 |  |
| B|ltafdb|V|Se | 78.63 | 77.02 |  |
| B|nsrdb|QRS|PP | 82.57 | 82.57 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 57.13 | 57.03 | 72 |
| B|svdb|S|Se | 84.57 | 84.71 |  |
| B|svdb|V|PP | 93.18 | 93.42 |  |
| B|svdb|V|Se | 89.78 | 88.74 |  |
| FPH|AFIB | 1.29 | 1.10 |  |
| FPH|AVB | 0.06 | 0.04 |  |
| FPH|SVT | 1.97 | 0.90 |  |
| FPH|VT | 0.14 | 0.14 |  |
| H|ltafdb|AFIB|D_PP | 99.63 | 99.59 | 90 |
| H|ltafdb|AFIB|D_Se | 94.60 | 94.63 | 98 |
| H|ltafdb|AFIB|E_PP | 89.95 | 90.96 | 92 |
| H|ltafdb|AFIB|E_Se | 69.41 | 71.29 | 90 |
| H|ltafdb|SVT|D_PP | 1.29 | 2.50 | 70 |
| H|ltafdb|SVT|D_Se | 10.79 | 10.79 | 70 |
| H|ltafdb|SVT|E_PP | 2.35 | 5.21 | 70 |
| H|ltafdb|SVT|E_Se | 10.58 | 10.58 | 70 |
| H|ltafdb|VT|D_PP | 62.39 | 66.67 | 70 |
| H|ltafdb|VT|D_Se | 41.48 | 44.32 | 70 |
| H|ltafdb|VT|E_PP | 86.84 | 94.74 | 70 |
| H|ltafdb|VT|E_Se | 32.73 | 35.45 | 70 |
| S|AFIB|D_PP | 96.60 | 96.46 | 90 |
| S|AFIB|D_Se | 83.47 | 83.94 |  |
| S|AFIB|E_PP | 91.09 | 90.75 | 92 |
| S|AFIB|E_Se | 86.42 | 86.49 |  |
| S|AVB|D_PP | 85.65 | 85.44 |  |
| S|AVB|D_Se | 88.74 | 88.90 |  |
| S|AVB|E_PP | 95.58 | 95.79 |  |
| S|AVB|E_Se | 89.70 | 89.59 |  |
| S|SVT|D_PP | 73.47 | 73.80 | 70 |
| S|SVT|D_Se | 82.13 | 81.20 |  |
| S|SVT|E_PP | 81.50 | 82.33 |  |
| S|SVT|E_Se | 75.90 | 73.29 | 70 |
| S|VT|D_PP | 71.87 | 74.35 | 70 |
| S|VT|D_Se | 82.99 | 82.44 |  |
| S|VT|E_PP | 84.40 | 84.71 |  |
| S|VT|E_Se | 73.12 | 74.72 | 82 |