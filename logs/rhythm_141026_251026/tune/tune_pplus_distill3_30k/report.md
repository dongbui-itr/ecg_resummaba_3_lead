# tune_pplus distill3_30k: validation only (Holter eval side + strip_val2), overlap as stored

score -174.80  target shortfall 174.6  guard violation 0.0

## knobs changed from official

- scale_AFIB: 1.0 -> 0.85
- min_AFIB: 3 -> 5
- min_AVB: 2 -> 3
- gap_AFIB: 3 -> 5
- gap_SVT: 0 -> 2
- gap_VT: 0 -> 2
- smooth_SVT: 1 -> 0.5
- smooth_VT: 1 -> 0.5
- smooth_AVB: 1 -> 5
- trim_VT: None -> 0.1
- svt_frac: 0.3 -> 0.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 97.06 | 97.06 | 99.8 |
| B|all|QRS|Se | 99.90 | 99.90 |  |
| B|all|S|PP | 30.66 | 30.65 | 72 |
| B|all|S|Se | 73.70 | 73.62 | 60 |
| B|all|V|PP | 83.21 | 83.21 | 95 |
| B|all|V|Se | 80.56 | 80.56 | 94 |
| B|incartdb|QRS|PP | 99.45 | 99.45 |  |
| B|incartdb|QRS|Se | 99.54 | 99.54 |  |
| B|incartdb|V|PP | 99.62 | 99.62 |  |
| B|incartdb|V|Se | 80.66 | 80.66 |  |
| B|ltaf+inc|S|PP | 50.48 | 50.54 | 40 |
| B|ltaf+inc|S|Se | 73.51 | 73.41 |  |
| B|ltafdb|QRS|PP | 98.97 | 98.97 |  |
| B|ltafdb|QRS|Se | 99.89 | 99.89 |  |
| B|ltafdb|S|PP | 50.92 | 50.98 |  |
| B|ltafdb|S|Se | 73.50 | 73.40 |  |
| B|ltafdb|V|PP | 81.81 | 81.81 |  |
| B|ltafdb|V|Se | 79.86 | 79.86 |  |
| B|nsrdb|QRS|PP | 79.81 | 79.81 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.97 | 99.97 |  |
| B|svdb|S|PP | 59.92 | 59.97 | 72 |
| B|svdb|S|Se | 75.39 | 75.48 |  |
| B|svdb|V|PP | 83.22 | 83.22 |  |
| B|svdb|V|Se | 93.29 | 93.29 |  |
| FPH|AFIB | 0.90 | 0.96 |  |
| FPH|AVB | 0.02 | 0.01 |  |
| FPH|SVT | 2.17 | 2.30 |  |
| FPH|VT | 0.17 | 0.18 |  |
| H|ltafdb|AFIB|D_PP | 99.74 | 99.71 | 90 |
| H|ltafdb|AFIB|D_Se | 93.13 | 94.03 | 98 |
| H|ltafdb|AFIB|E_PP | 97.23 | 95.84 | 92 |
| H|ltafdb|AFIB|E_Se | 67.29 | 68.94 | 90 |
| H|ltafdb|SVT|D_PP | 1.60 | 1.82 | 70 |
| H|ltafdb|SVT|D_Se | 7.47 | 8.71 | 70 |
| H|ltafdb|SVT|E_PP | 3.15 | 3.51 | 70 |
| H|ltafdb|SVT|E_Se | 6.73 | 8.65 | 70 |
| H|ltafdb|VT|D_PP | 46.23 | 48.97 | 70 |
| H|ltafdb|VT|D_Se | 52.27 | 53.98 | 70 |
| H|ltafdb|VT|E_PP | 76.36 | 76.67 | 70 |
| H|ltafdb|VT|E_Se | 40.00 | 43.64 | 70 |
| S|AFIB|D_PP | 96.51 | 96.64 | 90 |
| S|AFIB|D_Se | 82.98 | 82.67 |  |
| S|AFIB|E_PP | 90.32 | 90.61 | 92 |
| S|AFIB|E_Se | 84.64 | 84.38 |  |
| S|AVB|D_PP | 86.80 | 86.58 |  |
| S|AVB|D_Se | 85.80 | 86.80 |  |
| S|AVB|E_PP | 96.97 | 96.96 |  |
| S|AVB|E_Se | 86.98 | 87.31 |  |
| S|SVT|D_PP | 74.27 | 73.82 | 70 |
| S|SVT|D_Se | 78.46 | 79.30 |  |
| S|SVT|E_PP | 81.57 | 80.55 |  |
| S|SVT|E_Se | 68.39 | 68.61 | 70 |
| S|VT|D_PP | 69.43 | 73.59 | 70 |
| S|VT|D_Se | 85.27 | 84.43 |  |
| S|VT|E_PP | 79.02 | 78.07 |  |
| S|VT|E_Se | 77.90 | 78.59 | 82 |