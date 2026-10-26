# tune_pplus beatft_e1: validation only (Holter eval side + strip_val2), overlap as stored

score -102.37  target shortfall 101.5  guard violation 0.1

## knobs changed from official

- scale_AFIB: 1.0 -> 0.85
- scale_SVT: 1.0 -> 1.25
- scale_VT: 1.0 -> 1.25
- prob_AVB: 0 -> 0.5
- gap_AFIB: 3 -> 5
- gap_SVT: 0 -> 1
- gap_VT: 0 -> 2
- smooth_SVT: 1 -> 0.5
- smooth_VT: 1 -> 0.5
- svt_frac: 0.3 -> 0.5
- long_inv: 10.0 -> 20
- v_margin: 0 -> 0.3

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 99.09 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 57.81 | 72 |
| B|all|S|Se | 91.33 | 89.91 |  |
| B|all|V|PP | 92.84 | 95.48 | 95 |
| B|all|V|Se | 77.74 | 75.78 |  |
| B|incartdb|QRS|PP | 99.83 | 99.83 |  |
| B|incartdb|QRS|Se | 99.91 | 99.91 |  |
| B|incartdb|V|PP | 99.54 | 99.62 |  |
| B|incartdb|V|Se | 91.44 | 89.08 |  |
| B|ltaf+inc|S|PP | 56.48 | 58.98 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 90.11 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.53 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 56.98 | 59.44 |  |
| B|ltafdb|S|Se | 91.46 | 90.11 |  |
| B|ltafdb|V|PP | 92.03 | 94.99 |  |
| B|ltafdb|V|Se | 75.16 | 73.17 |  |
| B|nsrdb|QRS|PP | 93.14 | 94.35 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 57.18 | 72 |
| B|svdb|S|Se | 90.19 | 88.08 |  |
| B|svdb|V|PP | 93.25 | 93.94 |  |
| B|svdb|V|Se | 93.55 | 92.90 |  |
| FPH|AFIB | 0.58 | 0.43 |  |
| FPH|AVB | 0.03 | 0.02 |  |
| FPH|SVT | 0.82 | 0.89 |  |
| FPH|VT | 0.15 | 0.15 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.43 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 94.97 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 86.89 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 76.47 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 3.78 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 19.50 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 8.78 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 22.12 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 58.22 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 74.43 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 87.65 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 67.27 | 70 |
| S|AFIB|D_PP | 93.35 | 93.06 | 90 |
| S|AFIB|D_Se | 89.52 | 89.46 |  |
| S|AFIB|E_PP | 84.80 | 84.26 | 92 |
| S|AFIB|E_Se | 91.84 | 91.59 |  |
| S|AVB|D_PP | 87.90 | 87.92 |  |
| S|AVB|D_Se | 85.65 | 85.65 |  |
| S|AVB|E_PP | 97.45 | 97.45 |  |
| S|AVB|E_Se | 87.42 | 87.42 |  |
| S|SVT|D_PP | 75.54 | 75.83 | 70 |
| S|SVT|D_Se | 78.87 | 77.87 |  |
| S|SVT|E_PP | 85.33 | 85.04 |  |
| S|SVT|E_Se | 80.58 | 82.08 | 70 |
| S|VT|D_PP | 77.85 | 79.84 | 70 |
| S|VT|D_Se | 83.20 | 82.10 |  |
| S|VT|E_PP | 89.18 | 88.41 |  |
| S|VT|E_Se | 77.68 | 78.59 | 82 |