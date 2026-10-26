# tune_pplus pplus3_e2: validation only (Holter eval side + strip_val2), overlap as stored

score -160.88  target shortfall 114.0  guard violation 4.7

## knobs changed from official

- scale_VT: 1.0 -> 1.25
- prob_VT: 0 -> 0.5
- gap_AFIB: 3 -> 5
- gap_SVT: 0 -> 2
- gap_VT: 0 -> 1
- smooth_SVT: 1 -> 0.5
- smooth_VT: 1 -> 0.5
- trim_VT: None -> 0.1
- svt_frac: 0.3 -> 0.5
- s_afib_n: True -> False

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 99.09 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 55.64 | 72 |
| B|all|S|Se | 91.33 | 89.91 | 60 |
| B|all|V|PP | 92.84 | 93.31 | 95 |
| B|all|V|Se | 77.74 | 77.75 | 94 |
| B|incartdb|QRS|PP | 99.83 | 99.83 |  |
| B|incartdb|QRS|Se | 99.91 | 99.91 |  |
| B|incartdb|V|PP | 99.54 | 99.60 |  |
| B|incartdb|V|Se | 91.44 | 91.08 |  |
| B|ltaf+inc|S|PP | 56.48 | 56.72 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 90.20 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.51 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 56.98 | 57.10 |  |
| B|ltafdb|S|Se | 91.46 | 90.20 |  |
| B|ltafdb|V|PP | 92.03 | 92.60 |  |
| B|ltafdb|V|Se | 75.16 | 75.21 |  |
| B|nsrdb|QRS|PP | 93.14 | 94.57 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 56.04 | 72 |
| B|svdb|S|Se | 90.19 | 87.27 |  |
| B|svdb|V|PP | 93.25 | 93.32 |  |
| B|svdb|V|Se | 93.55 | 93.62 |  |
| FPH|AFIB | 0.58 | 0.65 |  |
| FPH|AVB | 0.03 | 0.02 |  |
| FPH|SVT | 0.82 | 0.92 |  |
| FPH|VT | 0.15 | 0.15 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.42 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 94.93 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 83.36 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 76.24 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 3.97 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 21.99 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 8.83 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 25.00 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 57.41 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 70.45 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 89.04 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 61.82 | 70 |
| S|AFIB|D_PP | 93.35 | 93.31 | 90 |
| S|AFIB|D_Se | 89.52 | 89.28 |  |
| S|AFIB|E_PP | 84.80 | 84.46 | 92 |
| S|AFIB|E_Se | 91.84 | 91.46 |  |
| S|AVB|D_PP | 87.90 | 88.10 |  |
| S|AVB|D_Se | 85.65 | 85.76 |  |
| S|AVB|E_PP | 97.45 | 97.69 |  |
| S|AVB|E_Se | 87.42 | 87.64 |  |
| S|SVT|D_PP | 75.54 | 75.48 | 70 |
| S|SVT|D_Se | 78.87 | 78.04 |  |
| S|SVT|E_PP | 85.33 | 84.65 |  |
| S|SVT|E_Se | 80.58 | 82.30 | 70 |
| S|VT|D_PP | 77.85 | 84.11 | 70 |
| S|VT|D_Se | 83.20 | 82.46 |  |
| S|VT|E_PP | 89.18 | 89.87 |  |
| S|VT|E_Se | 77.68 | 80.18 | 82 |