# tune_pplus pplus3_e3: validation only (Holter eval side + strip_val2), overlap as stored

score -159.23  target shortfall 106.1  guard violation 5.3

## knobs changed from official

- scale_AFIB: 1.0 -> 0.7
- scale_VT: 1.0 -> 1.25
- prob_VT: 0 -> 0.5
- prob_AVB: 0 -> 0.5
- gap_SVT: 0 -> 2
- smooth_SVT: 1 -> 0.5
- smooth_VT: 1 -> 0.5
- smooth_AVB: 1 -> 3
- trim_VT: None -> 0.1
- svt_frac: 0.3 -> 0.5
- onset: 1.25 -> 1.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20
- avb_maxhr: 60.0 -> 65
- s_afib_n: True -> False
- v_margin: 0 -> 0.1

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 98.99 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 56.65 | 72 |
| B|all|S|Se | 91.33 | 89.97 | 60 |
| B|all|V|PP | 92.84 | 93.86 | 95 |
| B|all|V|Se | 77.74 | 77.22 | 94 |
| B|incartdb|QRS|PP | 99.83 | 99.83 |  |
| B|incartdb|QRS|Se | 99.91 | 99.93 |  |
| B|incartdb|V|PP | 99.54 | 99.66 |  |
| B|incartdb|V|Se | 91.44 | 90.60 |  |
| B|ltaf+inc|S|PP | 56.48 | 57.73 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 90.28 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.50 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.93 |  |
| B|ltafdb|S|PP | 56.98 | 58.13 |  |
| B|ltafdb|S|Se | 91.46 | 90.27 |  |
| B|ltafdb|V|PP | 92.03 | 93.22 |  |
| B|ltafdb|V|Se | 75.16 | 74.66 |  |
| B|nsrdb|QRS|PP | 93.14 | 93.57 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 56.33 | 72 |
| B|svdb|S|Se | 90.19 | 87.09 |  |
| B|svdb|V|PP | 93.25 | 93.12 |  |
| B|svdb|V|Se | 93.55 | 93.42 |  |
| FPH|AFIB | 0.58 | 0.36 |  |
| FPH|AVB | 0.03 | 0.01 |  |
| FPH|SVT | 0.82 | 0.87 |  |
| FPH|VT | 0.15 | 0.17 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.58 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 95.08 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 87.18 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 76.00 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 3.87 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 20.75 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 8.89 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 24.04 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 56.09 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 73.30 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 85.54 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 67.27 | 70 |
| S|AFIB|D_PP | 93.35 | 94.05 | 90 |
| S|AFIB|D_Se | 89.52 | 89.46 |  |
| S|AFIB|E_PP | 84.80 | 85.37 | 92 |
| S|AFIB|E_Se | 91.84 | 91.59 |  |
| S|AVB|D_PP | 87.90 | 87.99 |  |
| S|AVB|D_Se | 85.65 | 86.62 |  |
| S|AVB|E_PP | 97.45 | 98.19 |  |
| S|AVB|E_Se | 87.42 | 88.61 |  |
| S|SVT|D_PP | 75.54 | 75.35 | 70 |
| S|SVT|D_Se | 78.87 | 79.52 |  |
| S|SVT|E_PP | 85.33 | 84.41 |  |
| S|SVT|E_Se | 80.58 | 81.64 | 70 |
| S|VT|D_PP | 77.85 | 85.40 | 70 |
| S|VT|D_Se | 83.20 | 82.63 |  |
| S|VT|E_PP | 89.18 | 89.30 |  |
| S|VT|E_Se | 77.68 | 80.18 | 82 |