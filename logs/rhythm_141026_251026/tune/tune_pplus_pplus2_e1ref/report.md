# tune_pplus pplus2_e1: validation only (Holter eval side + strip_val2), overlap as stored

score -142.69  target shortfall 108.6  guard violation 3.4

## knobs changed from official

- scale_AFIB: 1.0 -> 0.7
- scale_SVT: 1.0 -> 1.25
- scale_VT: 1.0 -> 1.25
- gap_SVT: 0 -> 2
- smooth_SVT: 1 -> 0.5
- smooth_VT: 1 -> 0.5
- smooth_AVB: 1 -> 3
- trim_VT: None -> 0.1
- svt_frac: 0.3 -> 0.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20
- avb_maxhr: 60.0 -> 65
- s_afib_n: True -> False

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 98.89 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 53.67 | 72 |
| B|all|S|Se | 91.33 | 90.72 | 60 |
| B|all|V|PP | 92.84 | 93.13 | 95 |
| B|all|V|Se | 77.74 | 77.63 | 94 |
| B|incartdb|QRS|PP | 99.83 | 99.83 |  |
| B|incartdb|QRS|Se | 99.91 | 99.90 |  |
| B|incartdb|V|PP | 99.54 | 99.48 |  |
| B|incartdb|V|Se | 91.44 | 91.16 |  |
| B|ltaf+inc|S|PP | 56.48 | 56.10 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 90.95 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.49 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 56.98 | 56.55 |  |
| B|ltafdb|S|Se | 91.46 | 90.94 |  |
| B|ltafdb|V|PP | 92.03 | 92.36 |  |
| B|ltafdb|V|Se | 75.16 | 75.05 |  |
| B|nsrdb|QRS|PP | 93.14 | 92.61 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 55.60 | 72 |
| B|svdb|S|Se | 90.19 | 88.66 |  |
| B|svdb|V|PP | 93.25 | 93.13 |  |
| B|svdb|V|Se | 93.55 | 93.62 |  |
| FPH|AFIB | 0.58 | 0.26 |  |
| FPH|AVB | 0.03 | 0.01 |  |
| FPH|SVT | 0.82 | 0.73 |  |
| FPH|VT | 0.15 | 0.14 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.60 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 95.06 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 89.21 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 73.88 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 4.41 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 20.33 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 10.09 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 23.08 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 58.01 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 76.14 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 87.80 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 69.09 | 70 |
| S|AFIB|D_PP | 93.35 | 94.01 | 90 |
| S|AFIB|D_Se | 89.52 | 88.95 |  |
| S|AFIB|E_PP | 84.80 | 86.25 | 92 |
| S|AFIB|E_Se | 91.84 | 90.95 |  |
| S|AVB|D_PP | 87.90 | 88.48 |  |
| S|AVB|D_Se | 85.65 | 85.32 |  |
| S|AVB|E_PP | 97.45 | 98.16 |  |
| S|AVB|E_Se | 87.42 | 87.09 |  |
| S|SVT|D_PP | 75.54 | 75.86 | 70 |
| S|SVT|D_Se | 78.87 | 77.48 |  |
| S|SVT|E_PP | 85.33 | 85.03 |  |
| S|SVT|E_Se | 80.58 | 79.74 | 70 |
| S|VT|D_PP | 77.85 | 84.64 | 70 |
| S|VT|D_Se | 83.20 | 82.59 |  |
| S|VT|E_PP | 89.18 | 88.32 |  |
| S|VT|E_Se | 77.68 | 78.36 | 82 |