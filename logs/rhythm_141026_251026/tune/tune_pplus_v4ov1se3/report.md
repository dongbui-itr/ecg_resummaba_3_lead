# tune_pplus v4ov1: validation only (Holter eval side + strip_val2), overlap as stored

score -106.01  target shortfall 106.0  guard violation 0.0

## knobs changed from official

- scale_SVT: 1.0 -> 1.25
- scale_VT: 1.0 -> 1.25
- prob_VT: 0 -> 0.5
- prob_AVB: 0 -> 0.5
- gap_SVT: 0 -> 2
- smooth_SVT: 1 -> 2
- smooth_VT: 1 -> 0.5
- trim_VT: None -> 0.1
- v_margin: 0 -> 0.1
- s_margin: 0 -> 0.1

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 98.96 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 55.11 | 72 |
| B|all|S|Se | 91.33 | 90.26 |  |
| B|all|V|PP | 92.84 | 93.23 | 95 |
| B|all|V|Se | 77.74 | 78.09 |  |
| B|incartdb|QRS|PP | 99.83 | 99.83 |  |
| B|incartdb|QRS|Se | 99.91 | 99.91 |  |
| B|incartdb|V|PP | 99.54 | 99.54 |  |
| B|incartdb|V|Se | 91.44 | 91.25 |  |
| B|ltaf+inc|S|PP | 56.48 | 57.99 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 90.39 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.52 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 56.98 | 58.48 |  |
| B|ltafdb|S|Se | 91.46 | 90.39 |  |
| B|ltafdb|V|PP | 92.03 | 92.48 |  |
| B|ltafdb|V|Se | 75.16 | 75.57 |  |
| B|nsrdb|QRS|PP | 93.14 | 93.14 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 54.95 | 72 |
| B|svdb|S|Se | 90.19 | 89.11 |  |
| B|svdb|V|PP | 93.25 | 93.05 |  |
| B|svdb|V|Se | 93.55 | 94.08 |  |
| FPH|AFIB | 0.58 | 0.56 |  |
| FPH|AVB | 0.03 | 0.02 |  |
| FPH|SVT | 0.82 | 0.90 |  |
| FPH|VT | 0.15 | 0.15 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.38 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 94.97 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 85.48 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 76.94 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 4.16 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 21.16 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 9.60 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 24.04 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 59.19 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 75.00 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 90.91 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 67.27 | 70 |
| S|AFIB|D_PP | 93.35 | 93.26 | 90 |
| S|AFIB|D_Se | 89.52 | 89.45 |  |
| S|AFIB|E_PP | 84.80 | 84.38 | 92 |
| S|AFIB|E_Se | 91.84 | 91.91 |  |
| S|AVB|D_PP | 87.90 | 87.88 |  |
| S|AVB|D_Se | 85.65 | 85.60 |  |
| S|AVB|E_PP | 97.45 | 97.44 |  |
| S|AVB|E_Se | 87.42 | 87.31 |  |
| S|SVT|D_PP | 75.54 | 75.75 | 70 |
| S|SVT|D_Se | 78.87 | 78.14 |  |
| S|SVT|E_PP | 85.33 | 84.64 |  |
| S|SVT|E_Se | 80.58 | 79.63 | 70 |
| S|VT|D_PP | 77.85 | 84.35 | 70 |
| S|VT|D_Se | 83.20 | 82.62 |  |
| S|VT|E_PP | 89.18 | 89.67 |  |
| S|VT|E_Se | 77.68 | 79.73 | 82 |