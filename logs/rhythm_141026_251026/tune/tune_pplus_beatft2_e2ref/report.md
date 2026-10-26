# tune_pplus beatft2_e2: validation only (Holter eval side + strip_val2), overlap as stored

score -121.84  target shortfall 106.7  guard violation 1.5

## knobs changed from official

- scale_AFIB: 1.0 -> 0.7
- scale_SVT: 1.0 -> 1.25
- scale_VT: 1.0 -> 1.25
- scale_AVB: 1.0 -> 0.5
- prob_VT: 0 -> 0.5
- prob_AVB: 0 -> 0.5
- gap_AFIB: 3 -> 10
- gap_SVT: 0 -> 2
- gap_VT: 0 -> 2
- smooth_SVT: 1 -> 0.5
- trim_VT: None -> 0.3
- svt_frac: 0.3 -> 0.5
- sinus: 8.0 -> 12
- avb_maxhr: 60.0 -> 65
- s_margin: 0 -> 0.1

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 99.08 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 55.58 | 72 |
| B|all|S|Se | 91.33 | 91.00 | 60 |
| B|all|V|PP | 92.84 | 93.86 | 95 |
| B|all|V|Se | 77.74 | 76.84 | 94 |
| B|incartdb|QRS|PP | 99.83 | 99.82 |  |
| B|incartdb|QRS|Se | 99.91 | 99.89 |  |
| B|incartdb|V|PP | 99.54 | 99.51 |  |
| B|incartdb|V|Se | 91.44 | 90.80 |  |
| B|ltaf+inc|S|PP | 56.48 | 57.36 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 91.17 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.51 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 56.98 | 57.83 |  |
| B|ltafdb|S|Se | 91.46 | 91.17 |  |
| B|ltafdb|V|PP | 92.03 | 93.11 |  |
| B|ltafdb|V|Se | 75.16 | 74.17 |  |
| B|nsrdb|QRS|PP | 93.14 | 94.41 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 55.59 | 72 |
| B|svdb|S|Se | 90.19 | 89.43 |  |
| B|svdb|V|PP | 93.25 | 93.93 |  |
| B|svdb|V|Se | 93.55 | 93.68 |  |
| FPH|AFIB | 0.58 | 0.32 |  |
| FPH|AVB | 0.03 | 0.02 |  |
| FPH|SVT | 0.82 | 0.92 |  |
| FPH|VT | 0.15 | 0.14 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.43 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 95.12 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 86.08 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 76.47 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 4.26 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 23.24 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 9.52 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 25.96 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 57.02 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 76.14 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 92.31 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 69.09 | 70 |
| S|AFIB|D_PP | 93.35 | 93.51 | 90 |
| S|AFIB|D_Se | 89.52 | 89.61 |  |
| S|AFIB|E_PP | 84.80 | 85.07 | 92 |
| S|AFIB|E_Se | 91.84 | 91.59 |  |
| S|AVB|D_PP | 87.90 | 88.51 |  |
| S|AVB|D_Se | 85.65 | 85.60 |  |
| S|AVB|E_PP | 97.45 | 97.80 |  |
| S|AVB|E_Se | 87.42 | 87.53 |  |
| S|SVT|D_PP | 75.54 | 75.49 | 70 |
| S|SVT|D_Se | 78.87 | 78.46 |  |
| S|SVT|E_PP | 85.33 | 84.52 |  |
| S|SVT|E_Se | 80.58 | 81.80 | 70 |
| S|VT|D_PP | 77.85 | 80.22 | 70 |
| S|VT|D_Se | 83.20 | 82.88 |  |
| S|VT|E_PP | 89.18 | 88.22 |  |
| S|VT|E_Se | 77.68 | 79.27 | 82 |