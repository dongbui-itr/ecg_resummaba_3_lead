# tune_pplus v4ov1: validation only (Holter eval side + strip_val2), overlap as stored

score -110.04  target shortfall 110.0  guard violation 0.1

## knobs changed from official

- scale_AFIB: 1.0 -> 0.7
- scale_VT: 1.0 -> 1.25
- scale_AVB: 1.0 -> 0.7
- min_AVB: 2 -> 3
- prob_VT: 0 -> 0.5
- gap_SVT: 0 -> 2
- trim_VT: None -> 0.1
- svt_frac: 0.3 -> 0.5

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 98.96 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 51.86 | 72 |
| B|all|S|Se | 91.33 | 91.56 | 60 |
| B|all|V|PP | 92.84 | 92.84 | 95 |
| B|all|V|Se | 77.74 | 77.74 | 94 |
| B|incartdb|QRS|PP | 99.83 | 99.83 |  |
| B|incartdb|QRS|Se | 99.91 | 99.91 |  |
| B|incartdb|V|PP | 99.54 | 99.54 |  |
| B|incartdb|V|Se | 91.44 | 91.44 |  |
| B|ltaf+inc|S|PP | 56.48 | 56.12 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 91.69 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.52 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 56.98 | 56.62 |  |
| B|ltafdb|S|Se | 91.46 | 91.69 |  |
| B|ltafdb|V|PP | 92.03 | 92.03 |  |
| B|ltafdb|V|Se | 75.16 | 75.16 |  |
| B|nsrdb|QRS|PP | 93.14 | 93.14 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 54.34 | 72 |
| B|svdb|S|Se | 90.19 | 90.33 |  |
| B|svdb|V|PP | 93.25 | 93.25 |  |
| B|svdb|V|Se | 93.55 | 93.55 |  |
| FPH|AFIB | 0.58 | 0.39 |  |
| FPH|AVB | 0.03 | 0.02 |  |
| FPH|SVT | 0.82 | 0.87 |  |
| FPH|VT | 0.15 | 0.16 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.34 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 94.89 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 89.16 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 75.76 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 4.08 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 21.99 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 9.51 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 25.00 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 56.20 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 77.27 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 87.06 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 70.91 | 70 |
| S|AFIB|D_PP | 93.35 | 93.79 | 90 |
| S|AFIB|D_Se | 89.52 | 88.71 |  |
| S|AFIB|E_PP | 84.80 | 85.56 | 92 |
| S|AFIB|E_Se | 91.84 | 91.27 |  |
| S|AVB|D_PP | 87.90 | 88.21 |  |
| S|AVB|D_Se | 85.65 | 84.84 |  |
| S|AVB|E_PP | 97.45 | 97.54 |  |
| S|AVB|E_Se | 87.42 | 86.55 |  |
| S|SVT|D_PP | 75.54 | 75.26 | 70 |
| S|SVT|D_Se | 78.87 | 79.22 |  |
| S|SVT|E_PP | 85.33 | 84.49 |  |
| S|SVT|E_Se | 80.58 | 80.86 | 70 |
| S|VT|D_PP | 77.85 | 84.04 | 70 |
| S|VT|D_Se | 83.20 | 82.78 |  |
| S|VT|E_PP | 89.18 | 89.65 |  |
| S|VT|E_Se | 77.68 | 79.50 | 82 |