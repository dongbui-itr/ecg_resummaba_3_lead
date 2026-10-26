# tune_pplus beatft2_e4: validation only (Holter eval side + strip_val2), overlap as stored

score -128.05  target shortfall 108.6  guard violation 2.0

## knobs changed from official

- scale_AFIB: 1.0 -> 0.7
- scale_SVT: 1.0 -> 1.25
- prob_AVB: 0 -> 0.5
- gap_SVT: 0 -> 2
- gap_VT: 0 -> 2
- smooth_VT: 1 -> 0.5
- trim_VT: None -> 0.3
- sinus: 8.0 -> 12
- s_margin: 0 -> 0.1

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 99.14 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 55.83 | 72 |
| B|all|S|Se | 91.33 | 90.94 | 60 |
| B|all|V|PP | 92.84 | 93.94 | 95 |
| B|all|V|Se | 77.74 | 76.70 | 94 |
| B|incartdb|QRS|PP | 99.83 | 99.83 |  |
| B|incartdb|QRS|Se | 99.91 | 99.90 |  |
| B|incartdb|V|PP | 99.54 | 99.51 |  |
| B|incartdb|V|Se | 91.44 | 90.74 |  |
| B|ltaf+inc|S|PP | 56.48 | 57.50 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 91.12 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.54 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 56.98 | 57.96 |  |
| B|ltafdb|S|Se | 91.46 | 91.11 |  |
| B|ltafdb|V|PP | 92.03 | 93.24 |  |
| B|ltafdb|V|Se | 75.16 | 74.01 |  |
| B|nsrdb|QRS|PP | 93.14 | 94.83 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 55.81 | 72 |
| B|svdb|S|Se | 90.19 | 89.29 |  |
| B|svdb|V|PP | 93.25 | 93.68 |  |
| B|svdb|V|Se | 93.55 | 93.55 |  |
| FPH|AFIB | 0.58 | 0.34 |  |
| FPH|AVB | 0.03 | 0.02 |  |
| FPH|SVT | 0.82 | 0.92 |  |
| FPH|VT | 0.15 | 0.15 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.52 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 95.05 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 87.91 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 75.53 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 4.20 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 23.24 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 9.39 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 25.96 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 54.07 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 75.57 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 85.37 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 67.27 | 70 |
| S|AFIB|D_PP | 93.35 | 93.37 | 90 |
| S|AFIB|D_Se | 89.52 | 89.55 |  |
| S|AFIB|E_PP | 84.80 | 84.73 | 92 |
| S|AFIB|E_Se | 91.84 | 91.52 |  |
| S|AVB|D_PP | 87.90 | 87.69 |  |
| S|AVB|D_Se | 85.65 | 86.38 |  |
| S|AVB|E_PP | 97.45 | 97.80 |  |
| S|AVB|E_Se | 87.42 | 87.53 |  |
| S|SVT|D_PP | 75.54 | 75.59 | 70 |
| S|SVT|D_Se | 78.87 | 78.31 |  |
| S|SVT|E_PP | 85.33 | 84.37 |  |
| S|SVT|E_Se | 80.58 | 81.47 | 70 |
| S|VT|D_PP | 77.85 | 80.72 | 70 |
| S|VT|D_Se | 83.20 | 82.68 |  |
| S|VT|E_PP | 89.18 | 88.32 |  |
| S|VT|E_Se | 77.68 | 78.36 | 82 |