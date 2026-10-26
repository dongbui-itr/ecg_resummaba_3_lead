# tune_pplus beatft2_e3: validation only (Holter eval side + strip_val2), overlap as stored

score -128.85  target shortfall 105.3  guard violation 2.4

## knobs changed from official

- scale_AFIB: 1.0 -> 0.7
- scale_VT: 1.0 -> 1.25
- scale_AVB: 1.0 -> 0.7
- prob_VT: 0 -> 0.5
- gap_SVT: 0 -> 2
- gap_VT: 0 -> 2
- smooth_SVT: 1 -> 0.5
- smooth_AVB: 1 -> 3
- trim_VT: None -> 0.1
- sinus: 8.0 -> 12
- s_margin: 0 -> 0.1

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 99.12 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 56.03 | 72 |
| B|all|S|Se | 91.33 | 90.86 | 60 |
| B|all|V|PP | 92.84 | 94.02 | 95 |
| B|all|V|Se | 77.74 | 76.62 | 94 |
| B|incartdb|QRS|PP | 99.83 | 99.83 |  |
| B|incartdb|QRS|Se | 99.91 | 99.90 |  |
| B|incartdb|V|PP | 99.54 | 99.51 |  |
| B|incartdb|V|Se | 91.44 | 90.57 |  |
| B|ltaf+inc|S|PP | 56.48 | 57.65 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 91.04 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.53 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 56.98 | 58.11 |  |
| B|ltafdb|S|Se | 91.46 | 91.03 |  |
| B|ltafdb|V|PP | 92.03 | 93.32 |  |
| B|ltafdb|V|Se | 75.16 | 73.94 |  |
| B|nsrdb|QRS|PP | 93.14 | 94.68 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 56.01 | 72 |
| B|svdb|S|Se | 90.19 | 89.29 |  |
| B|svdb|V|PP | 93.25 | 93.80 |  |
| B|svdb|V|Se | 93.55 | 93.62 |  |
| FPH|AFIB | 0.58 | 0.35 |  |
| FPH|AVB | 0.03 | 0.02 |  |
| FPH|SVT | 0.82 | 0.84 |  |
| FPH|VT | 0.15 | 0.14 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.52 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 95.06 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 87.43 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 75.76 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 4.22 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 21.99 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 9.58 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 25.00 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 57.14 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 77.27 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 89.02 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 70.00 | 70 |
| S|AFIB|D_PP | 93.35 | 93.23 | 90 |
| S|AFIB|D_Se | 89.52 | 89.61 |  |
| S|AFIB|E_PP | 84.80 | 84.43 | 92 |
| S|AFIB|E_Se | 91.84 | 91.52 |  |
| S|AVB|D_PP | 87.90 | 88.00 |  |
| S|AVB|D_Se | 85.65 | 85.75 |  |
| S|AVB|E_PP | 97.45 | 98.04 |  |
| S|AVB|E_Se | 87.42 | 87.31 |  |
| S|SVT|D_PP | 75.54 | 75.88 | 70 |
| S|SVT|D_Se | 78.87 | 78.04 |  |
| S|SVT|E_PP | 85.33 | 84.61 |  |
| S|SVT|E_Se | 80.58 | 80.86 | 70 |
| S|VT|D_PP | 77.85 | 85.21 | 70 |
| S|VT|D_Se | 83.20 | 82.30 |  |
| S|VT|E_PP | 89.18 | 88.31 |  |
| S|VT|E_Se | 77.68 | 79.73 | 82 |