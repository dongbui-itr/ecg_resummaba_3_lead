# tune_pplus beatft_e1: validation only (Holter eval side + strip_val2), overlap as stored

score -170.04  target shortfall 105.2  guard violation 6.5

## knobs changed from official

- scale_AFIB: 1.0 -> 0.7
- scale_SVT: 1.0 -> 1.25
- min_AVB: 2 -> 3
- prob_AVB: 0 -> 0.5
- gap_SVT: 0 -> 2
- smooth_VT: 1 -> 0.5
- trim_VT: None -> 0.3
- svt_frac: 0.3 -> 0.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20
- s_afib_n: True -> False
- v_margin: 0 -> 0.1

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 99.09 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 56.91 | 72 |
| B|all|S|Se | 91.33 | 89.55 | 60 |
| B|all|V|PP | 92.84 | 94.01 | 95 |
| B|all|V|Se | 77.74 | 77.92 | 94 |
| B|incartdb|QRS|PP | 99.83 | 99.83 |  |
| B|incartdb|QRS|Se | 99.91 | 99.91 |  |
| B|incartdb|V|PP | 99.54 | 99.53 |  |
| B|incartdb|V|Se | 91.44 | 90.23 |  |
| B|ltaf+inc|S|PP | 56.48 | 57.94 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 89.77 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.53 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 56.98 | 58.38 |  |
| B|ltafdb|S|Se | 91.46 | 89.77 |  |
| B|ltafdb|V|PP | 92.03 | 93.33 |  |
| B|ltafdb|V|Se | 75.16 | 75.48 |  |
| B|nsrdb|QRS|PP | 93.14 | 94.35 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 57.25 | 72 |
| B|svdb|S|Se | 90.19 | 87.58 |  |
| B|svdb|V|PP | 93.25 | 93.17 |  |
| B|svdb|V|Se | 93.55 | 94.21 |  |
| FPH|AFIB | 0.58 | 0.34 |  |
| FPH|AVB | 0.03 | 0.02 |  |
| FPH|SVT | 0.82 | 0.85 |  |
| FPH|VT | 0.15 | 0.14 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.42 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 95.08 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 87.45 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 76.00 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 4.24 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 21.99 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 9.73 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 25.00 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 56.36 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 75.57 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 87.80 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 69.09 | 70 |
| S|AFIB|D_PP | 93.35 | 93.16 | 90 |
| S|AFIB|D_Se | 89.52 | 89.85 |  |
| S|AFIB|E_PP | 84.80 | 84.15 | 92 |
| S|AFIB|E_Se | 91.84 | 91.65 |  |
| S|AVB|D_PP | 87.90 | 87.68 |  |
| S|AVB|D_Se | 85.65 | 86.29 |  |
| S|AVB|E_PP | 97.45 | 97.92 |  |
| S|AVB|E_Se | 87.42 | 87.31 |  |
| S|SVT|D_PP | 75.54 | 75.70 | 70 |
| S|SVT|D_Se | 78.87 | 77.98 |  |
| S|SVT|E_PP | 85.33 | 84.48 |  |
| S|SVT|E_Se | 80.58 | 81.25 | 70 |
| S|VT|D_PP | 77.85 | 80.95 | 70 |
| S|VT|D_Se | 83.20 | 82.44 |  |
| S|VT|E_PP | 89.18 | 87.69 |  |
| S|VT|E_Se | 77.68 | 78.13 | 82 |