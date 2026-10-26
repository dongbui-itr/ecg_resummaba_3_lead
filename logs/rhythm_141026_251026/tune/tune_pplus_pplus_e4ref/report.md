# tune_pplus pplus_e4: validation only (Holter eval side + strip_val2), overlap as stored

score -113.87  target shortfall 93.3  guard violation 10.3

## knobs changed from official

- scale_AFIB: 1.0 -> 0.7
- scale_SVT: 1.0 -> 0.6
- scale_VT: 1.0 -> 1.25
- prob_VT: 0 -> 0.6
- gap_SVT: 0 -> 2
- gap_VT: 0 -> 2
- smooth_AFIB: 1 -> 3
- smooth_SVT: 1 -> 0.5
- smooth_AVB: 1 -> 5
- trim_VT: None -> 0.1
- svt_frac: 0.3 -> 0.5
- onset: 1.25 -> 1.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20
- runs_vt: False -> True
- runs_svt: False -> True
- avb_maxhr: 60.0 -> 65
- s_afib_n: True -> False
- v_margin: 0 -> 0.2

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 98.92 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 56.72 | 72 |
| B|all|S|Se | 91.33 | 89.50 | 60 |
| B|all|V|PP | 92.84 | 94.60 | 95 |
| B|all|V|Se | 77.74 | 76.97 | 94 |
| B|incartdb|QRS|PP | 99.83 | 99.83 |  |
| B|incartdb|QRS|Se | 99.91 | 99.91 |  |
| B|incartdb|V|PP | 99.54 | 99.54 |  |
| B|incartdb|V|Se | 91.44 | 91.47 |  |
| B|ltaf+inc|S|PP | 56.48 | 58.48 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 89.75 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.51 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.93 |  |
| B|ltafdb|S|PP | 56.98 | 59.11 |  |
| B|ltafdb|S|Se | 91.46 | 89.74 |  |
| B|ltafdb|V|PP | 92.03 | 93.99 |  |
| B|ltafdb|V|Se | 75.16 | 74.26 |  |
| B|nsrdb|QRS|PP | 93.14 | 92.77 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 55.59 | 72 |
| B|svdb|S|Se | 90.19 | 87.18 |  |
| B|svdb|V|PP | 93.25 | 93.48 |  |
| B|svdb|V|Se | 93.55 | 93.29 |  |
| FPH|AFIB | 0.58 | 0.26 |  |
| FPH|AVB | 0.03 | 0.01 |  |
| FPH|SVT | 0.82 | 1.09 |  |
| FPH|VT | 0.15 | 0.23 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.62 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 95.07 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 90.00 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 77.18 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 5.55 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 33.61 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 11.90 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 39.42 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 55.00 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 81.25 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 84.31 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 80.00 | 70 |
| S|AFIB|D_PP | 93.35 | 93.85 | 90 |
| S|AFIB|D_Se | 89.52 | 89.09 |  |
| S|AFIB|E_PP | 84.80 | 85.31 | 92 |
| S|AFIB|E_Se | 91.84 | 91.71 |  |
| S|AVB|D_PP | 87.90 | 88.62 |  |
| S|AVB|D_Se | 85.65 | 85.77 |  |
| S|AVB|E_PP | 97.45 | 97.46 |  |
| S|AVB|E_Se | 87.42 | 87.64 |  |
| S|SVT|D_PP | 75.54 | 74.76 | 70 |
| S|SVT|D_Se | 78.87 | 80.33 |  |
| S|SVT|E_PP | 85.33 | 83.46 |  |
| S|SVT|E_Se | 80.58 | 87.92 | 70 |
| S|VT|D_PP | 77.85 | 86.01 | 70 |
| S|VT|D_Se | 83.20 | 83.83 |  |
| S|VT|E_PP | 89.18 | 87.25 |  |
| S|VT|E_Se | 77.68 | 86.79 | 82 |