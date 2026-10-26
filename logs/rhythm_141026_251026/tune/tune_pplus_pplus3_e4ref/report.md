# tune_pplus pplus3_e4: validation only (Holter eval side + strip_val2), overlap as stored

score -159.46  target shortfall 103.6  guard violation 5.6

## knobs changed from official

- scale_AFIB: 1.0 -> 0.85
- scale_SVT: 1.0 -> 1.25
- scale_VT: 1.0 -> 1.25
- gap_SVT: 0 -> 2
- smooth_SVT: 1 -> 0.5
- smooth_VT: 1 -> 0.5
- trim_VT: None -> 0.1
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20
- avb_maxhr: 60.0 -> 65
- s_afib_n: True -> False
- v_margin: 0 -> 0.1

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 99.09 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 57.12 | 72 |
| B|all|S|Se | 91.33 | 89.91 | 60 |
| B|all|V|PP | 92.84 | 93.87 | 95 |
| B|all|V|Se | 77.74 | 77.31 | 94 |
| B|incartdb|QRS|PP | 99.83 | 99.83 |  |
| B|incartdb|QRS|Se | 99.91 | 99.91 |  |
| B|incartdb|V|PP | 99.54 | 99.66 |  |
| B|incartdb|V|Se | 91.44 | 90.35 |  |
| B|ltaf+inc|S|PP | 56.48 | 58.14 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 90.21 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.52 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 56.98 | 58.58 |  |
| B|ltafdb|S|Se | 91.46 | 90.20 |  |
| B|ltafdb|V|PP | 92.03 | 93.24 |  |
| B|ltafdb|V|Se | 75.16 | 74.79 |  |
| B|nsrdb|QRS|PP | 93.14 | 94.48 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 56.64 | 72 |
| B|svdb|S|Se | 90.19 | 87.09 |  |
| B|svdb|V|PP | 93.25 | 93.18 |  |
| B|svdb|V|Se | 93.55 | 93.42 |  |
| FPH|AFIB | 0.58 | 0.36 |  |
| FPH|AVB | 0.03 | 0.01 |  |
| FPH|SVT | 0.82 | 0.88 |  |
| FPH|VT | 0.15 | 0.16 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.37 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 95.11 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 86.30 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 77.88 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 3.94 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 21.58 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 9.09 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 25.00 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 58.18 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 72.73 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 88.61 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 66.36 | 70 |
| S|AFIB|D_PP | 93.35 | 93.60 | 90 |
| S|AFIB|D_Se | 89.52 | 89.45 |  |
| S|AFIB|E_PP | 84.80 | 85.09 | 92 |
| S|AFIB|E_Se | 91.84 | 91.52 |  |
| S|AVB|D_PP | 87.90 | 88.01 |  |
| S|AVB|D_Se | 85.65 | 85.48 |  |
| S|AVB|E_PP | 97.45 | 98.05 |  |
| S|AVB|E_Se | 87.42 | 87.74 |  |
| S|SVT|D_PP | 75.54 | 75.41 | 70 |
| S|SVT|D_Se | 78.87 | 79.06 |  |
| S|SVT|E_PP | 85.33 | 84.54 |  |
| S|SVT|E_Se | 80.58 | 82.36 | 70 |
| S|VT|D_PP | 77.85 | 85.11 | 70 |
| S|VT|D_Se | 83.20 | 82.28 |  |
| S|VT|E_PP | 89.18 | 89.25 |  |
| S|VT|E_Se | 77.68 | 79.95 | 82 |