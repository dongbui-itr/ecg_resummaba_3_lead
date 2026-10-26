# tune_pplus beatft2_e2: validation only (Holter eval side + strip_val2), overlap as stored

score -98.49  target shortfall 98.5  guard violation 0.0

## knobs changed from official

- scale_AFIB: 1.0 -> 0.7
- scale_SVT: 1.0 -> 1.25
- scale_VT: 1.0 -> 1.25
- scale_AVB: 1.0 -> 0.5
- prob_AVB: 0 -> 0.6
- gap_SVT: 0 -> 2
- gap_VT: 0 -> 2
- trim_VT: None -> 0.3
- svt_frac: 0.3 -> 0.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20
- avb_maxhr: 60.0 -> 65
- v_margin: 0 -> 0.3
- s_margin: 0 -> 0.2

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 99.08 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 57.91 | 72 |
| B|all|S|Se | 91.33 | 89.88 |  |
| B|all|V|PP | 92.84 | 95.13 | 95 |
| B|all|V|Se | 77.74 | 76.54 |  |
| B|incartdb|QRS|PP | 99.83 | 99.82 |  |
| B|incartdb|QRS|Se | 99.91 | 99.89 |  |
| B|incartdb|V|PP | 99.54 | 99.53 |  |
| B|incartdb|V|Se | 91.44 | 89.19 |  |
| B|ltaf+inc|S|PP | 56.48 | 59.11 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 90.06 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.51 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 56.98 | 59.57 |  |
| B|ltafdb|S|Se | 91.46 | 90.06 |  |
| B|ltafdb|V|PP | 92.03 | 94.64 |  |
| B|ltafdb|V|Se | 75.16 | 74.03 |  |
| B|nsrdb|QRS|PP | 93.14 | 94.41 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 56.52 | 72 |
| B|svdb|S|Se | 90.19 | 88.17 |  |
| B|svdb|V|PP | 93.25 | 93.31 |  |
| B|svdb|V|Se | 93.55 | 93.49 |  |
| FPH|AFIB | 0.58 | 0.34 |  |
| FPH|AVB | 0.03 | 0.01 |  |
| FPH|SVT | 0.82 | 0.86 |  |
| FPH|VT | 0.15 | 0.14 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.41 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 95.08 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 87.01 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 76.71 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 4.22 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 21.99 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 9.54 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 25.00 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 58.74 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 74.43 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 90.00 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 68.18 | 70 |
| S|AFIB|D_PP | 93.35 | 93.07 | 90 |
| S|AFIB|D_Se | 89.52 | 89.76 |  |
| S|AFIB|E_PP | 84.80 | 84.11 | 92 |
| S|AFIB|E_Se | 91.84 | 91.78 |  |
| S|AVB|D_PP | 87.90 | 88.36 |  |
| S|AVB|D_Se | 85.65 | 85.58 |  |
| S|AVB|E_PP | 97.45 | 98.04 |  |
| S|AVB|E_Se | 87.42 | 87.42 |  |
| S|SVT|D_PP | 75.54 | 75.78 | 70 |
| S|SVT|D_Se | 78.87 | 77.92 |  |
| S|SVT|E_PP | 85.33 | 84.43 |  |
| S|SVT|E_Se | 80.58 | 80.91 | 70 |
| S|VT|D_PP | 77.85 | 81.66 | 70 |
| S|VT|D_Se | 83.20 | 82.36 |  |
| S|VT|E_PP | 89.18 | 88.47 |  |
| S|VT|E_Se | 77.68 | 79.04 | 82 |