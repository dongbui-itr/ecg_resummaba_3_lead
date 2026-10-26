# tune_pplus beatft_e4: validation only (Holter eval side + strip_val2), overlap as stored

score -93.68  target shortfall 93.5  guard violation 0.0

## knobs changed from official

- scale_AFIB: 1.0 -> 0.85
- scale_SVT: 1.0 -> 1.25
- scale_VT: 1.0 -> 1.25
- min_SVT: 1 -> 1.5
- prob_SVT: 0 -> 0.6
- prob_VT: 0 -> 0.5
- prob_AVB: 0 -> 0.7
- gap_SVT: 0 -> 2
- smooth_SVT: 1 -> 2
- trim_VT: None -> 0.1
- svt_frac: 0.3 -> 0.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20
- runs_svt: False -> True
- avb_maxhr: 60.0 -> 65

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 98.96 | 99.12 | 99.8 |
| B|all|QRS|Se | 99.93 | 99.93 |  |
| B|all|S|PP | 52.13 | 59.25 | 72 |
| B|all|S|Se | 91.33 | 89.30 |  |
| B|all|V|PP | 92.84 | 93.15 | 95 |
| B|all|V|Se | 77.74 | 78.27 |  |
| B|incartdb|QRS|PP | 99.83 | 99.83 |  |
| B|incartdb|QRS|Se | 99.91 | 99.89 |  |
| B|incartdb|V|PP | 99.54 | 99.54 |  |
| B|incartdb|V|Se | 91.44 | 90.66 |  |
| B|ltaf+inc|S|PP | 56.48 | 60.50 | 40 |
| B|ltaf+inc|S|Se | 91.46 | 89.49 |  |
| B|ltafdb|QRS|PP | 99.52 | 99.53 |  |
| B|ltafdb|QRS|Se | 99.92 | 99.92 |  |
| B|ltafdb|S|PP | 56.98 | 60.99 |  |
| B|ltafdb|S|Se | 91.46 | 89.49 |  |
| B|ltafdb|V|PP | 92.03 | 92.39 |  |
| B|ltafdb|V|Se | 75.16 | 75.84 |  |
| B|nsrdb|QRS|PP | 93.14 | 94.71 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 54.31 | 57.34 | 72 |
| B|svdb|S|Se | 90.19 | 87.49 |  |
| B|svdb|V|PP | 93.25 | 92.95 |  |
| B|svdb|V|Se | 93.55 | 94.40 |  |
| FPH|AFIB | 0.58 | 0.37 |  |
| FPH|AVB | 0.03 | 0.01 |  |
| FPH|SVT | 0.82 | 0.92 |  |
| FPH|VT | 0.15 | 0.14 |  |
| H|ltafdb|AFIB|D_PP | 99.27 | 99.47 | 90 |
| H|ltafdb|AFIB|D_Se | 94.98 | 95.11 | 98 |
| H|ltafdb|AFIB|E_PP | 85.17 | 86.37 | 92 |
| H|ltafdb|AFIB|E_Se | 77.18 | 77.65 | 90 |
| H|ltafdb|SVT|D_PP | 3.90 | 5.97 | 70 |
| H|ltafdb|SVT|D_Se | 19.50 | 31.54 | 70 |
| H|ltafdb|SVT|E_PP | 9.47 | 12.97 | 70 |
| H|ltafdb|SVT|E_Se | 22.12 | 37.50 | 70 |
| H|ltafdb|VT|D_PP | 52.82 | 57.21 | 70 |
| H|ltafdb|VT|D_Se | 74.43 | 74.43 | 70 |
| H|ltafdb|VT|E_PP | 86.08 | 89.87 | 70 |
| H|ltafdb|VT|E_Se | 65.45 | 68.18 | 70 |
| S|AFIB|D_PP | 93.35 | 93.13 | 90 |
| S|AFIB|D_Se | 89.52 | 90.13 |  |
| S|AFIB|E_PP | 84.80 | 83.76 | 92 |
| S|AFIB|E_Se | 91.84 | 92.03 |  |
| S|AVB|D_PP | 87.90 | 87.83 |  |
| S|AVB|D_Se | 85.65 | 86.87 |  |
| S|AVB|E_PP | 97.45 | 98.06 |  |
| S|AVB|E_Se | 87.42 | 88.07 |  |
| S|SVT|D_PP | 75.54 | 75.34 | 70 |
| S|SVT|D_Se | 78.87 | 79.19 |  |
| S|SVT|E_PP | 85.33 | 84.36 |  |
| S|SVT|E_Se | 80.58 | 87.15 | 70 |
| S|VT|D_PP | 77.85 | 84.77 | 70 |
| S|VT|D_Se | 83.20 | 82.37 |  |
| S|VT|E_PP | 89.18 | 88.31 |  |
| S|VT|E_Se | 77.68 | 79.50 | 82 |