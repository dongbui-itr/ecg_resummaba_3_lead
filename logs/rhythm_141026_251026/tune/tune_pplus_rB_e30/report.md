# tune_pplus rB_e30: validation only (Holter eval side + strip_val2), overlap as stored

score -140.07  target shortfall 140.1  guard violation 0.0

## knobs changed from official

- scale_AFIB: 1.0 -> 1.25
- prob_VT: 0 -> 0.5
- prob_AVB: 0 -> 0.6
- gap_AFIB: 3 -> 5
- gap_SVT: 0 -> 2
- gap_VT: 0 -> 1
- smooth_SVT: 1 -> 0.5
- smooth_VT: 1 -> 0.5
- smooth_AVB: 1 -> 5
- trim_VT: None -> 0.3
- svt_frac: 0.3 -> 0.5
- onset: 1.25 -> 1.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20
- avb_dur: 2.5 -> 4
- avb_maxhr: 60.0 -> 65
- v_margin: 0 -> 0.2
- s_margin: 0 -> 0.1

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 97.06 | 97.06 | 99.8 |
| B|all|QRS|Se | 99.90 | 99.90 |  |
| B|all|S|PP | 42.41 | 50.67 | 72 |
| B|all|S|Se | 81.90 | 79.09 |  |
| B|all|V|PP | 92.72 | 93.58 | 95 |
| B|all|V|Se | 69.87 | 68.78 |  |
| B|incartdb|QRS|PP | 99.69 | 99.69 |  |
| B|incartdb|QRS|Se | 99.54 | 99.54 |  |
| B|incartdb|V|PP | 99.56 | 99.62 |  |
| B|incartdb|V|Se | 76.58 | 74.58 |  |
| B|ltaf+inc|S|PP | 50.26 | 56.02 | 40 |
| B|ltaf+inc|S|Se | 82.80 | 79.95 |  |
| B|ltafdb|QRS|PP | 98.77 | 98.77 |  |
| B|ltafdb|QRS|Se | 99.90 | 99.90 |  |
| B|ltafdb|S|PP | 50.50 | 56.27 |  |
| B|ltafdb|S|Se | 82.80 | 79.94 |  |
| B|ltafdb|V|PP | 91.89 | 92.92 |  |
| B|ltafdb|V|Se | 67.94 | 66.94 |  |
| B|nsrdb|QRS|PP | 81.23 | 81.23 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.97 | 99.97 |  |
| B|svdb|S|PP | 57.16 | 58.66 | 72 |
| B|svdb|S|Se | 73.41 | 70.99 |  |
| B|svdb|V|PP | 91.93 | 91.76 |  |
| B|svdb|V|Se | 89.78 | 89.13 |  |
| FPH|AFIB | 1.19 | 0.97 |  |
| FPH|AVB | 0.02 | 0.00 |  |
| FPH|SVT | 2.09 | 1.57 |  |
| FPH|VT | 0.09 | 0.10 |  |
| H|ltafdb|AFIB|D_PP | 99.71 | 99.65 | 90 |
| H|ltafdb|AFIB|D_Se | 93.87 | 94.59 | 98 |
| H|ltafdb|AFIB|E_PP | 97.74 | 95.64 | 92 |
| H|ltafdb|AFIB|E_Se | 71.76 | 74.59 | 90 |
| H|ltafdb|SVT|D_PP | 1.84 | 2.31 | 70 |
| H|ltafdb|SVT|D_Se | 12.86 | 13.69 | 70 |
| H|ltafdb|SVT|E_PP | 2.71 | 3.50 | 70 |
| H|ltafdb|SVT|E_Se | 13.46 | 14.42 | 70 |
| H|ltafdb|VT|D_PP | 60.32 | 68.18 | 70 |
| H|ltafdb|VT|D_Se | 43.18 | 42.61 | 70 |
| H|ltafdb|VT|E_PP | 91.43 | 100.00 | 70 |
| H|ltafdb|VT|E_Se | 30.91 | 30.00 | 70 |
| S|AFIB|D_PP | 96.49 | 96.09 | 90 |
| S|AFIB|D_Se | 83.48 | 84.58 |  |
| S|AFIB|E_PP | 90.58 | 89.83 | 92 |
| S|AFIB|E_Se | 86.36 | 86.93 |  |
| S|AVB|D_PP | 86.69 | 86.72 |  |
| S|AVB|D_Se | 86.25 | 87.34 |  |
| S|AVB|E_PP | 96.28 | 97.00 |  |
| S|AVB|E_Se | 87.64 | 88.29 |  |
| S|SVT|D_PP | 72.90 | 73.77 | 70 |
| S|SVT|D_Se | 80.00 | 79.47 |  |
| S|SVT|E_PP | 83.03 | 83.20 |  |
| S|SVT|E_Se | 73.85 | 73.18 | 70 |
| S|VT|D_PP | 71.84 | 73.81 | 70 |
| S|VT|D_Se | 80.77 | 80.36 |  |
| S|VT|E_PP | 79.80 | 79.60 |  |
| S|VT|E_Se | 70.16 | 70.16 | 82 |