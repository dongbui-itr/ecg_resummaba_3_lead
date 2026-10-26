# tune_pplus distill2_30k: validation only (Holter eval side + strip_val2), overlap as stored

score -185.60  target shortfall 185.6  guard violation 0.0

## knobs changed from official

- scale_VT: 1.0 -> 1.25
- prob_VT: 0 -> 0.5
- gap_SVT: 0 -> 2
- gap_VT: 0 -> 1
- smooth_SVT: 1 -> 0.5
- smooth_VT: 1 -> 0.5
- smooth_AVB: 1 -> 3
- trim_VT: None -> 0.3
- svt_frac: 0.3 -> 0.5
- sinus: 8.0 -> 12
- long_inv: 10.0 -> 20
- avb_dur: 2.5 -> 4

## validation metrics (official -> tuned)

| metric | official | tuned | target |
|---|---|---|---|
| B|all|QRS|PP | 96.89 | 96.89 | 99.8 |
| B|all|QRS|Se | 99.90 | 99.90 |  |
| B|all|S|PP | 33.30 | 33.72 | 72 |
| B|all|S|Se | 72.14 | 72.07 |  |
| B|all|V|PP | 82.89 | 82.89 | 95 |
| B|all|V|Se | 76.46 | 76.46 |  |
| B|incartdb|QRS|PP | 99.44 | 99.44 |  |
| B|incartdb|QRS|Se | 99.48 | 99.48 |  |
| B|incartdb|V|PP | 99.67 | 99.67 |  |
| B|incartdb|V|Se | 75.37 | 75.37 |  |
| B|ltaf+inc|S|PP | 52.96 | 53.19 | 40 |
| B|ltaf+inc|S|Se | 71.65 | 71.57 |  |
| B|ltafdb|QRS|PP | 98.84 | 98.84 |  |
| B|ltafdb|QRS|Se | 99.90 | 99.90 |  |
| B|ltafdb|S|PP | 53.40 | 53.63 |  |
| B|ltafdb|S|Se | 71.64 | 71.57 |  |
| B|ltafdb|V|PP | 81.62 | 81.62 |  |
| B|ltafdb|V|Se | 75.83 | 75.83 |  |
| B|nsrdb|QRS|PP | 79.32 | 79.32 |  |
| B|nsrdb|QRS|Se | 100.00 | 100.00 |  |
| B|svdb|QRS|PP | 99.86 | 99.86 |  |
| B|svdb|QRS|Se | 99.98 | 99.98 |  |
| B|svdb|S|PP | 62.41 | 62.41 | 72 |
| B|svdb|S|Se | 76.79 | 76.79 |  |
| B|svdb|V|PP | 88.15 | 88.15 |  |
| B|svdb|V|Se | 90.56 | 90.56 |  |
| FPH|AFIB | 1.51 | 1.30 |  |
| FPH|AVB | 0.01 | 0.00 |  |
| FPH|SVT | 0.83 | 0.85 |  |
| FPH|VT | 0.14 | 0.14 |  |
| H|ltafdb|AFIB|D_PP | 99.79 | 99.75 | 90 |
| H|ltafdb|AFIB|D_Se | 93.72 | 93.83 | 98 |
| H|ltafdb|AFIB|E_PP | 94.77 | 94.59 | 92 |
| H|ltafdb|AFIB|E_Se | 64.71 | 66.82 | 90 |
| H|ltafdb|SVT|D_PP | 1.51 | 1.78 | 70 |
| H|ltafdb|SVT|D_Se | 5.81 | 7.05 | 70 |
| H|ltafdb|SVT|E_PP | 3.43 | 3.95 | 70 |
| H|ltafdb|SVT|E_Se | 5.77 | 7.69 | 70 |
| H|ltafdb|VT|D_PP | 34.16 | 37.13 | 70 |
| H|ltafdb|VT|D_Se | 47.16 | 50.00 | 70 |
| H|ltafdb|VT|E_PP | 67.31 | 73.08 | 70 |
| H|ltafdb|VT|E_Se | 34.55 | 37.27 | 70 |
| S|AFIB|D_PP | 97.20 | 97.05 | 90 |
| S|AFIB|D_Se | 80.05 | 80.92 |  |
| S|AFIB|E_PP | 91.64 | 91.23 | 92 |
| S|AFIB|E_Se | 82.15 | 82.41 |  |
| S|AVB|D_PP | 86.30 | 86.32 |  |
| S|AVB|D_Se | 86.21 | 86.80 |  |
| S|AVB|E_PP | 96.86 | 96.85 |  |
| S|AVB|E_Se | 87.20 | 87.09 |  |
| S|SVT|D_PP | 74.25 | 73.96 | 70 |
| S|SVT|D_Se | 73.96 | 74.24 |  |
| S|SVT|E_PP | 79.93 | 79.67 |  |
| S|SVT|E_Se | 64.44 | 64.66 | 70 |
| S|VT|D_PP | 70.15 | 73.75 | 70 |
| S|VT|D_Se | 79.42 | 79.44 |  |
| S|VT|E_PP | 78.42 | 81.97 |  |
| S|VT|E_Se | 72.44 | 73.12 | 82 |