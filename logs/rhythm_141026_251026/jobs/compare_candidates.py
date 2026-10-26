"""compare_candidates.py [ref_key] : every tune_pplus_* candidate (its tuned decode) against the
REFERENCE = v4 official decode at overlap 1 (tune_pplus_<ref>/base_metrics.json), on validation only.

  violations : metrics the 5.4.1 table / the user's rule pin (Se cells, QRS, AF false-episode rate)
               that fall below the reference by more than the tolerance
  shortfall  : the tuner's target shortfall (lower = closer to the 5.4.1 goals)
"""
import glob
import json
import os
import sys

S = '/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/strip_eval'
ref_key = sys.argv[1] if len(sys.argv) > 1 else 'v4ov1_tight'
ref = json.load(open(f'{S}/tune_pplus_{ref_key}/base_metrics.json'))
# metrics that may not drop vs the reference (tolerance in points; FPH: relative)
PIN = {'B|all|QRS|Se': 0.05, 'B|all|QRS|PP': 0.1, 'B|all|V|Se': 0.3, 'B|all|S|Se': 0.3, 'B|svdb|S|Se': 0.5,
       'H|ltafdb|AFIB|D_Se': 0.5, 'H|ltafdb|AFIB|E_Se': 1.0, 'H|ltafdb|VT|E_Se': 2.0, 'H|ltafdb|VT|D_Se': 2.0,
       'S|AFIB|E_Se': 0.5, 'S|AFIB|D_Se': 0.5, 'S|AVB|E_Se': 0.3, 'S|AVB|D_Se': 0.5, 'S|SVT|E_Se': 1.0, 'S|SVT|D_Se': 1.0,
       'S|VT|E_Se': 1.0, 'S|VT|D_Se': 1.0, 'S|AVB|E_PP': 0.5, 'S|AVB|D_PP': 0.5}
SE_TOL = float(os.environ['SE_TOL']) if os.environ.get('SE_TOL') else None
if SE_TOL is not None:
    for k in list(PIN):
        if k.startswith('B|') and (k.endswith('|V|Se') or k.endswith('|S|Se')):
            PIN[k] = SE_TOL
GAIN = ['B|all|V|PP', 'B|all|S|PP', 'B|svdb|S|PP', 'B|ltaf+inc|S|PP', 'H|ltafdb|AFIB|E_PP', 'H|ltafdb|AFIB|D_PP',
        'S|AFIB|E_PP', 'S|VT|D_PP', 'S|SVT|D_PP', 'H|ltafdb|VT|D_PP', 'FPH|AFIB', 'FPH|SVT', 'FPH|VT', 'FPH|AVB']
rows = []
for d in sorted(glob.glob(f'{S}/tune_pplus_*')):
    key = os.path.basename(d)[len('tune_pplus_'):]
    try:
        best = json.load(open(f'{d}/best.json'))
    except OSError:
        continue
    m = best['metrics']
    viol = {k: round(ref[k] - m[k], 2) for k, tol in PIN.items() if k in m and m[k] < ref[k] - tol}
    for c in ('AFIB', 'SVT', 'VT', 'AVB'):
        k = f'FPH|{c}'
        if m[k] > ref[k] * 1.1 + 0.02:
            viol[k] = round(m[k] - ref[k], 3)
    rows.append((key, best['short'], best['guard'], viol, m))
rows.sort(key=lambda r: (len(r[3]), r[1]))
print(f'reference: {ref_key} official decode (overlap 1)\n')
for key, short, guard, viol, m in rows:
    print(f'{key:16s} shortfall {short:6.1f}  own-guard {guard:4.1f}  pinned violations vs ref: {len(viol)}  {viol}')
print('\ngains vs reference (tuned candidate - ref):')
hdr = f'{"metric":22s}' + ''.join(f'{r[0][:10]:>11s}' for r in rows)
print(hdr)
for k in GAIN:
    print(f'{k:22s}' + ''.join(f'{r[4][k] - ref[k]:+11.2f}' for r in rows))
print('\nabsolute (tuned):')
print(f'{"metric":22s}{"ref":>9s}' + ''.join(f'{r[0][:10]:>11s}' for r in rows))
for k in sorted(set(GAIN) | set(PIN)):
    print(f'{k:22s}{ref[k]:9.2f}' + ''.join(f'{r[4][k]:11.2f}' for r in rows))
