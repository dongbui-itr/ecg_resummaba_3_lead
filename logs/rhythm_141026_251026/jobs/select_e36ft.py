"""select_e36ft.py : per-epoch validation (eval split) rhythm Se/+P per class + beat N/S/V Se/+P for the e36 fine-tunes,
against r30k_A_e36 (run A epoch 36, must not drop) and the teacher beatft2_e2 (AFIB must exceed)."""
import re, glob
R='/media/Project/ECG/Model_Dong/ecgr_rhythm'
def parse(path):
    E={}; txt=open(path).read()
    for blk in re.split(r'(?=EPOCH \d+  )', txt):
        m=re.match(r'EPOCH (\d+)  (val_rhythm_f1|beat)', blk)
        if not m: continue
        ep=int(m[1]); d=E.setdefault(ep,{})
        for row in re.finditer(r'^(SINUS|AFIB|SVT|VT|AVB|N|S|V)\s+\d+\s+([\d.]+)\s+([\d.]+)', blk, re.M):
            tag=('R|' if m[2].startswith('val') else 'B|')+row[1]
            d[tag+'|Se']=float(row[2]); d[tag+'|PP']=float(row[3])
    return E
A=parse(f'{R}/201026_rhythm_dual_30k_A/eval/dualunet_rhythm_30k/confusion_log.txt')[36]
T=parse(glob.glob(f'{R}/191026_rhythm_dual_1m_v4_beatft2/eval/*/confusion_log.txt')[0])[2]
keys=[k for k in A]
print('teacher AFIB', T['R|AFIB|Se'], T['R|AFIB|PP'], '| e36 AFIB', A['R|AFIB|Se'], A['R|AFIB|PP'])
for v in ['F1','F2','F2b','R1','R2']:
    f=glob.glob(f'{R}/2[34]1026_rhythm_dual_30k_e36*_{v}/eval/*/confusion_log.txt')
    if not f: continue
    for ep,d in sorted(parse(f[0]).items()):
        if not all(k in d for k in keys): continue
        drops={k:round(d[k]-A[k],2) for k in keys if not k.startswith('R|AFIB') and d[k]<A[k]}
        afib_ok=d['R|AFIB|Se']>T['R|AFIB|Se'] and d['R|AFIB|PP']>T['R|AFIB|PP']
        ok='PASS' if afib_ok and not drops else '    '
        print(f'{ok} {v} e{ep:02d} AFIB {d["R|AFIB|Se"]:.2f}/{d["R|AFIB|PP"]:.2f} drops vs e36: {len(drops)} {drops}')
