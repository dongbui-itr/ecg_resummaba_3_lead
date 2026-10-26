import pandas as pd, sys
S='/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026/ec57'
ks=sys.argv[1:]; d=None
for k in ks:
    x=pd.read_csv(f'{S}/{k}/rhythm_ec57_summary.csv')[['db','class','D_Se','D_+P','D_F1','E_Se','E_+P','E_F1']]
    x=x.rename(columns={c:f'{c}|{k}' for c in x.columns if c not in('db','class')})
    d=x if d is None else d.merge(x,on=['db','class'])
d=d[~((d['db']=='escdb')&(d['class']=='SVT'))]
for m in ['D_Se','D_+P','D_F1','E_Se','E_+P','E_F1']:
    print(m.ljust(6), '|'.join(k[-12:].rjust(12) for k in ks))
    for _,r in d.iterrows(): print(f"  {r['db']:5s} {r['class']:5s}", ' '.join(f"{r[f'{m}|{k}']:>12}" for k in ks))
for k in ks:
    try:
        b=pd.read_csv(f'{S}/{k}/beats/ec57_summary.csv'); print(k, b[b.db=='mitdb'][['Q_Se','Q_+P','V_Se','V_+P','S_Se','S_+P']].to_string(index=False,header=False))
    except Exception as e: print(k,e)
