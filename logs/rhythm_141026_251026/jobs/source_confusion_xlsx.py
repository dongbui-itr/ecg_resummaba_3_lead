"""source_confusion_xlsx.py : strip_eval/source_confusion.json -> summary_111026/confusion_by_source.xlsx"""
import json
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter as L
T = '/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026'
import sys
IN = sys.argv[1] if len(sys.argv) > 1 else f'{T}/strip_eval/source_confusion.json'
OUTX = sys.argv[2] if len(sys.argv) > 2 else f'{T}/confusion_by_source.xlsx'
TITLE = sys.argv[3] if len(sys.argv) > 3 else 'eval split (held-out studies) + rhythm_eval (whole 10 s windows)'
res = json.load(open(IN))
C = ['SINUS', 'AFIB', 'SVT', 'VT', 'AVB']
ORDER = ['rhythm_eval', 'dataset-1', 'dataset-4', 'dataset-5', 'rhythm-2', 'rhythm-3', 'afib-2', 'sinus-2',
         'sinus-3', 'sinus-4', 'sinus-4p', 'ltafdb', 'nsrdb', 'svdb', 'incartdb', 'ptbxl', 'challenge2020']
HDR = Font(name='Arial', bold=True, color='FFFFFF'); HFILL = PatternFill('solid', start_color='1F4E78')
B = Font(name='Arial', bold=True); N = Font(name='Arial')
thin = Side(style='thin', color='BFBFBF'); BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
wb = Workbook(); ws = wb.active; ws.title = 'Recall by source'
models = list(res)
srcs = [s for s in ORDER if any(s in res[m]['cm'] for m in models)] + \
       sorted({s for m in models for s in res[m]['cm']} - set(ORDER))

def stats(cm):
    cm = np.array(cm, float); sup = cm.sum(1); rec = np.where(sup > 0, np.diag(cm) / np.maximum(sup, 1) * 100, np.nan)
    acc = np.trace(cm) / max(cm.sum(), 1) * 100
    return rec, sup, acc

ws['A1'] = 'Raw per-step recall (%) by data source - argmax, no TTA, no post-process; ' + TITLE
ws['A1'].font = Font(name='Arial', bold=True, size=12)
ws['A2'] = 'Support = labelled seconds of the class in that source; blank = class absent. Step = 8 ms (1250 steps / 10 s).'
ws['A2'].font = Font(name='Arial', italic=True, color='595959')
r0 = 4
ws.cell(r0, 1, 'Source'); ws.cell(r0 + 1, 1, '')
col = 2
for m in models:
    ws.cell(r0, col, m); ws.merge_cells(start_row=r0, start_column=col, end_row=r0, end_column=col + len(C))
    for j, c in enumerate(C + ['Acc']):
        ws.cell(r0 + 1, col + j, c)
    col += len(C) + 1
ws.cell(r0, col, 'Support (s)'); ws.merge_cells(start_row=r0, start_column=col, end_row=r0, end_column=col + len(C) - 1)
for j, c in enumerate(C):
    ws.cell(r0 + 1, col + j, c)
last = col + len(C) - 1
for rr in (r0, r0 + 1):
    for cc in range(1, last + 1):
        x = ws.cell(rr, cc); x.font = HDR; x.fill = HFILL; x.alignment = Alignment(horizontal='center'); x.border = BOX
for i, s in enumerate(srcs):
    r = r0 + 2 + i; ws.cell(r, 1, s).font = B; ws.cell(r, 1).border = BOX
    col = 2; sup0 = None
    for m in models:
        cm = res[m]['cm'].get(s)
        if cm is not None:
            rec, sup, acc = stats(cm); sup0 = sup
            for j in range(len(C)):
                if not np.isnan(rec[j]):
                    ws.cell(r, col + j, round(float(rec[j]), 2))
            ws.cell(r, col + len(C), round(float(acc), 2))
        for j in range(len(C) + 1):
            x = ws.cell(r, col + j); x.font = Font(name='Arial', bold=(j == len(C))); x.number_format = '0.00'; x.border = BOX
        col += len(C) + 1
    for j in range(len(C)):
        x = ws.cell(r, col + j, int(round(sup0[j] / res[models[0]]['step_hz'])) if sup0 is not None else None)
        x.number_format = '#,##0'; x.font = N; x.border = BOX
rl = r0 + 1 + len(srcs)
col = 2
for m in models:
    ws.conditional_formatting.add(f'{L(col)}{r0 + 2}:{L(col + len(C))}{rl}',
        ColorScaleRule(start_type='num', start_value=50, start_color='F8696B', mid_type='num', mid_value=85,
                       mid_color='FFEB84', end_type='num', end_value=100, end_color='63BE7B'))
    col += len(C) + 1
ws.column_dimensions['A'].width = 15
for cc in range(2, last + 1):
    ws.column_dimensions[L(cc)].width = 8.5
ws.freeze_panes = ws.cell(r0 + 2, 2)

for m in models:
    w = wb.create_sheet(m[:31]); w['A1'] = f'{m}: confusion by source (rows = reference, cols = prediction; row %, seconds in brackets)'
    w['A1'].font = Font(name='Arial', bold=True, size=12); w['A2'] = res[m]['ckpt']; w['A2'].font = Font(name='Arial', italic=True, color='595959')
    r = 4
    for s in srcs:
        cm = res[m]['cm'].get(s)
        if cm is None:
            continue
        cm = np.array(cm, float); rec, sup, acc = stats(cm); hz = res[m]['step_hz']
        w.cell(r, 1, f'{s}   (accuracy {acc:.2f} %, {cm.sum() / hz:,.0f} s)').font = B; r += 1
        heads = ['Ref \\ Pred'] + C + ['Recall %', 'Support (s)']
        for j, h in enumerate(heads):
            x = w.cell(r, 1 + j, h); x.font = HDR; x.fill = HFILL; x.alignment = Alignment(horizontal='center'); x.border = BOX
        for i, c in enumerate(C):
            rr = r + 1 + i; w.cell(rr, 1, c).font = B; w.cell(rr, 1).border = BOX
            for j in range(len(C)):
                v = cm[i, j] / sup[i] * 100 if sup[i] else None
                x = w.cell(rr, 2 + j, round(v, 2) if v is not None else None); x.number_format = '0.00'; x.border = BOX
                x.font = Font(name='Arial', bold=(i == j))
                if i == j and v is not None:
                    x.fill = PatternFill('solid', start_color='E2EFDA')
            x = w.cell(rr, 2 + len(C), round(float(rec[i]), 2) if sup[i] else None); x.number_format = '0.00'; x.font = B; x.border = BOX
            x = w.cell(rr, 3 + len(C), int(round(sup[i] / hz))); x.number_format = '#,##0'; x.font = N; x.border = BOX
        r += len(C) + 2
    w.column_dimensions['A'].width = 14
    for cc in range(2, 10):
        w.column_dimensions[L(cc)].width = 11
wb.save(OUTX); print('saved', OUTX, len(srcs), 'sources', models)
