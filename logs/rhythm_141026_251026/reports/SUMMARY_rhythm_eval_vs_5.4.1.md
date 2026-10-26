# rhythm_eval vs bảng tham khảo (5.4.1 + event confusion) — 12/10/2026

Giao thức: chấm bên trong cửa sổ bác sĩ duyệt (ecgr/rhythm/strip_eval.py), 36 ô target.
Chọn checkpoint + decode chỉ trên validation strip_val2 (split eval, 4.719 sự kiện, 0 study trùng rhythm_eval).
rhythm_eval được chấm 2 lần: lần 1 (validation v1, thiếu cơn SVT/VT ngắn) và lần 2 (validation v2) — bảng dưới là lần 2.

| | 1m | 100k | 30k |
|---|---|---|---|
| Checkpoint | fine-tune lần 2, best (121026_rhythm_dual_1m_ftstrip2) | fine-tune lần 2, epoch 6 | fine-tune lần 1, epoch 3 |
| Decode | tunedv2_dual1m_ft2_best.json | tunedv2_dual100k_ft2_last.json | tunedv2_dual30k_ft_e3.json |
| rhythm_eval, decode cũ, model gốc | 29/36 | 31/36 | 29/36 |
| rhythm_eval, sau tune | **30/36** | **33/36** | **33/36** |

Ô còn thiếu trên rhythm_eval:
- 1m: AFIB E+P 92/94, AFIB D+P 95/96, VT E+P 89/90, VT D+P 85/86, SINUS 92.7/92.89, SVT recall 71.9/72.6
- 100k: AFIB E+P 92/94, VT E+P 88/90, VT D+P 81/86
- 30k: AFIB E+P 88/94, AFIB D+P 94/96, VT E+P 88/90

Event-level (hình 2): cả 3 model vượt mọi ô (AFIB F1 ~93-95 vs 83.3; SVT ~94-95 vs 83.8; VT ~90 vs 60.6; AVB ~96-97 vs 48.9).
Chi tiết từng model: final_<key>_v2_rhythm_eval.md / final_<key>_v2_validation.md.
