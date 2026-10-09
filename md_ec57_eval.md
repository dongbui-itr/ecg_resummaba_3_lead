# Eval EC57 cho model rhythm đang train — `eval_rhythm.py` + post-process theo beat

Code: [eval_rhythm.py](eval_rhythm.py) (cửa vào), [ecgr/rhythm/beats.py](ecgr/rhythm/beats.py)
(post-process theo beat), [ecgr/rhythm/tune.py](ecgr/rhythm/tune.py) (chọn tham số trên
validation), [ecgr/rhythm/ec57.py](ecgr/rhythm/ec57.py) (inference → npz → decode → epicmp/bxb).
Test: `tests/test_rhythm_beats.py` (17 test).

## 1. Chạy

```bash
# sửa khối CONFIG đầu file (checkpoint, RUN_TAG, TAG, DATABASES, ...) rồi:
python3 eval_rhythm.py
```

Không có tham số dòng lệnh (giống `evaluate.py` của task beat): khối CONFIG là bản ghi của lần
đo. Thứ tự chạy:

| # | Bước | Ghi chú |
|---|---|---|
| 1 | **Snapshot checkpoint** | copy `best_model.keras` → `<out>/checkpoint_snapshot.keras` + md5, mtime. Train đang ghi đè file gốc ở epoch tốt hơn không làm hỏng lần eval |
| 2 | **Validation** | inference trên record eval của ltafdb / nsrdb / incartdb / svdb (chưa từng train) → `ec57/<VALIDATION_TAG>/_ann/*.npz` |
| 3 | **Tune** (`tune.py`) | 3 lưới trên npz validation: (a) decode AFIB (gap, min, ngưỡng p, prior alpha) với ngân sách 0.4 FP-episode/giờ; (b) tiêu chí + tùy chọn post-process beat; (c) luật ký hiệu beat (S→N trong AFIB, cổng prematurity). Ghi `tuned_decode.json` |
| 4 | **EC57** | mitdb / afdb / escdb: inference → decode với tham số đã chọn → `epicmp -A` từng lớp (AFIB có `-x`) + `bxb` cho beat |
| 5 | **Report** | bảng class × (Duration/Episode) kèm target sản phẩm và dấu `*`/`!`, bảng bxb, A/B không có beat stage, `records_audit.csv` (từng record × lớp), record lỗi nhiều nhất |

Mọi thứ ghi dưới `<WORK_DIR>/<RUN_TAG>/ec57/<TAG>/` (log: `eval_rhythm.log`, `decode_used.json`).

**Chế độ 1 record:** `RECORDS = ['mitdb/201', 'mitdb/231']` → predict, in episode tham chiếu /
dự đoán, số beat sau post-process, rồi epicmp + bxb trên đúng record đó.

**GPU dùng chung với train:** `GPU_MEMORY_MB = 6000` giới hạn bộ nhớ của tiến trình eval (job
train đang giữ ~13.5 GB / 24.5 GB). Inference dual U-Net: ~1 s / record mitdb sau khi compile.

## 2. Post-process theo beat (`beats.py`) — ánh xạ với tài liệu phân tích

Mỗi beat (từ decoder beat, đỉnh p(beat) ≥ 0.5, refractory 200 ms) mang 1 rhythm = đa số của
track trong ô của nó (trung điểm tới 2 beat kề, như export sản phẩm). Thứ tự xử lý:

| Bước | Hàm | Lỗi trong tài liệu được sửa |
|---|---|---|
| 1 | `rhythm_per_beat` | — |
| 2 | `_runs_to_rhythm` | **K**: chuỗi V (S) tự thỏa tiêu chí VT (SVT) → VT (SVT) dù track không nói — tương đương VES_RUN / SVES_RUN mà export EC57 cộng vào VT. Bật/tắt theo validation (`runs_to_rhythm`) |
| 3 | `_extend_along_runs` | như sản phẩm: chuỗi V có 1 beat VT → cả chuỗi VT |
| 4 | `region_valid` | **A**: `num_beat` có tác dụng; **J**: VT/SVT cần HR ≥ `min_hr`; AVB là HR ≤ `max_hr` (**F**: không còn ngưỡng 40 bpm vô lý); SVT nhận beat N khi có run S, tỉ lệ S, hoặc nhảy HR (§7.3) |
| 5 | `_long_invalid_to_sinus` | **C/H**: đoạn invalid *liên tục tối đa* (mọi lớp, gap xét trên beat chứ không trên mask valid) ≥ `long_invalid` giây → SINUS; ngưỡng tuyệt đối thay cho ¼ record |
| 6 | `_merge_invalid` | **E**: thứ tự ưu tiên tường minh `BEAT_PP_PRIORITY`; 2 bên đều spec (VT/SVT/AVB) → SINUS; **I**: gap ≤ 2 beat giữa 2 VT → VT chỉ khi *cả hai* bên là VT |
| 7 | `_split_by_beats` | **D**: beat không khớp trong VT/SVT lấy rhythm nền xung quanh (AFIB giữ AFIB), không ép SINUS; beat N có RR ≤ 0.6 s trong SVT được giữ |
| 8 | `_afib_svt_windows` | **B/G**: cửa sổ đóng cả hai phía, chỉ gồm AFIB/SVT/SINUS ngắn; tỉ lệ `svt_ratio` tách khỏi `AFIB.duration`. Mặc định tắt (validation) |
| 9 | `beat_symbols` | S trong AFIB → N (§5.3); cổng prematurity cho S ngoài SVT (tùy chọn, chọn trên validation) |

Tham số mặc định ở `config.BEAT_PP_*`; `tune.py` chọn và `eval_rhythm.py` truyền qua
`decode['beat_criteria'] / ['beat_options']`.

## 3. Kết quả lần chạy đầu (07/10/2026, dual U-Net `071026_rhythm_dual_1m_s1`, snapshot 17:22, train vẫn đang chạy)

Output: `/media/Project/ECG/Model_Dong/ecgr_rhythm/071026_rhythm_dual_1m_s1/ec57/{val_dual_s1,ec57_dual_s1}/`.

**Tune trên validation** (ltafdb 15 / nsrdb 2 / incartdb 9 / svdb 16 record eval-side):
- AF decode: gap 5 s, min 3 s, p ≥ 0.6, prior alpha 0.5 (0.31 FP-episode/giờ).
- Beat-PP: VT/SVT HR ≥ 120, SVT min_frac 0.5, onset 1.25, SINUS ≥ 8 s, long-invalid 10 s,
  `runs_to_rhythm` VT **và** SVT bật. Objective ltafdb (ΣE_F1+D_F1 AFIB/SVT/VT) 319.5 → 361.2;
  VT E-F1 61.7 → 86.0, D-F1 54.6 → 65.8; AFIB E-F1 79.6 → 81.4; SVT 16.2 → 19.6.
- Ký hiệu beat: S trong AFIB → N, cổng prematurity 0.85 (S F1 62.1 → 64.2).

**EC57** (epicmp, D = duration, E = episode, F1; `*` đạt target sản phẩm, `!` chưa):

| Ô | D-F1 | E-F1 | Không beat-stage (D / E) |
|---|---|---|---|
| mitdb AFIB | 93.0 ! (target 93.8) | 81.3 ! (85.6) | 92.5 / 81.3 |
| afdb AFIB | 97.5 * | 93.0 * | 97.5 / 93.0 |
| mitdb SVT | 49.0 * | 55.4 * | 36.4 / 29.8 |
| mitdb VT | 27.3 ! (64.8) | 60.0 ! (67.9) | 47.3 / 64.4 |
| escdb VT | 84.0 | 91.9 | 65.9 / 80.3 |
| mitdb AVB | 97.4 * | 61.1 ! (100) | 95.8 / 47.3 |

bxb (beat): mitdb Q 99.96/99.96, V 96.9/93.2, S 30.9/70.5; escdb Q 99.99/99.97, V 97.8/95.6,
S 61.5/36.3; afdb Q 97.6/95.5 (tham chiếu .qrs, không có loại beat).

**Nhận xét**
- Beat-stage giúp mọi ô trừ **mitdb VT**: D-Se 60 → 25. Nguyên nhân gần như chỉ ở record **223**
  (113 s VT tham chiếu, HR ~100–110 bpm): tiêu chí HR ≥ 120 chọn trên validation loại hết;
  với HR ≥ 100 record này giữ lại 87 s. Trên validation, 120 hơn 100 khoảng 10 điểm objective
  (ltafdb VT +P), nên đây là đánh đổi validation ↔ mitdb, không đổi tham số theo mitdb.
  Tổng mitdb: VT tham chiếu 274 s; TP 165 s không PP, 90 s PP (HR 120), 204 s PP (HR 100).
- mitdb AVB E +P 44 %: 5 mảnh AVB giả 4–18 s (231 ×3, 201, 232). Validation không có AVB
  nên min-episode AVB (2 s / 2.5 s) chưa được chọn trên dữ liệu; mitdb (BII đều ≥ 41 s.
- FP AFIB mitdb: 222 (+125 s, flutter/nodal), 200 (+109 s, PVC nhiều); afdb 04043 (+1029 s).
- SVT mitdb vượt target nhờ `runs_to_rhythm` SVT + chấp nhận beat N trong SVT (E-F1 29.8 → 55.4).
- Checkpoint còn đang train (epoch ~26/40); chạy lại `python3 eval_rhythm.py` khi train xong
  (đặt `TUNED_DECODE = None` để tune lại trên checkpoint cuối).
