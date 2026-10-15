# Rhythm model – data train / eval và quy tắc chọn dataset

Cập nhật: 2026-10-14. Run hiện tại: `141026_rhythm_dual_1m_v4_scratch` (model `rhythm_dual_1m`).

- Repo: `/media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead`
- Work dir: `/media/Project/ECG/Model_Dong/ecgr_rhythm`
- Interpreter: `python3` (system 3.12). Luôn set `ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/`

## 0. Bài toán / model

- Input: 1 cửa sổ 10 s, 3 lead, 250 Hz (2500 × 3, float16). Không dùng input dài hơn.
- Model: Dual U-Net `rhythm_dual_1m` (`ecgr/rhythm/dualunet.py`), không attention/transformer.
  - rhythm (1250, 5): SINUS, AFIB (gồm AFL), SVT, VT, AVB (= AVB2 + AVB3)
  - beat (1250, 4): heatmap + N/S/V
  - channel (5, 4): NOISE/CH1..3 mỗi 2 s. Noise là output riêng, không phải một lớp rhythm.
- Train từ đầu (không `--init-from`), lưu checkpoint từ epoch 3.

### Lệnh train v4 (`datasel_131026/chain_v4.sh`)

```bash
ECGR_RHYTHM_NPY_DIR=/media/Project/ECG/Model_Dong/ecgr_rhythm/npy_curated_v4_131026 \
ECGR_RHYTHM_RUN_TAG=141026_rhythm_dual_1m_v4_scratch \
ECGR_RHYTHM_CKPT_START_EPOCH=3 ECGR_RHYTHM_STRAT_LONG=VT:5 ECGR_RHYTHM_STRAT_LONG_BOOST=3 \
python3 -m ecgr.rhythm train --model rhythm_dual_1m --schedule cosine --sampler stratified --epochs 40 --patience 8
```

- Batch 64, cosine LR (warmup 1 epoch, floor 2 %), stratified sampler. Window có VT ≥ 5 s được boost ×3 (1,318 window).
- Class weights: SINUS 0.24, AFIB 0.53, SVT 1.0, VT 2.13, AVB 1.80.
- Train xong, chain tự chạy EC57 bằng decode chính thức (`summary_111026/strip_eval/jobs/ec57_official.sh`).

## 1. Database TRAIN (split train, `npy_curated_v4_131026/train`)

Tổng: 278,534 window từ 263,468 event.
Số giây nhãn: SINUS 2,143,649 · AFIB 446,051 · SVT 124,215 · VT 27,344 · AVB 38,181 · IGNORE 5,900.

### Portal Holter strip (gốc `/media/MegaDataSet/DATA_4TINYML/Holter_report_strip/`)

| Source | Thư mục | Nội dung | Event train v4 |
|---|---|---|---|
| dataset-1 | `dataset-1` | strip 10 s, không có caliper | 70,407 |
| dataset-4 | `dataset-4` | strip 10 s, span lấy từ `.hea` | 47,759 |
| dataset-5 | `dataset-5` | strip 10 s, có caliper | 23,518 |
| rhythm-2 | `dataset-rhythm/dataset-2-vt-svt-avb2-avb3` | record 60 s có caliper | 2,500 |
| rhythm-3 | `dataset-rhythm/dataset-3-vt-svt-avb2-avb3` | record 60 s có caliper | 321 |
| afib-2 | `dataset-rhythm/dataset-afib-2` | record 60 s AFIB | 4,515 |
| sinus-2 | `dataset-sinus/dataset-2-tachy-brady-pause` | sinus tachy/brady/pause, 60 s | 476 |
| sinus-3 | `dataset-sinus/dataset-3-tachy-brady-pause` | sinus tachy/brady/pause, 60 s | 12,983 |
| sinus-4 | `dataset-sinus/dataset-4-tachy-brady` | strip 10 s | 11,269 |
| sinus-4p | `dataset-sinus/dataset-4-tachy-brady-pause` | strip 10 s | 916 |

### Database ngoài (không thuộc EC57)

| Source | Đường dẫn | Phần dùng cho train | Window |
|---|---|---|---|
| PTB-XL 1.0.3 | `/media/MegaDataSet/ECG/physionet_org/ptb-xl/1.0.3` | strat_fold 1–8 (8,822 record × 2) | 17,644 |
| ltafdb | `/media/MegaDataSet/University/private_projects/physionet_data/ltafdb` | 69 record | 41,915 |
| incartdb | `/media/MegaDataSet/ECG/physionet_org/incartdb/1.0.0` | 66 record | 8,503 |
| svdb | `/media/MegaDataSet/University/private_projects/physionet_data/svdb` | 62 record | 399 |
| Challenge-2020 | `/media/MegaDataSet/ECG/physionet_org/challenge-2020/1.0.2/training` | `cpsc_2018`, `cpsc_2018_extra`, `georgia` (85 %) | 22,568 |

Run v4 không có nsrdb trong train. Bản npy gốc `npy_250hz_10s_3lead_10sec` có nsrdb (16 record, 4,533 window).

## 2. Database EVAL

| Mục đích | Database | Chi tiết |
|---|---|---|
| Validation (chọn checkpoint, early stop) | Split eval gốc `npy_250hz_10s_3lead_10sec/eval` (symlink), 65,883 window / 62,236 event | Portal eval (~20 % số study), PTB-XL fold 9, phía eval của ltafdb (15 record), nsrdb (2), incartdb (9), svdb (16), 15 % Challenge-2020. Metric: `val_rhythm_f1` (macro F1 theo từng giây) |
| Tune decode / post-process (Holter) | Phía eval của ltafdb, nsrdb, incartdb, svdb, chạy nguyên record | `eval_rhythm.py` → `ecgr/rhythm/tune.py` |
| Tune theo bảng 5.4.1 | strip_val2 (3,778 event) | Split eval của afib-2, dataset-5, rhythm-2, dataset-4, rhythm-3 (`strip_tune.py`) |
| Test 1: rhythm_eval | `Holter_report_strip/dataset-eval/rhythm_eval` (npy `test`, 7,233 window) | 1,966 study, 4,909 event (AFIB 1673, SVT 1803, VT 439, AVB2 686, AVB3 308), record 60 s. Chỉ chấm một lần cho mỗi config cuối |
| Test 2: EC57 | `/media/Project/ECG/PhysionetData/`: mitdb, afdb, escdb, nstdb, ahadb | epicmp từng lớp (AFIB chạy `-x`, AFL = AF) + bxb cho beat. Bỏ 4 record paced của mitdb (102/104/107/217) |
| Dự trữ | PTB-XL fold 10 | Không đọc |

### Cách chấm

- Per-second trên test: `ecgr.rhythm.evaluate.evaluate(ckpt, split='test')` → `summary_111026/eval/<key>/`
- Kiểu bảng 5.4.1 (epicmp E/D Se/+P bên trong vùng mark của reviewer, 36 ô target): `ecgr/rhythm/strip_eval.py`
- EC57 decode chính thức: `summary_111026/strip_eval/jobs/ec57_official.sh <key> <ckpt>` (hop 5 s, TTA id,flip,swap, taper 0)
- EC57 đầy đủ (validation tune → EC57): `eval_rhythm.py` ở root repo
- Target EC57 (rhythm 3.0.6, D/E F1): AFIB mitdb 93.8/85.6, afdb 97.5/86.6; SVT mitdb 34.4/29.1; VT mitdb 64.8/67.9; AVB2 mitdb 92.2/100

### Mốc so sánh (test split, macro F1 per-second)

| Model | Macro F1 |
|---|---|
| dual1m_s2_e30 (bản gốc, data đầy đủ) | 83.01 |
| curated_v3_e1 (fine-tune từ e26) | 83.86 |
| v3_scratch_best (chỉ portal v3, train từ đầu, best val 0.857 @ e34) | 83.37 |
| v4_scratch_best (portal v3 + PTB-XL/ltafdb/svdb/incartdb/Challenge, train từ đầu, best val 0.8822 @ e36) | 83.39 |

EC57 v4_scratch_best (decode chính thức, D/E F1): mitdb AFIB 90.0/77.8, SVT 54.7/58.6, VT 63.5/66.5,
AVB 96.9/52.9; afdb AFIB 97.0/92.5; escdb VT 67.5/70.1. Beat mitdb QRS 99.96/99.95, VEB 97.6/91.9, SVEB 47.5/63.4.

## 3. Quy tắc chọn dataset

### A. Cấm tuyệt đối (chống leak)

- Không bao giờ train hoặc tune trên EC57 (mitdb, afdb, escdb, nstdb, ahadb, cudb) và rhythm_eval. `physionet_train` tự từ chối các DB EC57.
- Holdout đọc thẳng từ thư mục `rhythm_eval/<study>`, không đọc từ list (110 study không có trong list v4).
- Các study trong `assets/list_studies_eval_v4.json` (holdout của model beat) cũng bị loại.
- Kiểm tra mới nhất: overlap train×eval = 0, train/eval×rhythm_eval = 0 (`split_verification.json`, `datasel_131026/selection_report.json`).

### B. Chia split

- Portal: theo study, 80 % train / 20 % eval, quyết định bằng hash của study id.
- PTB-XL: theo patient (strat_fold), fold 1–8 train, fold 9 eval, fold 10 dự trữ.
- PhysioNet: theo record (incartdb theo patient) bằng hash, 15 % eval.
- Challenge-2020: 15 % eval. Bỏ bản sao PTB-XL và INCART trong đó.

### C. Gán nhãn portal (`eventType` → lớp)

- SINUS, TACHY, BRADY, PAUSE và các loại ectopy (SINGLE_SVE/VE, COUPLET, BIGEMINY, TRIGEMINAL, QUADRIGEMINY) → SINUS (hard negative).
- AFIB → AFIB; SVT, SVE_RUN → SVT; VT, VE_RUN → VT; AVB2, AVB3 → AVB.
- OTHERS, ARTIFACT, AVB1 → bỏ (ví dụ dataset-1 bỏ 41,845 OTHERS, dataset-4 bỏ 66,539).
- Strip không có caliper (dataset-1/4): SVT/VT lấy theo run ≥ 3 nhịp S/V từ `.atr`, không gán cả 10 s.
- Cắt window 10 s với hop 5 s; mỗi event có ít nhất 1 window.

### D. Giới hạn SINUS

- Tối đa 3 strip SINUS mỗi study (chọn bằng hash của event id).
- Cộng thêm tối đa 8 strip "hard sinus" mỗi study: SVE/VE bigeminy, trigeminy, couplet, single SVE, BRADY, PAUSE.
- Các nguồn sinus-* được ưu tiên giữ trước (`sinus_first`).

### E. Lọc chất lượng portal (curated v3)

File: `datasel_131026/train_events_v3.csv` (174,664 event), nạp qua `ECGR_RHYTHM_EVENT_LIST`.
Script: `datasel_131026/score_events.py`, `select_events.py`.

- Model 1m e26 chấm từng event, bỏ 0.3 s ở hai biên.
- Giữ event khi thỏa tất cả:
  - rhythm Se và P+ ≥ 0.8 (riêng SINUS: Se ≥ 0.9)
  - QRS Se và P+ ≥ 0.95
  - V Se/P+ ≥ 0.8, S Se/P+ ≥ 0.7 (matching 150 ms), chỉ áp dụng khi event có nhịp V hoặc S
- Kết quả `selected`: SINUS 121,853 · SVT 28,474 · VT 9,263 · AFIB 5,733 · AVB 2,858
- v3 thêm lại một số event bị loại:
  - `added_raw_ppv`: SINUS 2,619 · SVT 2,056 · VT 549 · AVB 316 · AFIB 117
  - `added_raw_long_se`: AFIB 761 · VT 65
- Lọc này chỉ áp dụng cho portal train. Split eval và test giữ nguyên bản gốc (symlink).

### F. PTB-XL

- Mã SCP → lớp: AFIB/AFLT → AFIB; SVTAC/PSVT → SVT; 2AVB/3AVB → AVB.
- Bỏ record có mã SVARR, record mang 2 lớp khác nhau, record mơ hồ.
- SINUS chỉ giữ khi có mã hard negative (PACE, CLBBB, CRBBB, IVCD, ILBBB, IRBBB, WPW, STACH, SBRAD, SARRH, PVC, PAC, BIGU, TRIGU, 1AVB), hoặc nằm trong 3,000 record "plain" chọn bằng hash.
- Mỗi record lấy 2 window, mỗi window là một tổ hợp 3 trong 12 lead.

### G. PhysioNet (ltafdb, svdb, incartdb, nsrdb)

- Nhãn từ `.atr`:
  - (AFIB, (AFL → AFIB; (SVTA → SVT; (VT → VT; (BII, (B3 → AVB
  - (N, (SBR, (AB, (B, (T, (PREX → SINUS
  - Mã khác → IGNORE
- nsrdb, incartdb: sinus suốt record; run ≥ 3 nhịp S/V được gán SVT/VT.
- svdb: chỉ gán nhãn cho các run SVT/VT, mọi giây còn lại là IGNORE.
- Giới hạn số window mỗi record theo loại: VT 80, SVT 80, AF_edge 150, AF_ectopy 400, AF 150, sinus_pac 250, sinus_tachy 80, sinus_brady 150, sinus_ectopy 80, sinus 60. Giữ hết đoạn hiếm, chỉ lấy mẫu một phần các giờ AF hoặc sinus thuần.

### H. Challenge-2020

- SNOMED → lớp: AF, AFL → AFIB; block độ 2 (Mobitz I/II) và độ 3 → AVB.
- Bỏ các mã paroxysmal hoặc không thuộc lớp nào (SVT, VT, PSVT…), record < 10 s, record mang nhiều lớp.
- SINUS chỉ giữ khi có mã hard negative (sinus tachy/brady, SA, PAC, PVC, BBB, block độ 1, long PR, WPW), hoặc nằm trong 1,500 record plain. Mỗi record lấy 2 window.

### I. Quy tắc chung của project

- AFL tính là AFIB. AVB = AVB2 + AVB3. Có 5 lớp; Noise là output riêng.
- Tham số decode và calibration chỉ chọn trên validation, không bao giờ trên EC57 hay rhythm_eval.
- Run mới luôn train từ đầu (không warm-start), lưu checkpoint từ epoch 3.
