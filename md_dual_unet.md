# Dual U-Net: một encoder chung, hai nhánh (thất / nhĩ), ba ngõ ra

Code: `ecgr/rhythm/dualunet.py`. Tên model: `rhythm_dual_{2m,1m,100k,30k}`. Test:
`tests/test_rhythm_dual.py`. Dữ liệu train/eval, nhãn, augmentation và cách chấm điểm giữ
nguyên theo [md_prompt.md](md_prompt.md).

## 1. Yêu cầu và cách đáp ứng

| Yêu cầu | Cách làm |
|---|---|
| Phân tích được QRS, T, P, sóng F/f | Nhánh **V** (QRS, T) và nhánh **A** (P, F, f). Nhánh A đọc tín hiệu **đã trừ QRST** ở 8 ms/bước, dùng average-pool và conv dilated |
| 2 nhánh U-Net cho 2 cụm đặc trưng | V: U-Net max-pool 1250 → 250 → 50 → 10. A: U-Net avg-pool, cùng 4 mức độ phân giải |
| Không dùng transformer | Chỉ có conv, pooling, SSM đường chéo (`ssm_block`) và tự tương quan RR. Không có `MultiHeadAttention` (có test kiểm tra) |
| Input 10 s, 3 kênh | `(2500, 3)` ở 250 Hz, z-score theo từng lead (như hiện tại) |
| 3 ngõ ra cùng lúc | `beat (1250, 4)`, `rhythm (1250, 5)`, `channel (5, 4)` |
| Dùng chung encoder, decoder riêng | Stem theo lead + nhánh V + nhánh A + neck SSM là encoder chung. Mỗi ngõ ra có decoder riêng |
| Giữ quy ước data | Cùng npy, cùng split, cùng target rhythm/beat. `channel` được tính từ đúng lượng nhiễu augmentation đã thêm vào |

**Lớp rhythm (đã chốt 07/10/2026):** 5 lớp **SINUS, AFIB (gồm AFL), SVT, VT, AVB (AVB2 +
AVB3)**.
- `config.CLASS_NAMES` đổi theo cho toàn project. Mọi nguồn nhãn (portal, PTB-XL, ltafdb,
  Challenge-2020) và reference EC57 (`(BII` + `(B3`) đều quy về `AVB`.
- Npy 6 lớp đang có được gộp nhãn khi tải. Checkpoint 6 lớp cũ vẫn predict và chấm được:
  hai cột AVB được cộng lại (`labels.to_current_classes`).
- Ô EC57 AVB trên mitdb chính là ô AVB2 cũ (mitdb không có `(B3`), nên target giữ nguyên.

## 2. Kiến trúc

```
input (2500, 3)
 │
 ├─ stem theo lead (cùng trọng số cho mọi lead)                 (3, 2500, s)
 │    └─ trọng số lead: softmax theo lead, mỗi 0.2 s
 │  lead pooling: Σ w·F ‖ max ‖ mean  → bất biến với thứ tự lead (2500, 3s)
 │
 ├─ NHÁNH V (QRS/T): U-Net max-pool  v1 1250 → v2 250 → v3 50 → vb 10, SSM tại 250
 │    └─ decoder VỊ TRÍ beat (chỉ dùng V) → p(beat) (1250)  ──stop-gradient──┐
 │                                                                         │
 ├─ HỦY QRST: template trung bình quanh các beat (−100 ms … +450 ms) ◄──────┘
 │    residual = tín hiệu − template, ở 125 Hz → còn lại P, F, f; kèm mask QRS
 │
 ├─ NHÁNH A (P/F/f): conv theo lead trên residual → lead pooling
 │    a1 1250 (dilation 1-2-4-8, nhìn ~0.75 s) → a2 250 (avg-pool) → a3 50 → ab 10
 │
 └─ NECK chung: (vb ‖ ab) đi lên 50 và 250 qua skip của CẢ HAI nhánh → 2 × SSM  (250, width)

DECODER
 beat    : p(beat) từ decoder vị trí × softmax(N/S/V) từ neck + skip 8 ms của V và A
           (beat S = sớm + P bất thường hoặc vắng, nên cần nhánh A)
 rhythm  : neck + xác suất beat (stop-gradient) + FiLM từ tự tương quan RR, qua SSM,
           rồi đi lên 1250 qua skip của V và A
 channel : stem theo lead → điểm của từng lead trên mỗi đoạn 2 s (CH1..CH3),
           còn NOISE lấy từ mean/max theo lead + neck
```

**Vì sao cần hủy QRST:** QRS cao 1–2 mV, trong khi P chỉ 0.1–0.25 mV và f 0.05–0.1 mV. Nếu
max-pool ngay sau stem thì gần như chỉ còn QRS, và model chỉ học được độ đều RR. Đây đúng là
nguồn lỗi còn lại trên mitdb:
- 232 (PAC) và 200/203 (PVC) bị gọi là AF;
- 215 (sinus nhanh) bị gọi là SVT;
- AVB2 bị bỏ sót.

Template chỉ bắt đầu từ 100 ms trước R, nên sóng P (120–250 ms trước R) vẫn nằm lại trong
residual. Sóng f không đồng pha với QRS nên bị trung bình hóa ra khỏi template, và cũng nằm lại
trong residual.

**Tách gradient:** loss rhythm không đi vào decoder beat (stop-gradient). Decoder vị trí beat
chỉ đọc nhánh V, nên template không phụ thuộc vòng vào nhánh A.

## 3. Ngõ ra `channel` (chọn kênh sạch)

- Mỗi đoạn 2 s nhận một trong 4 lớp: `NOISE / CH1 / CH2 / CH3`.
- **NOISE** dùng đúng luật CLEAN của head `noise` cũ: cả 2 giây phải có ít nhất 2 lead với
  SNR ≥ 6 dB. Vì vậy p(NOISE) dùng trực tiếp làm cổng nhiễu khi decode.
- **CHk** là lead tốt nhất trong đoạn, xếp theo thứ tự: số giây đọc được, rồi SNR của đoạn,
  rồi độ nhọn QRS (`augment.channel_scores`). Target đổi chỗ theo hoán vị lead.
- Trong `predict_signal`: `p_noise` lấy từ `channel[..., NOISE]`. Mỗi window ghi thêm
  `channel` (lead sạch nhất của từng đoạn).

## 4. Kích thước

| Model | Tham số | Ngân sách |
|---|---|---|
| `rhythm_dual_2m` | 1,899,421 | 2M |
| `rhythm_dual_1m` | 988,773 | 1M |
| `rhythm_dual_100k` | 80,189 | 100k |
| `rhythm_dual_30k` | 23,361 | 30k |

## 5. Đã kiểm chứng

- `tests/test_rhythm_dual.py`: 10/10 test pass.
  - Shape và tổng xác suất bằng 1.
  - Không có attention.
  - `beat` và `rhythm` **bất biến chính xác** với thứ tự lead; `channel` đổi chỗ theo lead.
  - Hủy QRST trên tín hiệu tổng hợp: QRS bị loại (đỉnh residual < 25 % đỉnh tín hiệu), sóng P
    còn lại, sóng f 6 Hz còn lại (tương quan > 0.8). Không có beat thì residual = tín hiệu.
  - Loss rhythm không có gradient vào decoder beat.
  - NOISE của `channel` trùng với NOISE của `noise`.
  - Loss và metric chạy được, save/load khớp, `predict_signal` trả đúng định dạng.
- Bộ test cũ: 228 pass, 11 skip. Một test fail có sẵn từ trước, không liên quan:
  `test_evaluate.py` cần `checkpoints/resumamba_1m.keras`, file này không có trên máy.
  (15/10/2026: `evaluate.py` beat và test đó đã được thay bằng `evaluate.py` rhythm và
  `tests/test_evaluate.py` mới cho checkpoint `dualunet_rhythm_1m_e26`.)
- Chạy thử trên GPU (dữ liệu thật, 2 epoch × 20 bước): pipeline thông suốt, khoảng 122 ms/bước
  ở batch 64, nghĩa là khoảng 6 phút/epoch với 2913 bước. **Chưa có kết quả train đầy đủ hay
  EC57.**

## 6. Train và chấm điểm

```bash
export ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/
export ECGR_RHYTHM_RUN_TAG=<DDMMYY>_rhythm_dual_1m
python3 -m ecgr.rhythm train --model rhythm_dual_1m --schedule cosine --lr 1e-3 \
    --sampler stratified --epochs 40 --patience 8
# chọn decode và beat-PP trên validation, sau đó chấm EC57 một lần (xem md_prompt.md)
python3 -m ecgr.rhythm ec57 --checkpoint <best.keras> --tag val_dual \
    --dbs ltafdb nsrdb incartdb --skip-rhythm-eval --sweep --sweep-on validation
python3 -m ecgr.rhythm ec57 --checkpoint <best.keras> --tag ec57_dual \
    --dbs mitdb nstdb escdb ahadb afdb
```

- **Không warm-start được** từ `rhythm_unet1250*`: tên và cấu trúc layer khác hoàn toàn, nên
  phải train từ đầu. Nên chạy 2 seed để so với baseline.
- Theo dõi cùng lúc: `val_rhythm_f1`, `val_beat_se/pp/type_f1` (bxb-like 150 ms),
  `val_channel_acc/noise_f1`.

## 7. Việc còn mở

1. ~~5 lớp rhythm~~: đã chốt. ~~Chấm beat afdb trên `.qrs`~~: đã sửa (2 record afdb:
   QRS 99.75 / 97.46).
2. Nếu sau khi train vẫn còn đường tắt RR: thêm head phân đoạn P/QRS/T/f
   ([md_atrial_activity.md](md_atrial_activity.md), mục 3.3) và probe / test đối chứng
   (mục 4 của file đó).

## 8. Kết quả (08/10/2026) — 2 seed, decode mặc định (chính thức)

| | Seed 1 (`071026_rhythm_dual_1m_s1`, e26) | Seed 2 (`081026_rhythm_dual_1m_s2`, e30) |
|---|---|---|
| val_rhythm_f1 | 0.8845 | 0.8842 |
| QRS Se/+P (150 ms) · beat type F1 | 99.72/99.38 · 0.820 | 99.65/99.45 · 0.833 |
| channel acc · NOISE F1 | 0.845 · 0.960 | 0.849 · 0.963 |

EC57 (D‑F1 / E‑F1; mục tiêu "rhythm 3.0.6"):

| DB · lớp | Seed 1 | Seed 2 | Mục tiêu |
|---|---|---|---|
| mitdb AFIB | 93.0 / 80.1 | 91.5 / 76.6 | 93.8 / 85.6 |
| mitdb SVT | 51.3 / 55.2 ✓ | 53.5 / 58.0 ✓ | 34.4 / 29.1 |
| mitdb VT | 54.9 / 68.8 (E ✓) | 60.0 / 66.5 | 64.8 / 67.9 |
| mitdb AVB (= AVB2) | 97.4 ✓ / 61.1 | 93.6 ✓ / 40.0 | 92.2 / 100 (5 episode tham chiếu) |
| afdb AFIB | 97.5 ✓ / 92.5 ✓ | 97.5 ✓ / 93.0 ✓ | 97.5 / 86.6 |
| escdb VT | 75.6 / 81.7 | 76.2 / 78.8 | – |
| rhythm_eval AFIB / SVT / VT / AVB (E‑F1) | 87.3 / 81.7 / 75.8 / 85.0 | 87.4 / 80.6 / 75.2 / 84.4 | – |

Beat mitdb (bxb): QRS 99.96/99.96 (✓ 99.95), VEB 96.9/93.2 → 97.5/93.9, SVEB 57.6/63.6 → 58.6/67.9
(mục tiêu >43 / >80: Se đạt, +P chưa). Hai seed chênh ≤ 2 điểm trừ AVB‑E (5 episode, 1 record).

**Decode:** bộ tham số validation‑tuned (`tune.py`) tăng validation 319→361 nhưng giảm mitdb VT‑D
(54.9→27.3) và SVEB Se (57.6→31) vì ltafdb không đại diện VT ngắn của mitdb; theo quy định không
chọn bằng EC57 nên **decode mặc định là chính thức**, bộ tuned giữ làm thí nghiệm (`release/tuning/`).
Lỗi prior scale trong `tune.py` (tiến trình chính dùng trọng số manifest) đã sửa, có test.

**Release:** `/media/Project/ECG/Model_Dong/ecgr_rhythm/release_dualunet_1m_e26/` (checkpoint seed 1,
code, log, EC57 reports; seed 2 ở `seed2_comparison/`).

## 9. Mô hình nhỏ 100k / 30k: SSL encoder + teacher–student (09/10/2026)

Code: `ecgr/rhythm/distill.py`, test `tests/test_rhythm_distill.py`, run `091026_rhythm_dual_small`.

| Model | Tham số | Input / output |
|---|---|---|
| `rhythm_dual_100k` | 80,189 | như 1M: `(2500,3)` → beat `(1250,4)`, rhythm `(1250,5)`, channel `(5,4)` |
| `rhythm_dual_30k` | 23,361 | như trên |

Cùng kiến trúc Dual U-Net (stem theo lead, nhánh V, nhánh A sau hủy QRST, neck SSM, 3 decoder), chỉ hẹp hơn.

**Giai đoạn 1 – SSL encoder** (`python3 -m ecgr.rhythm ssl --model rhythm_dual_100k`):
- Masked reconstruction: che 35 % thời gian theo đoạn ~240 ms + che nguyên 1 lead (p = 0.3); loss MSE
  chỉ trên phần bị che; theo dõi `nmse` (1.0 = không hơn đoán trung bình).
- Encoder = stem + nhánh V + nhánh A + neck (đồ thị `features_only`, tên layer trùng student).
- Hủy QRST cần vị trí beat: dùng p(beat) của **teacher frozen** trên cửa sổ chưa che (stop-gradient,
  chỉ là input) — mục tiêu học vẫn chỉ là tái tạo tín hiệu, không dùng nhãn.
- Head tái tạo nhỏ (1 conv k7 + 1×1) để việc học dồn vào encoder. Lưu `ssl_encoder.keras`.
- 12 epoch × 1500 bước, AdamW 1e-3 cosine, augmentation như train.

**Giai đoạn 2 – teacher → student** (`python3 -m ecgr.rhythm distill --model rhythm_dual_100k`):
- Teacher: Dual 1M seed 1 e26 (`release_dualunet_1m_e26`), frozen, chạy trên cùng batch đã augment.
- Student khởi tạo encoder từ SSL (`init_matching`), fine-tune toàn bộ.
- Loss mỗi head h ∈ {rhythm, beat, channel}: `w_h·[(1−α)·loss_nhãn + α·T²·KL(teacher_T ‖ student_T)]`,
  α = 0.5, T = 2; KL tính trên **mọi** bước (cả giây IGNORE — nơi teacher là nguồn tín hiệu duy nhất).
- Cùng sampler stratified, class weight, early stop (val_rhythm_f1, patience 8), lưu student vào
  `best_model.keras` → các công cụ predict / EC57 dùng như mọi checkpoint khác.

**Kết quả validation (eval split, epoch best, 10/10/2026):**

| | Teacher 1M (e26) | Student 100k SSL+KD (e40) | Student 30k SSL+KD (e38) |
|---|---|---|---|
| Tham số | 988,773 | 80,189 (8.1 %) | 23,361 (2.4 %) |
| val_rhythm_f1 | 0.8845 | **0.8727** (−1.2) | **0.8462** (−3.8) |
| F1 SINUS / AFIB / SVT / VT / AVB | 97.9 / 91.2 / 82.2 / 82.9 / 88.0 | 97.7 / 90.8 / 81.3 / 80.4 / 86.1 | 97.2 / 89.2 / 79.5 / 74.5 / 82.7 |
| QRS Se / +P (150 ms) | 99.72 / 99.38 | 99.51 / 99.34 | 99.25 / 99.18 |
| beat type F1 (N/S/V) | 0.820 | 0.865 | 0.882 |
| channel acc / NOISE F1 | 0.845 / 0.960 | 0.824 / 0.947 | 0.806 / 0.934 |

SSL: val_nmse 0.85 → ~0.83 (100k) / ~0.835 (30k), best epoch 10. Student 100k vẫn tăng ở epoch 40
(chạm giới hạn epoch). Chưa có baseline không SSL/KD cùng cỡ để tách đóng góp; chưa chấm EC57.
