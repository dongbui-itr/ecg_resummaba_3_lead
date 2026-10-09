# Prompt: Train model Beat + Rhythm (shared encoder, 2 decoder)

> Dán toàn bộ phần dưới vào Claude Code (hoặc agent tương đương) chạy trong repo
> `ecg_resummaba_3_lead`, branch `resumamba_rhythm_10s`. Sửa các ô `<...>` trước khi dán.

---

## Vai trò

Bạn là kỹ sư ML cấp cao chuyên ECG (Holter 3 lead, 250 Hz), thành thạo TensorFlow/Keras 2.20,
WFDB/PhysioNet và chuẩn ANSI/AAMI EC57 (bxb cho beat, epicmp cho rhythm). Bạn làm việc như một
nhà nghiên cứu cẩn thận: mỗi thay đổi có giả thuyết, có A/B, có số liệu trên validation, và
không bao giờ tune trên tập test.

## Mục tiêu

Huấn luyện **một** model `rhythm_unet1250b_*` (encoder chung, 2 decoder) cho:

1. **Beat decoder** – output `(1250, 4)` = none/N/S/V ở 125 step/s (heatmap Gaussian vị trí R +
   loại beat masked). Pick beat bằng `ecgr/rhythm/beats.py::pick_beats`.
2. **Rhythm decoder** – output `(1250, 5)` 5 lớp rhythm (SINUS, AFIB gồm AFL, SVT, VT,
   AVB = AVB2 + AVB3), **conditioned trên output beat**
   (cascade, stop-gradient).
3. **Noise head riêng** – `(5, 2)`; Noise **không bao giờ** là lớp rhythm thứ 7/8.

Mục tiêu đo theo **ANSI/AAMI EC57** bằng công cụ WFDB: **`bxb`** cho beat, **`epicmp`** cho
rhythm, cả hai tổng hợp bằng **`sumstats`**. Chi tiết ở mục "Mục tiêu AAMI EC57" bên dưới.

Ưu tiên lần train này: `<VD: AFIB mitdb duration +P, SVEB, VT duration>`.

## Mục tiêu AAMI EC57 (bxb + epicmp)

### Quy ước chung của cả hai tool

- Reference là `.atr` của database. Hypothesis do pipeline ghi ra: beat là `.bti`
  (`RHYTHM_BEAT_AI_EXTENSION`), rhythm là file annotation theo từng lớp.
- **Learning period 5 phút:** với record PhysioNet dài, 5 phút đầu không tính điểm (driver
  `bxb-script.sh` / `epicmp-script.sh`). Strip ngắn (`rhythm_eval`, 10–60 s) dùng
  `*-script2.sh` với `-f 0`.
- **Loại 4 record paced của mitdb** (102, 104, 107, 217) như EC57 quy định
  (`--include-paced` chỉ dùng khi phân tích).
- Số dùng để báo cáo là dòng **Gross** của `sumstats` (gộp mọi beat / episode của database).
  Dòng Average (trung bình theo record) chỉ để tham khảo.
- F1 = 2·Se·(+P) / (Se + (+P)). Mỗi ô được tính là **đạt** khi F1 ≥ F1 target. Vẫn phải báo
  cáo Se và +P, vì một ô có thể đạt F1 nhờ Se cao trong khi +P sụt.
- Không có bộ chấm nào khác thay được bxb/epicmp. Metric per-step và per-window trong lúc train
  (val_rhythm_f1, BeatMatchLog) chỉ dùng để xếp hạng checkpoint.
- Trước khi chạy GPU phải kiểm tra `which bxb epicmp sumstats`. Thiếu tool thì script không
  ghi report và không báo lỗi gì.

### Beat – `bxb` (so từng nhịp)

- **Ghép nhịp:** một beat hypothesis khớp với beat reference khi lệch nhau ≤ **150 ms** (cửa sổ
  ghép của bxb). Khi train, `BEAT_MATCH_TOLERANCE_SECONDS = 0.15` mô phỏng đúng cửa sổ này.
- **Lớp AAMI:** N = {N, L, R, B, e, j, n}, SVEB (S) = {A, a, J, S}, VEB (V) = {V, E},
  F = fusion, Q = paced/unclassifiable. Model chỉ ra N/S/V. F và Q được coi là IGNORE khi train,
  còn khi chấm thì bxb tự xử lý F/Q theo luật EC57. **Không được** tự lọc beat F/Q ra khỏi
  reference.
- **Metric:** QRS Se / +P (phát hiện), VEB Se / +P, SVEB Se / +P (phân loại).
- **Database:** mitdb là DB chính và có target. **Không** chấm EC57 trên incartdb, vì đó là nguồn train.
  Báo cáo thêm escdb, nstdb (độ bền với nhiễu) và afdb (reference `.qrs`, chỉ QRS) nếu chạy;
  các DB này không có target.

| DB | Metric | Target (`evaluate.py::TARGETS`) | Hiện có (v2, beat-PP bật) |
|---|---|---|---|
| mitdb | QRS Se | ≥ 99.95 | 99.95 |
| mitdb | QRS +P | ≥ 99.95 | 99.95 |
| mitdb | VEB Se / +P | *chưa có target trong repo* → không thấp hơn baseline (95 / 93) | 95 / 93 |
| mitdb | SVEB Se | > 43 | 43 |
| mitdb | SVEB +P | > 80 | 42 ✗ |
| dataset-v4-beat (holdout portal) | SVEB Se / +P | > 88 / > 92 | chưa chấm với rhythm model |

#### Bảng AAMI beat – 5 database EC57 (`ecgr/config.py::EC57_DBS`)

Thông tin từng database (dữ liệu nằm ở `$ECGR_PHYSIONET_DIR`):

| DB | Record chấm | Tần số gốc → model | Reference | Lớp AAMI chấm được | Đặc điểm / lưu ý |
|---|---|---|---|---|---|
| **mitdb** | 44 / 48 (bỏ paced 102, 104, 107, 217) | 360 → 250 Hz, 2 lead thật + 1 lead 0 | `.atr` | QRS, VEB, SVEB | DB chính. 30 phút/record, ~84k QRS → Q 99.95 nghĩa là chỉ được ≤ 42 sót và ≤ 42 nhầm. Lỗi Q dồn ở 203 và 105 (nhiễu); lỗi S dồn ở 232, 222, 215, 200 |
| **nstdb** | 12 (118e*, 119e* ở SNR 24 → −6 dB; 3 record nhiễu `bw/em/ma` không chấm) | 360 → 250 Hz | `.atr` (của 118/119) | QRS, VEB, SVEB | Đo độ bền với nhiễu: 5 phút đầu sạch, sau đó các đoạn nhiễu 2 phút xen kẽ. +P thấp là điều có thể đoán trước |
| **escdb** | 90 | 250 Hz | `.atr` | QRS, VEB, SVEB | 2 h/record (ST-T). Lớp S rất hiếm (~0.14 % beat), nên +P của S nhảy mạnh khi chỉ thêm vài FP |
| **ahadb** | 79 | 250 Hz | `.atr` | QRS, VEB (S không có trong ref AHA) | Có record VF/flutter dài. bxb tự xử lý các đoạn reference không có nhịp |
| **afdb** | 23 | 250 Hz | **`.qrs`** (`EC57_BEAT_REF_EXT`) – chỉ vị trí, không có loại | **chỉ QRS** | 10 h/record, AF chiếm đa số. VEB/SVEB không chấm được (`-`, không phải 0) |

Target và kết quả (Se / +P, dòng Gross của bxb + sumstats):

| DB · lớp | Target | Beat model tốt nhất (beat task, ensemble native) | Rhythm model v2 (decoder beat) |
|---|---|---|---|
| mitdb · QRS | **≥ 99.95 / ≥ 99.95** | 99.92 / 99.89 ✗ | 99.95 / 99.95 ✓ |
| mitdb · VEB | không có target → ≥ baseline | 94.66 / 97.13 | 95 / 93 |
| mitdb · SVEB | **> 43 / > 80** | 55.99 / 68.34 ✗ (+P) | 43 / 42 ✗ |
| nstdb · QRS | không có target → ≥ baseline | 96.79 / 86.36 | chưa chấm |
| nstdb · VEB | không có target → ≥ baseline | – | chưa chấm |
| nstdb · SVEB | không có target → ≥ baseline | 75.97 / 39.12 | chưa chấm |
| escdb · QRS | không có target → ≥ baseline | 99.96 / 99.91 | chưa chấm |
| escdb · VEB | không có target → ≥ baseline | 96.87 / 97.65 | chưa chấm |
| escdb · SVEB | không có target → ≥ baseline | 55.53 / 37.36 | chưa chấm |
| ahadb · QRS | không có target → ≥ baseline | 99.91 / 99.84 | chưa chấm |
| ahadb · VEB | không có target → ≥ baseline | 89.79 / 99.47 | chưa chấm |
| afdb · QRS | không có target → ≥ baseline | 97.49 / 95.05 | chưa chấm |

Quy định khi chấm beat trên 5 DB:

1. Chấm đủ 5 DB, mỗi model một lần, bằng `bxb-script.sh` (learning period 5 phút). Tham chiếu
   là `.atr`; riêng afdb dùng `.qrs`.
2. "Baseline" của các ô không có target là **cột beat model tốt nhất** ở trên. Rhythm model có
   decoder beat dùng chung encoder thì không được kém hơn model beat chuyên dụng quá 0.5 điểm ở
   QRS, hoặc quá 2 điểm ở VEB/SVEB, trừ khi được ghi rõ là đánh đổi có chủ đích.
3. Ngoài Gross, báo cáo thêm **số beat sót (FN) và nhầm (FP) của QRS** trên mitdb, và top-3
   record gây lỗi cho mỗi lớp.
4. Ô không có beat reference ghi `-`, không ghi 0.00.
5. Lead mode `native`: đọc 2 lead thật của record, lead thứ 3 lấp 0 (`EC57_LEAD_MODE`). Không
   dùng cách nhân ba một lead.
6. Lệnh chấm: mặc định `ec57` của rhythm chỉ chạy mitdb, afdb, escdb, nên phải chỉ định đủ 5 DB:
   `python3 -m ecgr.rhythm ec57 --checkpoint <best.keras> --tag ec57_<tên> --dbs mitdb nstdb escdb ahadb afdb`.
7. **afdb chấm beat trên `.qrs`** (đã sửa ngày 07/10/2026): `ec57.score_beats` lấy phần mở
   rộng reference từ `config.EC57_BEAT_REF_EXT` (afdb → `qrs`, các DB khác → `atr`). Kiểm tra
   trên 2 record afdb: QRS Se 99.75 / +P 97.46. `.qrs` không có loại beat, nên VEB/SVEB của
   afdb luôn là `-` hoặc 0 và không được dùng.

Điểm nghẽn hiện tại là **SVEB +P trên mitdb**. Record 232 chiếm ~43 % gánh nặng SVEB của mitdb
(sinus brady + PAC dày).

### Rhythm – `epicmp` (so từng episode)

- **Chế độ:** `epicmp -A` chỉ chấm các episode có nhãn `(AFIB`. Để chấm từng lớp, mỗi lớp được
  chạy một lượt riêng: lớp cần chấm ghi thành `(AFIB`, mọi thứ khác ghi thành `(N`, cho cả
  reference lẫn hypothesis (`wfdb_ann.write_class_annotations`). Mã reference tương ứng:
  AFIB = `(AFIB`, SVT = `(SVTA`, VT = `(VT`, **AVB = `(BII` + `(B3`** (một lớp; mitdb không
  có `(B3`, nên ô AVB trên mitdb chính là ô AVB2 cũ và dùng target AVB2).
- **AFL = AF:** lượt AFIB chạy `epicmp -x`, nên lúc reference là `(AFL` thì gọi AFIB không bị
  tính FP. Số chấm trước 29/09/2026 không có `-x` nên không so sánh được với số hiện tại.
- **Metric (dòng Gross, 4 số):**
  - **Episode** Se / +P (E): episode reference có được phát hiện không, episode dự đoán có
    trùng reference không.
  - **Duration** Se / +P (D): tỷ lệ thời gian trùng nhau.
- **Lưới chấm 1 s** (`EC57_GRID_HZ = 1`): độ phân giải output dưới 1 s không làm đổi điểm
  epicmp.
- **Ma trận lớp × DB:** AFIB → mitdb, afdb; SVT → mitdb, escdb; VT → mitdb, escdb;
  AVB → mitdb (+ `rhythm_eval` cho mọi lớp).

Target sản phẩm "rhythm 3.0.6" (`ec57.TARGET_TABLE`, Se / +P → F1):

| Lớp | DB | Duration Se / +P | **D-F1** | Episode Se / +P | **E-F1** | Hiện có D-F1 / E-F1 (v2) |
|---|---|---|---|---|---|---|
| AFIB | mitdb | 98 / 90 | **93.8** | 80 / 92 | **85.6** | 88.4 ✗ / 76.4 ✗ |
| AFIB | afdb | 96 / 99 | **97.5** | 81 / 93 | **86.6** | 97.5 ✓ / 92.5 ✓ |
| SVT | mitdb | 61 / 24 | **34.4** | 62 / 19 | **29.1** | 40.2 ✓ / 48.2 ✓ |
| VT | mitdb | 75 / 57 | **64.8** | 82 / 58 | **67.9** | 36.6 ✗ / 62.0 ✗ |
| AVB (= AVB2 trên mitdb) | mitdb | 98 / 87 | **92.2** | 100 / 100 | **100** | 91.3 ✗ / 49.6 ✗ |
| VT | escdb | – | – | – | – | 65.3 / 68.4 (chỉ báo cáo) |
| mọi lớp | rhythm_eval | – | – | – | – | **không được giảm** so với baseline (điều kiện chặn) |

Lưu ý về độ tin cậy thống kê: AVB mitdb chỉ có **5 episode** `(BII` (đều ở record 231), nên E-F1 nhảy
theo bậc ~20 điểm. VT mitdb có 60 episode, median 1.8 s (51/60 ngắn hơn 3 s). Ngưỡng tối thiểu
khi decode phải tính theo beat, không theo giây.

### Tiêu chí nghiệm thu một model

1. Tất cả ô rhythm đang ✓ vẫn giữ ✓. Ô ✗ được cải thiện theo ưu tiên của lần train này.
2. Không ô nào tụt quá 2 điểm F1 so với baseline, trừ khi được giải thích là do nhiễu
   (episode ít, khác biệt giữa seed).
3. Beat mitdb: QRS Se / +P ≥ 99.95. VEB không thấp hơn baseline. SVEB tiến dần về 43 / 80.
4. `rhythm_eval` không giảm.
5. Bảng kết quả chỉ chấm **một lần**, với tham số decode và beat-PP đã chọn trên validation.

## Ràng buộc cứng (không được vi phạm)

- **Input luôn là 1 cửa sổ 10 s** `(2500, 3)`. Không đề xuất cửa sổ 30 s hay stage record-level;
  ngữ cảnh dài chỉ đến từ inference chồng lấp (hop 5 s) + decoding.
- **AFL = AFIB**, một lớp. Không thêm lớp AFL.
- **Dữ liệu train được phép:** portal, PTB-XL, ltafdb, nsrdb, svdb, incartdb, Challenge-2020
  (đã bỏ bản sao PTB-XL/INCART). **Cấm** mọi DB EC57 (mitdb, afdb, escdb, nstdb, ahadb, cudb) và
  `rhythm_eval` trong train/tune.
- **Tham số decode / calibration / beat-PP chỉ chọn trên validation** (record held-out của
  ltafdb/nsrdb/incartdb): `ec57 --dbs ltafdb nsrdb incartdb --skip-rhythm-eval --sweep-on validation`,
  ngân sách 0.4 false-AF-episode/giờ. **EC57 chỉ chấm 1 lần** cho mỗi model cuối.
- Ngưỡng tối thiểu của decode tính theo **beat**, không theo giây (VT mitdb median 1.8 s).
- So sánh kiến trúc/ý tưởng: **cùng data, ≥ 2 seed**; không kết luận từ 1 run.
- Không commit/push nếu chưa được yêu cầu; commit không có trailer Co-Authored-By của Claude.

## Quy định dataset train / eval / test

### 4 tầng dữ liệu – mỗi tầng một mục đích, không dùng lẫn

| Tầng | Dùng để | Nguồn | Được tune? |
|---|---|---|---|
| **train** | cập nhật trọng số | split `train` của mọi nguồn được phép (bảng dưới) | – |
| **eval** (val per-window) | early-stop, chọn epoch, so sánh run | split `eval` của cùng các nguồn | có (chọn checkpoint) |
| **validation record-level** | chọn decode / calibration / beat-PP | record phía `eval` của ltafdb, nsrdb, incartdb, chạy liên tục qua pipeline epicmp | có (sweep) |
| **test** | báo cáo cuối, **chỉ 1 lần/model** | `rhythm_eval` (holdout portal, 4909 event) + EC57: mitdb, afdb, escdb (+ nstdb, ahadb, cudb) | **không bao giờ** |

### Nguồn được phép và cách chia split

| Nguồn | Nội dung | Chia train/eval | Nhãn rhythm | Nhãn beat |
|---|---|---|---|---|
| Portal Holter (`rhythm-2/3`, `afib-2`, `sinus-2/3/4/4p`, `dataset-1/4/5`) | strip 10 s / caliper 60 s, 3 lead | theo **study**, hash study id, 80/20 (`TRAIN_FRACTION`) | `EVENT_TYPE_TO_CLASS`; ectopy → SINUS; run ≥ 3 beat S/V → SVT/VT | có (.atr) |
| PTB-XL 1.0.3 | 10 s 12 lead, 500 Hz → resample 250 Hz, 3-of-12 lead | theo **bệnh nhân**, `strat_fold` 1–8 train, 9 eval, **10 không đọc** (dự trữ) | SCP code; AFLT → AFIB; SVARR bỏ | **IGNORE** |
| ltafdb | 24 h, có rhythm mark | theo **record**, hash, 15 % eval | (AFIB/(AFL → AFIB, (SVTA, (VT; (N/(SBR/(B/(T… → SINUS | có |
| nsrdb | sinus toàn bộ | theo record, 15 % eval | SINUS + run S/V | có |
| incartdb | 12 lead | theo **bệnh nhân**, 15 % eval | SINUS, (AFIB, (PREX → SINUS | có |
| svdb | không rhythm mark | theo record, 15 % eval | chỉ run SVT/VT, còn lại IGNORE | có |
| Challenge-2020 (cpsc_2018, cpsc_2018_extra, georgia) | 12 lead, nhãn cả record | hash, 15 % eval; **bỏ bản sao PTB-XL/INCART** | SNOMED → AFIB/AVB2/AVB3; mã paroxysmal bị skip | **IGNORE** |

### Danh mục dataset portal Holter (`$ECGR_RHYTHM_DATA_ROOT` = `/media/MegaDataSet/DATA_4TINYML/Holter_report_strip`)

Số liệu lấy từ file inventory `.xlsx` và từ `npy_250hz_10s_3lead_10sec/manifest.json` (lần build
gần nhất). Các cột:
- **sự kiện**: số event còn lại sau khi ánh xạ loại event.
- **giữ**: số event vào train/eval.
- **holdout**: số event bị loại vì thuộc study trong `rhythm_eval` hoặc list v4.

**A. Đang dùng** (`ecgr/rhythm/config.py::SOURCES`; thứ tự trong bảng là thứ tự ưu tiên khi một
event bị trùng):

| Source key | Thư mục / inventory | Dạng | Dòng xlsx (loại event chính) | Sự kiện → giữ / holdout | Nhãn rhythm | Nhãn beat |
|---|---|---|---|---|---|---|
| `rhythm-2` | `dataset-rhythm/dataset-2-vt-svt-avb2-avb3` | caliper 60 s | 5,082 (AVB2 2,234 · AVB3 1,478 · VT 790 · SVT 580) | 4,558 → 3,492 / 1,066 | span caliper; ngoài span: VT/SVT → SINUS, AVB → IGNORE | `.atr` |
| `rhythm-3` | `dataset-rhythm/dataset-3-vt-svt-avb2-avb3` | caliper 60 s | 705 (SVT 379 · AVB2 213 · VT 89 · AVB3 24) | 686 → 468 / 218 | như trên | `.atr` |
| `afib-2` | `dataset-rhythm/dataset-afib-2` | caliper 60 s | 7,962 (AFIB) | 7,962 → 6,129 / 1,833 | AFIB trong span, ngoài span IGNORE | `.atr` |
| `sinus-2` | `dataset-sinus/dataset-2-tachy-brady-pause` | caliper 60 s | 1,074 (TACHY 674 · BRADY 369 · PAUSE 31) | 1,055 → 993 / 62 | SINUS (hard negative, ưu tiên khi áp cap) | `.atr` |
| `sinus-3` | `dataset-sinus/dataset-3-tachy-brady-pause` | caliper 60 s | 24,152 (TACHY 15,022 · BRADY 9,124) | 24,036 → 22,789 / 1,247 | SINUS (hard negative) | `.atr` |
| `sinus-4p` | `dataset-sinus/dataset-4-tachy-brady-pause` | strip 10 s | 1,262 (TACHY 606 · PAUSE 509 · BRADY 147) | 1,262 → 1,227 / 35 | SINUS + run S/V ≥ 3 beat → SVT/VT | `.atr` |
| `sinus-4` | `dataset-sinus/dataset-4-tachy-brady` | strip 10 s | 16,067 (TACHY 12,324 · BRADY 2,412 · OTHERS 1,331 bỏ) | 14,736 → 14,276 / 460 | như trên | `.atr` |
| `dataset-5` | `dataset-5` | strip + caliper | 55,340 (SVE_RUN 6,899 · SVT 3,294 · VE_RUN 2,564 · ectopy… · OTHERS 14,326 bỏ) | 41,011 → 34,313 / 6,698 | `EVENT_TYPE_TO_CLASS`; ectopy → SINUS | `.atr` |
| `dataset-1` | `dataset-1` | strip 10 s | 145,756 (SVE_RUN 15,864 · SVT 6,126 · TACHY 11,623 · ectopy… · OTHERS 41,845 bỏ) | 103,904 → 98,748 / 5,156 | như trên + run S/V từ `.atr` | `.atr` |
| `dataset-4` | `dataset-4` | strip 10 s (span đọc từ `.hea`) | 156,400 (OTHERS 66,539 bỏ · TACHY 13,058 · SVE_RUN 9,411 · ectopy…) | 89,852 → 67,800 / 7,288 (14,764 trùng nguồn trên) | như trên | `.atr` |

Tổng sau khi build: **train** 294,960 window từ 46,487 study. **eval** 65,919 window từ 11,698
study. **test `rhythm_eval`** 7,233 window từ 4,909 event và 1,966 study. 4,154 study bị loại
theo list v4.

Số giây có nhãn ở split train:

| SINUS | AFIB | SVT | VT | AVB (AVB2 28,747 + AVB3 13,590) | IGNORE |
|---|---|---|---|---|---|
| 2,278,026 | 457,652 | 131,933 | 30,517 | 42,337 | 9,135 |

Npy hiện có được build với 6 lớp (manifest `class_names` cũ). `pipeline.load_arrays` tự gộp
nhãn AVB2/AVB3 → AVB khi tải, còn class weight được tính lại từ số giây đã gộp
(`pipeline.manifest_class_weights`). Vì vậy không bắt buộc build lại; nếu build lại thì manifest
mới sẽ ghi 5 lớp.

Cap SINUS đã bỏ 17,799 strip; budget "hard sinus" giữ thêm 41,377 strip.

**B. Không dùng trực tiếp:**

| Dataset | Kích thước | Lý do không dùng |
|---|---|---|
| `dataset-2` (export MCT gốc) | 17,491 event, 5,338 study; không có `eventType` chuẩn, chỉ có `Event Type` (AFib/Tachy/Brady/Pause/Manual) + `Event Tag1..11` + `AF/V/S Presence` | Nhãn rhythm là tag đa nhãn ở mức event, không có span. Phần có ích đã được lọc thành `rhythm-2`, `afib-2`, `sinus-2`. Beat `.atr` của nó **chưa** được decoder beat dùng |
| `dataset-3` (export MCT gốc) | 114,195 event, 41,120 study; `Event Type` chủ yếu Manual, `List Event Types` | Như trên; đã lọc thành `rhythm-3`, `sinus-3`, `afib-2`. Beat `.atr` **chưa** dùng cho decoder beat |
| `dataset-2-filter-vt-svt-avb2-avb3` (ở thư mục gốc) | 4,246 | Bản cũ, đã thay bằng `dataset-rhythm/dataset-2-…` (5,082) |
| `dataset-3-filter-vt-svt-avb2-avb3` (ở thư mục gốc) | 366 | Bản cũ, đã thay bằng `dataset-rhythm/dataset-3-…` (705). Beat task vẫn liệt kê bản này trong `TRAIN_DATASETS` |
| `dataset-AFib/dataset 2_3_4 - AFib - v2` | 7,962 | Trùng hoàn toàn với `afib-2` |
| `dataset_ivcd` | – | Beat task có liệt kê nhưng không có trên máy này |
| `beat-eval-dataset-2`, `RL-Beats`, `260519`, `260521`, `260522`, `260727`, `ecg_norm_1_*`, `Eval_data` | – | Chưa rõ nguồn và split. **Không dùng** khi chưa được xác nhận |
| `dataset-eval/rhythm_eval` | 1,966 study / 4,909 event | **Test holdout của rhythm**, cấm train |
| `dataset-eval/v4/beat-eval-dataset` (`dataset-v4-beat`) | 5,227 record | **Test holdout của beat** (target SVEB > 88 / > 92). Study của nó đã bị loại khỏi train qua `list_studies_eval_v4.json` |

**C. Cần người dùng quyết định trước khi train:**

1. **Có đưa beat `.atr` của `dataset-2` và `dataset-3` gốc (~131k event) vào decoder beat
   không?** Hiện decoder beat chỉ học từ các nguồn trong bảng A. Beat task chuyên dụng thì học
   cả dataset-1..5. Nếu đưa vào, các event này chỉ mang **nhãn beat**: rhythm = IGNORE, trừ khi
   tag là "Sinus Rhythm" đơn nhãn. Ngoài ra phải giữ split theo study và vẫn loại các study
   holdout.
2. Có dùng các thư mục chưa rõ nguồn ở mục B (`RL-Beats`, `2605xx/2607xx`, …) không?

Không tự thêm dataset nào khi chưa có câu trả lời. Sau mỗi lần thêm nguồn: chạy lại
`python3 -m ecgr.rhythm data` và `audit`, rồi báo cáo bảng sự kiện → giữ / holdout giống bảng A.

### Luật cấm rò rỉ (leakage)

1. **Cấm tuyệt đối trong train/eval/sweep:** mitdb, afdb, escdb, nstdb, ahadb, cudb
   (`EC57_DATABASES`) và mọi study dưới `rhythm_eval` (`ECGR_RHYTHM_EVAL_DIR`).
   `physionet_train` tự từ chối các DB này – không được tắt kiểm tra đó.
2. Holdout đọc **từ thư mục** `rhythm_eval/<study id>/`, không từ list (list v4 thiếu 110/1966 study);
   các study trong `assets/list_studies_eval_v4.json` cũng bị loại (`EXCLUDE_V4_EVAL_STUDIES=True`)
   để beat model và rhythm model dùng chung holdout.
3. Split luôn ở mức **study / bệnh nhân / record**, không bao giờ ở mức window. Một event xuất
   hiện ở nhiều export → lấy nguồn có ưu tiên cao nhất (thứ tự trong `SOURCES`).
4. Study id các nguồn ngoài được cộng offset (PTB-XL 1e9, PhysioNet 2e9, Challenge 3e9) –
   không được đổi, vì `audit` dựa vào đó để kiểm tra chồng chéo.
5. Sau mỗi lần `data`: chạy `python3 -m ecgr.rhythm audit`; audit fail = dừng, không train.
6. Không thêm nguồn mới (vd. thêm DB PhysioNet, export portal mới) mà chưa hỏi; nếu được duyệt
   thì phải có split theo study/bệnh nhân + offset id riêng + cập nhật audit.

### Quy định nhãn

- 5 lớp rhythm: SINUS, AFIB, SVT, VT, AVB. **AFL → AFIB; AVB2 và AVB3 → AVB.** Giây không ai xác nhận = IGNORE (255, weight 0).
- Ngoài caliper: SVT/VT/SINUS → SINUS; AFIB/AVB → IGNORE (`OUTSIDE_SPAN`).
- Một giây mang lớp khi span phủ ≥ 50 % giây đó.
- Beat: N = {N,L,R,B,e,j,n}, S = {A,a,J,S}, V = {V,E}; F, Q, /, f, !, r, ? và ±0.1 s quanh chúng = IGNORE.
- Noise/Clean: từ SNR augmentation (≥ 6 dB, ≥ 2/3 lead đọc được), **head riêng**, không phải lớp rhythm.

### Quy định cân bằng / lấy mẫu (không đổi nếu không có giả thuyết)

- SINUS portal: tối đa 3 strip/study + 8 strip "hard sinus" (ectopy, brady, pause)/study.
- PhysioNet: cap window theo category (`PHYSIONET_TRAIN_CAPS`), giữ hết VT/SVT/AF-edge/AF-ectopy.
- PTB-XL plain SINUS ≤ 3000, Challenge plain ≤ 1500; hard negative (BBB, paced, WPW, tachy/brady, PAC/PVC) giữ hết.
- Batch stratified 64: SINUS 24, AFIB 12, SVT 10, VT 7, AVB 11; class weight chia √(oversampling).
- Mọi thay đổi cap/quota phải ghi vào báo cáo kèm số window mỗi lớp trước/sau.

### Quy định chấm điểm

- **eval per-window:** `val_rhythm_f1` (macro per-second, chỉ giây clean), beat F1 qua
  BeatMatchLog (khớp 150 ms như bxb), noise accuracy.
- **validation record-level:** `ec57 --dbs ltafdb nsrdb incartdb --skip-rhythm-eval --sweep-on validation`;
  ô mục tiêu ltafdb AFIB/SVT/VT; nsrdb/incartdb chỉ đếm FP; ngân sách ≤ 0.4 false-AF-episode/giờ.
- **test:** EC57 đủ bộ + `rhythm_eval`, inference hop 5 s + TTA (id, flip, swap), lưới chấm 1 s,
  loại 4 record paced mitdb (102, 104, 107, 217) trừ khi `--include-paced`. AFIB chấm `epicmp -x`.
  Beat chấm bằng `bxb` (QRS, VEB, SVEB).
- Nếu đã nhìn số test rồi đổi bất kỳ tham số nào → kết quả mới **không còn là test**; phải ghi rõ.

## Môi trường (máy i9-gpu, RTX A5000)

- Interpreter: `python3` (3.12, packages trong `~/.local`). Không có conda / `python`.
- `export ECGR_PHYSIONET_DIR=/media/Project/ECG/PhysionetData/`
- `export ECGR_RHYTHM_RUN_TAG=<DDMMYY>_rhythm_u1250b_<tên>` → output ở
  `/media/Project/ECG/Model_Dong/ecgr_rhythm/<run_tag>/`.
- Data npy: `/media/Project/ECG/Model_Dong/ecgr_rhythm/npy_250hz_10s_3lead_10sec`. Nếu thiếu hoặc
  đổi nguồn/label → `python3 -m ecgr.rhythm data` (~10 phút) rồi `python3 -m ecgr.rhythm audit`.
- Warm-start: checkpoint baseline
  `/media/Project/ECG/Model_Dong/ecgr_rhythm/011026_rhythm_u1250/checkpoints/unetmamba1250_rhythm_1m/best_model.keras`
  (hoặc run v2 `061026_rhythm_u1250b_v2`).
- Thời gian tham khảo: ~5.5 phút/epoch (1M params), early-stop quanh epoch 18–38; EC57 đủ bộ
  ~35 phút, `--decode-only` ~5 phút.

## Quy trình bắt buộc

### Bước 0 – Đọc trước khi sửa
Đọc `ecgr/rhythm/{config,model,unet,objectives,train,labels,beats,ec57}.py` và
`tests/test_rhythm*.py`. Tóm tắt lại (≤ 15 dòng): kiến trúc 1250b, loss từng head, sampler,
cách beat-PP chạy sau `labels.decode_track`. Kiểm tra `git status` – có thay đổi chưa commit,
**không ghi đè**.

### Bước 1 – Giả thuyết
Viết 1–3 giả thuyết cụ thể gắn với lỗi đã biết, ví dụ:
- mitdb 215 (sinus tachy → SVT), 232 (brady/pause → AVB2), 207 (BBB + VFL → VT/AFIB),
  200/203 (PVC dày → AFIB), 222/203 (flutter).
- SVEB thấp → S-beat thiếu dữ liệu / class weight S; AFIB +P duration → prior scale AFIB.
Mỗi giả thuyết: thay đổi gì (data / loss / weight / sampler / kiến trúc), đo bằng metric
validation nào, kết quả nào thì bác bỏ.

### Bước 2 – Smoke test
```bash
python3 -m pytest tests/test_rhythm.py tests/test_rhythm_beats.py -q
python3 -m ecgr.rhythm train --model rhythm_unet1250b_1mw --epochs 1 --max-windows 512 \
    --init-matching <baseline.keras>
```
Kiểm tra: shape output, loss không NaN, beat metric (BeatMatchLog, 150 ms) được log,
số tensor transfer khi warm-start (kỳ vọng ~232/271 với `_1mw`).

### Bước 3 – Train
```bash
python3 -m ecgr.rhythm train --model rhythm_unet1250b_1mw \
    --init-matching <baseline.keras> --freeze-epochs 2 \
    --schedule cosine --lr 2e-4 --sampler stratified \
    --epochs 40 --patience 8
```
Chạy nền, log ra file trong `<run_tag>/`. Theo dõi `val_rhythm_f1`, beat F1 (N/S/V), noise acc.
Lưu ý: `val_rhythm_f1` là macro per-second F1, trần ~0.8 do lớp hiếm + nhiễu label – **không**
diễn giải là "chưa hội tụ". Chạy seed thứ 2 khi giả thuyết cần so sánh.

### Bước 4 – Chọn tham số trên validation
```bash
python3 -m ecgr.rhythm ec57 --checkpoint <best.keras> --tag val_<tên> \
    --dbs ltafdb nsrdb incartdb --skip-rhythm-eval --sweep --sweep-on validation
```
Chọn decode (smoothing, minimums theo beat, prior scale, AF confidence floor) và tiêu chí
beat-PP (`BEAT_PP_*`, `BEAT_PICK_THRESHOLD`) theo ngân sách false-AF. Ghi rõ giá trị đã chọn.

### Bước 5 – Chấm EC57 một lần
```bash
python3 -m ecgr.rhythm ec57 --checkpoint <best.keras> --tag ec57_<tên>
python3 -m ecgr.rhythm ec57 --checkpoint <best.keras> --tag ec57_<tên>_nopp --decode-only --no-beat-pp
```
Báo cáo gồm 2 bảng. Mỗi bảng có cột target, cột baseline, ký hiệu ✓/✗ và cột A/B beat-PP
bật/tắt:
- **epicmp** (dòng Gross): lớp × DB × {Duration, Episode} × {Se, +P, F1}, cho AFIB/SVT/VT/AVB
  trên mitdb/afdb/escdb và mọi lớp trên `rhythm_eval`.
- **bxb** (dòng Gross): đủ 5 DB (mitdb, nstdb, escdb, ahadb, afdb) × {QRS, VEB, SVEB} × {Se, +P}, theo đúng bảng AAMI beat ở trên. Trên mitdb báo cáo thêm số beat
  reference của mỗi lớp.

Đính kèm đường dẫn report gốc (`<run_tag>/ec57/<tag>/<db>/*_report_line.out`) để có thể truy
ngược từng số.

### Bước 6 – Phân tích lỗi
Liệt kê top-5 record mitdb đóng góp FP/FN theo giây cho mỗi class, đối chiếu với danh sách lỗi
đã biết; lỗi nào là do model, lỗi nào do decode (so phân bố độ dài episode reference với
minimums trước khi đổ cho model).

## Định dạng báo cáo cuối

1. Giả thuyết → kết quả (xác nhận / bác bỏ), 1 dòng mỗi cái.
2. Bảng EC57 + bxb so với baseline và target.
3. Đường dẫn checkpoint, run tag, tham số decode đã chọn.
4. Rủi ro / điều chưa chắc chắn (vd chỉ 1 seed, record ít episode như AVB chỉ 5 episode ở 231).
5. Đề xuất bước tiếp theo (tối đa 3), xếp theo kỳ vọng lợi ích / chi phí.

Báo cáo trung thực: nếu metric giảm, test fail hoặc bước nào bị bỏ qua thì nói rõ kèm output.
