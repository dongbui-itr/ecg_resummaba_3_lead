# XAI cho dual U-Net rhythm: phân tích từng layer, chẩn đoán lỗi EC57, đề xuất hiệu chỉnh

Code: [ecgr/rhythm/xai.py](ecgr/rhythm/xai.py). Test: `tests/test_rhythm_xai.py` (6 test).
Chạy (CPU là đủ, ~5 phút với 300 window/lớp):

```bash
python3 -m ecgr.rhythm xai --checkpoint <best_model.keras> \
    --explain-ec57 <run>/ec57/<tag>          # giải thích các đoạn lỗi EC57 dài nhất + tách lỗi model / post-process
    [--explain /path/mitdb/223:605:VT]      # một giây bất kỳ
    [--parts probe layers experiment explain] [--per-class 300]
```

Output: `<out>/xai_report.md` (bảng + đề xuất xếp hạng), `xai_report.json`, `explain_<rec>_<s>.png`.
Mọi số đo trên split **eval** của dữ liệu train (chưa từng train, có nhãn theo giây) và trên
output EC57 đã có. XAI **không** chọn tham số trên EC57: knob nào thắng ở đây phải qua
`eval_rhythm.py` (tune trên validation, rồi chấm EC57 một lần).

## 1. XAI làm gì

| Phần | Cách đo | Trả lời câu hỏi |
|---|---|---|
| **probe** | Can thiệp phản thực tế vào layer có tên, beat giữ nguyên: `qrst_cancel` = 0 (nhánh A mù), residual / RR descriptor lấy từ window lớp khác (donor), `rr_ac` = trung bình, điều kiện beat = 'none', lead weight đều, mọi `DiagSSM1D` = identity, chèn sóng f 4–8 Hz vào SINUS (kèm đối chứng nhánh A mù) | Quyết định AFIB/AVB/VT dựa vào hoạt động nhĩ, RR hay ngữ cảnh? |
| **layers** | Lead weight (entropy, khớp với `channel`, phản ứng khi 1 lead bị nhiễu 0 dB); residual QRST (năng lượng, tỉ lệ 4–9 Hz theo lớp); kernel SSM (bộ nhớ 90 % L1, tỉ lệ bị cắt ở cuối kernel); ReLU chết; gradient × activation tại các điểm nối nhánh; **F1 theo vị trí giây trong window**; TTA nào bất biến | Layer học được gì, capacity có bị phí không |
| **experiment** | Quét knob không cần train lại: cửa sổ QRST pre/post, ngưỡng peak, nhiệt độ lead weight, cắt kernel SSM | Có tham số suy luận nào tốt hơn giá trị đã train không |
| **explain** | Integrated gradients (thời gian × lead) + beat + p(rhythm) + residual nhĩ + lead weight cho từng giây | Vì sao record này sai |
| **error_sources** | Mỗi giây FP/FN của EC57 (mitdb) được gán *model* hay *post-process* theo argmax thô của xác suất record mà decoder đã thấy | Sửa decode hay sửa model |
| **suggest** | Luật đọc các số đo → danh sách thí nghiệm xếp hạng | Làm gì tiếp |

## 2. Kết quả trên checkpoint s1 (epoch 26, val_rhythm_f1 0.884)

Báo cáo đầy đủ: `/media/Project/ECG/Model_Dong/ecgr_rhythm/071026_rhythm_dual_1m_s1/xai/s1_best/xai_report.md`.

**Phát hiện chính**

1. **Hiệu ứng mép window (đòn bẩy lớn nhất, không cần train lại).** F1 macro theo giây là 92.9 ở
   giữa window và 86 ở hai mép; SVT 91.7 → 72.1, VT 94.4 → 80.5. Ví dụ mitdb 221 s1041: window
   có giây đó ở giữa → AFIB 0.96, window bắt đầu đúng giây đó → SINUS 0.71; với hop 5 s mỗi
   giây chỉ được 2 window "bỏ phiếu" ngang nhau. → hop 2 s + trung bình có trọng số theo vị trí
   (`rc.PREDICT_TAPER_FLOOR`, `eval_rhythm.py PREDICT_TAPER`). Đang chấm (mục 3).
2. **TTA 'swap' vô dụng với dual U-Net.** Lead pooling bất biến thứ tự lead (|Δp| 1e-6), nên
   `swap` chỉ tốn 1/3 thời gian suy luận. 'flip' thì có tác dụng (|Δp| tới 0.8).
3. **Quyết định hình thành từ nhánh V + ngữ cảnh SSM, không từ bằng chứng nhĩ đặc hiệu.**
   - SSM → identity: macro F1 90.4 → 25.9 (SVT về 0). Bộ nhớ kernel trung vị 3.4–4.0 s, chỉ
     ≤ 10 % kênh chạm cuối kernel 5.1 s ⇒ `kernel_len` không phải nút thắt.
   - Thay residual nhĩ / RR descriptor bằng của window lớp khác: p(lớp) đổi ≤ 0.05 cho mọi lớp.
   - Nhưng làm mù nhánh A: AVB Se 89.4 → 74.7, VT 84.6 → 73.1 ⇒ nhánh A mang thông tin
     *không đặc hiệu* (cần "một residual", không cần P/f của chính window đó).
   - RR descriptor về trung bình: −1.2 điểm; tắt điều kiện beat: −0.2 ⇒ hai module gần như không dùng.
4. **Sóng f đi vào qua nhánh V (tín hiệu thô).** Chèn sóng f 10 % biên độ QRS vào sinus đều:
   AFIB 31.5 % số giây; nhánh A mù vẫn 19.1 %. Nhiễu nền 4–9 Hz vì thế bị đọc thành AF — khớp
   với FP AFIB trên record nhiễu (mitdb 200, afdb 04043).
5. **Lead weight nhánh A bỏ qua lead nhiễu.** Lead 1 nhiễu 0 dB: trọng số nhánh V 0.35 → 0.00
   (đúng), nhánh A 0.36 → 0.33 (không phản ứng). Hình `explain_200_149.png` cho thấy đúng cơ chế
   này trên FP AFIB của mitdb 200.
6. **Nguồn lỗi EC57 (mitdb, giây):**

   | Lớp | FP model / post-proc | FN model / post-proc | Record chính |
   |---|---|---|---|
   | AFIB | 304 / 134 | 606 / 35 | FN: 221, 219; FP: 200, 222 |
   | SVT | 58 / 41 | 79 / 30 | 202, 207, 234 |
   | VT | 134 / 20 | 73 / 83 | FP: 207 (VFL, model); FN: 223 (60 s do luật HR ≥ 120 của beat-PP) |
   | AVB | 42 / 5 | 4 / 0 | 231, 201 |

   Phần lớn lỗi là của model; ngoại lệ là VT FN trên 223 do post-process.
7. Không có ReLU chết (tệ nhất 2 %); residual QRST giữ được sóng f (tỉ lệ 4–9 Hz AFIB/SINUS 1.67);
   không knob QRST / lead temperature nào hơn giá trị đã train quá 0.25 điểm.

## 3. Đề xuất (xếp theo khả năng đổi EC57 / chi phí)

| # | Thí nghiệm | Loại | Trạng thái |
|---|---|---|---|
| 1 | hop 2 s + taper 0.1 + TTA id,flip (bỏ swap) | suy luận | **không dùng**: decode thô tốt hơn (validation obj 319.5 → 332.8) nhưng sau beat-PP thua (361.2 → 351.6, AF FP/h 0.30 → 0.38); EC57 khớp: mitdb VT 27.3/60.0 → 28.9/64.3 nhưng AVB E 61.1 → 47.3, SVT, AFIB E giảm. Chỉ giữ: bỏ 'swap' khỏi TTA |
| 2 | Train: augment nhiễu nền 4–9 Hz trên window SINUS (dạy V-branch không đọc nhiễu thành AF) | train lại | đề xuất |
| 3 | Train: dùng chung lead logits nhánh V cho nhánh A, hoặc giám sát `alead_w` bằng target `channel` | train lại | đề xuất |
| 4 | Train: head phân đoạn P/QRS/T/f cho nhánh A hoặc V-branch dropout, để AF/AVB phải đọc từ hoạt động nhĩ (md_atrial_activity.md 3.3); chạy lại probe donor để kiểm chứng | train lại | đề xuất |
| 5 | Beat-PP: luật HR VT 120 làm mất 60 s VT trên 223 — chỉ đổi nếu validation ủng hộ (HR 100 kém hơn ~10 điểm objective) | decode | ghi nhận |
| 6 | Bỏ hoặc regularize `rr_film` / điều kiện beat (gần như không được dùng) | train lại | thấp |

## 4. Webservice XAI tương tác

Code: [ecgr/rhythm/xai_web.py](ecgr/rhythm/xai_web.py) + [xai_web.html](ecgr/rhythm/xai_web.html)
(stdlib `http.server`, canvas thuần, không CDN). Test: `tests/test_rhythm_xai_web.py`.

```bash
R=/media/Project/ECG/Model_Dong/ecgr_rhythm/071026_rhythm_dual_1m_s1
python3 -m ecgr.rhythm xai-web --checkpoint $R/checkpoints/dualunet_rhythm_1m/best_model.keras \
    --ec57-out $R/ec57/ec57_dual_s1 --xai-report $R/xai/s1_best/xai_report.json   # http://<máy>:8777/
```

CPU mặc định (`--gpu` để dùng GPU). ~8 s / lần phân tích (2 lần chạy, gradient 35 layer, IG 16 bước).

- **Input**: record PhysioNet + giây bắt đầu (dải cả record: tham chiếu vs argmax EC57 đã lưu, nhấp để nhảy),
  window eval split theo lớp, hoặc upload CSV. Chỉnh sửa: sóng f, nhiễu trắng / dải 4–9 Hz theo SNR,
  trôi nền, điện lưới, gain / mất / đảo lead, hoán vị lead, đảo thời gian, làm phẳng đoạn.
- **File .dat bất kỳ** (tab *File .dat*): duyệt thư mục trên server (gốc mặc định: PhysionetData,
  Holter_report_strip, physionet_data, challenge-2020, ptb-xl; thêm bằng `--browse-root <dir>`, ngoài
  các gốc thì bị chặn 403) hoặc tải lên .hea + .dat (+ annotation). Chọn kênh record cho từng lead
  model (`—` = lead trống, đệm 0 như khi rơi điện cực), file annotation (.atr, .qrs, …), giây bắt đầu
  (bước 0.1 s), dải toàn record để nhảy. Nhãn tham chiếu: mốc rhythm của annotation, nếu không có thì
  `eventType` + caliper `eventStartSample/StopSample` trong .hea (Holter strip), hoặc `Dx:` SNOMED
  (Challenge 2020). Link trực tiếp: `http://<máy>:8777/?file=<record>&start=<s>[&gain=10]`.
- **ECG 3 layer** (bật/tắt riêng): *lưới AAMI* (25 mm/s; ô nhỏ 1 mm = 0.04 s, ô lớn 5 mm = 0.2 s;
  5 / 10 / 20 mm/mV, lane là bội 5 mm), *tín hiệu* (mV sau lọc dải của pipeline, xung chuẩn 1 mV × 0.2 s,
  tô IG; "tự co" = z-score như cũ), *nhãn* (tham chiếu / argmax model theo giây, mốc rhythm + mã, dải
  event / Dx của .hea, ký hiệu beat và annotation khác như `~`, beat của model, chú thích .hea).
  Tooltip cho biên độ mV từng lead tại con trỏ.
- **Dải HR** dưới tín hiệu (layer tín hiệu): HR tức thời = 60 / RR cho từng khoảng R-R, vẽ thành đoạn ngang từ
  beat này tới beat sau; *atr* (beat của file annotation) = dải xám dày, *model* (beat head, lần chạy đã sửa) =
  đường xanh mảnh ở trên, nên chỗ lệch (beat thừa / thiếu) hiện ra ngay. Lề trái: HR trung bình window
  = 60 · (số beat − 1) / khoảng từ beat đầu tới beat cuối, và Δ model − atr. Bỏ RR < 0.2 s hoặc > 3 s.
  Tooltip: HR tại con trỏ + trung bình + số beat của cả hai.
- **Can thiệp layer**: nhánh A mù, SSM→identity từng layer, lead weight đều / tắt lead (V, A riêng),
  điều kiện beat, RR = trung bình, donor residual / RR từ window eval khác, QRST pre/post/ngưỡng,
  nhiệt độ lead weight, cắt kernel SSM.
- **Heatmap**: grad×activation của Σ log p(lớp | giây chọn) cho 33 layer theo thời gian (gốc / đã sửa / hiệu),
  chi tiết kênh × thời gian của layer được chọn, IG tô trên ECG, lead weight, residual nhĩ, head nhiễu, RR/FiLM.
- **Output**: p(rhythm) gốc vs đã sửa, bảng theo giây, Δp; quét mép window; chạy cùng chỉnh sửa trên n window eval.
- **Đánh giá phát hiện** (n = 60 / lớp, 2026-10-08, s1):

  | Phát hiện | Kết luận | Số đo |
  |---|---|---|
  | 1 Hiệu ứng mép | xác nhận | macro F1 giữa 93.0 vs mép 88.4 |
  | 2 TTA swap vô dụng | xác nhận | max \|Δp\| swap 1.2e-6, đảo dấu 0.63 |
  | 3 V + SSM quyết định | xác nhận | SSM→id −66.4, A mù −5.5, RR mean −1.1, beat cond 0.0 điểm |
  | 3b Donor không đặc hiệu | một phần | Δp tối đa 0.057 (ngưỡng 0.05) |
  | 4 Sóng f qua nhánh V | xác nhận | f 10 %: AFIB 27.7 % giây SINUS, A mù 22.0 % |
  | 5 Lead weight A bỏ qua nhiễu | xác nhận | V 0.34→0.00, A 0.35→0.32 |
  | 6 Lỗi EC57 chủ yếu do model | xác nhận | 79 % giây lỗi; 223 VT FN post-process 60 vs model 46 |

## 5. Tự động hiệu chỉnh tham số theo layer (`POST /api/tune`, [xai_tune.py](ecgr/rhythm/xai_tune.py))

Tìm tham lam (coordinate ascent) trên các can thiệp layer của mục 4: nhánh A mù, nhiệt độ lead,
tắt lead V / A, cắt kernel SSM, từng `DiagSSM1D` → identity, điều kiện beat, RR = trung bình.
Mục tiêu: trung bình log p(lớp | giây) trên n window eval của lớp đó (tập **chọn**); kết quả đo trên
n window khác (tập **kiểm tra**, không tham gia tìm). `side_weight` > 0 phạt việc làm lệch p trung bình
của các lớp khác. Vòng 0 cho bảng độ nhạy một-tham-số (layer nào gánh quyết định).

```bash
curl -X POST localhost:8777/api/tune -d '{"cls":"SVT","n":8,"max_rounds":3,"side_weight":1.0}'
```

Kết quả s1 (2026-10-13, n = 8 + 8, ~30 s / lớp trên CPU):

| Lớp | Cấu hình tìm được | p(lớp) kiểm tra gốc → tune | Ghi chú |
|---|---|---|---|
| AFIB | không đổi | 0.958 → 0.958 | đã bão hoà, mọi knob \|Δ\| < 0.001 |
| SVT | `neck_ssm0` identity + tắt lead V 1 + nhiệt độ 2 | 0.324 → 0.382 | AFIB 0.08 → 0.03, SINUS giữ nguyên |
| VT | `rhythm_ssm` + `vs_ssm` identity + nhiệt độ 2 | 0.274 → 0.261 | **không tổng quát**: AFIB 0.10 → 0.32 trên tập kiểm tra |

Đọc kết quả: các bước lớn nhất luôn là "tắt một SSM" — đó là layer **gánh** ngữ cảnh (khớp phát hiện 3),
không phải "núm lấy nét" đặc trưng; bỏ nó làm dịch prior giữa các lớp. Tune trên 8 window là công cụ
chẩn đoán (layer nào cản lớp nào), không phải cách chọn tham số suy luận: mọi cấu hình phải qua
validation tune + EC57 như mục 3.
