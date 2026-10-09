# Phase B – Phân tích hoạt động nhĩ (P, QRST, T, sóng f/F) với input 10 s + U-Net

> Tài liệu đề xuất phương pháp, đồng thời dùng làm prompt cho agent thực hiện.
> Đọc cùng [md_prompt.md](md_prompt.md): mọi ràng buộc dữ liệu, chấm điểm và nghiệm thu ở đó
> vẫn áp dụng ở đây.

## 1. Vấn đề

**Câu hỏi:** U-Net với input 10 s có phân tích được P, QRST, T và sóng f/F không?

**Trả lời: có thể, nhưng model hiện tại gần như chắc chắn chưa làm.**

Về độ phân giải thì đủ:
- Input 250 Hz.
- Nhánh beat chạy ở 8 ms/bước.
- Lưới gộp ở 40 ms vẫn bắt được sóng f 4–9 Hz.

Vấn đề là model **không bị buộc phải nhìn** các đặc trưng đó:

1. **Biên độ lấn át.** QRS cao 1–2 mV, P chỉ 0.1–0.25 mV, sóng f còn 0.05–0.1 mV. Stem có
   2 lớp conv rồi max-pool 5 (`ecgr/rhythm/unet.py`), nên gần như chỉ giữ lại đỉnh QRS.
2. **Chỉ có nhãn rhythm.** Đường dễ nhất để giảm loss là nhìn độ đều của RR. Decoder beat còn
   cấp thẳng tự tương quan của chuỗi beat (`beat_ac`), làm đường tắt này mạnh hơn.
3. **Lỗi EC57 khớp với việc chỉ nhìn RR:**
   - mitdb 232 (sinus chậm + PAC) và 200/203 (PVC dày) bị gọi là AF: RR không đều nhưng vẫn có P.
   - 215 (sinus nhanh) bị gọi là SVT: RR đều nhưng P bình thường.
   - 222/203 (flutter) bị nhầm sang lớp khác.
   - AVB2 mitdb chỉ đạt E-F1 49.6: muốn nhận ra cần thấy **P không có QRS đi sau**, điều mà
     RR không chứa.

Muốn model dùng P/f thì phải đưa chúng vào **cấu trúc model** và **nhãn giám sát**.

## 2. Ràng buộc (không đổi)

- Input luôn là **1 cửa sổ 10 s** `(2500, 3)`, không dùng ngữ cảnh dài hơn.
- **AFL = AFIB**, vẫn một lớp rhythm. Head phân đoạn ở mục 3.3 chỉ phân biệt *loại hoạt động
  nhĩ* (P / f / F), không tạo thêm lớp rhythm.
- Noise là head riêng.
- Chỉ dùng nguồn dữ liệu đã được phép. Chọn tham số trên validation, chấm EC57 một lần mỗi
  model, so sánh với ≥ 2 seed.

## 3. Các hướng đề xuất (xếp theo lợi ích so với chi phí)

### 3.1. Trừ template QRST ngay trong model (không cần nhãn mới)

Đây là ABS (average beat subtraction) làm khả vi, dùng heatmap của decoder beat:

```
p(t)      = p(beat) từ decoder beat, stop-gradient            (1250,)
template  = Σ_t p(t)·x[t-w : t+w] / Σ p(t)                    (2w, 3), w ≈ 0.1 s trước / 0.45 s sau R
x̂         = p ⊛ template                                       (tích chập lại tại vị trí beat)
residual  = x − x̂                                             → kênh "hoạt động nhĩ" (2500, 3)
```

- Dùng **2 template theo loại beat** (N/S và V), trọng số là xác suất N/S/V. Nếu chỉ dùng một
  template, PVC để lại phần dư lớn và bị hiểu nhầm thành sóng f.
- `residual` đi qua một encoder nhỏ ở 4–8 ms/bước, không max-pool sớm. Ghép kết quả vào lưới
  250 bước cạnh nhánh `beat_ctx`.
- Toàn bộ là phép tensor, khoảng +30–60k tham số. Với 10–20 beat trong 10 s, template đủ ổn định.
- **Nhắm tới:** AFIB +P mitdb (200/203/232), flutter (222/203).

### 3.2. Gộp theo khoảng TQ và token theo từng beat

Khoảng TQ của beat i là `[R_i + QT ước lượng, R_{i+1} − 60 ms]`. Mỗi beat tạo một token:

```
[RR_i, RR_i/RR_{i-1}, loại beat N/S/V, P-present_i (pool trên TQ của residual),
 năng lượng 4–9 Hz trong TQ, tương quan TQ_i ↔ TQ_{i-1}]
```

Một SSM/transformer nhỏ chạy qua khoảng 10–25 token, cho ra rhythm theo beat, rồi chiếu ngược
về trục thời gian. Cách này khớp với stage beat-PP (`ecgr/rhythm/beats.py`).

- **AF:** không có P và TQ có năng lượng f không đều.
- **Flutter:** TQ có sóng F đều 4–6 Hz, giống nhau giữa các beat.
- **AVB2 / AVB3:** đếm số P nhiều hơn số QRS; PP đều trong khi PR thay đổi. Cần phát hiện P
  **độc lập với QRS**, nên làm sau 3.3.

### 3.3. Head phân đoạn P / QRS / T / f-F (giám sát trực tiếp, hướng mạnh nhất)

Thêm decoder thứ tư trên nhánh 8 ms: mỗi bước thuộc `none / P / QRS / T / f-F`.

| Nguồn nhãn | Ưu | Nhược |
|---|---|---|
| **LUDB** (200 bản ghi, đúng 10 s, 12 lead, P/QRS/T do bác sĩ đánh dấu) | Đúng khung 10 s, chất lượng cao | 500 Hz cần resample; ít AF. **Chưa có trong danh sách nguồn được phép, cần duyệt** |
| **QTDB** (105 bản ghi, có đoạn annotate P/QRS/T) | Có loạn nhịp | Chỉ annotate một phần. **Cần duyệt; phải loại các record lấy từ mitdb** |
| **Nhãn yếu suy từ nhãn rhythm** | Không cần dữ liệu mới; áp cho cả ~295k window train | Nhiễu: SINUS + beat N thì có P trước R; AFIB thì TQ = `f`; AFL (nguồn có phân biệt) thì TQ = `F` |
| Pseudo-label từ delineator cổ điển (`ecgpuwave`, NeuroKit2 DWT) trên strip sinus sạch | Nhiều | Sai ở chính các ca khó |

- **Đề xuất:** nhãn yếu cho toàn bộ dữ liệu, cộng LUDB làm "neo" chất lượng nếu được duyệt.
  Chỗ không chắc thì mask loss (IGNORE), giống nhãn beat.
- **Ràng buộc nhất quán** giữa các head (loss phụ, trọng số nhỏ):
  - rhythm = AFIB thì p(P) thấp trên TQ;
  - rhythm = SINUS và beat = N thì phải có P trong khoảng 120–300 ms trước R.

### 3.4. Augmentation phá đường tắt RR

- **RR không đều nhưng có P:** strip SVE_BIGEMINY, SINGLE_SVE, PAUSE (đã có trong hard-sinus),
  cộng time-warp từng đoạn RR mà giữ nguyên hình P-QRS-T của từng beat. Nhãn vẫn SINUS.
- **RR đều kèm sóng f/F:** AF đáp ứng thất đều, flutter dẫn 2:1.
- **Cắt-dán TQ:** ghép TQ của strip AF vào strip sinus đã trừ P (dựa trên 3.1), nhãn thành
  AFIB, và làm ngược lại. Phải kiểm tra bằng mắt một mẫu trước khi dùng rộng.

### 3.5. Chỉnh kiến trúc để không mất P ngay từ stem

- Ở nhánh "nhĩ", bỏ max-pool 5 ngay sau stem; dùng stride 2 kèm conv dilated
  (dilation 1–2–4–8, nhìn được khoảng 300 ms).
- Xử lý từng lead bằng cùng một bộ trọng số, rồi dùng attention chọn lead. P rõ nhất ở lead
  kiểu II/V1, mà thứ tự lead bị xáo khi augment.
- Pool trên TQ bằng **trung bình hoặc attention**, không dùng max.

## 4. Kiểm chứng model thực sự dùng P/f (không chỉ nhìn F1)

1. **Probe tuyến tính:** đóng băng encoder, train lớp tuyến tính đoán "có P" và "có f" trên
   validation. Không học được nghĩa là encoder không mã hóa thông tin đó.
2. **Đối chứng (counterfactual)** trên validation:
   - Thay TQ của strip AF bằng đường thẳng hoặc bằng TQ sinus: p(AFIB) phải giảm mạnh.
   - Ép RR đều nhưng giữ TQ có sóng f: vẫn phải ra AFIB.
   - Nếu p(AFIB) chủ yếu chạy theo RR thì model vẫn đang đi đường tắt.
3. **Integrated gradients:** đo tỷ lệ attribution rơi vào TQ, so giữa ca AF và ca PAC-sinus.
4. **Theo dõi record cụ thể:** số giây FP trên mitdb 232, 200, 203, 215, 222 và số episode
   AVB2 ở 231, trước và sau mỗi bước. Chỉ xem sau khi đã chốt model, không dùng để tune.

## 5. Lộ trình

| Bước | Nội dung | Dữ liệu mới | Kỳ vọng |
|---|---|---|---|
| 1 | 3.1 + 3.5 (trừ QRST khả vi, nhánh nhĩ độ phân giải cao) + probe/đối chứng | Không | AFIB +P mitdb, flutter |
| 2 | 3.3 với nhãn yếu + ràng buộc nhất quán | Không | 215 (SVT/sinus), 232 |
| 3 | 3.2 (token theo beat, đếm P/QRS) | Tốt nhất có LUDB | AVB2 E-F1, AVB3 |
| 4 | 3.4 (cắt-dán TQ) nếu bước 1–3 vẫn còn đường tắt RR | Không | Độ bền |

Mỗi bước:
- Warm-start từ checkpoint `rhythm_unet1250b_1mw` tốt nhất.
- Chạy 2 seed.
- Chọn decode và beat-PP trên validation (ltafdb/nsrdb/incartdb).
- Chấm EC57 một lần, có A/B với bước trước.
- Báo cáo theo bảng AAMI (epicmp + bxb) trong `md_prompt.md`, kèm kết quả probe và đối chứng.

## 6. Cần quyết định trước khi làm

1. Có cho dùng **LUDB** (và **QTDB**, loại các record lấy từ mitdb) làm nguồn train nhãn
   P/QRS/T không? Nếu không, bước 2–3 chỉ dùng nhãn yếu.
2. Có tách nhãn yếu `F` (flutter) khỏi `f` (AF) ở các nguồn có phân biệt AFL (ltafdb `(AFL`,
   PTB-XL `AFLT`, Challenge-2020) không? Điều này chỉ ảnh hưởng head phân đoạn; lớp rhythm vẫn
   là AFIB.
3. Ngân sách tham số: giữ khoảng 1.1M (`_1mw`) hay cho phép đến khoảng 1.3M cho nhánh nhĩ?
