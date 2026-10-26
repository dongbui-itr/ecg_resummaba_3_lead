# Kế hoạch bài báo Q1: một mạng, một lần suy luận, ba ngõ ra cho Holter 3 chuyển đạo

Cập nhật: 2026-10-19. Tài liệu này đặt ra tiêu chuẩn, yêu cầu và nội dung bắt buộc để bài báo
về Dual U-Net rhythm/beat (`ecgr/rhythm/dualunet.py`) được nhận ở tạp chí Q1. Mọi số liệu
hiện có lấy từ [md_dual_unet.md](md_dual_unet.md), [md_data_train_eval.md](md_data_train_eval.md),
[md_ec57_eval.md](md_ec57_eval.md) và [md_xai.md](md_xai.md).

Nguyên tắc viết của tài liệu này và của bài báo: mỗi khẳng định phải có một bảng hoặc một
thí nghiệm đi kèm. Khẳng định nào chưa có bằng chứng được đánh dấu **[chưa có]**.

---

## 1. Tạp chí mục tiêu và tiêu chuẩn của reviewer

### 1.1. Tạp chí

| Ưu tiên | Tạp chí | Lý do phù hợp |
|---|---|---|
| 1 | IEEE Journal of Biomedical and Health Informatics (JBHI) | Nhận bài có góc hệ thống và triển khai, quen với EC57 |
| 2 | Computers in Biology and Medicine (CIBM) | Nhận bài kiến trúc có ablation đầy đủ, chu kỳ review nhanh |
| 3 | Computer Methods and Programs in Biomedicine (CMPB) | Nhận bài pipeline đầy đủ, đòi hỏi mã nguồn công khai |
| 4 | IEEE Transactions on Biomedical Circuits and Systems (TBCAS) | Nếu phần đo trên vi điều khiển trở thành trọng tâm |

### 1.2. Bốn câu hỏi reviewer Q1 luôn hỏi

1. Đóng góp là gì, và điều gì trong văn liệu chưa trả lời được?
2. Bằng chứng nào cho thấy từng thành phần của phương pháp có tác dụng?
3. Kết quả có lặp lại được không, trên dữ liệu công khai, với mã nguồn công khai?
4. So với phương pháp tốt nhất đã công bố, bài này hơn ở đâu và kém ở đâu?

Bài báo chỉ được gửi đi khi cả bốn câu đều có câu trả lời bằng bảng số.

---

## 2. Luận điểm trung tâm và chuỗi lập luận

### 2.1. Luận điểm

Một mạng duy nhất, nhận cửa sổ 10 giây chạy tự do từ Holter 3 chuyển đạo, trong một lần suy
luận trả về đồng thời ba ngõ ra: vị trí và loại beat (N/S/V), sự kiện rhythm theo giây
(SINUS, AFIB, SVT, VT, AVB) được giải mã thành episode, và chất lượng từng kênh. Mạng này đạt
mức EC57 chấp nhận được trên năm database chưa từng dùng để train, với chi phí tính toán cho
một Holter 24 giờ thấp hơn nhiều lần so với chuỗi xử lý theo từng beat, và vẫn giữ được phần
lớn độ chính xác khi thu nhỏ xuống dưới 100 nghìn tham số.

### 2.2. Chuỗi lập luận (mỗi mắt xích là một mục trong bài)

| # | Mắt xích | Bằng chứng cần có | Trạng thái |
|---|---|---|---|
| L1 | Pipeline theo beat trong văn liệu có chi phí tỉ lệ với số beat và cần nhiều mạng | Bảng so sánh văn liệu (mục 3) + đo chi phí pipeline đối chứng trên cùng máy | **[chưa có]** đo đối chứng |
| L2 | Xử lý theo cửa sổ 10 s có chi phí cố định theo thời lượng | Số lần suy luận, FLOPs, giây GPU cho 24 h ở hop 5 / 2 / 1 s | **[chưa có]** bảng FLOPs |
| L3 | Một encoder chung với hai nhánh lọc khác nhau đủ sức làm ba việc | Ablation: single U-Net, dual không hủy QRST, không SSM, không beat-conditioning | **[chưa có]** |
| L4 | Train chung ba head không làm hại từng head | Ablation: train riêng từng head so với train chung, cùng tham số | **[chưa có]** |
| L5 | Kết quả giữ được trên dữ liệu ngoài phân phối | EC57 trên mitdb, afdb, escdb, nstdb, ahadb; đã có cho 1M | Có 1M, 2 seed |
| L6 | Thu nhỏ xuống 100k / 30k vẫn giữ phần lớn chất lượng | EC57 cho student + baseline cùng cỡ train từ đầu | **[chưa có]** EC57 và baseline |
| L7 | Người khác lặp lại được | Biến thể chỉ dữ liệu công khai + mã nguồn + checkpoint | **[chưa có]** biến thể công khai |
| L8 | Chi phí chính xác thấp hơn là chấp nhận được trong quy trình Holter có bác sĩ đọc lại | Mốc EC57 nội bộ + phân tích FP theo thời gian review + AF gate | Có một phần (AF gate) |

Mắt xích nào thiếu thì luận điểm 2.1 không đứng được. Thứ tự ưu tiên thực hiện: L3, L4 trước
(quyết định claim), rồi L1, L2, L6, L7.

---

## 3. Vị trí trong văn liệu

Phần Related Work phải có một bảng đối chiếu theo đúng năm tiêu chí của luận điểm. Các bài
dưới đây là những bài gần nhất đã tìm thấy (tra cứu 2026-10-19) và bắt buộc phải trích dẫn.

| Bài | Input | Beat vị trí + loại | Rhythm theo giây, episode | Chất lượng kênh | Tham số | EC57 cross-DB |
|---|---|---|---|---|---|---|
| Multitask 12-lead delineation + rhythm + median beat (EHJ Digital Health 2025) | 12 lead, 10 s | delineation, không N/S/V | có theo điểm thời gian | không | lớn | không |
| QRS detection + classification in one step (Sci Rep 2022) | 1 lead Holter | có | không | không | lớn | không |
| DeepBeat (npj Digit Med 2020) | PPG | không | AF theo cửa sổ | có | trung bình | không |
| Tiny Transformer on MCU (2024) | beat đã cắt quanh R | loại beat | không | có tầng nhiễu | 6k | không, intra-patient |
| ECG-Mamba / S2M2ECG (2025) | 12 lead, 10 s | không | nhãn cả strip | không | trung bình | không |
| **Bài này** | 3 lead Holter, 10 s chạy tự do | có | có, 5 lớp | có | 1M / 80k / 23k | có, 5 DB |

Câu claim tính mới được phép viết: *"Theo hiểu biết của chúng tôi, đây là mạng một lần suy
luận đầu tiên đồng thời phát hiện và phân loại beat, phân đoạn rhythm theo giây có chấm
episode theo EC57, và ước lượng chất lượng kênh trên Holter 3 chuyển đạo chạy tự do, với dưới
100 nghìn tham số."* Không được viết claim rộng hơn câu này.

---

## 4. Yêu cầu về phương pháp

### 4.1. Gọi đúng tên thành phần

- Khối `DiagSSM1D` là SSM chéo tuyến tính bất biến theo thời gian, tính bằng tích chập FIR, hai
  chiều. Trong bài gọi là **diagonal state-space block (S4D-style)**. Không gọi là Mamba, vì
  không có cơ chế chọn lọc phụ thuộc đầu vào.
- Nếu muốn có từ "Mamba" trong bài, phải thêm một biến thể neck dùng S6 thật và đưa vào bảng
  ablation. Đây là lựa chọn, không bắt buộc.
- Hủy QRST gọi là **differentiable average-beat subtraction** với template lấy từ heatmap beat
  và stop-gradient.

### 4.2. Mô tả kiến trúc phải trả lời "vì sao" ở mỗi khối

| Khối | Câu hỏi reviewer | Câu trả lời phải có trong bài |
|---|---|---|
| Stem theo lead + lead pooling | Vì sao bất biến thứ tự lead? | Holter 3 lead không chuẩn hóa vị trí điện cực; test chứng minh bất biến chính xác |
| Nhánh V max-pool | Vì sao max-pool? | Giữ đỉnh QRS 1–2 mV |
| Nhánh A avg-pool + dilated trên residual | Vì sao cần residual? | P 0.1–0.25 mV, f 0.05–0.1 mV bị max-pool xóa; ablation L3 phải cho thấy chênh lệch |
| Neck SSM | Vì sao không attention? | FLOPs tuyến tính theo thời gian, xuất được TFLite, bộ nhớ kernel đo được 3.4–4.0 s |
| Ba decoder + stop-gradient | Vì sao tách gradient? | Loss rhythm không được phá beat; ablation L4 |

### 4.3. Dữ liệu và chống rò rỉ

- Bảng dữ liệu train đầy đủ theo nguồn, số window, số giây mỗi lớp (đã có trong
  md_data_train_eval.md mục 1).
- Chia split theo study hoặc bệnh nhân, công bố quy tắc hash.
- Bảng kiểm rò rỉ: train × eval = 0, train × rhythm_eval = 0, không DB EC57 nào trong train.
- Hai cấu hình train: **Full** (portal + công khai) và **Public-only** (PTB-XL, ltafdb,
  incartdb, svdb, Challenge-2020). Mọi bảng kết quả chính báo cả hai. **[chưa có]** Public-only.

### 4.4. Giao thức đánh giá

- EC57 bằng `epicmp` và `bxb` của WFDB, nêu rõ phiên bản. Nói rõ epicmp chính thức chỉ chấm
  AF/AFL; SVT, VT, AVB chấm bằng cách đổi nhãn từng lớp và gọi là "EC57-style per-class
  episode scoring".
- Tham số decode chỉ chọn trên validation (ltafdb, nsrdb, incartdb, svdb phần eval). EC57 chạy
  đúng một lần cho mỗi cấu hình cuối. Viết câu này rõ trong bài.
- Bỏ 4 record paced của mitdb, nêu lý do.
- Mỗi số EC57 đi kèm: số episode tham chiếu, số record có lớp đó, và khoảng tin cậy bootstrap
  theo record (1000 lần). Lớp có dưới 10 episode tham chiếu (AVB trên mitdb) phải ghi chú là
  không đủ thống kê.
- Ít nhất 3 seed cho cấu hình chính. Báo trung bình ± độ lệch chuẩn. **[chưa có]** seed 3.

---

## 5. Thí nghiệm bắt buộc

### 5.1. Ablation kiến trúc (L3)

Cùng ngân sách 1M tham số, cùng data, cùng decode mặc định, 3 seed.

| Biến thể | Thay đổi | Câu hỏi trả lời |
|---|---|---|
| A0 | Dual U-Net đầy đủ | mốc |
| A1 | Single U-Net, chỉ nhánh V, mở rộng width cho đủ 1M | Hai nhánh có cần không |
| A2 | Dual, cả hai nhánh max-pool, không hủy QRST | Bộ lọc khác nhau có cần không |
| A3 | Dual, có nhánh A nhưng không hủy QRST | Hủy QRST có cần không |
| A4 | Neck SSM thay bằng conv dilated cùng tham số | SSM có cần không |
| A5 | Bỏ beat-conditioning và RR FiLM vào rhythm | Luồng beat → rhythm có cần không |

Kết quả XAI hiện tại dự báo A3 có thể chênh dưới 1 điểm. Nếu đúng như vậy, bài phải nói thẳng
và hạ claim về nhánh A xuống "hai bộ lọc theo biên độ", hoặc thêm head phân đoạn P/f
(md_atrial_activity.md mục 3.3) rồi chạy lại A3.

### 5.2. Ablation đa nhiệm (L4)

| Biến thể | Mô tả |
|---|---|
| M0 | Train chung ba head (mốc) |
| M1 | Chỉ head beat |
| M2 | Chỉ head rhythm |
| M3 | Chỉ head channel |
| M4 | Beat + rhythm, không channel |

Báo cho từng head: metric validation và EC57. Kết luận được phép: "train chung không làm giảm
head nào quá X điểm và tiết kiệm Y lần suy luận".

### 5.3. Đối chứng pipeline theo beat (L1, L2)

Tái hiện trên cùng máy một pipeline gồm: QRS detector công bố (ví dụ Pan-Tompkins hoặc
một detector học sâu công khai), classifier beat công bố (ví dụ Tiny Transformer 6k hoặc CNN
1D chuẩn), model rhythm theo cửa sổ công khai. Train lại trên cùng data Full để công bằng.

Đo cho cả hai phương án trên cùng 24 h record (chọn 3 record mitdb hoặc ltafdb nối dài, hoặc
record Holter thật):

| Chỉ số | Cách đo |
|---|---|
| Số lần suy luận cho 24 h | đếm thực |
| FLOPs cho 24 h | profiler |
| Giây GPU và giây CPU cho 24 h | đồng hồ tường, batch đầy, 5 lần lấy trung vị |
| RAM đỉnh | đo tiến trình |
| Số Holter 24 h mỗi giờ trên 1 GPU, trên 1 CPU core | suy ra từ trên |
| EC57 tương ứng | cùng giao thức mục 4.4 |

Kịch bản tải: 500 Holter/ngày, tính số GPU hoặc CPU cần cho từng phương án. Bảng này là
bảng quan trọng nhất của góc hệ thống.

### 5.4. Đường Pareto chính xác và tài nguyên (L2, L6)

Lưới: 3 cỡ model (1M, 100k, 30k) × 3 hop (5 s, 2 s, 1 s). Trục ngang là giây GPU cho 24 h,
trục dọc là EC57 F1 từng lớp. Đánh dấu điểm cấu hình chính thức.

Thêm cho 100k và 30k: baseline cùng cỡ train từ đầu không SSL, không KD; biến thể chỉ SSL;
biến thể chỉ KD. Bốn ô này tách được đóng góp của SSL và KD. **[chưa có]** cả bốn.

### 5.5. Đo trên thiết bị nhúng (L6)

Ít nhất một vi điều khiển hoặc máy tính nhúng (ví dụ STM32H7, ESP32-S3, Raspberry Pi):
latency cho 1 cửa sổ 10 s, RAM, flash, năng lượng nếu đo được, cho 100k và 30k sau lượng tử
hóa int8. Báo chênh lệch chính xác trước và sau lượng tử hóa.

### 5.6. Độ bền của xử lý theo cửa sổ (L8)

- Hiệu ứng mép window: F1 theo vị trí giây trong window (đã có từ XAI), và cách overlap hoặc
  taper giảm nó.
- Nhiễu: EC57 trên nstdb theo mức SNR.
- Mất một lead: EC57 khi một lead bằng 0.

---

## 6. Cấu trúc bài báo và nội dung từng mục

| Mục | Nội dung bắt buộc | Độ dài gợi ý |
|---|---|---|
| Abstract | Luận điểm 2.1, ba con số: EC57 chính, hệ số tiết kiệm tài nguyên, số tham số nhỏ nhất | 250 từ |
| 1. Introduction | Bài toán Holter khối lượng lớn, chi phí pipeline theo beat, khoảng trống (bảng mục 3 rút gọn), ba đóng góp | 1.5 trang |
| 2. Related Work | Phân loại beat, phân loại rhythm, multi-task ECG, SSM cho ECG, model nhỏ; kết bằng bảng mục 3 | 1.5 trang |
| 3. Method | Hình kiến trúc tổng thể; 3.1 stem và lead pooling; 3.2 nhánh V; 3.3 hủy QRST và nhánh A; 3.4 neck SSM; 3.5 ba decoder và luồng gradient; 3.6 loss và sampler; 3.7 decode thành episode và AF gate; 3.8 SSL và KD cho model nhỏ | 4 trang |
| 4. Data and Protocol | Bảng dữ liệu, split, chống rò rỉ, EC57-style scoring, validation-only tuning, seed, CI | 2 trang |
| 5. Results | 5.1 EC57 chính (Full và Public-only, 3 seed); 5.2 ablation kiến trúc; 5.3 ablation đa nhiệm; 5.4 đối chứng pipeline theo beat và kịch bản tải; 5.5 Pareto và model nhỏ; 5.6 thiết bị nhúng; 5.7 độ bền | 5 trang |
| 6. Discussion | Vì sao chính xác thấp hơn bài cắt đoạn ngắn và vì sao chấp nhận được; giới hạn: dữ liệu portal tư nhân, AVB ít episode, nhánh A chưa đọc P/f đặc hiệu (dẫn chiếu bài XAI riêng), AFL gộp vào AFIB | 1.5 trang |
| 7. Conclusion | Ba đóng góp, một câu về hướng tiếp theo | 0.3 trang |
| Supplementary | Chi tiết layer, bảng hyper-parameter, EC57 từng record, mã nguồn và checkpoint | không giới hạn |

Hình bắt buộc: (1) kiến trúc, (2) ví dụ ba ngõ ra trên một đoạn 10 s thật, (3) đường Pareto,
(4) biểu đồ chi phí 24 h của hai phương án, (5) F1 theo vị trí trong window.

---

## 7. Yêu cầu về cách viết

- Mỗi đoạn mở bằng câu khẳng định, sau đó là bằng chứng, sau đó là hệ quả. Không mở đoạn bằng
  mô tả thủ tục.
- Mỗi số trong phần Results có nguồn là một bảng hoặc một hình được đánh số. Không có số nào
  chỉ xuất hiện trong văn bản.
- Mỗi khẳng định so sánh ("tốt hơn", "nhỏ hơn", "nhanh hơn") phải có đối tượng so sánh, điều
  kiện so sánh và mức chênh lệch kèm khoảng tin cậy.
- Thuật ngữ cố định suốt bài: "window" cho cửa sổ 10 s, "episode" cho đoạn rhythm sau decode,
  "beat" cho nhịp, "head" cho ngõ ra, "branch" cho nhánh V/A. Không đổi tên giữa chừng.
- Viết câu đầy đủ chủ ngữ và vị ngữ. Không viết câu cụt, không viết câu nối nhiều ý bằng dấu
  phẩy, không viết tắt chưa định nghĩa.
- Kết quả âm tính được viết thẳng, trong cùng mục với kết quả dương tính, không giấu xuống
  phụ lục.
- Không dùng từ "novel", "state-of-the-art", "significantly" nếu không có kiểm định thống kê
  đi kèm.

---

## 8. Yêu cầu về tái lập

- Repo công khai với: mã train, mã eval EC57, script decode, test.
- Checkpoint công khai cho cấu hình Public-only ở cả ba cỡ.
- File cấu hình ghi đầy đủ hyper-parameter, seed, phiên bản thư viện, phiên bản WFDB.
- Script tái tạo từng bảng của bài từ checkpoint, không cần train lại.
- Danh sách record và split của mọi DB công khai, dạng CSV.

---

## 9. Danh sách kiểm trước khi gửi

- [ ] L1–L8 đều có bảng số.
- [ ] Ablation 5.1 và 5.2 chạy 3 seed, có CI.
- [ ] Đối chứng pipeline theo beat đo trên cùng máy, cùng data, cùng EC57.
- [ ] Pareto 3 cỡ × 3 hop, kèm 4 ô SSL/KD.
- [ ] Public-only có đủ bảng chính.
- [ ] Đo trên ít nhất một thiết bị nhúng.
- [ ] Không có chữ "Mamba" trừ khi có biến thể S6 thật trong ablation.
- [ ] Claim tính mới đúng nguyên văn mục 3.
- [ ] Bảng đối chiếu văn liệu mục 3 có trong bài.
- [ ] Giới hạn về nhánh A và XAI được nêu trong Discussion.
- [ ] Mã nguồn, checkpoint, split CSV đã công khai và được dẫn link.

---

## 10. Ước lượng khối lượng

| Hạng mục | GPU-giờ ước tính | Ghi chú |
|---|---|---|
| Ablation 5.1: 6 biến thể × 3 seed × ~4 h | 72 | Lớn nhất |
| Ablation 5.2: 5 biến thể × 3 seed × ~4 h | 60 | |
| Public-only: 1 cấu hình × 3 seed | 12 | |
| Model nhỏ: 4 ô × 2 cỡ | 16 | SSL + KD đã có |
| Đối chứng pipeline theo beat | 12 | Train lại 2–3 model công khai |
| EC57 cho mọi biến thể ở overlap 1 s | 10 | ~8 phút mỗi lần |
| Đo nhúng và profiling | 0 | CPU và thiết bị |

Tổng khoảng 180 GPU-giờ, tương đương 4–6 tuần trên một GPU nếu chạy nối tiếp, 2–3 tuần nếu
có hai GPU. Viết bài song song với phần chạy ablation.

---

## 11. Bài XAI tách riêng

Các phát hiện trong [md_xai.md](md_xai.md) (hiệu ứng mép, nhánh V và SSM quyết định, sóng f đi
qua nhánh V, nhánh A không đặc hiệu) không phục vụ luận điểm 2.1 và sẽ làm loãng bài này. Chúng
thành bài thứ hai với câu hỏi riêng: "Mô hình rhythm học sâu có đọc hoạt động nhĩ không?".
Bài này chỉ dẫn chiếu bài XAI trong Discussion để nêu giới hạn của nhánh A.
