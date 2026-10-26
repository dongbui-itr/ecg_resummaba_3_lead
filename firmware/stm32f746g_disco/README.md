# Kit STM32F746G-Discovery: model rhythm chạy trên MCU, giao tiếp PC qua UART

```
PC: webservice/app.py, tab "STM32 kit"                     STM32F746G-Discovery
  record WFDB -> read_leads (250 Hz, lọc như lúc train)        USART1 = ST-LINK VCP (USB)
  CHUNK 0: 10 s + nhãn   ───────── UART 921600 8N1 ───────▶  cửa sổ 10 s (int16 µV, 3 lead)
  CHUNK k: 9 s  + nhãn                                         z-score từng lead
                                                               net A (X-CUBE-AI) -> stage C -> net B
                                                               giây chồng lấp = trung bình 2 cửa sổ
  RESULT: dự đoán + nhãn gửi lại ◀──────────────────────────  beats, HR, noise; LCD 3 layer
  vẽ lại LCD trên web (3 canvas)
```

Code: [App/](App/) (phần chạy giống hệt trên board và trên PC), [Core/Src/main.c](Core/Src/main.c)
(phần riêng của board), [export_model.py](export_model.py), [host/](host/) (kit ảo),
[webservice/stm32_link.py](../../webservice/stm32_link.py) (phía PC), test
[tests/test_stm32_kit.py](../../tests/test_stm32_kit.py).

## 1. Model trên kit

`r30k_A_e36` (student 30k, 23 361 tham số, `201026_rhythm_dual_30k_A/release/e36.keras`), float32.
Đổi checkpoint: sửa CONFIG trong `export_model.py`. Model 1M không vừa (riêng weights 4 MB, flash 1 MB).

X-CUBE-AI 10.2 (stedgeai 2.2) dịch đúng các phần conv nhưng **sai** hai layer xây từ frame/gather:
`QRSTCancel` và `RhythmDescriptor` (export nguyên khối: crash khi chạy C trên host; viết lại bằng
tile/reshape: chạy được nhưng `rr_ac` ra sai, cos ≈ 0). Nên model được cắt làm 3 phần
([ecgr/rhythm/mcu_export.py](../../ecgr/rhythm/mcu_export.py)):

| phần | chạy bằng | MACC | flash | activations |
|---|---|---:|---:|---:|
| net A: input → p(beat), encoder V, kênh | X-CUBE-AI | 6.20 M | 41 KB | 352 KB |
| stage: QRST cancellation + RR autocorrelation | C viết tay [App/ecg_stage.c](App/ecg_stage.c) | ~0.5 M | – | ~40 KB |
| net B: nhánh A, neck, 3 decoder | X-CUBE-AI | 7.39 M | 72 KB | 261 KB |

DiagSSM1D (FFT) được thay bằng đúng bộ lọc FIR hai phía của nó (depthwise conv 255 tap).
Kiểm chứng mỗi lần export: A → stage → B (TFLite) so với Keras: max |Δ| 2.6e-5; code C sinh ra
chạy trên PC (`stedgeai validate --mode host`) so với TFLite: cos = 1.000000 cho cả 11 output —
export dừng nếu < 0.99999.

## 2. Luồng dữ liệu, overlap, nhãn

- Chunk 0 = 2500 mẫu (10 s), chunk k ≥ 1 = 2250 mẫu (9 s). Kit giữ 1 s cuối của cửa sổ trước:
  cửa sổ k = [9k, 9k+10) s, luôn đúng 10 s đầu vào của model.
- Giây 0 của cửa sổ k (k ≥ 1) = trung bình xác suất của 2 cửa sổ (như `predict.py`, taper tắt),
  áp dụng cho cả rhythm và beat. PC ghép record: giây [9k, 9k+9) lấy từ cửa sổ k.
- Nhãn đi kèm chunk: rhythm theo giây (mốc `+` trong `.atr`, AFL = AFIB, mã khác = SINUS, 255 = không
  có nhãn) và beat AAMI (1 N, 2 S, 3 V, 4 khác). Kit trả nhãn rhythm về trong RESULT và vẽ nhãn
  beat lên LCD dưới beat của AI.
- Quyết định rhythm trên kit = argmax từng giây (thô), **không** có decode episode như EC57.
  Beat = `beats.pick_beats` (ngưỡng 0.5, refractory 0.2 s), HR = 60 / median RR.
- PC gửi tín hiệu đã qua `build.read_leads` (resample 250 Hz + band-pass như dữ liệu train), int16
  µV. Kit chỉ z-score từng lead.

## 3. Giao thức (USART1 qua ST-LINK, mặc định 921600 8N1)

Frame: `A5 5A | type u8 | seq u8 | len u16 | payload | crc16 u16` (little-endian, CRC-16/CCITT-FALSE
trên type..payload). Chi tiết payload: docstring của `stm32_link.py` và `App/ecg_proto.h`.

| type | chiều | nội dung |
|---|---|---|
| 0x01 HELLO | PC→kit | reset stream, kit trả INFO |
| 0x02 CHUNK | PC→kit | index, số mẫu, 3 lead, int16 µV, nhãn rhythm/giây, nhãn beat |
| 0x81 INFO | kit→PC | model, MACC, activations, fs, window/hop, CPU Hz (0 = kit ảo) |
| 0x82 RESULT | kit→PC | 10 × {pred, nhãn, prob[5]}, noise/2 s, lead tốt nhất, HR, beats, thời gian A/stage/B |
| 0x83 ERROR | kit→PC | sai kích thước chunk, sai thứ tự (gửi lại HELLO + chunk 0), lỗi model |

PC chờ RESULT của chunk trước rồi mới gửi chunk sau (ring nhận 32 KB trong DTCM không bao giờ tràn).
Nhịp "real time": chunk k gửi lúc 9k s; "fast": gửi ngay khi có RESULT.

## 4. LCD 480×272: 3 layer

| layer | nội dung | phần cứng |
|---|---|---|
| 1 lưới | giấy ECG chuẩn 25 mm/s, 10 mm/mV (1 mm = 1.92 px; ô nhỏ 1 mm, ô lớn 5 mm), nền header/panel | LTDC layer 0, vẽ 1 lần |
| 2 tín hiệu | 3 lead của cửa sổ 10 s (envelope min/max mỗi cột), tên lead | ARGB trong suốt |
| 3 kết quả | header (#cửa sổ, thời gian, ms AI), dải màu giây non-sinus, vạch overlap 1 s, NOISE, beat AI (ô màu; vàng = không khớp nhãn ±150 ms) + nhãn beat, hàng REF / AI theo giây (gạch đỏ = sai), HR + số beat, số giây khớp | ARGB trong suốt |

Layer 3 được DMA2D blend lên layer 2 vào LTDC layer 1 (double buffer, đổi ở vertical blanking);
LTDC blend layer 1 lên lưới. Tab web vẽ cùng bố cục bằng 3 canvas chồng nhau (hằng số trong
[App/ecg_view.h](App/ecg_view.h) và `const K` trong `webservice/static/index.html` — sửa một bên
thì sửa cả bên kia). Với kit ảo, web hiện thêm frame buffer thật của firmware để so.

## 5. Build, nạp, chạy

```bash
cd firmware/stm32f746g_disco
./fetch_deps.sh                        # 1 lần: STM32CubeF7 v1.17.4 (HAL, BSP, font) -> third_party/
python3 export_model.py                # khi đổi model: Model/ (cần TF + X-CUBE-AI 10.2); ~3 phút
make                                   # build/fw/ecg_kit.{elf,bin,hex}: flash 250 KB, RAM 193 KB
make flash                             # STM32_Programmer_CLI qua ST-LINK trên board
make APP_BAUD=115200                   # tốc độ khác (chọn cùng baud trên web)
make -C host                           # kit ảo: build/host/ecgkit_sim (cần export_model.py đã chạy)

python3 webservice/app.py              # tab "STM32 kit": Port = /dev/ttyACM0 (ST-LINK) hoặc "virtual"
```

Bộ nhớ: SDRAM 8 MB — 4 MB dưới (MPU non-cacheable) cho 3 layer + 2 buffer hiển thị, 4 MB trên
(cacheable WB) cho activations của net A (0xC0400000) và net B. DTCM: ring nhận UART (DMA2
stream 2, circular) + stack. Clock 216 MHz (HSE 25 MHz), I/D-cache bật. LED1 nháy mỗi lần nhận data;
nháy liên tục = `Error_Handler` (HardFault, init lỗi).

## 6. Đã kiểm chứng / chưa

Đã chạy (2026-10-25), không có board cắm vào máy:
- Firmware build sạch (arm-none-eabi-gcc 13.2, không warning).
- **Kit ảo** = toàn bộ `App/` + `Model/ecga*.c ecgb*.c` build cho x86 với runtime X-CUBE-AI x86
  mà `stedgeai validate` dùng, nói đúng giao thức qua pty. So với pipeline Python độc lập (TFLite +
  `c_stage` + trung bình overlap + `beats.pick_beats` của repo): mitdb 201, 25 cửa sổ: 0 giây sai
  pred, prob lệch ≤ lượng tử u8, 371/371 beat trùng vị trí và lớp. Test: `tests/test_stm32_kit.py`.
- Tab web với kit ảo, mitdb 203, 30 cửa sổ: khớp nhãn 95.9 % số giây có nhãn (AFIB Se/+P 97.4/98.5,
  VT 1/5 giây) — chỉ là kiểm tra chức năng, không phải đánh giá model.

**Chưa** kiểm trên board thật: clock/MPU/DMA2D/LTDC, UART DMA từ DTCM, thời gian suy luận trên
Cortex-M7 (activations ở SDRAM; chưa đo — RESULT gửi về thời gian A/stage/B để đo ngay lần chạy đầu).
Nếu kit không trả INFO: kiểm port (`/dev/ttyACM*`), baud, và LED1.
