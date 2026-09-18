# ecgr — phát hiện & phân loại nhịp ECG 3 chuyển đạo bằng ResUMamba

Một họ model (ResUMamba thích ứng cho contract seq2seq) ở **bốn** kích thước, một đường đi
duy nhất qua dữ liệu:

```
record portal ──► npy ──► tfrecord ──► ssl ──► cpc ──► train ──► step eval ──► beat eval (EC57/bxb)
                                    └── tự giám sát, không dùng nhãn ──┘
```

**Contract:** vào `(2500, 3)` = 10 s @ 250 Hz trên **3 chuyển đạo**, ra `(500, 4)` softmax —
phát hiện **và** phân loại ở độ phân giải 20 ms, không cần cho trước đỉnh R. 4 lớp AAMI:
`None / N / V / S`.

Hai chặng `ssl` và `cpc` là thứ phân biệt họ model này: cả hai đều huấn luyện **không nhãn**.
`ssl` tiền huấn luyện **backbone** (nơi chứa gần như toàn bộ tham số) bằng tái tạo tín hiệu bị
che; `cpc` huấn luyện bộ mã hóa ngữ cảnh bằng InfoNCE rồi **đóng băng** nó, nên hàm mất mát của
bài toán nhịp không bao giờ nắn lại được nó.

---

## 0. Cài đặt

```bash
cd ecg_resumamba
pip install -r requirements.txt          # hoặc dùng env conda `beat` có sẵn trên máy này
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
which bxb sumstats                       # trống = chưa cài WFDB apps, bước ec57 sẽ báo lỗi
./run_pipeline.sh test                   # 71 test, ~2 phút
```

> **`libdevice not found at ./libdevice.10.bc`.** Bánh xe `tensorflow[and-cuda]` có cuDNN và
> cuBLAS nhưng **không** có `nvvm/libdevice/libdevice.10.bc` của CUDA, mà TF vẫn đẩy một số
> gradient hàm kích hoạt hợp nhất qua XLA bất kể `jit_compile` hay
> `TF_XLA_FLAGS=--tf_xla_auto_jit=0` — nên **mọi** chặng huấn luyện chết ngay bước đầu tiên.
> [`ecgr/xla.py`](ecgr/xla.py) tự tìm file đó (CUDA toolkit của hệ thống, bánh xe
> `nvidia-cuda-nvcc`, hay bản Triton mang kèm) và trỏ `XLA_FLAGS` vào. Không tìm được thì
> `pip install nvidia-cuda-nvcc-cu12`.

## 1. Một cổng vào duy nhất

```bash
python -m ecgr config                            # cấu hình đang hiệu lực
python -m ecgr models                            # họ model + số tham số
python -m ecgr data     --step all               # record → npy → tfrecord
python -m ecgr ssl      --model resumamba_30k    # backbone, tự giám sát
python -m ecgr cpc      --model resumamba_30k    # bộ mã hóa ngữ cảnh, tự giám sát
python -m ecgr train    --model resumamba_30k
python -m ecgr stepeval --model resumamba_30k
python -m ecgr ec57     --model resumamba_30k
python -m ecgr all      --model resumamba_30k    # 5 bước trên, liền mạch
python -m ecgr compare  resumamba_2m resumamba_1m resumamba_100k resumamba_30k
```

Hoặc chạy cả 4 kích thước tự động: [`run_pipeline.sh`](run_pipeline.sh).

## 2. Cấu hình — [`ecgr/config.py`](ecgr/config.py)

Một run được mô tả trọn vẹn bởi file này cộng với các biến môi trường nó đọc. Đặt biến môi
trường trong launcher, **đừng sửa file khi job đang chạy**:

| biến | ý nghĩa | mặc định |
|---|---|---|
| `ECGR_DATA_DIR` | dataset portal (record + `dataset_info_full.csv`) | `/mnt/md0/Dong_data/portal_data/` |
| `ECGR_PHYSIONET_DIR` | mitdb/nstdb/escdb/ahadb/afdb | `/mnt/md0/Dong_data/physionet/` |
| `ECGR_WORK_DIR` | nơi ghi npy/tfrecord/checkpoint/report | `<DATA_DIR>/train` |
| `ECGR_RUN_TAG` | tên thư mục run | hôm nay `yymmdd_ecgr` |
| `ECGR_IN_CHANNELS` | **số chuyển đạo** đưa vào model | `3` |
| `ECGR_EC57_LEAD_MODE` | `auto` / `duplicate` / `native` (mục 6b) | `auto` |
| `ECGR_SSL_RUN` | dùng lại backbone tự giám sát của run khác | — |
| `ECGR_CPC_RUN` | dùng lại bộ mã hóa CPC của run khác | — |
| `ECGR_WORKERS` | số process khi build npy | `cpu/2` |

> `RUN_TAG` mặc định theo ngày và **đổi lúc nửa đêm** — một training bắt đầu hôm qua sẽ ghi
> vào thư mục hôm qua còn eval sáng nay tìm thư mục hôm nay và không thấy gì. Ghim nó.

## 3. Ba chuyển đạo — trục kênh mang gì

Trục kênh mang **chuyển đạo ECG**, không phải băng tần lọc (bản trước dùng kênh 1 = băng QRS,
kênh 2 = băng P/T). Mọi record portal vốn đã là **3 chuyển đạo @ 250 Hz** (CH0/CH1/CH2), nên
model thấy cả montage thay vì một chuyển đạo mà người review tình cờ chọn: sóng P vô hình trên
một chuyển đạo thường vẫn thấy được trên chuyển đạo khác, và đó chính là bằng chứng phân biệt
một nhịp ngoại vị nhĩ **không đến sớm** (mitdb 232, tỉ số R–R ≈ 0.99) với nhịp xoang.

Toàn bộ hợp đồng nằm trong [`signal_ops.build_leads`](ecgr/signal_ops.py), và chỉ có hai quy
tắc:

1. **Chuyển đạo được gán nhãn luôn là kênh 0.** Các chuyển đạo còn lại giữ thứ tự vòng sau nó.
   Nhãn, việc tìm đỉnh R trong `decode_beats`, phép thử phẳng và bộ mô tả nhịp — tất cả đọc
   kênh 0, và nếu không có quy tắc này thì tất cả sẽ đọc một chuyển đạo không ai gán nhãn.
2. **Record thiếu chuyển đạo được bù bằng cách lặp lại kênh 0.** Cả 5 database EC57 đều có 2
   chuyển đạo và được gán nhãn trên chuyển đạo đầu, nên chấm điểm chúng nghĩa là đưa cho model
   **cùng một chuyển đạo ba lần**.

Vì việc "một chuyển đạo lặp ba lần" là *đầu vào thật* ở chặng EC57, huấn luyện phải tái hiện
nó: `config.AUGMENT_LEAD_DUPLICATE_PROB = 0.25` gộp 1/4 cửa sổ train về đúng dạng đó, và
`AUGMENT_LEAD_DROP_PROB = 0.15` làm phẳng một chuyển đạo phụ (điện cực rơi). Không có phần này
thì benchmark EC57 là một **dịch chuyển phân phối** chứ không còn là đánh giá.

> Thứ tự trong [`pipeline.augment`](ecgr/data/pipeline.py) có ý nghĩa: hệ số biên độ được nhân
> **trước** khi nhân bản chuyển đạo. Nhân sau thì mỗi chuyển đạo của một mẫu đã nhân bản lại bị
> nhân một hệ số khác nhau, ba chuyển đạo ra **tỉ lệ** chứ không **bằng nhau**, và trường hợp
> mà phép tăng cường này tồn tại để tạo ra thì chưa từng đến được model.
>
> Đo trên dữ liệu thật: **10,8% cửa sổ vốn đã có một chuyển đạo phụ phẳng**. `is_flat` chỉ xét
> kênh 0 là có chủ ý — nhãn đến từ chuyển đạo đó, nên cửa sổ chỉ vô dụng khi *chuyển đạo đó*
> chết, dù hai chuyển đạo kia có sống động thế nào.

## 4. Tạo dữ liệu — [`ecgr/data/`](ecgr/data/)

Cửa sổ 10 s trượt 1 s trong khoảng đã review, nhãn 500 bước với block bất đối xứng
`LABEL_STEPS_BEFORE=8` / `LABEL_STEPS_AFTER=2`, chống rò rỉ dữ liệu kiểm ba lần
(`held_out_studies` → `verify_split_integrity` → `audit_written_data`), split theo hàm băm MD5
của `study_id` chứ không theo seed ngẫu nhiên.

```bash
python -m ecgr data --step npy --db dataset-1 --limit 300   # chạy thử nhanh
python -m ecgr data --step all --workers 32                 # đầy đủ
python -m ecgr data --step all --db dataset-2               # build lại đúng một dataset
```

Build lại một dataset là việc bình thường (chính bản sửa lỗi 3 dưới đây cần nó), nên
`dataset_manifest.json` được **hợp nhất** chứ không ghi đè: các dataset mà lần build này không
chạm đến vẫn còn trong manifest, miễn tfrecord của chúng còn trên đĩa.

Build npy chạy **đa tiến trình** (`imap` có thứ tự, nên rebuild cùng input cho ra batch giống
hệt): ~5000 record/s với 32 worker, tức cả nửa triệu record trong vài phút thay vì gần một giờ.

### 4a. Ba lỗi tạo dữ liệu đã sửa — chúng ảnh hưởng hơn 1/5 corpus

Đây là phần đáng đọc nhất của bản cập nhật này.

**Lỗi 1 — khoảng đã review bị cắt xuống giây nguyên.** Bản trước chuyển đổi bằng
`start // fs * fs`. Cả hai tần số đều là 250 Hz, nên phép chuyển đổi là phép đồng nhất *ngoại
trừ* việc cắt đó — thứ đã đẩy ~95k cửa sổ ra ngoài khoảng mà người review chứng nhận, lệch tới
249 mẫu, và làm 19,763 record ngắn đi đủ để **không sinh ra cửa sổ nào cả**.

**Lỗi 2 — 114,920 record (23,1%) có khoảng review dài đúng 2499 mẫu**, thiếu một mẫu so với
10 s. Quy tắc "cửa sổ phải nằm hẳn trong khoảng" loại bỏ từng cái một trong số đó.
`SPAN_SLACK_SAMPLES` (0,1 s — ngắn hơn nửa phức bộ QRS và ngắn hơn chính block nhãn, nên không
thể kéo vào một nhịp chưa gán nhãn) nhận chúng lại; khoảng ngắn thật thì vẫn bị từ chối và
được đếm.

Điều làm việc này tốn kém chứ không chỉ là không gọn: các record bị mất là **~14% toàn bộ
event strip SVE và VE**, tức tập trung đúng vào hai lớp model yếu nhất.

| event_type | bị mất | tổng |
|---|---|---|
| `SINUS, SINGLE VE` | 1,401 | 8,379 |
| `SINUS, SINGLE SVE` | 781 | 5,442 |
| `SINGLE VE` | 272 | 1,708 |
| `SINUS, VE BIGEMINY` | 155 | 1,076 |
| `SINGLE SVE` | 129 | 858 |
| `VE BIGEMINY` | 118 | 563 |

**Lỗi 3 — dataset-2 lưu mỗi đoạn hai lần.** Lỗi này nằm trong *dữ liệu*, không phải trong số
học: mọi thư mục event của `dataset-2` chứa **hai file `.dat` giống nhau từng byte**, dưới hai
tiền tố event-id khác nhau (200/200 event được lấy mẫu). Vì `process_record` xử lý mọi `.dat`
khớp, mỗi cửa sổ của dataset-2 được phát ra **hai lần** — và dataset-2 là nguồn segment lớn
nhất, nên khoảng **một phần tư** tập train là bản sao chính xác của một phần tư khác.

Đây **không** phải rò rỉ train/eval (split theo study), nhưng nó âm thầm nhân đôi trọng số của
một dataset trong hàm mất mát, làm lệch cân bằng lớp, và nhân đôi lượng RAM mà `cache()` cần.
`_record_files` giờ gộp các file **giống nhau theo nội dung** (khóa `(size, md5)`) — một thư mục
có hai bản ghi *thật sự khác nhau* vẫn giữ cả hai, tức hành vi cũ được bảo toàn đúng ở chỗ nó
đúng. Chi phí: đọc thêm một file 90 kB mỗi event, không nhìn thấy được trong thời gian build.

Kiểm tra lại bằng test, không phải bằng niềm tin:
[`tests/test_signal_and_labels.py`](tests/test_signal_and_labels.py) ghim cả ba mẫu
`(start, stop)` phổ biến nhất và cả phép gộp bản sao (kể cả việc **không** gộp hai bản ghi khác
nhau).

> **Đã biết, chưa sửa — trùng chéo dataset.** 8,310 cặp (study, event) được **hai** dataset liệt
> kê: `dataset 2_3_4 - AFib - v2` và `dataset-3-filter-vt-svt-avb2-avb3` là bản re-curation của
> event trong dataset-2/3/4. Đo trên 300 cặp: `.dat` và `.atr` **giống nhau từng byte** (300/300),
> chỉ 1,555/8,310 hàng CSV trùng (channel, start, stop). Nghĩa là cùng một đoạn được cắt hai lần
> với cùng nhãn nhưng có thể khác cửa sổ 10 s hoặc khác chuyển đạo chính — nhân đôi nhẹ trọng số
> của 1,7% event, **không** xung đột nhãn, và cùng nằm một split (split theo study) nên không rò
> rỉ. Để nguyên vì chi phí một lần build + train lại không đổi được gì đo được. Phần bxb trên
> split (mục 7b) thì **có** gộp: một bản ghi chỉ được chấm một lần, theo review của dataset đứng
> trước trong `TRAIN_DATASETS`.

### 4b. Định dạng tfrecord: bytes thay vì typed list

```
signal  bytes  float32[2500, 3]   row-major
labels  bytes  uint8[500]         (4 lớp vừa một byte)
```

`float_list` lưu mỗi mẫu thành một field protobuf riêng. Với 3 chuyển đạo là 7.500 field mỗi
example, và cả lúc ghi lẫn `parse_example` đều trả giá theo từng field; một feature `bytes` là
một `memcpy` mỗi chiều. Nó cũng nhỏ hơn trên đĩa, và nhãn co lại 8 lần khi đi từ int64 sang
uint8.

Kèm theo là [`dataset_manifest.json`](ecgr/data/build_tfrecord.py) ghi shape/dtype/histogram
lớp của những gì **thực sự đã ghi**. `pipeline.check_manifest` đối chiếu nó với config đang
chạy, nên lệch cấu hình là một thông báo rõ ràng lúc khởi động thay vì
`Input to reshape is a tensor with 1250000 values but the requested shape has 3750000` từ bên
trong bước huấn luyện đầu tiên.

## 5. Model — [`ecgr/models/`](ecgr/models/)

```
python -m ecgr models
```

| `--model` | tham số | ngân sách | backbone | ctx | head |
|---|---|---|---|---|---|
| `resumamba_2m` | 1,977,660 | 2,000,000 | 1,458,296 | 179,552 | 339,812 |
| `resumamba_1m` | 934,476 | 1,000,000 | 639,200 | 101,832 | 193,444 |
| `resumamba_100k` | 93,517 | 100,000 | 50,588 | 6,485 | 36,444 |
| `resumamba_30k` | 29,741 | 30,000 | 15,044 | 2,605 | 12,092 |

Ngân sách được [`tests/test_models.py`](tests/test_models.py) kiểm: mỗi kích thước phải **dưới**
ngân sách của nó và không **quá thấp** so với tên nó mang.

Chi phí đo được trên một RTX 3090, batch 128, đầu vào `(2500, 3)`:

| `--model` | train s/step | ssl s/step | VRAM đỉnh |
|---|---|---|---|
| `resumamba_2m` | 0.324 | 0.253 | 6,595 MiB |
| `resumamba_1m` | 0.208 | 0.153 | 4,164 MiB |
| `resumamba_100k` | 0.117 | 0.075 | 1,708 MiB |
| `resumamba_30k` | 0.102 | 0.058 | 986 MiB |

Cả bốn đều vừa một card 24 GB với chỗ dư lớn, nên hai queue song song trên hai card là cách
dùng máy này hợp lý nhất — xem phần đầu [`run_pipeline.sh`](run_pipeline.sh).

Gốc: **Heo và cộng sự**, *Patient-conditioned ECG beat classification via self-supervised
embeddings*, Expert Systems With Applications 331:133149, 2026 — danh mục tài liệu tham khảo
đầy đủ ở [`docs/references/README.md`](docs/references/README.md).

Bài toán của họ khác hẳn: phân loại **từng nhịp một**, cửa sổ 720 mẫu căn giữa tại đỉnh R đã
biết. Ở đây không có đỉnh R — model phải vừa phát hiện vừa phân loại trên đoạn 10 s chạy tự do.
Bốn thành phần vì vậy được chuyển thể chứ không bê nguyên (lý luận đầy đủ trong docstring của
[`resumamba.py`](ecgr/models/resumamba.py)):

1. **Mamba S6 → SSM chéo, hai chiều.** Quét chọn lọc là một hồi quy tuần tự; trong TensorFlow
   nó là `tf.scan` qua 500 bước — chậm khi huấn luyện và không export được. SSM **chéo** có
   nghiệm đóng `k_l = c·b·a^l`, nên toàn bộ hồi quy bằng đúng một tích chập FIR: cùng độ phức
   tạp tuyến tính, cùng trí nhớ dài, nhưng song song theo thời gian và hạ xuống thành conv
   thường khi export. Và nó chạy **hai chiều**: Mamba của bài báo nhân quả vì nó stream, còn
   model này thấy trọn đoạn 10 s — mà vùng T–P phân biệt S với N nằm **trước** nhịp cần gán nhãn.
2. **Embedding bệnh nhân → embedding ngữ cảnh của đoạn.** Bài báo cần 60 s ECG hiệu chuẩn không
   nhãn cho mỗi bệnh nhân; tfrecord ở đây không có định danh bệnh nhân lẫn đoạn hiệu chuẩn. Cùng
   bộ máy CPC được chạy trên chính đoạn 10 s: cắt thành 9 cửa sổ 2 s chồng 50%. Nó vẫn làm đúng
   việc AdaIN cần — mang biên độ, đường nền và chế độ nhịp.
3. **Thống kê R–R → tự tương quan envelope.** Thống kê R–R đòi hỏi vị trí nhịp, tức chính đầu
   ra của model. Tự tương quan của tín hiệu chỉnh lưu trong dải trễ 30–220 bpm mang cùng thông
   tin nhịp. 88 độ trễ trên ~400 bước ≈ 35k phép nhân cộng, bản 30K cũng gánh được.
4. **Cross-attention đảo chiều.** Bài báo lấy vector lâm sàng làm *query* vì nó cần một đầu ra;
   ở đây đặc trưng per-step là query còn token nhịp là key/value, vì cần 500 đầu ra.

Giữ nguyên: khối ResU, AdaIN, Poly-2 loss, bố cục hai nhánh.

### 5a. `RhythmDescriptor` đọc **một** chuyển đạo, không phải max trên cả ba

Trên đoạn 3 chuyển đạo thì `max |x|` qua các chuyển đạo giàu thông tin hơn. Nhưng ở chặng EC57,
layer này nhận cùng một chuyển đạo ba lần, và ở đó phép max **suy biến về đúng chuyển đạo đó**:
bộ mô tả sẽ được tính từ hai đại lượng khác nhau lúc đánh giá và lúc huấn luyện. Đọc một chuyển
đạo cố định (kênh 0) thì hai trường hợp **giống hệt nhau**. Bất biến này có test riêng.

### 5b. Hai sub-model lồng nhau

`backbone` (stem + hai nhánh + phép hợp nhất) và `context_encoder` được dựng thành `keras.Model`
riêng chứ không inline, vì **cả hai đều được tiền huấn luyện không nhãn rồi nạp lại theo tên**.
Kiến trúc không đổi — cùng những layer đó, cùng thứ tự đó, chỉ lồng thêm một mức.

### Ngân sách tham số đi đâu

Nguyên tắc cắt giảm: **không đụng vào chiều sâu của nhánh state-space**. Trường tiếp nhận mà nó
mua chính là thứ phân biệt S với N, trong khi một nhân SSM chỉ tốn 4 số thực cho mỗi cặp
(kênh, trạng thái) — ở bản 30K là 576 trọng số cho cả nhánh. Ngân sách bị cắt ở chiều rộng và ở
các tích chập dày của nhánh ResU, vốn được thay bằng tích chập tách chiều sâu dưới mức 100k.

> **`layers.PKG` phải mãi là `"resumamba_seq2seq"`.** Keras ghi `"<package>>ClassName"` vào
> trong mỗi file `.keras` và tra cứu bằng đúng chuỗi đó. Tên package là một **định dạng lưu
> trữ**, không phải namespace.

## 6. Tự giám sát rồi train — [`ecgr/training/`](ecgr/training/)

### 6a. Chặng 1: backbone — [`ssl.py`](ecgr/training/ssl.py)

```bash
python -m ecgr ssl --model resumamba_30k
```

Đây là chặng tiền huấn luyện **nơi tham số thực sự nằm**. Chặng CPC huấn luyện bộ mã hóa ngữ
cảnh, chỉ 5–9% model; toàn bộ phần còn lại trước đây khởi tạo ngẫu nhiên và chỉ có hàm mất mát
có nhãn để học.

Mục tiêu, trên tín hiệu không nhãn từ chính tfrecord đó:

```
làm hỏng đoạn 3 chuyển đạo → backbone → một decoder nhỏ → tái tạo phần bị che
```

Hai kiểu làm hỏng, và kiểu thứ hai là lý do làm việc này trên đầu vào 3 chuyển đạo:

1. **Che theo khoảng.** 35% bước đầu ra, theo các khoảng liền nhau 12 bước (~240 ms, cỡ một
   nhịp), bị làm 0 trên mọi chuyển đạo. Điền lại một nhịp bị che từ các nhịp lân cận chính là
   ngữ cảnh nhiều nhịp mà nhánh state-space tồn tại để cung cấp — nên bài toán giả và bài toán
   thật muốn cùng một trường tiếp nhận.
2. **Che theo chuyển đạo.** Với xác suất 0,5, một chuyển đạo bị làm 0 hoàn toàn và phải được
   tái tạo từ hai chuyển đạo kia. Đây dạy đúng tính dư thừa liên chuyển đạo làm cho 3 chuyển đạo
   có giá trị — và nó chính là phép toán chặng EC57 thực hiện, nên model đến benchmark đã thông
   thạo việc đọc một montage suy biến.

Đích là đầu vào ở độ phân giải **bước**: 5 mẫu thô sau mỗi bước đầu ra, trên mọi chuyển đạo, tức
`5·C` số mỗi bước. Dự đoán cả patch thay vì một giá trị trung bình giữ được hình thái mịn trong
mục tiêu — một bản tái tạo chỉ khớp trung bình 20 ms có thể làm phẳng mọi QRS mà vẫn điểm cao.

Loss tính **chỉ trên các bước bị che**. Lấy trung bình trên mọi bước sẽ cho model ghi điểm bằng
cách sao lại 65% đầu vào nó vẫn thấy.

`nmse` là sai số tái tạo chia cho phương sai của phần bị che: **1.0 = không hơn gì việc đoán
giá trị trung bình**, nên đó là con số đáng theo dõi chứ không phải MSE thô.

### 6b. Chặng 2: bộ mã hóa ngữ cảnh — [`cpc.py`](ecgr/training/cpc.py)

```bash
python -m ecgr cpc --model resumamba_30k
```

InfoNCE trên chính tập train, **nhãn bị bỏ đi hoàn toàn**: từ ngữ cảnh quá khứ `c_i`, dự đoán
latent tương lai `v_{i+j}` với `j ∈ {1, 2}`, cạnh tranh với mọi latent khác trong batch — cả cửa
sổ khác lẫn **đoạn khác**. Chính phần "đoạn khác" ép biểu diễn phải mang đặc trưng riêng của
từng đoạn.

Chặng này bật `lead_jitter`: bộ mã hóa phải mô tả một đoạn có montage suy biến dễ dàng như một
đoạn 3 chuyển đạo sạch. Chặng `ssl` thì **không** bật, vì nó tự che chuyển đạo — bật cả hai là
làm hỏng cùng một đầu vào hai lần.

Hai chặng tách nhau vì hai sub-network muốn hai thứ trái ngược từ một biểu diễn: backbone phải
**giữ** hình thái nhịp, bộ mã hóa ngữ cảnh phải **tóm gọn** biến thiên gây nhiễu mà AdaIN sau đó
chia đi.

Cả hai chỉ phụ thuộc kiến trúc nên dùng lại được: `ECGR_SSL_RUN` / `ECGR_CPC_RUN` trỏ sang run
cũ, hoặc `--ssl-weights` / `--ctx-weights` trỏ thẳng vào file.

### 6c. Huấn luyện — [`train.py`](ecgr/training/train.py)

```bash
python -m ecgr train --model resumamba_30k --epochs 30 --lr 1.4e-3
```

Hai sub-model tự giám sát được đối xử **khác nhau, có chủ ý**:

* **bộ mã hóa ngữ cảnh bị đóng băng**, như bài báo đóng băng bộ mã hóa bệnh nhân, nên hàm mất
  mát nhịp không bao giờ nắn một mô tả về biến thiên gây nhiễu thành một bộ phát hiện nhịp;
* **backbone được fine-tune**, vì nó *chính là* bộ trích đặc trưng mà classifier đọc.
  `--freeze-backbone-epochs N` giữ nó bất động N epoch đầu để cái head ngẫu nhiên không rửa
  trôi phần tiền huấn luyện trước khi nó kịp tốt lên.

**Loss mặc định là `poly2`** — đúng công thức bài báo, `CE + 0.3(1−Pt) − 0.5(1−Pt)²`. Nó huấn
luyện tốt, nhưng **giá trị của nó là thước đo xếp hạng tồi**: đạo hàm của số hạng đa thức theo
`u = 1−Pt` là `0.3 − u`, nên khi `Pt < 0.7` một dự đoán *tệ hơn* lại làm *giảm* số hạng đó. Đo
được: `val_loss` chạm đáy ngay **epoch 1** rồi tăng đơn điệu, trong khi weighted F1 vẫn leo tới
epoch 7. Tính chất này có test riêng, để nếu ai đổi `POLY2_EPS` thì test sẽ nhắc kiểm lại
`MONITOR`.

**Nên monitor mặc định là `val_weighted_f1`, không phải `val_loss`.**

### 6d. `val_weighted_f1` giờ là một Keras *metric*, không phải callback

Trước đây `WeightedF1Checkpoint` tự chạy **một lượt đầy đủ qua tập eval** để tính F1, ngoài lượt
validation mà Keras đã chạy. Giờ [`step_metrics.StepConfusion`](ecgr/evaluation/step_metrics.py)
là một metric được compile, tích lũy ma trận nhầm lẫn 4×4 ngay trong lượt validation đó.

Hai thứ từng quan trọng nay không còn quan trọng nữa:

* **tập eval được đi qua một lần mỗi epoch thay vì hai.**
* **thứ tự callback.** `val_weighted_f1` có mặt trong `logs` trước khi bất kỳ callback nào chạy,
  nên `EarlyStopping` / `ReduceLROnPlateau` / `ModelCheckpoint` monitor được nó bất kể chúng
  được liệt kê ở đâu. Việc công bố một khóa monitor **từ** một callback từng làm thứ tự danh
  sách trở thành thứ chịu lực, và một khóa monitor thiếu ở epoch đầu làm các callback đó bỏ qua
  **âm thầm** chứ không báo lỗi.

### 6e. Chỉ lưu checkpoint từ epoch 10

Cần **ba mảnh khớp nhau**, chặn ghi file thôi là chưa đủ:

1. `WeightedF1Checkpoint` vẫn **đo** F1 mỗi epoch (bắt buộc, vì nó là chỉ số dừng) nhưng chỉ
   **ghi** từ `save_start_epoch`.
2. `DelayedModelCheckpoint` **bỏ qua hẳn** các epoch khởi động. Nếu chỉ chặn ghi file mà vẫn để
   callback chạy thì epoch đầu tiên sẽ **đặt mốc `best`** mà mọi epoch sau phải vượt.
3. `EarlyStopping(start_from_epoch=…)` — không có nó, một run patience ngắn có thể dừng ở epoch
   7 và kết thúc **không có checkpoint nào**.

Vì sao epoch 10: đo trên bản 1 chuyển đạo trước đây, `resumamba_1m` dao động từ 0.8267 đến
0.8519 trong 9 epoch đầu, rồi ổn định trong dải hẹp 0.8467–0.8523 suốt 14 epoch sau.

Output:

```
<RUN_DIR>/checkpoints/<model>/best_model.keras            best theo --monitor
<RUN_DIR>/checkpoints/<model>/BEST_F1/*.keras             best weighted F1, F1 nằm trong tên
<RUN_DIR>/checkpoints/<model>/ssl_backbone.weights.h5     backbone tự giám sát
<RUN_DIR>/checkpoints/<model>/cpc_context.weights.h5      bộ mã hóa ngữ cảnh tự giám sát
<RUN_DIR>/eval/<model>/confusion_log.txt                  mỗi epoch xuất một khối
<RUN_DIR>/logs/<model>/                                   tensorboard --logdir đây
```

## 7. Eval

**7a. Mức bước** — chỉ số **rẻ** để chọn checkpoint, **không phải** chỉ số đánh giá cuối:

```bash
python -m ecgr stepeval --model resumamba_30k
```

**7b. Mức nhịp, chuẩn EC57 (bxb)** — cái thực sự quyết định:

```bash
python -m ecgr ec57 --model resumamba_30k
python -m ecgr ec57 --model resumamba_30k --bxb-only            # chấm lại, không predict lại
python -m ecgr ec57 --model resumamba_30k --lead-mode duplicate # một chuyển đạo, ở mọi nơi
```

`--lead-mode` quyết định trục chuyển đạo được lấp thế nào:

| chế độ | physionet (2 chuyển đạo) | portal beat-eval (3 chuyển đạo) |
|---|---|---|
| `auto` (mặc định) | `duplicate` — chuyển đạo 0 lặp ba lần | `native` — montage thật, giống lúc train |
| `duplicate` | chuyển đạo 0 lặp ba lần | chuyển đạo được review, lặp ba lần |
| `native` | 2 chuyển đạo thật + 1 bản lặp | montage thật |

Chấm **bốn nhóm nguồn**, cùng một cách:

| nguồn | là gì | đọc thế nào |
|---|---|---|
| `mitdb` `nstdb` `escdb` `ahadb` `afdb` | 5 database EC57 | benchmark độc lập, chưa từng train |
| `dataset-v4-beat` | 5,227 record portal beat-eval, giữ riêng | holdout đã được curate, chưa từng thấy dưới bất kỳ dạng nào |
| `portal-eval` | mẫu 5,000 event của **split eval** | cùng các study mà F1 mức bước dùng để chọn checkpoint — ở mức nhịp |
| `portal-train` | mẫu 5,000 event của **split train** | dữ liệu model **đã thấy** — chỉ để đo overfitting (khoảng cách tới `portal-eval`), **không bao giờ** là con số hiệu năng |

Hai split được lấy mẫu **xác định theo thứ tự md5 của tên record**: cùng một tập con cho mọi
model, mọi lần chạy, mọi máy, và thêm dataset vào CSV cũng không xáo lại. 5,000 là cỡ của chính
bộ beat-eval (5,227) và tốn ~3 phút mỗi split mỗi model; toàn bộ split là 365,787 / 91,342
event (`--split-records 0` = tất cả, `--splits` không kèm gì = bỏ qua).

Ba dataset portal đánh chỉ số cửa sổ review theo **ba cách khác nhau** trong `.hea`
(`startSample` / `startMarkSample` / `eventStartSample`), và nửa số `.hea` của dataset-2 trỏ tới
một `.dat` **khác tên** vì bản sao — nên với hai split, `ec57` **tự viết lại `.hea`** trong thư
mục chấm điểm: đổi tên record (tên gốc theo giờ ghi, trùng nhau giữa các study), giữ nguyên
từng trường tín hiệu, và ghi cửa sổ review **từ CSV** vào ba comment mà script mark-window đọc.
Test kiểm rằng `.hea` viết lại được `wfdb` parse với cùng gain/baseline/fs như bản gốc.

Dự đoán `.ain` lưu ở `_ann/<nguồn>/` và **được giữ lại**, việc chấm diễn ra trong symlink farm
dùng-một-lần ở `_work/<nguồn>/` — không bao giờ đụng vào thư mục dữ liệu gốc.

> Suy luận trên portal đi qua một `tf.function` cache trên model chứ không qua `model.predict`:
> với một record 60 s (7 cửa sổ), `predict` tốn **113 ms** còn lời gọi compiled tốn **8 ms** — đo
> trên bản 30k. Nhân với 5,227 + 5,000 + 5,000 record là ~30 phút overhead thuần mỗi model.

### 7c. Giải mã nhịp: mỗi vùng chồng lấn được chia đôi

Các cửa sổ suy luận chồng lấn nhau, nên không có quy tắc thì một nhịp trong vùng chồng bị giải
mã hai lần. Bản trước so với **vị trí đã nhận trước đó**; cách đó giả định các phát hiện đến
theo thứ tự tăng dần giữa các cửa sổ (không đúng, ngay trong một vùng chồng) và nó giữ lại bản
sao từ cửa sổ mà nhịp nằm **gần biên nhất**, tức bản sao có ít ngữ cảnh nhất về một phía — mà
model này **hai chiều**, nên cả hai phía đều quan trọng.

Giờ [`labels.core_bounds`](ecgr/labels.py) chia mỗi vùng chồng tại **trung điểm**: đoạn `i` sở
hữu tới trung điểm vùng chồng với `i+1`, đoạn `i+1` sở hữu từ đó trở đi. Các miếng **lát kín
record chính xác** — không khe hở, không đếm hai lần — với bất kỳ khoảng cách `starts` nào.

Đồng thời [`signal_ops.segment_starts`](ecgr/signal_ops.py) không còn sinh ra cửa sổ cuối cùng
có thể **hơn 90% là đệm lặp-mẫu-cuối**: vì mỗi cửa sổ được z-score độc lập, phần đệm đó quay lại
thành một vệt phẳng biên độ đầy đủ mà model nhìn như tín hiệu. Cửa sổ cuối giờ được kéo về để
kết thúc đúng tại hết record.

Cả hai tính chất — lát kín chính xác, và round-trip `nhãn → dự đoán oracle → giải mã` trả lại
đúng số nhịp ban đầu với sai số vị trí ≤ 2 mẫu — đều có test.

## 8. So sánh nhiều model

```bash
python -m ecgr compare resumamba_2m resumamba_1m resumamba_100k resumamba_30k
```

Cột **FP/1k** — số báo nhầm trên 1000 nhịp tham chiếu của lớp đó — tồn tại vì **+P không so được
giữa các database**: mật độ lớp S chạy từ 0,14% (escdb) đến 10,8% (bộ portal), và ở mật độ thấp
thì +P gần như chỉ phản ánh tỉ lệ báo nhầm.

### Cảnh báo: đừng chọn model bằng mitdb S_Se

Đo trên họ 1 chuyển đạo trước đây: một checkpoint có mitdb S_Se **30.49**, thấp hơn nhiều so với
47.91 của checkpoint trước nó — nhưng trên portal nó lại *tốt hơn* (S F1 87.42 so với 87.08) và
+P cao hơn hẳn ở mọi database nhiễu (nstdb S_+P 35.94 → 51.40).

Lý do: mitdb S bị chi phối bởi bản ghi **232**, chiếm ~43% toàn bộ gánh nặng SVEB của database
đó, và các nhịp ngoại vị nhĩ của nó **không đến sớm** (tỉ số R–R ≈ 0.99). Chỉ cần ranh giới
quyết định N/S dịch nhẹ theo hướng thận trọng là chỉ số này sụp, trong khi trên dữ liệu thật sự
thận trọng ấy lại có lợi. **F1 mức bước không quyết định được chất lượng lớp S ở mức nhịp.**

### 8b. Kết quả đo được — run `260917_3lead`

Cả bốn kích thước, cùng data, cùng cấu hình, cùng mẫu 5,000 record cho hai split. File gốc:
`<RUN_DIR>/ec57/<model>/ec57_summary.csv`; bốn checkpoint tốt nhất và bộ weights tự giám sát
tương ứng nằm trong [`checkpoints/`](checkpoints/) kèm [`manifest.json`](checkpoints/manifest.json).

**Train và tự giám sát**

| model | tham số | epoch tốt nhất / đã chạy | `val_wF1` | SSL `val_nmse` tốt nhất | CPC `val_top1` (ngẫu nhiên 0.09%) |
|---|---|---|---|---|---|
| `resumamba_2m` | 1,977,660 | 13 / 21 | **0.9262** | **0.4875** | 4.15% |
| `resumamba_1m` | 934,476 | 10 / 18 | 0.9249 | 0.5131 | 3.91% |
| `resumamba_100k` | 93,517 | 10 / 18 | 0.9145 | 0.6149 † | 3.33% † |
| `resumamba_30k` | 29,741 | 22 / 30 | 0.9102 | 0.6895 † | 2.76% † |

† weights lưu ở epoch **cuối** (30k: 0.7067; 100k: 0.6157), không phải epoch tốt nhất — xem mục 6a.
`val_nmse` giảm đều theo kích thước (0.69 → 0.61 → 0.51 → 0.49): backbone càng lớn tái tạo phần
bị che càng tốt, đúng như một mục tiêu tiền huấn luyện phải hành xử.

**Lớp S, bxb** — Se / +P / F1, và FP trên 1000 nhịp S tham chiếu:

| nguồn | S | `2m` | `1m` | `100k` | `30k` |
|---|---|---|---|---|---|
| `portal-train` (đã thấy) | 6,450 | 88.99 / 87.79 / **88.39** · 124 | 87.33 / 85.66 / 86.49 · 146 | 83.63 / 85.71 / 84.66 · 139 | 87.13 / 81.86 / 84.41 · 193 |
| `portal-eval` | 6,071 | 89.72 / 85.27 / **87.44** · 155 | 89.66 / 82.51 / 85.94 · 190 | 85.21 / 83.35 / 84.27 · 170 | 88.40 / 78.40 / 83.10 · 244 |
| `dataset-v4-beat` | 7,888 | 85.86 / 90.92 / **88.32** · 86 | 85.17 / 89.07 / 87.08 · 105 | 80.35 / 89.20 / 84.54 · 97 | 84.23 / 85.86 / 85.04 · 139 |
| mitdb | 2,745 | 45.79 / 61.17 / 52.37 · 291 | 49.44 / 36.63 / 42.08 · 855 | 32.02 / 69.10 / 43.76 · 143 | 50.75 / 62.49 / **56.01** · 305 |
| nstdb | 516 | 78.88 / 23.94 / 36.73 · 2506 | 75.00 / 27.49 / **40.23** · 1978 | 65.89 / 22.12 / 33.12 · 2320 | 72.48 / 16.21 / 26.49 · 3747 |
| escdb | 1,075 | 51.26 / 21.49 / **30.28** · 1873 | 64.56 / 19.53 / 29.99 · 2660 | 43.53 / 19.94 / 27.35 · 1748 | 50.98 / 13.55 / 21.41 · 3253 |

Q F1 ≥ 99.4 trên mọi nguồn portal và mitdb/escdb/ahadb cho cả bốn model; V F1 trên
`dataset-v4-beat` 95.1 (30k) → 96.1 (2m).

Bốn điều rút ra:

1. **Không có dấu hiệu overfitting.** Khoảng cách `portal-train` → `portal-eval` ở S F1 là
   0.4–1.3 điểm cho cả bốn model, với dữ liệu model đã thấy đúng từng strip. Nó học đặc trưng,
   không học mẫu.
2. **Holdout curate (`dataset-v4-beat`) ngang `portal-eval`** (88.32 vs 87.44 ở `2m`): không có
   dịch chuyển phân phối giữa split băm và holdout.
3. **So với họ 1 chuyển đạo cũ** trên `dataset-v4-beat`: `1m` S F1 87.42 → **87.08**, `30k`
   85.14 → 85.04 — ngang, dù `val_wF1` mức bước tăng 0.85 → 0.92. `2m` mới đạt **88.32**, cao
   nhất mọi model của cả hai package (`beat_seq2seq_multistream` baseline 87.05). Cả họ dịch về
   phía **nhạy**: `30k` mitdb S_Se 43.35 → **50.75**. Đúng cảnh báo cũ: F1 mức bước không quyết
   định S F1 mức nhịp — và cũng đúng rằng data cũ thiếu 19,763 record nên hai bảng không cùng
   điều kiện.
4. **EC57 với một chuyển đạo nhân ba vẫn là điểm yếu**: nstdb/escdb S_+P 14–27%, FP/1k
   1,700–3,700 — ở mật độ S 0.14% mọi báo nhầm đều thành +P. Đòn bẩy là `CLASS_WEIGHTS[0]` và
   `--s-boost < 1`, **hiệu chuẩn trên `portal-eval`**, không trên mitdb.

`30k` nhạy nhất trên mọi database và `100k` chính xác nhất; hai model nhỏ nằm ở hai điểm khác
nhau của cùng đường cong Se/+P chứ không cái nào trội hẳn.

## 9. Test — [`tests/`](tests/)

```bash
./run_pipeline.sh test          # hoặc: python -m pytest tests/ -q
```

71 test, không cái nào cần dataset thật trừ [`test_ec57.py`](tests/test_ec57.py) (tự skip khi
thiếu database). Chúng ghim đúng những thứ đã từng sai âm thầm:

| file | ghim cái gì |
|---|---|
| [`test_signal_and_labels.py`](tests/test_signal_and_labels.py) | chuyển đạo được gán nhãn là kênh 0; bù bằng nhân bản **chính xác**; hình học cửa sổ; lát kín vùng chồng; ba mẫu `(start, stop)` của lỗi 2499 mẫu; round-trip nhãn ↔ giải mã |
| [`test_models.py`](tests/test_models.py) | 4 ngân sách tham số; contract vào/ra; hai sub-model tồn tại và backbone đủ lớn để đáng tiền huấn luyện; bất biến một-chuyển-đạo của `RhythmDescriptor`; round-trip checkpoint theo **từng trọng số** |
| [`test_pipeline_and_training.py`](tests/test_pipeline_and_training.py) | parse bytes **chính xác từng bit**; mọi bước nhãn có đúng một lớp sau augment; nhân bản chuyển đạo cho ra bản sao **bằng nhau**; manifest lệch bị từ chối; tính phi đơn điệu của poly2; `StepConfusion` khớp tham chiếu numpy; `ssl` học được thứ học được; `cpc` vượt mức ngẫu nhiên |
| [`test_ec57.py`](tests/test_ec57.py) | mitdb 2 chuyển đạo @360 Hz → 3 chuyển đạo @250 Hz với bản sao **bit-identical**; `--lead-mode` thực sự khác nhau; chuyển đạo được đọc từ header; `.ain` ghi ra đọc lại được ở đúng fs và không có nhịp nào vượt quá record; mẫu split **xác định và độc lập thứ tự CSV**; `.hea` viết lại parse được với cùng gain/baseline/fs; `portal-eval` chạy hết predict → bxb → report trên record thật |

## 10. Cấu trúc

```
ecg_resumamba/
├── ecgr/
│   ├── config.py           mọi đường dẫn & siêu tham số
│   ├── xla.py              tìm libdevice.10.bc cho XLA (xem mục 0)
│   ├── signal_ops.py       lọc, chọn chuyển đạo, z-score, cắt cửa sổ ── dùng CHUNG cho build data và inference
│   ├── labels.py           annotation → nhãn 500 bước, và ngược lại (giải mã nhịp)
│   ├── checkpoints.py      tìm checkpoint tốt nhất theo F1 trong tên file
│   ├── data/               splits.py · build_npy.py · build_tfrecord.py · pipeline.py
│   ├── models/             layers.py (DiagSSM1D, AdaIN, RhythmDescriptor) · resumamba.py
│   ├── training/           losses.py · callbacks.py · ssl.py · cpc.py · train.py
│   ├── evaluation/         step_metrics.py · bxb.py · ec57.py · report.py
│   └── cli.py              `python -m ecgr <stage>`
├── tests/                  71 test, chạy ở đâu cũng được
├── docs/references/        danh mục tài liệu tham khảo
├── checkpoints/            4 checkpoint 3 chuyển đạo + weights ssl/cpc + manifest.json; legacy_1lead/ = 3 bản cũ
├── logs/tensorboard_legacy_1lead/  log train của 3 run 1 chuyển đạo cũ
├── scripts/                driver shell của WFDB (bxb, sumstats, …)
├── assets/                 danh sách study eval v4 (đi kèm project)
└── run_pipeline.sh
```

> Log TensorBoard của các run **mới** nằm trong `<RUN_DIR>/logs/<model>/`
> (`tensorboard --logdir <RUN_DIR>/logs`). `logs/tensorboard_legacy_1lead/` là 3 run 1 chuyển
> đạo cũ, giữ lại để đối chiếu — xem `epoch_loss` cùng đường weighted F1 trong
> `confusion_log.txt` của run đó: chúng đi ngược chiều nhau từ epoch 2, và đó chính là bằng
> chứng cho việc monitor phải là F1 chứ không phải loss (mục 6c).

## 11. Những gì đã thay đổi so với bản 1 chuyển đạo

| | trước | giờ |
|---|---|---|
| đầu vào | `(2500, 1)` băng QRS (tùy chọn 2: + băng P/T) | `(2500, 3)` **ba chuyển đạo**, chuyển đạo gán nhãn ở kênh 0 |
| kích thước model | 3 (1M / 100K / 30K) | **4** (2M / 1M / 100K / 30K) |
| tự giám sát | chỉ CPC, 5–9% tham số | **`ssl` (backbone, che khoảng + che chuyển đạo) + `cpc`** |
| EC57 | 1 chuyển đạo, khớp tự nhiên | 1 chuyển đạo **lặp ba lần**, và train tái hiện đúng trường hợp đó |
| dữ liệu | thiếu 19,763 record; ~95k cửa sổ lệch ngoài khoảng review; ~1/4 tập train là bản sao chính xác (dataset-2) | cả ba đã sửa (mục 4a) |
| tfrecord | `float_list` (7.500 field/example) + nhãn int64 | `bytes` + nhãn uint8, kèm manifest được kiểm |
| lượt eval mỗi epoch | 2 (Keras + callback) | **1** (`StepConfusion` là metric) |
| giải mã nhịp qua vùng chồng | so với vị trí đã nhận trước đó | chia đôi vùng chồng, lát kín chính xác |
| cửa sổ cuối khi sweep | có thể >90% là đệm, rồi bị z-score | kết thúc đúng tại hết record |
| build npy | một tiến trình | **đa tiến trình** (~5000 record/s với 32 worker) |
| `bxb` | `shell=True` không quote, bỏ qua exit status | quote đầy đủ + kiểm exit status |
| test | không có | **71** |

### Bất định của GPU — đã biết, đã đo

Một checkpoint vừa `fit` xong và cùng checkpoint đó nạp lại từ file cho ra dự đoán lệch tới
`3.8e-3` **dù trọng số giống nhau từng bit**: TF đẩy một phần đồ thị qua XLA, và quyết định gom
cụm khác nhau giữa một đồ thị đã trace cho train và một đồ thị chỉ để predict. Trên CPU, cùng
file nạp hai lần cho ra kết quả **giống nhau chính xác**.

Điều này không ảnh hưởng tính nhất quán của việc đánh giá — `stepeval` và `ec57` luôn nạp từ
file — nhưng nó là lý do một con số EC57 đo lại có thể lệch ở chữ số thập phân thứ hai.
