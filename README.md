# ecgr — phát hiện & phân loại nhịp ECG 3 chuyển đạo bằng ResUMamba

Một họ model (ResUMamba thích ứng cho contract seq2seq) ở **bốn** kích thước, một đường đi
duy nhất qua dữ liệu:

```
record portal ──► npy ──► tfrecord ──► ssl ──► cpc ──► train ──► refine ──► step eval ──► beat eval (EC57/bxb)
                                    └── tự giám sát, không dùng nhãn ──┘      └ head thời gian trên base đóng băng
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
./run_pipeline.sh test                   # 111 test, ~2 phút
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
| `ECGR_EC57_LEAD_MODE` | bao nhiêu chuyển đạo **thật**: `native` / `single` / `auto` (mục 7b) | `native` |
| `ECGR_LEAD_FILL` | lấp kênh còn thiếu bằng gì: `zero` / `duplicate` (mục 3) | `zero` |
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
2. **Record thiếu chuyển đạo được lấp bằng 0** (`LEAD_FILL_MODE='zero'`). Kênh im lặng là thứ
   model vốn đã thấy mỗi khi một điện cực rơi ra, và huấn luyện tạo ra nó **có chủ ý**
   (`AUGMENT_LEAD_DROP_PROB`) — nên "chuyển đạo này không tồn tại" được nói bằng đúng từ vựng
   model đã học. Nó cũng không thể bị nhầm thành bằng chứng: một chuyển đạo nhân bản là **lá
   phiếu thứ hai cho đúng điều chuyển đạo thứ nhất vừa nói**, thứ mà một model đa chuyển đạo
   không nên được đưa. `'duplicate'` vẫn dùng được (`--lead-fill duplicate`) và là cách mọi
   bảng EC57 trước 20/09/2026 được tạo.

Hai câu hỏi tách rời nhau, mỗi câu một nút: **dùng bao nhiêu chuyển đạo thật** (`EC57_LEAD_MODE`)
và **lấp phần còn lại bằng gì** (`LEAD_FILL_MODE`).

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

### 5c. So với bài báo gốc — giống gì, khác gì, và vì sao

Đọc trực tiếp từ [`Heo2026_ResUMamba_ESWA331-133149.pdf`](docs/references/) (Fig. 1–5, Bảng 8,
mục 3.1–3.6, 4.2).

**Giữ nguyên, kể cả chi tiết:**

| thành phần | bài báo | ở đây |
|---|---|---|
| Thân hai nhánh ResUNet + state-space, hợp nhất bằng concat theo kênh | Fig. 2a | như vậy |
| Khối ResU, **2 khối nối tiếp độ sâu r = 3 rồi r = 2** (Hwang 2023) | Fig. 2b | `resu_depths=(3, 2)` — trùng khít |
| Bộ mã hóa ngữ cảnh huấn luyện bằng CPC/InfoNCE rồi **đóng băng** | mục 3.3 | như vậy |
| AdaIN: `Conv1D → AdaIN → LeakyReLU`, lặp **×2 với kernel k = {3, 7}** | Fig. 4 | `adain_kernels=(3, 7)` — trùng cả kích thước kernel |
| Cross-attention hợp nhất đặc trưng nhịp với đặc trưng sâu, **4 đầu** | Fig. 5, L = 4 | `attn_heads=4` (2 ở bản 100k/30k) |
| Poly-2 loss, **ε₁ = 0.3, ε₂ = −0.5** | eq. 12, Fig. 10 | `POLY2_EPS = (0.3, -0.5)` — đúng số bài báo |
| Cắt gradient theo chuẩn ℓ₂ = 1.0 | mục 4.2 | như vậy trong `ssl.py` / `cpc.py` |
| 4 lớp đầu ra | N/SVEB/VEB/F | None/N/V/S (xem khác biệt 1) |

**Khác, và mỗi khác biệt đều do contract bài toán chứ không phải do tuỳ ý:**

| # | bài báo | ở đây | vì sao |
|---|---|---|---|
| 1 | Phân loại **một nhịp** đã cho đỉnh R; 4 lớp N/SVEB/VEB/**F** | **Phát hiện + phân loại** 500 bước; lớp 0 là **nền (None)**, còn N/V/S | Không có đỉnh R cho trước. Lớp F (fusion) bị thay bằng lớp nền vì phải trả lời "ở đây có nhịp không" |
| 2 | Vào `720 × 1` = 2 s @ 360 Hz, **căn giữa đỉnh R** | Vào `2500 × 3` = 10 s @ 250 Hz, **chạy tự do** | Dữ liệu portal là 3 chuyển đạo @ 250 Hz; Holter thực tế không có đỉnh R sẵn |
| 3 | **S6** (SSM chọn lọc, phụ thuộc đầu vào), **nhân quả** | `DiagSSM1D`: SSM **chéo** nghiệm đóng, tính bằng **FFT**, **hai chiều** | Quét S6 là hồi quy tuần tự → `tf.scan` 500 bước, chậm và không export được. Và vùng T–P phân biệt S với N nằm **trước** nhịp cần gán nhãn nên cần hai chiều |
| 4 | Bộ mã hóa **bệnh nhân**: 60 s ECG hiệu chuẩn không nhãn, M ≈ 59 cửa sổ 720 mẫu | Bộ mã hóa **ngữ cảnh đoạn**: chính đoạn 10 s, 9 cửa sổ 500 mẫu chồng 50% | tfrecord không có định danh bệnh nhân lẫn đoạn hiệu chuẩn |
| 5 | Chân trời CPC **j = 2** cố định | `j ∈ {1, 2}` | Thêm j = 1 tốn đúng một ma trận `W_j` |
| 6 | **32 đặc trưng lâm sàng thủ công**: 10 nhịp (R–R trước/sau, mean/var/skew/kurtosis, thống kê 20 nhịp trước) + 3 hình thái + 19 mẫu sóng tại mốc | `RhythmDescriptor`: tự tương quan envelope, 88 độ trễ dải 30–220 bpm, **trong graph** | **Toàn bộ** 32 đặc trưng đó cần vị trí nhịp — tức chính đầu ra của model này |
| 7 | Cross-attention: đặc trưng lâm sàng là **query**, đặc trưng sâu là key/value | **Đảo lại**: đặc trưng per-step là query, token nhịp là key/value | Bài báo cần **một** đầu ra; ở đây cần **500** |
| 8 | `AvgPool` trước classifier, rồi `Linear(256,32) → Linear(32,4)` | Không pool; `Conv1D(4, k=1)` per-step | Phải giữ 500 bước |
| 9 | Tự giám sát **chỉ** cho bộ mã hóa bệnh nhân (0.345M / 1.592M ≈ 22%) | **Thêm** chặng `ssl`: tái tạo có che (span + chuyển đạo) cho **backbone** | Ở quy mô này bộ mã hóa ngữ cảnh chỉ là 5–9% tham số; phần còn lại trước đó khởi tạo ngẫu nhiên (mục 6a) |
| 10 | Train/test MIT-BIH DS1/DS2 (liên bệnh nhân) | Train trên 542,721 đoạn portal; **MIT-BIH là benchmark giữ riêng, chưa từng train** | mục 7e |
| 11 | Một model (1.247M suy luận) | **Bốn** kích thước, tích chập tách chiều sâu dưới mức 100k | Mục tiêu nhúng |
| 12 | batch 2048, AdamW, lr 5e-4, wd 1e-2, cosine | batch 128, Adam, lr 1.4e-3, ReduceLROnPlateau | A100 vs 3090; và `val_weighted_f1` là chỉ số dừng (mục 6c) |
| 13 | Đánh giá: macro/weighted F1 mức nhịp, có đỉnh R | EC57/bxb Se và +P — **đo cả phát hiện** | Contract khác nên chỉ số khác; hai bảng không so trực tiếp được |

### Bảng so sánh 5 model — tham số theo đúng cách chia module của Bảng 8 bài báo

| module | **ref** (bài báo) | **2m** | **1m** | **100k** | **30k** |
|---|---|---|---|---|---|
| stem (ConvBlock 1→C) | — (gộp) | 54,968 | 31,328 | 3,188 | 1,184 |
| nhánh ResU | 696,000 | 779,712 | 386,048 | 21,576 | 6,324 |
| nhánh state-space (Mamba/DiagSSM) | 235,000 | 549,120 | 188,544 | 21,024 | 6,288 |
| hợp nhất hai nhánh | — (gộp) | 74,496 | 33,280 | 4,800 | 1,248 |
| điều biến AdaIN | 115,000 | 254,464 | 138,624 | 23,520 | 7,776 |
| đặc trưng nhịp + cross-attn | 201,000 | 84,832 | 54,432 | 12,760 | 4,216 |
| classifier | — (gộp) | 516 | 388 | 164 | 100 |
| bộ mã hóa ngữ cảnh (CPC) | 345,000 | 179,552 | 101,832 | 6,485 | 2,605 |
| **TỔNG** | **1,592,000** | **1,977,660** | **934,476** | **93,517** | **29,741** |
| **suy luận (trừ CPC)** | **1,247,000** | 1,798,108 | **832,644** | 87,032 | 27,136 |
| vào → ra | `720×1` → `4` | `2500×3` → `500×4` | ⟵ | ⟵ | ⟵ |
| FLOPs | 0.700 G / **nhịp** | — | — | — | — |
| s/step (batch 128, 3090) | — | 0.324 | 0.208 | 0.117 | 0.102 |
| VRAM đỉnh | A100 (batch 2048) | 6,595 MiB | 4,164 MiB | 1,708 MiB | 986 MiB |

Bốn điều đọc ra từ bảng:

1. **`resumamba_1m` là bản tương đương quy mô với bài báo**: 832,644 tham số suy luận so với
   1,247,000 — cùng bậc, và nhánh ResU (386,048) lẫn nhánh state-space (188,544) đều cùng bậc
   với 696,000 / 235,000. `2m` thì **lớn hơn** bài báo.
2. **Nhánh state-space của chúng ta tốn nhiều hơn tương đối** (549k so với 235k ở `2m`): nó
   **hai chiều** nên có hai bank nhân, và chạy ở 192 kênh thay vì 128.
3. **AdaIN tốn hơn** (254k so với 115k): bài báo điều biến **một** vector đã pool, ở đây điều
   biến **500 bước** ở chiều rộng lớn hơn.
4. **Khối đặc trưng nhịp lại RẺ hơn** (84,832 so với 201,000) dù làm cùng việc: tự tương quan
   envelope **không có tham số** — nó là phép tính, không phải lớp học được — còn bài báo phải
   nuôi `Linear(H,64)` cho 32 đặc trưng thủ công cộng MLP `Linear(256,32) → Linear(32,4)`.

FLOPs không điền cho cột của chúng ta vì **không so được**: 0.700 GFLOPs của bài báo là cho
**một nhịp** (cửa sổ 720 mẫu), còn một lần chạy ở đây xử lý **10 s** và trả 500 nhãn — quy về
"mỗi nhịp" cần giả định số nhịp mỗi đoạn, và con số đó sẽ nói nhiều về nhịp tim hơn về model.
Cột s/step và VRAM là số **đo được**, dùng được để so sánh giữa bốn kích thước với nhau.


### Ngân sách tham số đi đâu

Nguyên tắc cắt giảm: **không đụng vào chiều sâu của nhánh state-space**. Trường tiếp nhận mà nó
mua chính là thứ phân biệt S với N, trong khi một nhân SSM chỉ tốn 4 số thực cho mỗi cặp
(kênh, trạng thái) — ở bản 30K là 576 trọng số cho cả nhánh. Ngân sách bị cắt ở chiều rộng và ở
các tích chập dày của nhánh ResU, vốn được thay bằng tích chập tách chiều sâu dưới mức 100k.

> **`layers.PKG` phải mãi là `"resumamba_seq2seq"`.** Keras ghi `"<package>>ClassName"` vào
> trong mỗi file `.keras` và tra cứu bằng đúng chuỗi đó. Tên package là một **định dạng lưu
> trữ**, không phải namespace.

## 6. Tự giám sát rồi train — [`ecgr/training/`](ecgr/training/)

### 6.0. Tổng quan: chặng nào không nhãn, chặng nào cần nhãn

Bốn chặng huấn luyện, **hai chặng đầu không đụng vào nhãn**:

```
ssl   ──► backbone            tái tạo có che            KHÔNG NHÃN
cpc   ──► context encoder     InfoNCE                   KHÔNG NHÃN   ──► đóng băng
train ──► backbone + head     poly2 trên 4 lớp          CẦN NHÃN
refine──► head thời gian      poly2, base đóng băng     CẦN NHÃN
```

**Bao nhiêu tham số học được mà không cần nhãn**

| model | `ssl` (backbone) | `cpc` (context) | **không nhãn** | chỉ có nhãn (head) |
|---|---|---|---|---|
| `2m` | 1,458,296 (74%) | 179,552 (9%) | **83%** | 339,812 (17%) |
| `1m` | 639,200 (68%) | 101,832 (11%) | **79%** | 193,444 (21%) |
| `100k` | 50,588 (54%) | 6,485 (7%) | **61%** | 36,444 (39%) |
| `30k` | 15,044 (51%) | 2,605 (9%) | **59%** | 12,092 (41%) |

Model càng lớn thì phần học được không nhãn càng lớn — vì head có kích thước gần như cố định
còn backbone nở theo chiều rộng.

**Mục tiêu không nhãn lấy "đáp án" từ đâu**

Một mục tiêu tự giám sát vẫn cần một đích để so; điều làm nó *tự* giám sát là đích ấy **lấy từ
chính tín hiệu**, không từ chú thích của con người.

| chặng | đích là gì | lấy từ đâu |
|---|---|---|
| `ssl` | 5 mẫu thô nằm sau mỗi bước đầu ra, trên cả 3 chuyển đạo (15 số/bước) | `reshape` của **chính đầu vào** — `_target()` không làm gì khác |
| `cpc` | latent `v_{i+j}` của cửa sổ tương lai | **chỉ số thời gian** `i+j`, không phải lớp |

**`ssl` — tái tạo có che.** Đầu vào bị làm hỏng hai kiểu, mỗi kiểu bật theo xác suất riêng cho
từng mẫu (≈ 80% batch mang ít nhất một loại):

* **che theo khoảng** — 35% trong 500 bước, theo các khoảng liền nhau 12 bước (240 ms ≈ một
  nhịp), làm 0 trên **mọi** chuyển đạo. Điền lại một nhịp bị che từ các nhịp lân cận chính là
  ngữ cảnh nhiều nhịp mà nhánh state-space tồn tại để cung cấp;
* **che theo chuyển đạo** — với xác suất 0,5, **một** chuyển đạo bị làm 0 hoàn toàn và phải
  được tái tạo từ hai chuyển đạo kia.

Loss là MSE **chỉ trên phần bị che**. Lấy trung bình trên mọi bước sẽ cho model ghi điểm bằng
cách chép lại 65% đầu vào nó vẫn thấy. Decoder (2 lớp conv) cố tình nhỏ và **bị vứt đi** sau
chặng này — một decoder có sức chứa riêng sẽ tái tạo được từ một biểu diễn yếu hơn, tức chuyển
việc học ra khỏi backbone.

**`cpc` — InfoNCE.** Đoạn 10 s cắt thành 9 cửa sổ 2 s chồng 50%; bộ mã hóa cho latent `v_i`; một
mô hình tự hồi quy **nhân quả** cho vector ngữ cảnh `c_i`. Từ `c_i` dự đoán `v_{i+j}` với
`j ∈ {1, 2}`, cạnh tranh với **mọi** latent khác trong batch — cả cửa sổ khác lẫn **đoạn khác**.
Chính phần "đoạn khác" ép biểu diễn phải mang đặc trưng riêng của từng đoạn.

**Khi nào KHÔNG cần nhãn**

* **Toàn bộ chặng `ssl` và `cpc`.** Không chỉ mục tiêu — cả **đầu vào**: `parse_signal` thậm chí
  không khai báo trường `labels`, nên hai chặng này đọc được một kho bản ghi **hoàn toàn không
  có nhãn** y như đọc tfrecord của dự án. (Trước đây nhãn bị parse rồi bỏ, tức trường ấy vẫn
  **bắt buộc** tồn tại — mục tiêu thì không nhãn nhưng đầu vào thì có, và điều đó vô hiệu hoá
  đúng cái lợi thực tế của tự giám sát. Đã sửa, có test.)
* **Hệ quả thực tiễn**: nếu anh có nhiều ECG chưa gán nhãn và ít ECG đã gán, hãy chạy `ssl` +
  `cpc` trên **toàn bộ** kho chưa gán rồi chỉ dùng phần đã gán cho `train`. Kiến trúc không đổi,
  chỉ cần trỏ `ECGR_TFRECORD_DIR` sang kho đó cho hai chặng đầu.
* **Bộ mã hóa ngữ cảnh thì không bao giờ cần nhãn**, kể cả về sau: `train` nạp nó rồi **đóng
  băng** (`freeze_ctx=True`). Đo trên `30k`: 2,605 tham số của nó có **0** tham số trainable khi
  train có nhãn. Đây đúng là cách bài báo giữ bộ mã hóa bệnh nhân bất biến — để hàm mất mát của
  bài toán nhịp không nắn một mô tả về biến thiên gây nhiễu thành một bộ phát hiện nhịp.

**Khi nào CẦN nhãn**

* **`train`** — đây là chặng duy nhất dạy model *ý nghĩa* của các lớp. Nhãn là 500 bước ×
  4 lớp one-hot, loss `poly2` có trọng số lớp. Đo trên `30k`: cập nhật **26,336** tham số
  (backbone 14,244 + head 12,092); backbone khởi tạo từ `ssl` rồi **được fine-tune**, nên nhãn
  *có* chạm vào nó — `ssl` là điểm khởi đầu, không phải một khối đóng băng.
* **`refine`** — đầu tinh chỉnh thời gian, cũng `poly2`, base đóng băng.
* **Chọn checkpoint** — `val_weighted_f1` và mọi lần chấm bxb đều cần nhãn tham chiếu.
* **Đánh giá** — EC57 và v4 beat-eval cần chú thích của chuyên gia. Không có cách nào quanh
  chuyện này: một con số Se/+P là một so sánh với nhãn.

Tóm lại: **nhãn cần cho việc gán *tên* lớp và cho việc *đo*, không cần cho việc học *biểu diễn***
— và ở bản `2m` thì 83% tham số học xong biểu diễn trước khi nhìn thấy nhãn đầu tiên.


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

### 6f. Chặng 4: đầu tinh chỉnh theo thời gian — [`refine`](ecgr/training/refine.py)

```bash
python -m ecgr refine --model resumamba_2m          # train head trên base đóng băng + chọn epoch
python -m ecgr ec57   --model resumamba_2m --checkpoint <run>/checkpoints/resumamba_seq2seq_2m/refined/refined_best.keras --tag resumamba_2m_refined
./run_pipeline.sh refine resumamba_2m               # cả hai bước
```

Model gốc gán nhãn từng bước 20 ms từ hình thái cộng với ngữ cảnh mà nhánh state-space mang
*ngầm*. Nhưng S/N phần lớn là câu hỏi về **thời điểm** — nhịp ngoại vị trên thất đến *sớm* so
với nhịp quanh nó — và bằng chứng đó nằm trong **chuỗi nhịp model vừa dự đoán**, không nằm trong
đặc trưng của một bước riêng lẻ. [`models/refine.py`](ecgr/models/refine.py) gắn một mạng
state-space hai chiều nhỏ (~22k tham số) đọc đúng thứ đó: đặc trưng trước softmax của base nối
với chính phân phối lớp `p₁(t)` của nó (và `log p₁`). Kernel 256 bước mỗi chiều = 5,1 s = 6–10
khoảng R–R — đúng khung mà bác sĩ đọc tính đến sớm.

Hai tính chất được **xây vào** chứ không hy vọng:

* **`p_None` được bảo toàn chính xác.** Head chỉ phát ra hiệu chỉnh cho ba lớp *nhịp*, và đầu ra
  được ghép `[p_None, (1 − p_None)·softmax(logit_nhịp + Δ)]`: nó phân phối lại khối lượng giữa
  N/V/S, không thể tự biến một phát hiện thành nền hay ngược lại. Phát hiện QRS vẫn là của base.
* **Khởi đầu là phép đồng nhất.** Lớp cuối khởi tạo bằng 0, nên `epoch_00` **chính là** base.
  Nhờ đó quy tắc chọn dưới đây luôn có ít nhất một ứng viên hợp lệ.

Base **đóng băng** khi train head: chỉ số của nó là sàn, và một base đóng băng không thể tụt khỏi
sàn. Sau khi train, **mọi epoch được chấm bxb trên mẫu 5,000 record `portal-eval`** (cùng mẫu mà
base đã được chấm) và epoch thắng là epoch có **S F1 cao nhất trong số các epoch mà Q, V, S Se và
+P đều ≥ base** (dung sai `REFINE_TOLERANCE_PP` = 0,1 điểm cho nhiễu chấm). Không epoch nào đủ →
base được chọn và `selection.json` nói rõ. Chọn trên `portal-eval`, **không** trên beat-eval hay
mitdb: đó là các holdout được báo cáo, chọn trên chúng là tự chấm.

### 6g. Hai đòn bẩy sau khi đã train: SWA, ensemble, và bộ lọc decoder

Ba công cụ *không cần train lại*, mỗi cái là một cách khác nhau để nâng **cả** Se và +P thay
vì đổi cái này lấy cái kia:

```bash
python -m ecgr swa  --model resumamba_2m --checkpoints <BEST_F1>/*.keras --out swa.keras
python -m ecgr ec57 --model resumamba_2m --checkpoint a.keras b.keras c.keras --tag ens   # ensemble
python -m ecgr ec57 --model resumamba_2m --min-run 2                                       # bộ lọc decoder
```

* **SWA** ([`training/swa.py`](ecgr/training/swa.py)) — trung bình trọng số của vài epoch tốt cuối
  cùng một run. Kéo nghiệm về vùng phẳng của mặt loss, thứ mà một checkpoint đơn lẻ — chọn tại
  một đỉnh nhiễu của chỉ số mức bước — không có. Thống kê BatchNorm được trung bình cùng; đó là
  xấp xỉ thường dùng thay cho việc ước lượng lại, và các epoch liên tiếp của một run đủ gần để
  nó đúng.
* **Ensemble** (`Ensemble` trong [`evaluation/ec57.py`](ecgr/evaluation/ec57.py)) — trung bình
  softmax của nhiều model theo từng bước. Báo nhầm *độc lập* của các thành viên triệt tiêu,
  phát hiện *chung* cộng dồn — đây là thay đổi lúc suy luận đáng tin nhất để nâng đồng thời
  Se và +P, trả bằng một lượt forward mỗi thành viên. Mọi stage eval coi nó như một model.
* **`--min-run`** — bỏ các run ngắn hơn N bước khi giải mã. Một phát hiện 1 bước (20 ms) gần như
  luôn là nhấp nháy artefact chứ không phải nhịp (block nhãn rộng 11 bước), nên bộ lọc mua +P
  với giá Se gần bằng không. Chỉnh trên `portal-eval`.

### 6h. Train lại có nhiễu tổng hợp — run `260918_3lead_noise`

Phân tích theo từng record của mitdb (từ bảng `-L` của bxb) cho `2m`: **74/82 nhịp bỏ sót và
99/121 báo nhầm nằm trong 4 record nhiễu** 203, 105, 108, 116; còn nstdb theo định nghĩa là
database stress nhiễu. Trong khi đó cửa sổ train là strip portal đã review — sạch — và pipeline
augment **không có** một dạng nhiễu nào. Model chưa từng học giữ nhịp qua artefact.

[`pipeline._noise`](ecgr/data/pipeline.py) thêm ba thành phần, mỗi cái bật theo xác suất riêng
cho từng mẫu (≈ 80% batch mang ít nhất một loại), biên độ theo đơn vị z-score (QRS ở 3–8):

| thành phần | dạng | xác suất / biên độ | mô phỏng |
|---|---|---|---|
| trôi đường nền | 2 sin/chuyển đạo, 0.05–0.6 Hz | 0.5 / ≤ 0.6 | hô hấp, trôi điện cực |
| nhiễu băng rộng | Gauss trắng | 0.5 / ≤ 0.25 | EMG, bộ khuếch đại |
| chuyển động điện cực | 1 bump Hann 0.2–1 s, **một** chuyển đạo, ±1–3 | 0.2 | dạng giống QRS nhất, tốn +P nhất |

Nhãn giữ nguyên. Cùng run này bật `SAVE_EVERY_EPOCH`: mọi epoch từ `CKPT_START_EPOCH` được lưu
(`<ckpt>/epochs/epoch_NN.keras`) để **chọn checkpoint bằng bxb trên `portal-eval`** thay vì bằng
F1 mức bước — đúng cảnh báo của chính README này rằng F1 mức bước không chọn được model tốt ở mức
nhịp. SSL/CPC được tái dùng từ `260917_3lead` (`ECGR_SSL_RUN` / `ECGR_CPC_RUN`), kiến trúc không
đổi.

### 6i. Hai chặng tự giám sát có thật sự **không dùng nhãn**? — chứng minh, không phải khẳng định

tfrecord mang nhãn **trong cùng một record** với tín hiệu, nên chỉ cần một dòng là bắt đầu dùng
nhãn một cách vô tình, và **không gì ở phía sau báo lỗi** — chặng đó chỉ đơn giản thôi không còn
là tự giám sát. Vì vậy [`tests/test_label_free.py`](tests/test_label_free.py) tấn công tính chất
này theo ba đường:

**1. Cấu trúc** — dataset mà hai chặng đọc trả về **một** tensor, tức nhãn **vắng mặt khỏi
graph** chứ không phải "có nhưng không dùng":

```
signal_only=True  -> TensorSpec(shape=(None, 2500, 3), dtype=tf.float32)     # một tensor
signal_only=False -> (TensorSpec(2500, 3), TensorSpec(500, 4))               # có nhãn
```

và `train_step` / `test_step` của cả hai trainer có chữ ký đúng `(self, signal)`.

**2. Từ vựng** — sau khi bỏ docstring và comment, code của `ssl.py`/`cpc.py` không chứa
`y_true`, `CLASS_WEIGHTS`, `SYMBOL_TO_LABEL`, `CLASS_NAMES`, `NUM_CLASSES`, `LOSSES`,
`labels_from_annotations`. (Chú ý: `ssl.py` **có** dùng `tf.one_hot`, nhưng là one-hot trên chỉ
số **kênh** để chọn chuyển đạo bị che — không liên quan lớp nhãn. Đây là một dương tính giả mà
chính lần chạy test đầu tiên bắt được, và danh sách từ khoá đã được sửa cho đúng.)

**3. Nhân quả — cái không thể lừa được.** Cùng một bộ tín hiệu, **hai bộ nhãn hoàn toàn khác
nhau** (toàn 0 so với ngẫu nhiên 0–3, khác nhau ở 11,930/16,000 bước), cùng seed, cùng trọng số
khởi tạo:

| chặng | loss với nhãn A | loss với nhãn B | |
|---|---|---|---|
| `ssl` | 0.5532110929 | 0.5532110929 | **giống từng bit** |
| `cpc` | 4.23913908 | 4.23913908 | **giống từng bit** |

Nếu một nhãn từng đến được một trong hai mục tiêu — trực tiếp, qua đường augment, hay qua một
trọng số lớp — hai lần chạy này đã phân kỳ. Test còn kiểm rằng loss **có di chuyển** giữa các
epoch, vì hai hằng số bằng nhau thì chẳng chứng minh điều gì.

Đích của hai mục tiêu cũng đến từ chính đầu vào, không từ nhãn: `ssl` tái tạo **5 mẫu thô sau
mỗi bước đầu ra** của chính tín hiệu (`_target` chỉ là một phép `reshape` của đầu vào), còn `cpc`
dự đoán latent `v_{i+j}` xác định bằng **chỉ số thời gian**, không bằng lớp.

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

| `--lead-mode` | physionet (2 chuyển đạo) | portal beat-eval (3 chuyển đạo) |
|---|---|---|
| **`native` (mặc định)** | **2 chuyển đạo thật + 1 kênh lấp** | montage thật, không phải lấp |
| `single` | chuyển đạo 0 + 2 kênh lấp | chuyển đạo được review + 2 kênh lấp |
| `auto` | như `native` | montage thật |

Kênh lấp là **0** theo mặc định (`--lead-fill zero`); `--lead-fill duplicate` lặp lại chuyển đạo
được gán nhãn, cách các bảng trước 20/09/2026 được tạo. `duplicate` vẫn được nhận như bí danh cũ
của `--lead-mode single`.

Mặc định là `native` từ 20/09/2026 — lý do và số đo ở mục 8f. `duplicate` giữ lại làm đối chứng
"một chuyển đạo" khắt khe: `--lead-mode duplicate`.

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

## 7d. Đánh giá tại local — [`evaluate.py`](evaluate.py)

Chấm một checkpoint trên EC57 + bộ v4 beat-eval. **Không có tham số dòng lệnh**: sửa khối
`CONFIG` ở đầu file rồi chạy.

```bash
~/miniconda3/envs/beat/bin/python evaluate.py      # env có tensorflow
```

Khối `CONFIG` là toàn bộ giao diện:

```python
CHECKPOINTS = ['checkpoints/resumamba_2m.keras']   # nhiều file = ensemble (trung bình softmax)
DATABASES   = ['mitdb', 'nstdb', 'escdb', 'ahadb', 'afdb']
SCORE_V4    = True        # bộ v4 beat-eval 5,227 record
LEAD_MODE   = 'native'    # bao nhiêu chuyển đạo THẬT: native / single / auto
LEAD_FILL   = 'zero'      # lấp kênh còn thiếu: zero / duplicate
MIN_RUN     = 1           # bỏ run ngắn hơn N bước 20 ms
S_BOOST     = 1.0         # hiệu chuẩn trên portal, KHÔNG trên mitdb
MAX_RECORDS = None        # đặt số nhỏ = chạy thử
GPU         = None        # ví dụ '0'
BXB_ONLY    = False       # True = chấm lại, không suy luận (giây thay vì phút)
BASELINE    = None        # một ec57_summary.csv để so, mọi chỉ số kèm delta
```

Không dùng argparse là **có chủ ý**: một lần đánh giá là một bản ghi về những gì đã đo, và một
khối hằng số có tên trong file chính là bản ghi đó. Một dòng shell thì trôi khỏi history, và
không để lại dấu vết nào cho biết bảng kết quả kia đến từ một hay ba chuyển đạo.

Nó là cửa vào mỏng của cùng bộ máy `python -m ecgr ec57` dùng, khác ở bốn điểm — mỗi điểm là một
thứ đã từng làm mất thời gian:

* **Checkpoint tường minh**, ghi vào `./eval_results/<tên>/`. `ecgr ec57` ghi vào thư mục của
  `ECGR_RUN_TAG` và mặc định lấy checkpoint mà một run huấn luyện tình cờ để lại.
* **Kiểm tra điều kiện TRƯỚC khi tốn GPU**: `bxb`/`sumstats` trên PATH, database có trên đĩa,
  file checkpoint tồn tại, giá trị `LEAD_MODE`/`LEAD_FILL` hợp lệ, và tensorflow import được bằng
  đúng interpreter đang chạy — chạy bằng `/usr/bin/python3` sẽ báo ngay *"tensorflow is not
  importable by /usr/bin/python3 … invoke it with ~/miniconda3/envs/beat/bin/python"* thay vì
  chết giữa chừng. Thiếu bất kỳ cái nào bình thường sẽ hiện ra dưới dạng một báo cáo rỗng sau
  một giờ.
* **Đánh dấu theo ngưỡng nghiệm thu**: `*` đạt, `!` chưa, kèm danh sách thiếu bao nhiêu. Ngưỡng
  ở hằng `TARGETS` (mitdb Q ≥ 99.95 hai chiều, S Se > 43 / +P > 80; v4-beat S Se > 88 / +P > 92);
  database không có ngưỡng thì không bị đánh dấu, và `-` của lớp không có nhịp tham chiếu được
  giữ nguyên chứ không thành 0.00.
* **Không chấm hai split portal** (mục 7b) — chúng là công cụ của một run huấn luyện, không phải
  của việc nghiệm thu một file.

Mặc định của script **bằng** mặc định của dự án (`LEAD_MODE`/`LEAD_FILL`), có test ghim, nên chạy
trần là đo đúng thứ `ecgr ec57` sẽ đo. Dự đoán `.ain` và báo cáo WFDB thô được giữ dưới thư mục
kết quả, nên mọi con số truy ngược được về từng record.

Ví dụ đầu ra:

```
database        records    Q_Se    Q_+P    V_Se    V_+P    S_Se    S_+P
-----------------------------------------------------------------------
dataset-v4-beat       3 100.00 100.00       -       -  70.00!  87.50!
mitdb                 3 100.00*  99.96* 100.00  33.33  91.18* 100.00*

* = meets the acceptance target, ! = below it
below target:
  dataset-v4-beat  S_Se   70.00  (target 88.0)
```

## 7e. Kiểm tra rò rỉ và shape — kết quả đo ngày 22/09/2026

### Shape, từ config xuống tới từng record

| | shape | ghi chú |
|---|---|---|
| contract | `(2500, 3)` → `(500, 4)` | 2500/500 = 5 mẫu/bước = 20 ms |
| 4 model dựng mới | `(None, 2500, 3)` → `(None, 500, 4)` | cả bốn kích thước |
| 6 checkpoint đã giao | `(2500, 3)` → `(500, 4)` | softmax tổng = 1 (lệch ≤ 2.4e-07) |
| tfrecord manifest | 2500 × 3, 500 × 4, float32/uint8 | khớp config, `check_manifest` chặn nếu lệch |
| batch thực tế | `(8, 2500, 3)` float32 · `(8, 500, 4)` float32 | đọc từ tfrecord |
| mitdb, nstdb | 2 ch @ 360 Hz → `(451389, 3)` @ 250 Hz | 2 kênh thật + 1 kênh lấp 0 |
| escdb, ahadb, afdb | 2 ch @ 250 Hz → `(N, 3)` | 2 kênh thật + 1 kênh lấp 0 |
| v4 beat-eval | 3 ch @ 250 Hz → `(15000, 3)` | 3 kênh thật, **không** lấp |

> **Một cái bẫy đã sửa nhờ chính lần kiểm này.** `read_leads` và `predict_record` mặc định
> `lead_mode='auto'`, còn `'auto'` trên record 2 chuyển đạo với model 3 kênh rơi về `'single'`
> — nên một lời gọi trực tiếp dùng **một** chuyển đạo mitdb trong khi `ecgr ec57` (resolve từ
> config, nay là `native`) dùng **hai**. Không con số nào đã công bố bị ảnh hưởng: cả bốn scorer
> đều resolve `lead_mode or config.EC57_LEAD_MODE` và mọi call site truyền tường minh. Nhưng
> mặc định giờ là `None` → lấy từ config, mô tả `'auto'` trong config đã sửa lại cho đúng, và
> [`tests/test_ec57.py`](tests/test_ec57.py) ghim rằng lời gọi trần phải trùng với đường pipeline.

### Rò rỉ dữ liệu — đo trên những gì có trên đĩa

| kiểm tra | kết quả |
|---|---|
| tfrecord train/eval nằm dưới `PHYSIONET_DIR` | **0** / 76 file |
| tfrecord có tên chứa `mitdb|nstdb|escdb|ahadb|afdb` | **0** |
| `train ∩ study-giữ-lại` | **0** (62,282 study train) |
| `eval ∩ study-giữ-lại` | **0** (15,609 study eval) |
| `train ∩ eval` | **0** |
| v4 beat-eval: 5,227 record → 2,242 study | **toàn bộ** nằm trong danh sách giữ lại |
| `v4 ∩ train`, `v4 ∩ eval` | **0**, **0** |
| `portal-eval ∩ v4` | **0** — hai holdout rời nhau |
| `portal-eval ∩ train` | **0** |

Bốn bất biến này giờ là test ([`tests/test_leakage.py`](tests/test_leakage.py)), đọc từ study id
**lưu trong npy** và từ đường dẫn tfrecord đã resolve — không phải từ ý định của code. Kèm một
test kiểm rằng chính cái guard `assert_no_benchmark_data` có báo lỗi khi được đưa đường dẫn
physionet, vì một guard không bao giờ nói "không" thì không phải guard.

### Rò rỉ khi tuning — hai quyết định đã dùng số benchmark

Tầng dữ liệu sạch, nhưng **quy trình thì không hoàn toàn**. Mọi lựa chọn tự động đều nằm trên
`portal-eval` (split eval của dữ liệu portal — rời hoàn toàn khỏi v4 và EC57): chọn checkpoint
([`select.py`](ecgr/training/select.py)), chọn epoch của head ([`refine.py`](ecgr/training/refine.py)),
hiệu chuẩn `s_boost` (0.75), quyết định loại bỏ công thức nhiễu (0/48 epoch hợp lệ). Hai quyết
định **do tôi** đưa ra thì có nhìn số EC57:

1. **`EC57_LEAD_MODE = 'native'` (mục 8f)** — tôi quyết dựa trên bảng mitdb/nstdb/escdb. Đây là
   chọn *chính sách đọc dữ liệu* bằng benchmark. Giảm nhẹ: lý lẽ là **cơ chế**, không phải tham
   số khớp — sóng P của record 232 chỉ có trên V5, và +297 trong +304 nhịp S bắt thêm nằm đúng ở
   record đó; `portal-eval` không bị ảnh hưởng (record portal vốn 3 chuyển đạo thật nên `native`
   không đổi gì ở đó); và đó là một lựa chọn nhị phân, không phải một cuộc tìm kiếm. Nhưng bằng
   chứng tôi trưng ra là bảng mitdb, nên phải nói rõ.
2. **`resumamba_2m_sonly_head.keras` = epoch 4** — quy tắc tự động **đã loại** mọi epoch của head
   này (giữ `epoch_00` = base, vì chúng đổi S Se lấy S +P). Tôi lấy epoch 4 ra thủ công sau khi
   xem mitdb. Giảm nhẹ: epoch 4 **cũng là** epoch có S F1 `portal-eval` cao nhất trong 6 epoch
   (88.14 so với 87.94–88.10), nên quyết định trùng với cái mà chỉ `portal-eval` cũng đưa ra —
   nhưng thứ tự việc làm thì không sạch.

Một điểm nữa, không phải rò rỉ nhưng nên nói: `AUGMENT_LEAD_DUPLICATE_PROB` và
`AUGMENT_LEAD_DROP_PROB` được đặt **vì biết trước** database EC57 có 2 chuyển đạo và được gán nhãn
trên chuyển đạo đầu. Đó là dùng kiến thức về **định dạng** của benchmark, không phải về nhãn hay
điểm số của nó — nhưng nó vẫn là lý do công thức augment có hình dạng như vậy.


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

### 8c. Track A đo được trên EC57 — và vì sao nó chưa đủ

Ba biến thể không-train-lại được chọn trên `portal-eval` (mục 6g) rồi chấm đủ 8 nguồn. Se / +P;
**đậm** = tốt hơn base `2m`, *nghiêng* = kém hơn:

| nguồn · lớp | base `2m` | SWA-`2m` (ep 10+11+13) | ensemble SWA-`2m`+`1m`+`100k_ref` + min-run 2 | ensemble `2m`+`1m`+`100k_ref` + min-run 2 (không SWA) |
|---|---|---|---|---|
| mitdb · Q | 99.90 / 99.86 (82 sót / 121 nhầm) | 99.90 / *99.85* | 99.90 / **99.89** (86 / **89**) | 99.90 / **99.90** (85 / **87**) |
| mitdb · S | 45.79 / 61.17 | *43.02* / *49.13* | *43.86* / *58.82* | *45.10* / **65.57** |
| nstdb · Q | 96.42 / 83.72 | *96.26* / **84.18** | *95.59* / **85.89** | *95.64* / **85.94** |
| nstdb · S | 78.88 / 23.94 | *78.68* / **26.45** | *75.19* / **38.61** | *75.97* / **39.56** |
| escdb · S | 51.26 / 21.49 | **59.44** / **25.93** | **57.40** / **34.39** | **53.86** / **33.35** |
| v4-beat · Q | 99.68 / 99.56 | *99.66* / *99.53* | **99.69** / **99.59** | **99.69** / **99.60** |
| v4-beat · S | 85.86 / 90.92 | **86.99** / *90.68* | *84.77* / **91.09** | *84.22* / **91.36** |
| v4-beat · V | 97.19 / 95.11 | **97.32** / *94.95* | *97.03* / **95.58** | *97.02* / **95.65** |

Ba điều rút ra, và cả ba đều là lý do không có biến thể nào được "ship" theo tiêu chí *không chỉ số
nào giảm*:

1. **SWA-`2m` sụp S trên mitdb** dù tăng cả Se và +P của S trên `portal-eval`: báo nhầm S 798 →
   1,221, trong đó **record 213** (xoang nhanh ~110 bpm, đều, có fusion) 34 → 350 và 202 18 →
   85. Tách `portal-eval` theo loại nhịp: lớp TACHY/SVT chỉ 228 → 229 báo nhầm — portal **không
   gây stress** chế độ này, nên không thể gác nó; gác bằng mitdb thì là tự chấm. Bài học: trung
   bình trọng số trên một base mạnh có thể dịch ranh giới N/S trong một chế độ mà dữ liệu chọn
   không phủ. Một biến thể chỉ được ship sau khi đối chiếu **mọi** holdout, không sau `portal-eval`.
2. **Ensemble cắt báo nhầm Q trong nhiễu** (mitdb 121 → 89: record 108 29 → 5, 105 33 → 25) và
   nâng +P của mọi lớp trên mọi database — nhưng **giảm Q Se ở nstdb** (96.42 → 95.59) và afdb:
   trong nhiễu các thành viên bất đồng, `p_None` trung bình thắng nhiều hơn. Control cùng
   ensemble ở min-run 1: Q Se 95.82 — tức −0.6 là của ensemble, −0.23 là của `--min-run 2`.
3. **Ensemble không SWA** là biến thể cân nhất: không còn lỗi record 213 (báo nhầm S mitdb 798 →
   650, +P 61.2 → 65.6), Q +P lên ở mọi database, v4-beat S +P 91.36 — nhưng vẫn trả bằng Se
   (mitdb S −0.7, v4-beat S −1.6, nstdb Q −0.8).
4. Vì thế **Track A không thể** thỏa "Q Se tăng ở mọi tập" hay "S +P mitdb > 80": nó chỉ đổi chỗ
   trên biên Pareto sẵn có của họ model. Hai mục tiêu đó cần một base tốt hơn ở đúng chỗ đang
   lỗi — 80% lỗi Q của mitdb nằm trong 4 record nhiễu, ~47% báo nhầm S nằm trong record AF/AFL —
   tức Track B (mục 6h) rồi đầu ngữ cảnh nhịp (mục 6f) trên base đó.

### 8d. Track B — kết quả đầu tiên: `1m` train lại với nhiễu (run `260918_3lead_noise`)

Cùng SSL/CPC, cùng kiến trúc, chỉ thêm `_noise` và lưu mọi epoch từ 10. Hai cách chọn checkpoint
trên cùng run — theo F1 mức bước (như trước) và **theo bxb trên `portal-eval`** (`ecgr select`,
mục 6h) — so với `1m` cũ:

| mitdb | sót / nhầm Q | Q Se / +P | S Se / +P | 203 · 105 · 108 · 116 (sót/nhầm) | 213 nhầm S |
|---|---|---|---|---|---|
| `1m` cũ (260917) | 95 / 109 | 99.89 / 99.87 | 49.44 / 36.63 | 35/43 · 12/31 · 7/9 · 19/4 | 1,295 |
| `1m`+nhiễu, chọn theo F1 bước (ep 11) | 85 / 135 | 99.90 / 99.84 | 53.70 / 40.30 | 31/48 · 15/36 · 4/12 · 15/2 | 1,139 |
| `1m`+nhiễu, **chọn theo bxb** (ep 14) | **79 / 101** | **99.91 / 99.88** | 44.70 / **41.65** | 30/39 · 12/26 · 6/11 · 15/4 | **806** |
| … cùng checkpoint, đọc `native` (mục 8f) | **74** / 108 | **99.91** / 99.87 | 45.03 / **50.82** | 30/36 · 15/32 · – · – | **316** |

Ba điều rút ra:

1. **Giả thuyết nhiễu đúng hướng nhưng chưa đủ liều**: trên các record nhiễu, checkpoint chọn
   theo bxb giảm ~15% cả sót lẫn nhầm (95 → 79, 109 → 101) và là lần đầu **cả** Q Se và Q +P cùng
   lên trên mitdb. Nhưng đích 99.95 cần ≤ 42 mỗi loại, tức cắt 50–65%. Nhiễu tổng hợp một mình
   không tới đó.
2. **Chọn theo bxb khác hẳn chọn theo F1 bước**, đúng như README này đã cảnh báo: cùng một run,
   epoch 11 (F1 bước cao nhất) cho 135 báo nhầm Q và +P S 40.3; epoch 14 cho 101 và 41.7.
3. Nhiễu làm model **nhạy hơn và kém chính xác hơn** một cách hệ thống: trên `portal-eval` mọi epoch
   đều giảm V +P và S +P so với base cũ (0/10 hợp lệ theo quy tắc không-giảm). Record 213 (xoang
   nhanh) là điểm yếu **của cả họ trừ `2m`**: `1m` cũ đã có 1,295 báo nhầm S ở đó.

Đọc `native` cộng thêm lên trên `1m`+nhiễu như trên `2m`: mitdb S +P 41.7 → **50.8**, nstdb Q +P
84.8 → **86.9**, escdb Q **99.96 / 99.91**, escdb S +P 14.8 → **29.9**; hai đòn bẩy (nhiễu khi
train, hai chuyển đạo thật khi chấm) **cộng được** vì chúng chữa hai lỗi khác nhau.

**`2m`+nhiễu, chọn theo F1 bước (ep 10), đọc `duplicate`** — một cảnh báo hơn là một kết quả: mitdb
Q 85 sót / 125 nhầm (cũ 82 / 121; record 108 nhầm 29 → 8 nhưng 203 37 → 50), và S **44.70 / 51.23**
(cũ 45.79 / 61.17) vì báo nhầm S ở **record 213 nổ từ 34 lên 507**. Khả năng "miễn nhiễm 213" của
`2m` cũ là tính chất của *một checkpoint*, không của kiến trúc hay công thức train — đúng loại mong
manh mà quy tắc chọn trên `portal-eval` không nhìn thấy. Đọc `native` đưa 213 của `2m` cũ về 1 báo
nhầm, nhưng **không** cứu được `2m`+nhiễu: 507 → 592 (Q sót lại giảm 85 → **64**, thấp nhất từ đầu).
Tức run nhiễu đã làm `2m` mất khả năng phân biệt xoang nhanh-đều với S ở cả hai chế độ đọc. Nghi phạm
đầu tiên là thành phần *chuyển động điện cực* (bump Hann 1–3 std trên một chuyển đạo) — **đã kiểm và
bác bỏ** (run `260918_2m_ft_nobump`): fine-tune `2m` ep13 với nhiễu **không** bump, LR 3e-4, chỉ **một
epoch**, chọn-bxb hợp lệ trên `portal-eval` (S 89.72/85.27 → 90.20/85.63) — mà trên mitdb `duplicate`
record 213 vẫn nổ 34 → **632** (S 43.68 / 48.23). Vậy bất kỳ nhiễu nào cũng đủ dịch ranh giới N/S ở
chế độ xoang nhanh-đều khi model chỉ có MLII: "miễn nhiễm 213" của ep13 là một lưỡi dao, không phải một
tính chất. Đọc `native`, cùng checkpoint ấy: 213 = **6**, S 50.35 / 65.84, Q 71 sót / 121 nhầm, nstdb
S +P 31 → **50.8**. Tức với V5 trong tay, 213 luôn phân biệt được và sự mong manh này **không còn quan
trọng** — một lý lẽ mạnh nữa cho chính sách `native`.

**`2m`+nhiễu, chọn theo bxb (ep 13), đọc `duplicate`** — một điểm vận hành khác hẳn: mitdb S **38.29 /
68.78** (+P +7.6, Se −7.5 so với `2m` cũ), escdb S +P 21.5 → **37.5**, nstdb S +P 23.9 → **67.7**
(gấp gần ba) nhưng V +P giảm (mitdb 94.1 → 91.8, nstdb 67.1 → 61.9) và Q sót/nhầm không đổi
(85 / 125). Trên `portal-eval` không epoch nào của run nhiễu là hợp lệ so với `2m` cũ (0/9). Nhiễu
theo công thức này đẩy `2m` về phía **S chính xác hơn / V kém chính xác hơn / S kém nhạy hơn** — không
phải một cải thiện đồng đều, và không đụng được vào lỗi Q của 203/105 (82 sót / 128 nhầm).

Nhưng giải phẫu S trên mitdb của checkpoint này đáng chú ý: báo nhầm S 798 → **477**; record 213 về
**13** (miễn nhiễm trở lại — khác với checkpoint chọn theo F1 bước, 507); và **record 222 (AFL)
243 → 125** — lần đầu một record rung/cuồng nhĩ bị cắt một nửa, đúng chỗ mà đầu ngữ cảnh nhịp chỉ
cắt được 15%. Giá: record 232 (APC không đến sớm) 366 → 265 TP. Mà đọc `native` chính là thứ cộng
+297 TP cho 232 trên `2m` cũ. Hai ứng viên đang chấm: checkpoint này ở `native`, và **ensemble chéo
run** `2m` cũ (giỏi 232) + `2m`-nhiễu (giỏi 222) + `1m` + `100k_refined` ở `native`.

**`100k`+nhiễu, chọn theo bxb (ep 16), đọc `duplicate`** — model nhỏ hưởng lợi rõ nhất từ nhiễu: so với
`100k` cũ, mitdb Q +P 99.84 → **99.86**, S Se 32.0 → **42.9** (+P 69.1 → 60.7), v4-beat S **84.23 /
90.20** (cũ 80.35 / 89.20 — **cả hai lên**), escdb S 43.5 / 19.9 → **56.2 / 21.3**, nstdb V +P 68.7 →
**87.2**, S Se 65.9 → **79.1**. Trả bằng nstdb Q Se (96.1 → 95.3) và S +P mitdb/nstdb. Trên
`portal-eval` không epoch nào hợp lệ so với `100k` cũ (0/11) — cùng mẫu "nhạy hơn, kém chính xác hơn"
như `1m`/`2m`. Đầu ngữ cảnh nhịp — thứ đã nâng cả hai chiều S trên `100k` cũ — đang được gắn lên
checkpoint này.

Hai head gắn lên checkpoint này: `s_only` trung tính → mọi epoch đổi S Se lấy +P, **không** epoch nào
hợp lệ (giữ base); `beats` với trọng số cũ → epoch 6 hợp lệ (2/7), S F1 `portal-eval` +0.18, và trên
mitdb **cả hai chiều S lên nhẹ**: 42.91 / 60.69 → 44.70 / 61.72 (`dup`), 43.72 / 58.91 → 44.55 / 59.20
(`native`). Thật, nhưng nhỏ — head trên `100k` đã cho hết những gì nó có.

Còn chờ: `30k`; các bảng `native` còn lại; và thí nghiệm không-bump (dưới).

### 8e. Đầu ngữ cảnh nhịp trên mitdb — cơ chế, và giới hạn của nó

Ba head của `2m` bị quy tắc chọn loại trên `portal-eval` (mục 6f) được chấm **chỉ trên mitdb** để
hiểu cơ chế — không phải để chọn (chọn trên mitdb là tự chấm). S Se / +P, và báo nhầm S theo record:

| `2m` | S Se / +P | TP / FP | 222 (AFL) | 219 (AF) | 215 | 213 (xoang nhanh) | 200 |
|---|---|---|---|---|---|---|---|
| base | 45.79 / 61.17 | 1,257 / 798 | 243 | 87 | 117 | 34 | 94 |
| head `s_only` ep4 | 41.35 / **66.57** | 1,135 / **570** | 211 | 72 | **72** | **12** | 68 |
| head `neutral` ep6 | 41.86 / 65.85 | 1,149 / 596 | 213 | 78 | 82 | 21 | 67 |
| head trọng số cũ ep6 | **48.82** / 60.12 | **1,340** / 889 | 250 | 90 | 127 | 80 | 103 |

Head làm đúng việc của nó ở **nhịp nhanh-đều** (213: −65%, 215: −38%) — giống `100k_refined`
gần sạch record 213 (1 báo nhầm) — nhưng chỉ cắt ~15% ở **AF/AFL** (222, 219), nguồn báo nhầm S
lớn nhất. Và trên mitdb nó vẫn là thanh trượt Pareto: mỗi điểm +P đổi bằng gần một điểm Se.

Cùng hai head đó đọc **`native`** (V5 cộng 232, head cắt 213/215/200):

| `2m`, `native` | mitdb S Se / +P | 232 TP | 222 · 215 · 200 | nstdb Q Se / S +P / V +P |
|---|---|---|---|---|
| không head | 56.87 / 65.64 | 663 | 250 · 140 · 103 | 96.79 / 31.13 / 71.23 |
| + head `s_only` ep4 | 49.80 / **70.36** | 525 | 222 · 74 · 64 | 96.97 / 47.82 / 73.40 |
| + head `neutral` ep6 | 51.91 / 69.78 | 568 | 220 · 94 · 73 | **97.07** / **50.71** / **80.76** |

`native` + head là **profile +P tốt nhất của một model đơn với Se > 43**: mitdb S +P 70.4, nstdb S +P
50.7 và Q Se cùng lên. Nhưng trên mitdb S F1 vẫn giảm (60.9 → 58–59) vì head vẫn đổi 5 điểm Se lấy 4
điểm +P, và 222/219 (AF) gần như không nhúc nhích.

Số học của mục tiêu *S Se > 43 và +P > 80 trên mitdb*: cần ≤ ~295 báo nhầm ở ≥ 1,180 TP. Head tốt
nhất ở `duplicate` đứng ở 570 / 1,135, ở `native` 576 / 1,367 — **cách đích gấp đôi về báo nhầm**. Không head hay ensemble nào trên họ
base hiện tại đóng được khoảng cách đó; nó đòi một base xử lý AF khác hẳn (nhịp N trong rung nhĩ
không được coi là "đến sớm"), và đó là hướng cho vòng sau — không phải một nút chỉnh.

### 8f. Chính sách chuyển đạo trên EC57 — phát hiện lớn nhất của vòng này

Từ đầu, EC57 được chấm với **một** chuyển đạo nhân ba (`duplicate`), theo đúng đặc tả. Mọi database
EC57 lại có **hai** chuyển đạo thật. `--lead-mode native` đưa cả hai vào (2 thật + 1 lặp), model
không đổi. Se / +P, chỉ đổi cách đọc dữ liệu:

| | mitdb Q | mitdb V | mitdb S | nstdb Q | nstdb V | nstdb S |
|---|---|---|---|---|---|---|
| `2m` duplicate | 99.90 / 99.86 | 95.64 / 94.10 | 45.79 / 61.17 | 96.42 / 83.72 | 83.42 / 67.11 | 78.88 / 23.94 |
| `2m` **native** | **99.91** / 99.86 | 95.12 / **96.21** | **56.87 / 65.67** | **96.79 / 84.30** | **85.28 / 71.23** | **80.43 / 31.11** |
| `100k_refined` duplicate | 99.90 / 99.84 | 94.29 / 95.31 | 41.49 / 69.41 | 96.12 / 83.32 | 83.16 / 74.44 | 69.96 / 31.15 |
| `100k_refined` **native** | **99.92** / 99.83 | 94.14 / **96.02** | **52.90** / 65.29 | **96.93 / 83.66** | **85.15 / 77.10** | **74.81** / 20.92 |

Trên nstdb `2m` lên **mọi** chỉ số; trên mitdb chỉ V Se −0.5. Giải phẫu theo record của `2m` cho
biết vì sao: +297 trong +304 nhịp S bắt thêm nằm ở **record 232** (366 → 663) — APC không đến sớm,
chỉ nhận ra bằng sóng P, thứ **V5 có mà MLII không có**; record 213 báo nhầm S 34 → **1**; record 108
báo nhầm Q 29 → 9. Chuyển đạo thứ hai chính là bằng chứng mà kiến trúc 3 chuyển đạo được xây để
dùng (mục 3), và chính sách nhân ba đã giấu nó đi.

Bảng đủ 5 database Physionet ở `native` (portal không đổi vì vốn đã 3 chuyển đạo thật). Se / +P;
**đậm** = tốt hơn `2m` duplicate:

| database · lớp | `2m` duplicate | `2m` native | ensemble `2m`+`1m`+`100k_ref` + min-run 2, native |
|---|---|---|---|
| mitdb · Q | 99.90 / 99.86 | **99.91** / 99.86 | **99.92 / 99.89** |
| mitdb · V | 95.64 / 94.10 | 95.12 / **96.21** | 94.66 / **97.13** |
| mitdb · S | 45.79 / 61.17 | **56.87 / 65.64** | **55.99 / 68.34** |
| nstdb · Q | 96.42 / 83.72 | **96.79 / 84.29** | **96.79 / 86.36** |
| nstdb · S | 78.88 / 23.94 | **80.43 / 31.13** | 75.97 / **39.12** |
| escdb · Q | 99.90 / 99.81 | **99.96 / 99.91** | **99.96 / 99.91** |
| escdb · V | 98.08 / 86.37 | 98.01 / **93.81** | 96.87 / **97.65** |
| escdb · S | 51.26 / 21.49 | **55.07 / 28.67** | **55.53 / 37.36** |
| ahadb · Q | 99.87 / 99.57 | **99.93 / 99.72** | **99.91 / 99.84** |
| ahadb · V | 89.21 / 97.25 | **89.31 / 98.24** | **89.79 / 99.47** |
| afdb · Q | 97.44 / 94.77 | **97.49 / 94.87** | **97.49 / 95.05** |

Số học của mốc Q 99.95 trên mitdb (83,978 QRS → tối đa **42 sót và 42 nhầm**): `2m` duplicate
82 / 121 → `2m` native 74 / 114 → ensemble native **65 / 93**. Phần còn lại nằm gần hết ở hai record
nhiễu: 203 (31 sót / 41 nhầm) và 105 (12 / 29) — riêng chúng đã là 43 sót và 70 nhầm, tức chính là
khoảng cách tới đích. Record 116 được `native` chữa (18 → 4 sót), record 108 được ensemble chữa
(29 → 3 nhầm); 203 và 105 thì chưa gì chữa được.

`2m native` là biến thể **gần nhất với "không chỉ số nào giảm"** trong toàn bộ vòng này: 30/32 ô lên,
hai ô xuống là V Se mitdb (−0.52) và escdb (−0.07). Q trên escdb chạm **99.96 / 99.91** — mốc
99.95 đã đạt ở một database. Ensemble native là biến thể +P mạnh nhất (mitdb Q 99.92 / 99.89, S +P
68.3 với Se 56) nhưng vẫn trả bằng Se của V và của S trên nstdb / v4-beat.

**Quyết định (20/09/2026): `EC57_LEAD_MODE` mặc định là `native`.** Ba lý do, tất cả đều có số:

1. Nó là cách dùng model đúng với dữ liệu sẵn có — kiến trúc 3 chuyển đạo được xây để đọc cả
   montage (mục 3), và mọi database EC57 đều có hai chuyển đạo thật nằm đó không dùng.
2. Nó **thêm bằng chứng**, không đổi ngưỡng: +297 trong +304 nhịp S bắt thêm nằm ở record 232, nơi
   APC không đến sớm chỉ nhận ra được bằng sóng P trên V5. Đây là thông tin, không phải đánh đổi.
3. Nó **bỏ đi một chỗ mong manh** thay vì che nó: ranh giới N/S ở xoang nhanh (record 213) là lưỡi
   dao khi chỉ có MLII — fine-tune nhiễu một epoch cũng lật nó từ 34 lên 632 báo nhầm (mục 8d) —
   và ổn định khi có V5 (34 → 1, và 632 → 6 cho chính checkpoint đã lật).

`duplicate` vẫn chạy được bằng `--lead-mode duplicate` và mọi bảng `duplicate` vẫn nằm trong
[`checkpoints/manifest.json`](checkpoints/manifest.json), vì nó là bài kiểm "một chuyển đạo"
khắt khe nhất và là đối chứng của phát hiện này.

### 8g. Kết luận vòng tuning (đến 20:45 ngày 18/09) — cái gì giao được, cái gì chưa

Hai deliverable, cả hai đều là model **đã có** trong [`checkpoints/`](checkpoints/), khác nhau ở
cách dùng:

| | `2m`, đọc `native` | ensemble `2m`+`1m`+`100k_refined`, min-run 2, `native` | `2m` + head `s_only`, `native` |
|---|---|---|---|
| lệnh | `ecgr ec57 --checkpoint checkpoints/resumamba_2m.keras --lead-mode native` | `ecgr ec57 --checkpoint checkpoints/resumamba_2m.keras checkpoints/resumamba_1m.keras checkpoints/resumamba_100k_refined.keras --min-run 2 --lead-mode native` | `ecgr ec57 --checkpoint checkpoints/resumamba_2m_sonly_head.keras --lead-mode native` |
| tính chất | **gần nhất với "không chỉ số nào giảm"**: 30/32 ô Physionet lên so với base, chỉ V Se mitdb −0.5, escdb −0.07 | **+P và Q mạnh nhất**: Q +P lên mọi database, V +P lên mọi database; trả bằng V Se (mitdb −1.0) và S Se (v4-beat −1.6, nstdb −2.9) | **+P của S cao nhất**: trả bằng S Se ở mọi nơi (mitdb −7, v4-beat −2.2, escdb −5) |
| mitdb Q Se / +P | 99.91 / 99.86 (74 sót / 114 nhầm) | **99.92 / 99.89** (65 / 93) | 99.91 / 99.86 |
| mitdb S Se / +P | **56.87** / 65.64 | 55.99 / **68.34** | 49.80 / **70.25** |
| escdb Q | **99.96 / 99.91** | **99.96 / 99.91** | **99.96 / 99.91** |
| v4-beat S | 85.86 / 90.91 | 84.22 / 91.36 | 83.62 / **92.85** |
| nstdb S +P · Q Se | 31.13 · 96.79 | 39.12 · 96.79 | **47.82 · 96.98** |

**Đối chiếu mục tiêu đặt ra:**

| mục tiêu | tốt nhất đạt được | đạt? |
|---|---|---|
| Q Se, +P tăng ở mọi tập | `2m native`: Q lên ở 5/5 database Physionet, cả Se và +P | ✓ (portal không đổi vì vốn 3 chuyển đạo) |
| mitdb Q Se > 99.95 | 99.92 (65 sót; đích ≤ 42) | ✗ — phần còn lại nằm ở record 203 (31) và 105 (12) |
| mitdb Q +P > 99.95 | 99.89 (93 nhầm; đích ≤ 42) | ✗ — 203 (41) và 105 (29) |
| mitdb S Se > 43 | 56.87 | ✓ |
| mitdb S +P > 80 | 70.25 (head `s_only`, native) · 68.34 (ensemble) | ✗ — hai record AF 222 / 219 gánh ~nửa số báo nhầm |
| v4-beat S Se > 88 · +P > 92 | Se 85.86 (`2m native`) · +P **92.85** (head `s_only`) | +P ✓ với head; Se ✗ ở mọi biến thể (tốt nhất 85.9) — hai đích này kéo ngược nhau trên họ base hiện tại |
| chỉ số khác không giảm | `2m native`: 2/32 ô giảm (V Se, ≤ 0.5) | gần — không tuyệt đối |

**Cái đã thử và không đủ**, mỗi cái có số đo trong các mục trên: SWA (8c — sụp S ở record 213),
ensemble ở `duplicate` (8c — Q Se nstdb giảm), đầu ngữ cảnh nhịp (8e — trượt dọc biên Pareto, AF
chỉ −15%), train lại có nhiễu theo công thức hiện tại (8d — Q sót giảm nhưng S đổi điểm vận hành,
record 213 mong manh).

**Ensemble chéo run** (`2m` cũ + `2m`-nhiễu chọn-bxb + `1m` + `100k_refined`, `native`, min-run 1) —
đã chấm, **không trội hơn** ensemble Track A: mitdb Q y hệt (99.92 / 99.89, 65 / 93) nhưng S
51.48 / 67.74 (Se −4.5: điểm mạnh 222 của thành viên nhiễu không sống qua phép trung bình, 222 về
258); đổi lại escdb S +P 37.4 → **43.4**, nstdb S +P 39.1 → **45.8**, ahadb V Se 89.8 → **91.9**,
v4-beat S +P **91.80**, `portal-eval` S 89.94 / 87.92 (lên cả hai). Một điểm khác trên biên, không
phải một bước lên. Ensemble Track A vẫn là ứng viên mitdb tốt nhất.

**Fine-tune không-bump** (8d) đã về: bác bỏ giả thuyết bump; xác nhận 213 chỉ mong manh ở `duplicate`.
Không thay đổi hai deliverable. **Còn chạy**: `30k` + nhiễu với bảng `native`, và bảng đủ 8 nguồn của
`2m` + head `s_only` ở `native` (lựa chọn "ưu tiên +P": mitdb S 49.8 / 70.4, nstdb S +P 47.8).

**Cái cần một vòng khác, không phải một nút chỉnh**: (1) Q ≤ 42 sót/nhầm trên mitdb — hai record
nhiễu 203/105 cần một model học được artefact thật hơn nhiễu tổng hợp hiện tại (nstdb có noise
records `em`/`ma`/`bw` nhưng dùng chúng để train là làm bẩn benchmark nstdb; cần nguồn nhiễu khác);
(2) S +P > 80 trên mitdb — cần một base không coi nhịp N trong rung nhĩ là "đến sớm", tức dữ liệu AF
với nhãn N được nhấn mạnh trong loss hoặc một đặc trưng độ đều nhịp tường minh đưa vào head.

### 8h. Bảng tổng hợp — mọi biến thể đã chấm, đối chiếu mục tiêu

Mười biến thể trên cùng dữ liệu, cùng cách chấm. `native` = 2 chuyển đạo thật + 1 lặp (mục 8f);
`mr2` = `--min-run 2`; `+head` = đầu tinh chỉnh `s_only` (mục 6f); `+noise bxb` = train lại có nhiễu,
checkpoint chọn bằng bxb (mục 6h/8d).

**mitdb** — mục tiêu: Q Se và +P > 99.95, S Se > 43, S +P > 80

| biến thể | Q Se | Q +P | V Se | V +P | S Se | S +P |
|---|---|---|---|---|---|---|
| `2m` (duplicate, gốc) | 99.90 | 99.86 | 95.64 | 94.10 | 45.79 | 61.17 |
| `2m` native | 99.91 | 99.86 | 95.12 | 96.21 | **56.87** | 65.64 |
| `2m`+head native | 99.91 | 99.86 | 95.08 | 96.28 | 49.80 | **70.25** |
| ens `2m`+`1m`+`100k_ref` mr2 (dup) | 99.90 | **99.90** | 95.27 | 95.48 | 45.10 | 65.57 |
| ens … native | **99.92** | 99.89 | 94.66 | **97.13** | 55.99 | 68.34 |
| ens chéo run native | **99.92** | 99.89 | 94.97 | 96.99 | 51.48 | 67.74 |
| `2m`+noise bxb native | **99.92** | 99.85 | 95.17 | 93.88 | 39.67 | 65.17 |
| `1m`+noise bxb native | 99.91 | 99.87 | **95.80** | 95.22 | 45.03 | 50.82 |
| `100k`+noise bxb native | **99.92** | 99.85 | 94.08 | 95.43 | 43.72 | 58.91 |
| `30k`+noise bxb native | 99.90 | 99.78 | 92.76 | 95.38 | 42.26 | 57.91 |

**dataset-v4-beat** — mục tiêu: S Se > 88, S +P > 92

| biến thể | Q Se | Q +P | V Se | V +P | S Se | S +P |
|---|---|---|---|---|---|---|
| `2m` native | 99.68 | 99.56 | 97.19 | 95.11 | **85.86** | 90.91 |
| `2m`+head native | **99.69** | 99.57 | 97.15 | 95.06 | 83.62 | **92.85** ✓ |
| ens … native | **99.69** | **99.60** | 97.02 | **95.65** | 84.22 | 91.36 |
| ens chéo run native | **99.69** | **99.60** | 97.14 | 95.42 | 84.14 | 91.80 |
| `2m`+noise bxb native | 99.67 | 99.53 | **97.67** | 94.52 | 83.96 | 91.72 |
| `1m`+noise bxb native | 99.66 | 99.51 | 96.98 | 94.94 | 85.32 | 89.57 |

**Train lại có nhiễu — kết luận trên cả bốn kích thước**: nó giúp `100k` (v4-beat S **cả hai chiều**
lên: 80.35/89.20 → 84.23/90.20) và giúp Q sót trên mitdb, nhưng **không** giúp `2m` (S Se 56.9 → 39.7)
và **hại** `30k` (v4-beat S 84.23/85.86 → 81.41/88.45, mitdb S Se 50.75 → 42.26). Không epoch nào của
bất kỳ kích thước nào hợp lệ theo quy tắc không-giảm trên `portal-eval` (0/10, 0/9, 0/11, 0/18). Công
thức nhiễu này là một **dịch điểm vận hành**, không phải một cải thiện — và các checkpoint cũ vẫn là
cơ sở cho ba deliverable ở mục 8g.

### 8i. `--s-boost` bão hoà — và vì sao mitdb S +P 80 không phải chuyện chỉnh ngưỡng

`s_boost` nhân xác suất S trước argmax, tức trượt model dọc đường cong Se/+P của chính nó mà
không train lại — đòn bẩy cuối cùng chưa thử. Quét trên hai biến thể +P tốt nhất (`native`), hiệu
chuẩn trên `portal-eval`, báo cáo mitdb:

| `s_boost` | portal-eval S Se/+P (F1) | mitdb S Se/+P | mitdb Q |
|---|---|---|---|
| 1.00 | 89.74 / 86.66 (88.17) | 55.99 / 68.34 | 99.92 / 99.89 |
| **0.75** | 88.52 / 88.21 (**88.36**) | 52.31 / 69.37 | 99.92 / 99.89 |
| 0.60 | 87.18 / 89.30 (88.23) | 49.33 / 70.59 | 99.92 / 99.89 |
| 0.45 | 85.60 / 90.95 (88.19) | 44.95 / **71.66** | 99.92 / 99.89 |

(ensemble `2m`+`1m`+`100k_ref`, `native`, min-run 2. Hiệu chuẩn đúng luật — chọn trên `portal-eval`
— cho **0.75**; Q không đổi ở mọi điểm vì `s_boost` chỉ đụng lớp S.)

**+P bão hoà quanh 71–72 ngay cả khi ép Se xuống sát 43.** Giải phẫu cho biết vì sao:

| `s_boost` | S TP | S FP | 222 | 219 | 215 | 200 | 202 |
|---|---|---|---|---|---|---|---|
| 1.00 | 1,537 | 712 | 260 | 94 | 62 | 67 | 74 |
| 0.45 | 1,234 | 488 | **220** | **69** | 37 | 23 | 44 |

Hạ ngưỡng 55% vẫn chỉ cắt 15% báo nhầm ở 222 và 27% ở 219; **59% số báo nhầm còn lại nằm ở hai
record rung/cuồng nhĩ đó**. Nghĩa là chúng không phải những lời gọi S lưỡng lự mà một ngưỡng gạt
được — model **tự tin** rằng nhịp ấy đến sớm, vì trong rung nhĩ khoảng R–R *thật sự* bất thường.
Số học: để đạt +P 80 ở Se 43 cần FP ≤ 295; điểm tốt nhất cho 488.

Thí nghiệm cuối cùng — **đưa head vào chính ensemble** (`2m`+head, `1m`, `100k_ref`, `native`) — chỉ
cho một điểm **nội suy** giữa hai model cha: mitdb S 53.30 / 69.17 (cha: 55.99/68.34 và 49.80/70.25),
Q 99.92 / 99.89. Không cộng hưởng. `--min-run 3` thay vì 2 nhích Q +P lên **99.90** (nhầm 87 → 85) mà
không đổi gì khác. Trên `portal-eval` tổ hợp này lại là điểm S cân nhất: 89.15 / 87.66, **F1 88.40** —
cao nhất mọi biến thể.

Kết luận: **mitdb S +P > 80 không đạt được bằng bất kỳ đòn bẩy sau-huấn-luyện nào** — SWA,
ensemble (5 tổ hợp), đầu tinh chỉnh (3 cấu hình × 2 chế độ chuyển đạo), `--min-run` (1/2/3),
`--lead-mode`, `s_boost` (4 điểm), và train lại có nhiễu trên cả 4 kích thước đều đã đo. Nó cần model học rằng *nhịp N trong
rung nhĩ không phải nhịp đến sớm*: một đặc trưng độ đều nhịp/AF tường minh đưa vào head, hoặc dữ
liệu AF có nhãn N được nhấn mạnh trong hàm mất mát. Đó là thay đổi ở mức bài toán, không phải ở
mức siêu tham số.

## 9. Test — [`tests/`](tests/)

```bash
./run_pipeline.sh test          # hoặc: python -m pytest tests/ -q
```

111 test, không cái nào cần dataset thật trừ [`test_ec57.py`](tests/test_ec57.py) (tự skip khi
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
│   ├── models/             layers.py (DiagSSM1D, AdaIN, RhythmDescriptor) · resumamba.py · refine.py
│   ├── training/           losses.py · callbacks.py · ssl.py · cpc.py · train.py · refine.py
│   ├── evaluation/         step_metrics.py · bxb.py · ec57.py · report.py
│   ├── rhythm/             rhythm theo từng giây + lead tốt nhất / NOISE theo cửa sổ (mục 12)
│   └── cli.py              `python -m ecgr <stage>`
├── evaluate.py             chấm EC57 + v4 beat-eval tại local, một lệnh (mục 7d)
├── tests/                  111 test, chạy ở đâu cũng được
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
| test | không có | **111** |

### Bất định của GPU — đã biết, đã đo

Một checkpoint vừa `fit` xong và cùng checkpoint đó nạp lại từ file cho ra dự đoán lệch tới
`3.8e-3` **dù trọng số giống nhau từng bit**: TF đẩy một phần đồ thị qua XLA, và quyết định gom
cụm khác nhau giữa một đồ thị đã trace cho train và một đồ thị chỉ để predict. Trên CPU, cùng
file nạp hai lần cho ra kết quả **giống nhau chính xác**.

Điều này không ảnh hưởng tính nhất quán của việc đánh giá — `stepeval` và `ec57` luôn nạp từ
file — nhưng nó là lý do một con số EC57 đo lại có thể lệch ở chữ số thập phân thứ hai.

## 12. Phân loại rhythm — [`ecgr/rhythm/`](ecgr/rhythm/)

Nhánh thứ hai, độc lập với model beat nhưng dùng chung tiền xử lý
([`signal_ops.py`](ecgr/signal_ops.py)) và kiến trúc ([`models/`](ecgr/models/)).

**Contract.** Vào `(2500, 3)` = 10 s × 3 lead @ 250 Hz. Hai output:

| output | shape | ý nghĩa |
|---|---|---|
| `rhythm` | `(10, 6)` | mỗi giây một softmax `SINUS, AFIB, SVT, VT, AVB2, AVB3` — cùng bộ lớp với project tham khảo `itr-ai-sensor_annotation-ae_ecg_classification` |
| `lead` | `(4,)` | cả cửa sổ 10 s một softmax: `0 = NOISE` (không lead nào đọc được), `1/2/3` = CH1/CH2/CH3 là lead có tín hiệu tốt nhất |

```bash
python -m ecgr.rhythm config
python -m ecgr.rhythm data                         # train / eval / test + audit holdout
python -m ecgr.rhythm train --model rhythm_1m [--backbone-from <beat checkpoint .keras>]
python -m ecgr.rhythm eval  --checkpoint <run>/checkpoints/resumamba_rhythm_1m/best_model.keras
python -m ecgr.rhythm predict --checkpoint ... --record <đường dẫn record, không đuôi>
```

Đường dẫn qua biến môi trường: `ECGR_RHYTHM_DATA_ROOT` (mặc định
`/media/MegaDataSet/DATA_4TINYML/Holter_report_strip`), `ECGR_RHYTHM_EVAL_DIR` (tập test,
mặc định `<DATA_ROOT>/dataset-eval/rhythm_eval`), `ECGR_RHYTHM_WORK_DIR`, `ECGR_RHYTHM_RUN_TAG`.

**Nhãn** ([`labels.py`](ecgr/rhythm/labels.py), [`inventory.py`](ecgr/rhythm/inventory.py)).
Như project tham khảo: loại event → lớp, vùng caliper `eventStartSample..eventStopSample` mang
lớp của event. Chia theo giây: một giây nhận lớp loạn nhịp khi bị phủ ≥ 50%; span ngắn hơn nửa
giây vẫn giữ được giây phủ nhiều nhất. Khác bản tham khảo ở ba chỗ, đều để nhãn đúng hơn:

- *Ngoài caliper* trên bản ghi 60 s: `SINUS` cho SVT/VT (cơn kịch phát trong nền xoang), nhưng
  **bỏ qua** (không tính loss) cho AFIB/AVB2/AVB3 — AF không dừng đúng chỗ caliper kết thúc.
- Strip ectopy (SINGLE_VE, *_COUPLET, *_BIGEMINY, …) được dùng làm `SINUS` — đúng loại âm tính
  khó nhất cho SVT/VT — và các run ≥ 3 nhịp S/V trong đó được **suy ra từ file `.atr`**. Cách này
  cũng định vị SVE_RUN/VE_RUN/SVT/VT của dataset-1/4 vốn không có caliper (một strip SVT của
  dataset-1 thường chỉ có nửa strip là S: `SSSSSSSSSSSSNNNNNNN`).
- Strip SINUS bị giới hạn 3 strip/study (chọn theo hash): ~200k strip xoang so với ~11k AFIB.
  Bộ `dataset-sinus` (TACHY/BRADY/PAUSE → SINUS; nguồn `sinus-2`, `sinus-3` là bản ghi 60 s có
  caliper, `sinus-4p`, `sinus-4` là strip 10 s) được **giữ trước** trong giới hạn này: đó là âm
  tính khó nhất — nhanh xoang vs SVT, chậm/pause vs AVB. Sau holdout, dedupe và giới hạn: 671 /
  14.476 / 1.177 / 14.261 event; `sinus-4` trùng 14.764 event ID với `dataset-4` và được tính
  một lần.
- **PTB-XL** ([`ptbxl.py`](ecgr/rhythm/ptbxl.py), nguồn `ptbxl`, `ECGR_PTBXL_DIR`): 21.799 bản
  ghi 12 lead × 10 s @ 500 Hz, không thuộc EC57 nên dùng để train là hợp lệ. Đây là nguồn duy nhất
  có các âm tính khó mà mitdb báo động giả — block nhánh, nhịp máy, WPW, nhanh/chậm xoang, ngoại
  tâm thu — portal chỉ có `OTHERS` cho chúng. Nhãn cho cả 10 s theo mã SCP: `AFIB → AFIB`,
  `AFLT → AFIB` (quy ước AFL = AF, từ 2026-09-29), `SVTAC/PSVT → SVT`, `2AVB → AVB2`,
  `3AVB → AVB3` (hai lớp khác nhau → bỏ); có `SVARR` → bỏ (không rõ lớp); còn lại → `SINUS` nhưng chỉ giữ khi có một trong
  `PACE, CLBBB, CRBBB, IVCD, ILBBB, IRBBB, WPW, STACH, SBRAD, SARRH, PVC, PAC, BIGU, TRIGU, 1AVB`
  hoặc nằm trong 3.000 bản ghi "xoang thường" chọn theo hash (`PTBXL_PLAIN_CAP`) — train đã có
  1,3 M giây SINUS. Chia theo bệnh nhân qua `strat_fold`: 1–8 train, 9 eval, 10 **không bao giờ
  đọc** (dự phòng). Study id = `10⁹ + patient_id` để không đụng study portal trong
  `studyids_*.npy`; `verify_split`/`audit_written` kiểm cả các id này. Mỗi bản ghi cho **2 cửa
  sổ** với 2 bộ 3-trong-12 lead khác nhau chọn theo hash `ecg_id` (bộ đầu luôn có một lead chi +
  một lead trước ngực, bộ sau ngẫu nhiên; trường `leads` trong event dict, `build.read_leads` chọn
  cột trước khi lọc), resample 500 → 250 Hz, lọc + z-score như strip portal. Giữ được: train
  8.773 bản ghi (SINUS 7.536 — trong đó 3.000 xoang thường —, AFIB 1.194, SVT 23, AVB2 10,
  AVB3 10) = 17.546 cửa sổ; eval 1.096 bản ghi (SINUS 939, AFIB 149, SVT 5, AVB2 1, AVB3 2);
  bỏ 205 AFLT/SVARR, 4 đa lớp, 9.523 xoang thường quá cap, 2.198 fold 10. Sau khi thêm: train
  186.435 cửa sổ, giây SINUS 1.479 M / AFIB 183,5 k / SVT 125,6 k / VT 29,1 k / AVB2 28,1 k /
  AVB3 13,1 k.

**Holdout.** Mọi study trong thư mục `rhythm_eval` (đọc từ tên thư mục *và* `studyId` trong
JSON) cùng list v4 bị loại **trước** khi chia; train/eval chia theo hash study
(`data/splits.study_split_side`). Kiểm tra trước khi ghi (`inventory.verify_split`) và kiểm lại từ
study id đã ghi ra đĩa (`build.audit_written`, `python -m ecgr.rhythm audit`). Đọc từ thư mục là
bắt buộc: 110/1.966 study của rhythm_eval **không** có trong list v4. Đo được: rhythm-2 bị loại
1.066/4.558 event, afib-2 1.833/7.962, dataset-5 6.698/41.011.

**Augmentation** ([`augment.py`](ecgr/rhythm/augment.py)), trên GPU theo batch: gain từng lead,
**đảo cực từng lead** (p = 0.3, `FLIP_LEAD_PROB` — nhịp không đổi khi lật ngược; SNR và kurtosis
chẵn theo dấu nên nhãn `lead`/NOISE không đổi, có test), rơi 1 lead (p = 0.2, `LEAD_DROP_PROB` —
mọi bản ghi Physionet EC57 chỉ có 2 lead thật + 1 lead 0), **đảo ngẫu nhiên thứ tự 3 kênh**, **nhiễu có cường độ ngẫu nhiên** — trộn
Gauss lọc 30 Hz / dao động 1–25 Hz / xung chuyển động — trên 1–3 lead, SNR mỗi lead rút đều trong
[-12, 18] dB, theo burst vị trí và độ dài ngẫu nhiên; thêm 12% cửa sổ hỏng cả 3 lead toàn cửa sổ.
Một lead *đọc được* trong một giây khi SNR ≥ 6 dB — tính chính xác từ nhiễu đã cộng.

- Nhãn `lead`: xếp hạng lead theo (1) số giây đọc được, (2) SNR cả cửa sổ (chặn ở 30 dB), (3) độ
  nhọn QRS (kurtosis, kSQI) — cái cuối phân định giữa các lead augmentation để sạch. Lead phẳng
  không bao giờ được chọn. `NOISE` khi lead tốt nhất đọc được < 8/10 giây. Nhãn được hoán vị
  **cùng** với kênh. Không dùng cột `channel` của reviewer: ở dataset-1/4/5 nó là CH2 cho 83–89%
  event — giá trị mặc định chứ không phải đánh giá chất lượng. Đo trên mẫu train: NOISE 14.6%,
  CH1/CH2/CH3 ≈ 28–29% mỗi kênh.
- Loss rhythm trên giây nhiễu (< 2 lead đọc được) nhân 0.25, và metric rhythm chỉ tính giây sạch.
- Tập eval dùng cùng nhiễu nhưng seed cố định, nên các epoch so sánh được.

**Model** ([`model.py`](ecgr/rhythm/model.py)): backbone ResUMamba nguyên bản (stem → ResU ‖
SSM chéo) ở 250 bước (40 ms), AdaIN theo context encoder, cross-attention với rhythm
descriptor (lấy max qua 3 lead — nhãn rhythm không thuộc về lead nào), rồi pool về 10 giây + một
khối SSM giữa các giây cho `rhythm`. Nhánh `lead` là một mạng conv nhỏ **dùng chung trọng số cho
từng lead** (Conv2D kernel `(1, k)` trên ảnh lead × thời gian): mỗi lead một điểm → logit CH1..CH3,
nên thứ hạng giữa các lead đổi chỗ đúng theo hoán vị (có test); logit NOISE đọc mean/max của 3
embedding lead + đặc trưng backbone. Bốn kích thước `rhythm_2m / 1m / 100k / 30k`
(1.960.608 / 941.176 / 99.589 / 29.983 tham số). `--backbone-from` nạp backbone từ checkpoint
beat cùng kích thước.

**UNet-Mamba, output theo mẫu** ([`unet.py`](ecgr/rhythm/unet.py), `--model rhythm_unet_2m /
1m / 100k / 30k`, 1.985.296 / 985.208 / 98.497 / 29.115 tham số, checkpoint
`unetmamba_rhythm_<size>`): input `(2500, 3)` → **`rhythm (2500, 6)`** + `lead (4,)`. Cùng cấu
trúc hai nhánh, thay nhánh ResU bằng **một U-Net đầy đủ**. Trunk chung: stem 2500 → pool 5 →
enc1 (500) → pool 2 → enc2 (250). Nhánh U-Net: pool 5 → enc3 (50, 200 ms) → pool 5 → bottleneck
(10 bước = 1 bước/giây, thấy cả strip) → up 5 ‖ skip enc3 → up 5 ‖ skip enc2 → (250, width).
Nhánh Mamba giữ nguyên (SSM hai chiều trên trunk 250 bước). Concat → conv 1 → AdaIN + rhythm
attention (dùng chung với ResUMamba) → **nửa trên U-Net** (`sample_head`): up 2 ‖ enc1 (500) →
up 5 ‖ stem (2500) → conv 1 softmax. Head lead dùng chung. Không có `--backbone-from`.

**UNet-Mamba 20 ms + noise** (`--model rhythm_unet500_2m / 1m / 100k / 30k`, 1.949.600 /
960.088 / 93.437 / 27.459 tham số, checkpoint `unetmamba500_rhythm_<size>`): input `(2500, 3)` →
**`rhythm (500, 6)`** (1 bước = 20 ms: SINUS / AFIB / SVT / VT / AVB2 / AVB3) + **`noise (5, 2)`**
(mỗi đoạn 2 s: CLEAN / NOISE). Cùng backbone UNet-Mamba; decoder dừng ở mức enc1 (up 2 ‖ enc1 →
500 bước), không lên tới stem. Head `lead` được thay bằng `noise_head`: conv chung trọng số cho
từng lead như nhánh lead, nhưng giữ 5 vị trí thời gian; mỗi đoạn đọc mean và max qua 3 lead
cộng đặc trưng backbone pool về 5 đoạn. Nhãn: rhythm = nhãn theo mẫu ở tâm mỗi khối 5 mẫu;
noise = đoạn 2 s là NOISE nếu một trong hai giây không đọc được (< `CLEAN_MIN_LEADS` lead đạt
`CLEAN_SNR_DB`, đúng định nghĩa đang đặt trọng số loss rhythm), sinh on-the-fly từ nhiễu augment.
Loss `noise` = CE theo đoạn, trọng số `NOISE_LOSS_WEIGHT = 0.5`; log `val_noise_f1` (F1 lớp NOISE)
và `val_noise_acc`. Predict/EC57: rhythm pool về 25 bước/giây, `p_noise` của mỗi bước = p(NOISE)
của đoạn 2 s chứa nó (cổng `DECODE_NOISE_THRESHOLD` như cũ).

**UNet-Mamba 8 ms** (`--model rhythm_unet1250_2m / 1m / 100k / 30k`, 1.949.600 / 960.088 /
93.437 / 27.459 tham số, checkpoint `unetmamba1250_rhythm_<size>`): như họ 500 nhưng trunk pool
(2, 5) thay vì (5, 2), nên enc1 nằm ở 1250 bước và head rhythm đi 250 → 1250 qua skip đó:
**`rhythm (1250, 6)`** + `noise (5, 2)`. Nhãn rhythm = nhãn theo mẫu ở tâm mỗi khối 2 mẫu. Lưu ý:
lưới chấm EC57 mặc định là 1 s (`EC57_GRID_HZ = 1`), nên biên 8 ms chỉ có giá trị khi sản phẩm
cần biên dưới 1 s hoặc khi chấm với `ECGR_RHYTHM_EC57_GRID_HZ=25`; chưa có so sánh 500 ↔ 1250
cùng dữ liệu cùng seed.

**UNet-Mamba 8 ms + beat decoder** (`--model rhythm_unet1250b_2m / 1m / 100k / 30k`, 1.974.796 /
983.108 / 99.181 / 24.347 tham số, checkpoint `unetmamba1250b_rhythm_<size>`, 2026-10-06): một
encoder chung, hai decoder. **Beat decoder** (`unet.beat_head`): up 250 → 1250 ‖ enc1 → 2×conv →
**`beat (1250, 4)`** softmax none / N / S / V mỗi 8 ms, nhãn từ `.atr` theo AAMI (L, R, B, e, j → N;
A, a, J, S → S; V, E → V; F, Q, nhịp máy → IGNORE ±0,1 s; bản ghi không có `.atr` → IGNORE cả
cửa sổ: PTB-XL, challenge-2020); mỗi nhịp phủ ±1 bước quanh mẫu R (`labels.beat_labels`,
`augment.beat_targets`), loss CE trọng số `BEAT_CLASS_WEIGHTS = (0.15, 1, 4, 3)` ×
`BEAT_LOSS_WEIGHT = 0.5`, metric `val_beat_f1` = macro F1 trên N/S/V theo bước. **Rhythm decoder
được điều kiện hoá theo beat** qua stop-gradient (`StopGradient`, loss rhythm không kéo head
beat): xác suất beat max-pool về lưới 250 bước ghép vào đặc trưng gộp → conv 1 → một khối SSM
nữa (`beat_ctx`); và `RhythmDescriptor` chạy trên **chuỗi xác suất nhịp** p(beat) = 1 − p(none)
(tự tương quan của chuỗi xung nhịp chính là cấu trúc RR: đỉnh sắc = đều, phẳng = AF) → thêm
`beat_tokens = 4` token vào rhythm attention bên cạnh 4 token từ đường bao tín hiệu. Build ghi
thêm `beats_<k>.npy` (n, 2500); pipeline chỉ nạp khi model có output `beat`. Lý do: FP VT/SVT trên
mitdb là các dự đoán tự tin trên BBB (207, 212) và nhanh xoang (215) — nhãn N cho nhịp block
nhánh và chuỗi S/V thật là tín hiệu đúng định nghĩa "≥ 3 nhịp V/S liên tiếp". Giai đoạn B (chưa
làm): cắt đoạn TQ giữa hai R và khử template QRST để đọc hoạt động nhĩ (F/f/P).

Hệ quả của output theo mẫu, xuyên suốt pipeline:
- **Nhãn**: `build` ghi thêm `labels_samples_<k>.npy` `(n, 2500)` (`labels.sample_labels`): mẫu
  trong span loạn nhịp mang lớp đó (chồng lấn: ưu tiên như decode, VT > SVT > AFIB > AVB3 >
  AVB2), mẫu trong vùng đã biết là SINUS, còn lại IGNORE — không ngưỡng phủ 50 % như theo giây.
  `labels_<k>.npy` theo giây giữ nguyên cho ResUMamba. Train/eval tự chọn loại nhãn theo shape
  output (`model.is_per_sample`); trọng số "giây sạch" lặp ra 250 mẫu (nhiễu đo theo giây).
- **`val_rhythm_f1` khi train là theo mẫu**, không so trực tiếp với model theo giây; so công
  bằng bằng cách trung bình 250 mẫu mỗi giây rồi chấm với nhãn theo giây.
- **Predict / EC57**: xác suất toàn bản ghi giữ ở `SAMPLE_PROBS_HZ = 25` bước/giây (trung bình
  mỗi 10 mẫu; file 10 h afdb ~11 MB). `.npz` lưu `step_hz` (file cũ = 1). `decode_episodes(...,
  step_hz)` đổi mọi tham số giây sang số bước, nên `DECODE_*` giữ nguyên nghĩa; episode có biên
  giây lẻ.
- **Lưới chấm EC57**: file annotation từng lớp (ref lẫn hyp) viết trên lưới
  `EC57_GRID_HZ` (env `ECGR_RHYTHM_EC57_GRID_HZ`, mặc định 1 = giây nguyên, đúng như mọi bảng
  trước — đã kiểm tra decode lại v2 ra đúng số cũ). Tăng lên (vd 25) để biên dưới 1 s được tính;
  khi đó chấm lại cả model theo giây trên cùng lưới.

**Eval** ([`evaluate.py`](ecgr/rhythm/evaluate.py)) trên tập test rhythm_eval: confusion theo
giây (Se/+P/F1 từng lớp, macro F1), theo strip 10 s (argmax thô, **không** hậu xử lý — xem ghi
chú dưới), và confusion 4×4 của
output `lead` (accuracy, Se/+P của NOISE, tỷ lệ chọn cùng lead khi cả hai đều coi là đọc được) —
ở dạng nguyên bản và với nhiễu cố định 18/12/6/0/−6 dB. `predict` ghi CSV theo giây (rhythm +
p_noise) và JSON gồm episode và lead tốt nhất của từng cửa sổ. Test:
[`tests/test_rhythm.py`](tests/test_rhythm.py) (23 test, không cần dữ liệu thật).

**EC57 thật** ([`ec57.py`](ecgr/rhythm/ec57.py), [`evaluation/epicmp.py`](ecgr/evaluation/epicmp.py)).
`evaluate.py` ở trên chỉ đo trên tập npy/tfrecord tự lưu của chính model — không phải một phép đo
độc lập. `ecgr.rhythm ec57` là phép đo đó: chạy model trên bản ghi WFDB thật, rồi so bằng chính công
cụ EC57 (`epicmp` + `sumstats`, không phải `bxb` — `bxb` so từng nhịp, `epicmp` so từng **episode**
loạn nhịp) — giống hệt cách `evaluation/ec57.py` dùng `bxb` cho model beat, chỉ đổi công cụ.

```bash
python -m ecgr.rhythm ec57 --checkpoint <run>/checkpoints/resumamba_rhythm_1m/best_model.keras \
    --tag mytag [--dbs mitdb afdb escdb] [--classes AFIB VT] [--max-records N] \
    [--skip-physionet] [--skip-rhythm-eval]
python -m ecgr.rhythm ec57 --tag mytag --decode-only          # giải mã lại từ npz, chạy lại epicmp
python -m ecgr.rhythm ec57 --tag mytag --decode-only --sweep  # quét tham số giải mã (mục dưới)
```

Ba tầng, tầng nào cũng chạy lại được riêng:

1. **Inference — một lần duy nhất.** Xác suất theo giây của mỗi bản ghi lưu vào
   `<EC57_DIR>/<tag>/_ann/<db>/<record>.npz` (`rhythm (T,6)` + `p_noise (T,)` float16, `fs`,
   `sig_len`). Đây là tầng tốn kém duy nhất.
2. **Giải mã** (`labels.decode_episodes`, tham số `rc.DECODE_*`) từ npz → file `.rhi` gộp mọi lớp
   (đọc bằng `rdann`) **và một file hypothesis riêng cho từng lớp**. `--decode-only` (alias cũ
   `--predict-only`) làm lại tầng này và tầng sau, không cần model.
3. **epicmp, một lượt cho mỗi (database, lớp)**, trong thư mục symlink dùng một lần
   `_work/<db>/<lớp>/`; báo cáo `<db>/<db>_<lớp>_report_line.out`.

**Vì sao chấm theo từng lớp.** `epicmp -A` chỉ so **một** nhịp: các episode có `aux_note` là
`(AFIB` (và biết `(AFL` phía tham chiếu). Bản trước ghi cả sáu lớp vào một file (`(AFIB/(SVTA/
(VT/…`) và chạy `epicmp -A` một lần — nên chỉ AFIB được chấm, SVT/VT/AVB2/AVB3 chưa bao giờ được
đo. Nay với mỗi lớp C, `wfdb_ann.write_class_annotations` ghi một cặp ref/hyp trong đó giây thuộc C
viết `(AFIB`, mọi giây khác (`SINUS`, lớp khác, `NOISE`, giây không ai gán) viết `(N`, một dấu `+` ở
giây 0 và tại mỗi lần đổi mã — đúng cách nhánh `dict_ext` của project tham khảo
`compare_HES_IES_ATR/utils/ec57_eval.py` làm. Đuôi file chỉ gồm chữ (wfdb-python từ chối `_` và
chữ số): `r<lớp>` / `a<lớp>` — `rafib/aafib`, `rsvt/asvt`, `rvt/avt`, `ravbii/aavbii`,
`ravbiii/aavbiii` (`rc.class_extensions`).

Hai nguồn tham chiếu:

- **Physionet** (`rc.EC57_DEFAULT_DBS` = mitdb, afdb, escdb; `--dbs` thêm nstdb/ahadb nhưng chúng
  không có nhịp nào chấm được). Bốn bản ghi nhịp máy của mitdb (102, 104, 107, 217 —
  `rc.EC57_EXCLUDE_RECORDS`) bị **loại mặc định**, in ra khi chấm; `--include-paced` giữ lại. EC57
  cũng loại chúng khỏi chấm beat, và với rhythm chúng chỉ sinh báo động giả AFIB/VT không có TP nào
  để đổi. Đọc dấu `+` trong `.atr`, ánh xạ mã → lớp qua
  `rc.PHYSIONET_AUX_TO_CLASS`, **chặt**: AFIB = `(AFIB` (riêng `(AFL` giữ nguyên trong file ref
  của bài AFIB), SVT = `(SVTA` (không tính `(NOD`, `(J`, `(PREX`),
  VT = `(VT` (không tính `(VFL`, `(IVR`), AVB2 = `(BII`, AVB3 = `(B3` (chỉ escdb có; mitdb không
  có block độ 3). Mã khác → `(N`. Cặp (lớp, db) chấm mặc định theo `rc.EC57_CLASS_DBS`: AFIB
  mitdb+afdb, SVT/VT mitdb+escdb, AVB2 mitdb — đúng bảng của sản phẩm tham khảo, escdb là phần
  thêm. Db không có episode tham chiếu của lớp → cột Se là `-`, không lỗi.
- **Tập test riêng của model rhythm** (`rc.EVAL_DIR`, chấm cả 5 lớp) — **không** có annotation
  WFDB cho nhịp, chỉ có nhãn theo giây suy ra từ caliper của reviewer (`labels.second_labels`,
  giống hệt lúc build dữ liệu train). Tham chiếu được **tổng hợp**: nhãn theo giây → episode
  (`labels.reference_episodes`, bỏ qua giây `IGNORE` mà không cắt đoạn) → cùng bộ ghi theo lớp ở
  trên (+ `.rhy` gộp để đọc). Bản ghi 10–60 s nên dùng `epicmp -f 0` (`epicmp-script2.sh`).

`start`/`stop` của episode tính theo **giây**, nên đổi sang sample chỉ là `second * fs` của bản
ghi gốc. Báo cáo: `report.summarize_episodes` gom mọi `<db>_<lớp>_report_line.out` thành
`rhythm_ec57_summary.csv` (một dòng mỗi db × lớp: `E_Se E_+P E_F1 D_Se D_+P D_F1`, F1 =
2·Se·PPV/(Se+PPV) từ hai token của dòng Gross; thêm `.xlsx` nếu có `openpyxl`) và in bảng
lớp × (Duration | Episode) × db × Se/PPV/F1 kèm dòng `target` của sản phẩm tham khảo
(`ec57.TARGET_TABLE`, "rhythm 3.0.6").

**Quy ước AFL = AF** (chốt 2026-09-29, `rc.EC57_AFL_AS_AF`, env `ECGR_RHYTHM_EC57_AFL_AS_AF=0`
để tắt): cuồng nhĩ được tính là AF. Train: PTB-XL `AFLT → AFIB` (+49 bản ghi train / 7 eval).
Chấm: lượt epicmp của lớp AFIB chạy với **`-x`** (EC38:1998) — `(AFL` của tham chiếu bị loại
khỏi so sánh +P, nên gọi AFIB trong đoạn cuồng nhĩ không còn là FP; AFL vẫn không bắt buộc cho
Se. Các lượt SVT/VT/AVB2 không đổi (AFL ở đó vốn là `(N`). Mặc định EC57/EC38:2007 (không `-x`)
thì khác: AFIB trong AFL là FP — mọi bảng trước 2026-09-29 là theo mặc định đó. Ví dụ đo trên
v2: 40 % giây FP AFIB mitdb (678 / 1.676) là `(AFL`.

**Hậu xử lý khi giải mã** (`labels.decode_episodes`, để tín hiệu liền mạch thay vì vụn). Chạy
**một lần trên toàn bản ghi**, sau khi mọi cửa sổ 10 s đã qua model và được ghép thành chuỗi
xác suất theo giây (`predict.predict_signal`: bước 10 s, cửa sổ cuối kéo lùi, giây bị phủ hai
lần lấy trung bình) — nên cầu nối, thời lượng tối thiểu đều xuyên qua ranh giới cửa sổ. Không
bao giờ áp dụng trên một cửa sổ 10 s rời (`ecgr.rhythm eval` theo strip dùng argmax thô).
Không tham gia vào train: loss/metric train là mức giây, thô. Theo thứ tự:

1. trung bình trượt xác suất **theo từng lớp**, cửa sổ `DECODE_SMOOTH_SECONDS[lớp]` (lẻ; 1 =
   tắt; SINUS giữ nguyên; một số thay cho dict = cùng cửa sổ cho mọi lớp);
2. argmax; giây có `p_noise > DECODE_NOISE_THRESHOLD` → `NOISE`;
3. **nối khoảng trống** từng lớp theo ưu tiên `DECODE_PRIORITY = VT > SVT > AFIB > AVB3 > AVB2`:
   hai episode cùng lớp cách nhau ≤ `DECODE_MERGE_GAP_SECONDS[lớp]` giây mà khoảng giữa chỉ
   gồm SINUS / NOISE / lớp ưu tiên thấp hơn thì nối thành một; không bao giờ ghi đè lớp ưu tiên cao;
4. **thời lượng tối thiểu** `DECODE_MIN_EPISODE_SECONDS`: episode ngắn hơn ngưỡng của lớp nhận
   lớp của hai láng giềng khi chúng trùng nhau (2 s SVT lọt giữa AFIB → AFIB), ngược lại → SINUS;
5. nối khoảng trống lần nữa (bước 4 có thể tạo hai đoạn cùng lớp sát nhau).

**Huấn luyện v2 (2026-10-06, sau khi run 1250b từ đầu bị dừng ở epoch 11).** Chẩn đoán: beat
decoder đã hội tụ theo nghĩa cần (3 bản ghi validation: QRS Se/+P 100/100 ở dung sai 150 ms,
loại N/S/V đúng 99.9 %) nhưng `val_beat_f1` theo bước 8 ms chỉ 0.55; rhythm bám đúng đường
cong baseline. Thay đổi (`train.py`, `pipeline.py`, `augment.py`, `objectives.py`):
- **Đích beat tách "ở đâu" và "là gì"**: target `(1250, 6)` = `[heat | N S V | w_heat | w_type]`;
  heat = Gauss σ = `BEAT_HEAT_SIGMA_STEPS` (20 ms) quanh R, loss BCE trên p(beat) = 1 − p(none)
  (dương × `BEAT_HEAT_POS_WEIGHT`); N/S/V one-hot trong ±`BEAT_TYPE_RADIUS_STEPS` (40 ms), loss
  CE trên xác suất N/S/V chuẩn hoá lại, chỉ trên các bước đó. `val_beat_f1` giờ là F1 loại
  trên vùng beat; **`BeatMatchLog`** ghép beat (`pick_beats`) với R tham chiếu ở 150 ms trên
  `BEAT_MATCH_WINDOWS` cửa sổ eval mỗi epoch → `val_beat_se / val_beat_pp / val_beat_type_f1`
  (đúng cách bxb chấm).
- **Batch phân tầng** (`SAMPLER=stratified`, `--sampler`): loại cửa sổ = lớp loạn nhịp hiếm nhất có
  ≥ `STRAT_MIN_SECONDS` (ưu tiên VT > AVB3 > AVB2 > SVT > AFIB), mỗi batch 64 = SINUS 24, AFIB 12,
  SVT 10, VT 7, AVB2 6, AVB3 5 (`STRAT_BATCH_QUOTA`), lớp lặp quá `STRAT_MAX_REPEAT` lần/epoch
  trả phần dư cho SINUS. Trọng số lớp chia cho √(tỉ lệ lấy mẫu dày) (`train.class_weights`),
  không bao giờ tăng; trọng số hiệu dụng lưu trong `train_config.json` và `ec57.prior_scale`
  đọc nó qua `ECGR_RHYTHM_TRAIN_CONFIG`.
- **Warm-start** `--init-matching <ckpt>`: mọi layer trùng tên và hình nhận trọng số (từ
  `rhythm_unet1250_1m` sang `rhythm_unet1250b_1m`: 169/271 tensor — encoder, rhythm decoder,
  noise head); `--freeze-epochs N` train riêng các layer mới ở `--lr`, rồi mở toàn bộ ở `--lr2`.
- **Lịch LR cosine** (`LR_SCHEDULE=cosine`, AdamW wd 1e-4, warm-up 1 epoch, sàn 2 %) thay
  plateau; `LogLR` ghi LR vào history.csv.
  Với `rhythm_unet1250b_1m` (filter U-Net thu nhỏ 28/44/56/72) chỉ 169/271 tensor khớp (U-Net
  học lại từ đầu), nên thêm **`rhythm_unet1250b_1mw`**: filter U-Net giữ nguyên 32/48/72/96 của
  `rhythm_unet1250_1m` → 232/271 tensor khớp, chỉ 15 lớp beat mới; 1.085.884 tham số (vượt
  ngân sách 1m một chút, budget riêng 1.3 M).
Run `061026_rhythm_u1250b_v2` (`rhythm_unet1250b_1mw`): 22 epoch, 2 epoch đóng băng ở 1e-3
(29 tensor beat), rồi mở toàn bộ ở 2e-4 cosine.

**Tầng hậu xử lý theo nhịp (beat)** (`rhythm/beats.py`, 2026-10-06, chỉ với model có đầu ra
`beat` — họ `rhythm_unet1250b_*`; tắt bằng `ECGR_RHYTHM_BEAT_PP=0` hoặc `ec57 --no-beat-pp`).
Làm lại tầng 2 của pipeline Holter sản xuất (tài liệu `rhythm-post-process-analysis.md` của
thư viện Bioflux) trên chính beat decoder của model, sửa các lỗi tài liệu đó liệt kê. Luồng:
`predict_signal` trung bình xác suất beat (125 Hz) qua các cửa sổ chồng → `pick_beats` (cực đại
cục bộ của 1 − p(none) > `BEAT_PICK_THRESHOLD`, khoảng trơ `BEAT_REFRACTORY_SECONDS`, nhãn N/S/V
= argmax ±2 bước) → lưu `beat_t/beat_cls/...` trong cùng `.npz` (`ec57.load_beats`) →
`ec57.decode_record`: `labels.decode_track` (5 bước trên) → `beats.postprocess` →
`episodes_from_track`. `postprocess`: (1) mỗi beat nhận lớp đa số của track trong ô của nó (biên
= trung điểm hai R); (2) chuỗi V (S) có bất kỳ beat VT (SVT) → cả chuỗi VT (SVT); (3) kiểm tra
tiêu chí `BEAT_PP_CRITERIA` từng vùng: AFIB ≥ 3 s; VT ≥ 3 beat, ≥ 3 V liên tiếp, HR ≥ 100
(lỗi A, J: số beat có hiệu lực, nhịp tự thất chậm không là VT); SVT ≥ 3 beat, HR ≥ 100 và
(≥ 3 S liên tiếp, hoặc ≥ 30 % S, hoặc HR nhảy ≥ 1.25× so với 4 beat trước — SVT kéo dài mất
tính "sớm" nên beat là N); AVB2/3 ≥ 2.5 s, HR ≤ 60 (lỗi F: chặn trên, không phải ≤ 40);
SINUS < 5 s coi là không hợp lệ; (4) vùng không hợp lệ nhận lớp láng giềng theo luật tường
minh (hai bên giống nhau → lớp đó; hai bên đều VT/SVT/AVB → SINUS; một bên → bên kia; còn
lại theo `BEAT_PP_PRIORITY` AFIB > SINUS > NOISE; vùng dài ≥ `BEAT_PP_LONG_INVALID_SECONDS`
→ SINUS — lỗi E, H: không xếp theo ID lớp, không dùng ¼ bản ghi); (5) tách VT/SVT theo beat:
beat không phải V trong VT (không phải S và R-R > `BEAT_PP_FAST_RR_SECONDS` trong SVT) nhận lớp
**nền xung quanh** (AFIB nếu kề AFIB, không mặc định SINUS — lỗi D), mảnh còn lại kiểm tra
lại tiêu chí; (6) tuỳ chọn trọng tài AFIB/SVT theo cửa sổ (`BEAT_PP_AFIB_SVT_MERGE`, mặc định
tắt; cửa sổ đóng hai phía — lỗi B, G); (7) beat S trong AFIB → N. Bước NOISE của track giữ
nguyên; bản ghi < 3 beat trả về track cũ. Hypothesis beat (N/S/V tại R, phần mở rộng
`RHYTHM_BEAT_AI_EXTENSION = bti`) được chấm bxb/sumstats so với `.atr` (`ec57.score_beats`,
báo cáo `<tag>/beats/<db>/<db>_QRS_report_line.out`) — báo cáo AAMI beat cho các DB
PhysioNet. Tiêu chí chọn trên validation (`val_beat_grid.py` trong scratchpad: lưới
HR VT/SVT × tỉ lệ S × ngưỡng khởi phát × SINUS tối thiểu × ngưỡng vùng dài, mục tiêu tổng
F1 episode + duration các ô ltafdb, giữ ngân sách AF giả 0.4/h), EC57 chấm một lần.
Test: `tests/test_rhythm_beats.py` (chuỗi tổng hợp SINUS|SVT|AFIB, AFIB|SINUS<5 s|AFIB,
VT|N|VT trong AFIB, 2 V không là VT, blip giữa SVT và VT → SINUS, ...).

**Giá trị mặc định (2026-09-29)**, chọn bằng `--sweep` (`ec57.sweep_decoding`: giải mã lại từ
npz cho từng tổ hợp smoothing {1, 3, 5, `split`} × khoảng trống {0, 1×} × bộ thời lượng tối thiểu
{`user`, `beats`, `baseline`}, một dòng mỗi tổ hợp trong `decode_sweep.csv`; mục tiêu = trung bình
F1 Duration+Episode trên các ô `rc.EC57_TARGET_CELLS`, rhythm_eval là điều kiện chặn):
`DECODE_SMOOTH_SECONDS = {AFIB 5, AVB2 5, AVB3 5, SVT 1, VT 1}`, `DECODE_MERGE_GAP_SECONDS = 0`
(không nối), `DECODE_MIN_EPISODE_SECONDS = {AFIB 4, SVT 1.5, VT 1, AVB2 6, AVB3 3}`,
`DECODE_NOISE_THRESHOLD = 0.5`. Lý do, đo trên mitdb (loại paced): độ dài episode tham chiếu
**(VT trung vị 1,8 s — 51/60 dưới 3 s**, (SVTA 2,4 s (16/26 dưới 3 s), (AFIB 25/83 dưới 7 s, còn
(BII cả 5 đều ≥ 7 s. Quy tắc cũ "AFIB ≥ 7 s, SVT/VT ≥ 3 s (3 nhịp)" xoá về mặt cấu trúc 85 %
episode VT tham chiếu — 3 nhịp VT ở 150–200 bpm chỉ là ~1 s — nên VT Episode Se kẹt ở 16–18 % dù
Duration Se 60 %; và lưới sweep cũ (độ lệch −1/0/+2) không bao giờ thử min 1 s. Làm mượt 5 s
cũng xoá cơn 2 s (trung bình 5 s kéo xác suất VT xuống dưới xoang) → mượt theo lớp. AVB2 tăng lên
6 s vì FP AVB2 quanh pause đều ngắn. Nối khoảng trống làm giảm mục tiêu ở mọi cặp cấu hình (52,7
so với 55,2) và không giúp rhythm_eval → bỏ. Kết quả trên UNet500 (`pp_u500`, AFL = AF): mục tiêu
Physionet 51,4 → 55,2; mitdb VT Episode F1 **20,6 → 47,2**, AFIB Episode 68,5 → 71,9, AVB2
63,0/13,1 → 65,8/14,8; rhythm_eval Episode F1 SVT +3,4, VT +9,5, AVB3 +1,5, AFIB −0,6. Bộ cũ
còn trong sweep dưới tên `user` (`ec57.USER_MIN_SECONDS / USER_MERGE_GAP`). Ngưỡng nhiễu không
đổi kết quả trên Physionet (`p_noise > 0.5` ở ~0,01 % số giây) — giữ cho dữ liệu Holter thật.

Lịch sử: sweep đầu (tag `pp_v1`, 2026-09-24, lưới smoothing {1,3,5} × ngưỡng nhiễu × khoảng trống
{0, 1, 1.5×} × độ lệch min {−1, 0, +2}) chọn mượt 5 s + bộ `user`; kết luận "min không quan trọng"
khi đó là do lưới chưa chạm vùng 1 s.

**UNet500 với decode mặc định mới so với mục tiêu** (`290926_rhythm_unet500`, epoch 25, val
rhythm F1 0,830 / noise F1 0,955; AFL = AF; F1 Duration / Episode, trong ngoặc Se / PPV):

| Ô | Decode cũ | **Decode mới** | Mục tiêu |
|---|---|---|---|
| AFIB afdb | 95,4 / 92,4 | 94,8 (91/99) / **92,9** (90/96) | 97,5 / 86,6 → Episode vượt |
| AFIB mitdb | 82,4 / 68,5 | 81,7 (77/87) / 71,9 (75/69) | 93,8 (98/90) / 85,6 (80/92) |
| SVT mitdb | 18,2 / 21,5 | 19,6 (40/13) / 20,1 (**62**/12) | 34,4 (61/24) / 29,1 (62/19) |
| VT mitdb | 38,6 / 20,6 | 42,7 (**74**/30) / **47,2** (65/37) | 64,8 (75/57) / 67,9 (82/58) |
| AVB2 mitdb | 63,0 / 13,1 | 65,8 (100/49) / 14,8 (100/8) | 92,2 (98/87) / 100 |
| VT escdb | 9,2 / 5,8 | **44,3** (30/85) / **55,2** (40/89) | – |
| rhythm_eval AFIB / SVT / VT / AVB2 / AVB3 (Episode) | 84,2 / 73,0 / 62,4 / 70,7 / 46,2 | 83,6 / 76,4 / 71,9 / 71,2 / 47,7 | – |

Sau decode mới, **Se đã đạt hoặc vượt mục tiêu ở SVT Episode (62), VT Duration (74) và AVB2
(100)**; khoảng cách còn lại ở mọi ô mitdb là **PPV** (SVT 12–13 so với 19–24, VT 30–37 so với
57–58, AVB2 8–49 so với 87–100, AFIB Episode 69 so với 92) — tức là dương tính giả, chủ yếu từ
các bản ghi 215 (nhanh xoang), 232 (chậm xoang/pause), 207 (BBB + `(VFL`), 200/203 (xoang nhiều
ngoại tâm thu). Phần việc tiếp theo là dữ liệu âm tính khó và ngữ cảnh > 10 s, không phải decode.

**v1 → v2** (`260924_rhythm_v2`, fine-tune 20 epoch từ v1 với lr 3e-4 trên dữ liệu có PTB-XL, đảo
cực p = 0.3, rơi lead p = 0.2; best epoch 17). Trên **cùng** tập eval mới (PTB-XL fold 9 thêm vào,
augment mới, `noisy`): val_rhythm_f1 v1 0.791 → v2 0.828; F1 theo giây từng lớp v1 → v2: SINUS
96.8 → 97.6, AFIB 82.9 → 86.9, SVT 80.5 → 83.1, VT 77.7 → 82.4, AVB2 73.6 → 77.9, AVB3 63.2 → 68.8.
EC57 (tag `pp_v1_nopaced` / `pp_v2`, cùng giải mã mặc định, cùng loại 4 bản ghi nhịp máy), Se/PPV/F1:

| lớp · db | | v1 Duration | v2 Duration | target D | v1 Episode | v2 Episode | target E |
|---|---|---|---|---|---|---|---|
| AFIB | mitdb | 61/65/62.9 | 76/75/**75.5** | 98/90/93.8 | 65/53/58.4 | 75/60/**66.7** | 80/92/85.6 |
| AFIB | afdb | 87/93/89.9 | 92/97/**94.4** | 96/99/97.5 | 93/81/86.6 | 93/89/**91.0** | 81/93/86.6 |
| SVT | mitdb | 48/8/13.7 | 38/8/13.2 | 61/24/34.4 | 50/10/16.7 | 38/12/18.2 | 62/19/29.1 |
| SVT | escdb | 3/0/0.0 | 6/1/1.7 | | 5/0/0.0 | 9/1/1.8 | |
| VT | mitdb | 56/3/5.7 | 58/32/**41.2** | 75/57/64.8 | 18/5/7.8 | 16/29/**20.6** | 82/58/67.9 |
| VT | escdb | 13/0/0.0 | 10/44/16.3 | | 11/1/1.8 | 7/62/12.6 | |
| AVB2 | mitdb | 100/25/40.0 | 100/42/**59.2** | 98/87/92.2 | 100/3/5.8 | 100/7/13.1 | 100/100/100 |
| AFIB | rhythm_eval | 90/59/71.3 | 90/61/72.7 | | 97/69/80.6 | 97/76/85.2 | |
| SVT | rhythm_eval | 56/66/60.6 | 56/69/61.8 | | 77/72/74.4 | 77/77/77.0 | |
| VT | rhythm_eval | 56/37/44.6 | 57/60/58.5 | | 61/42/49.7 | 61/69/64.8 | |
| AVB2 | rhythm_eval | 76/36/48.9 | 78/38/51.1 | | 93/50/65.0 | 94/57/71.0 | |
| AVB3 | rhythm_eval | 65/28/39.1 | 62/34/43.9 | | 76/32/45.0 | 75/41/53.0 | |

Báo động giả theo giây trên mitdb (không nhịp máy) v1 → v2: AFIB 2.202 → 1.676 (giây có nhịp
L/R 779 → 326; phần còn lại chủ yếu là `(AFL` 495 → 678 — flutter bị gọi là AFIB, bản ghi 222/203),
SVT 1.396 → 1.063 (bản ghi 215 nhanh xoang 922 → 906: **chưa giải quyết**), VT 4.258 → 301
(bản ghi 112: 1.770 → 0; nhịp L/R 486 → 79; còn lại gần nửa là `(VFL` của 207), AVB2 2.359 → 1.058
(nhịp L/R 1.076 → 231; bản ghi 232 chậm xoang/pause 928 → 861: **chưa giải quyết**). Quét giải mã
lại trên v2: điểm tốt nhất (smooth 3, không nối, min {6,2,2,1,1}) 51.6 so với 49.3 của mặc định
(+2.3, v1 là 37.2) nhưng rhythm_eval mất AFIB E −1.2, AVB2 E −4.0, AVB3 E −1.9 (SVT/VT E +2.0/+5.5)
→ **giữ mặc định**; file của `pp_v2` đã giải mã lại theo mặc định.

Test: [`tests/test_rhythm_ec57.py`](tests/test_rhythm_ec57.py) — annotation round-trip qua
`wfdb.rdann`, bộ ghi theo lớp (gộp mã trùng, `(AFL` chỉ ở ref AFIB, đuôi file toàn chữ), đọc `.atr`
→ episode với ánh xạ chặt, parser Gross-line + F1, và một lượt `epicmp`/`sumstats` thật với hai lớp
trên bản ghi tổng hợp (không cần mitdb/afdb/rhythm_eval); [`tests/test_rhythm.py`](tests/test_rhythm.py)
có test riêng cho nối khoảng trống theo ưu tiên, thời lượng tối thiểu và lượt nối thứ hai. Phần cần
dữ liệu thật bị `skipif` khi không có trên máy.
