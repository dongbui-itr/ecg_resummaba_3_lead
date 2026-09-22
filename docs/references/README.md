# Tài liệu tham khảo

## Bài báo gốc của model

**Heo, J., Cha, J., Park, Y., Cho, S. P., & Kim, M. (2026).**
*Patient-conditioned ECG beat classification via self-supervised embeddings for robust
inter-patient arrhythmia detection.*
**Expert Systems With Applications 331, 133149.**
https://doi.org/10.1016/j.eswa.2026.133149
(Đại học Yonsei · MEZOO Co. Ltd., Wonju, Hàn Quốc)

Bốn thành phần, và trạng thái của từng cái trong bản port này:

| thành phần bài báo | ở đây |
|---|---|
| Thân hai nhánh ResUNet + Mamba | giữ ý tưởng, thay S6 bằng SSM chéo hai chiều (`DiagSSM1D`) |
| Bộ mã hóa bệnh nhân CPC, đóng băng | giữ nguyên cơ chế, đổi nguồn: 10 s của chính đoạn thay vì 60 s hiệu chuẩn |
| — (bài báo không có) | **thêm**: tiền huấn luyện backbone bằng tái tạo có che (`training/ssl.py`), xem mục dưới |
| Điều biến AdaIN | **giữ nguyên** |
| Cross-attention với đặc trưng R–R | đảo vai trò query/key, đặc trưng thay bằng tự tương quan envelope |
| Poly-2 loss (ε₁ = 0.3, ε₂ = −0.5) | **giữ nguyên** |

Kết quả họ công bố (MIT-BIH, liên bệnh nhân DS1/DS2, 30 lần chạy): macro F1 86.58 ± 0.05%,
weighted F1 98.54 ± 0.08%. Bảng 8 của bài báo chia tham số như sau — đây là cột `ref` trong bảng
so sánh 5 model ở README chính, mục 5c:

| module | tham số (M) | FLOPs (G) | pha |
|---|---|---|---|
| ResU-Mamba · ResU | 0.696 | 0.354 | runtime |
| ResU-Mamba · Mamba | 0.235 | 0.129 | runtime |
| Patient-Conditioning (AdaIN) | 0.115 | 0.055 | runtime |
| Clinical Feature Fusion | 0.201 | 0.162 | runtime |
| Patient-specific Feature Extraction (CPC) | 0.345 | 0.183 | **chỉ lúc hiệu chuẩn** |
| **Tổng suy luận trực tuyến** | **1.247** | **0.700** | — |

Tổng kể cả bộ mã hóa bệnh nhân là **1.592M**. FLOPs tính cho **một nhịp** (cửa sổ L = 720 mẫu).
**Không so trực tiếp được với bảng EC57 trong README chính** — họ phân loại từng nhịp với đỉnh
R cho trước, ở đây model phải tự phát hiện nhịp trên đoạn 10 s chạy tự do.

## Chặng tự giám sát được thêm vào (không có trong bài báo)

Bài báo chỉ tự giám sát **bộ mã hóa bệnh nhân**, tức 5–9% tham số ở quy mô này; thân model được
huấn luyện hoàn toàn bằng nhãn. [`training/ssl.py`](../../ecgr/training/ssl.py) thêm một chặng
tiền huấn luyện cho **backbone** bằng tái tạo tín hiệu bị che. Nó không phải một phần của bài
báo và không được bài báo viện dẫn, nên nguồn gốc của nó ghi riêng ở đây:

* **Devlin, J., và cộng sự (2019).** *BERT: pre-training of deep bidirectional transformers for
  language understanding.* NAACL. — nguồn của ý tưởng che rồi dự đoán phần bị che, và lý do
  việc che theo **khoảng liền nhau** chứ không theo từng bước rời rạc mới tạo ra bài toán khó.
* **He, K., và cộng sự (2022).** *Masked autoencoders are scalable vision learners.* CVPR. —
  hai lựa chọn thiết kế lấy trực tiếp từ đây: tỉ lệ che cao, và **decoder cố tình nhỏ** (một
  decoder có sức chứa riêng sẽ tái tạo được từ một biểu diễn yếu hơn, tức chuyển việc học ra
  khỏi backbone vào một cái head rồi bị bỏ đi).
* **Baevski, A., và cộng sự (2020).** *wav2vec 2.0.* NeurIPS. — tiền lệ cho việc che theo khoảng
  trên tín hiệu một chiều lấy mẫu dày, thay vì trên token rời rạc.
* **Zhang, W., Yang, L., Geng, S., & Hong, S. (2023).** *Self-supervised time series
  representation learning via cross reconstruction transformer.* IEEE TNNLS. — tiền lệ cho phần
  **che theo chuyển đạo**: tái tạo một chuyển đạo từ các chuyển đạo còn lại, thứ mà bản 1 chuyển
  đạo trước đây không thể có.

Phần che theo chuyển đạo không chỉ là một phép tăng cường: nó chính là phép toán mà chặng EC57
thực hiện khi đưa cho model cùng một chuyển đạo ba lần (README mục 7b), nên nó vừa là mục tiêu
tự giám sát vừa là cách làm cho benchmark không còn là một dịch chuyển phân phối.

## Được bài báo viện dẫn, và thực sự dùng trong bản port

* **Gu, A., & Dao, T. (2024).** *Mamba: Linear-time sequence modeling with selective state
  spaces.* COLM. — nguồn của nhánh state-space. `DiagSSM1D` giữ cấu trúc khối (hai nhánh,
  cổng SiLU) và bỏ phần chọn lọc phụ thuộc đầu vào, xem docstring của `layers.py`.
* **Gu, A., và cộng sự (2021).** *Combining recurrent, convolutional, and continuous-time
  models with linear state space layers.* NeurIPS 34, 572–585. — cơ sở của dạng nghiệm đóng
  cho SSM chéo, tức lý do toàn bộ hồi quy trở thành một tích chập.
* **Huang, X., & Belongie, S. (2017).** *Arbitrary style transfer in real-time with adaptive
  instance normalization.* ICCV. — nguồn của AdaIN.
* **van den Oord, A., Li, Y., & Vinyals, O. (2018).** *Representation learning with
  contrastive predictive coding.* arXiv:1807.03748. — mục tiêu InfoNCE trong
  [`training/cpc.py`](../../ecgr/training/cpc.py).
* **Leng, Z., và cộng sự (2022).** *PolyLoss: a polynomial expansion perspective of
  classification loss functions.* ICLR. — Poly-2 là một trường hợp của họ loss này.
* **Hwang, S., Cha, J., Heo, J., Cho, S., & Park, Y. (2023).** *Multi-label ECG abnormality
  classification using a combined ResNet-DenseNet architecture with ResU blocks.* IEEE EMBS
  DSEHMB. — khối ResU của nhánh hình thái.
* **Perez, E., và cộng sự (2018).** *FiLM: visual reasoning with a general conditioning
  layer.* AAAI 32. — bài báo dùng làm đối chứng cho AdaIN; điểm khác biệt là FiLM không chuẩn
  hóa trước khi điều biến, nên không khử được chênh lệch biên độ giữa các bản ghi.

## Nền tảng của bài toán, không phải của model

* **De Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004).** *Automatic classification of
  heartbeats using ECG morphology and heartbeat interval features.* IEEE TBME 51(7),
  1196–1206. — định nghĩa giao thức liên bệnh nhân DS1/DS2 mà bài báo dùng.
* **Moody, G. B., & Mark, R. G. (2001).** *The impact of the MIT-BIH arrhythmia database.*
  IEEE EMB Magazine 20(3), 45–50.
* **ANSI/AAMI EC57** — chuẩn quy định cách chấm Se/+P theo lớp, cửa sổ khớp, và việc loại các
  bản ghi có máy tạo nhịp khỏi phần chấm nhịp. `bxb` và `sumstats` của bộ WFDB hiện thực
  chuẩn này; xem `scripts/`.
