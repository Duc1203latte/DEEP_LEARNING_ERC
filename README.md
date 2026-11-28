# SECNet: Nhận diện cảm xúc trong hội thoại (ERC) trên MELD

Bài tập lớn môn Deep Learning, nhóm 4 người.

**Phần D, tuần 1 (Nguyễn Minh Tuấn):** [Hướng dẫn chạy PELD, prev_idx và demo bước 1–4](docs/WEEK1_README.md). [Gói mã](deliverables/week1/Nguyen_Minh_Tuan_Tuan1_Code.zip) · [3 video mẫu](deliverables/week1/Tuan1_Video_Mau.zip).

**Bài toán:** cho một hội thoại tiếng Anh nhiều lượt, dự đoán cảm xúc của **từng câu** (7 lớp). Mạng dùng văn bản, âm thanh, thông tin người nói và (nếu có) tính cách người nói.

**Demo:** nhận một video hội thoại, hiển thị cảm xúc của từng người theo thời gian.

Tài liệu thiết kế (đọc trước khi sửa kiến trúc):
- Kiến trúc mạng, pipeline demo, phân công: https://docs.google.com/document/d/1-AQBqYhuDb0q4z3LIPQDjTQ80bcezY6vykfuEWACeFY/edit?tab=t.0#heading=h.6p6yixq6ntzj
- Hướng dẫn chuẩn bị dữ liệu: https://claude.ai/artifact/K6aXLd8j3qgru6AZsAThUn


## 1. Ràng buộc bắt buộc (yêu cầu của giảng viên)

1. **Mạng nhận diện cảm xúc phải tự xây** từ các lớp cơ bản của PyTorch (`nn.Embedding`, `nn.GRU`, `nn.Conv2d`, `nn.Linear`, `nn.LayerNorm`, `nn.Dropout`, `nn.BatchNorm2d`, …).
   - Attention, multi-head attention, Transformer **tự viết**. Không dùng `nn.MultiheadAttention`, `nn.TransformerEncoder`, `nn.TransformerEncoderLayer`.
   - Không dùng mô hình tiền huấn luyện (BERT, RoBERTa, wav2vec, HuBERT, Whisper encoder, …) bên trong mạng.
   - Không dùng đặc trưng trích sẵn của MELD (các file `.pkl` trong repo MELD).
2. **Được phép:** GloVe (`glove.6B.300d`) để khởi tạo embedding; `librosa`/`torchaudio` để tính log-mel; ffmpeg.
3. **Demo** được dùng Whisper/faster-whisper, pyannote.audio hoặc WhisperX để tách người nói và chuyển giọng nói thành chữ, vì đó là tiền xử lý đầu vào chứ không phải mạng cảm xúc.
4. **Không rò rỉ dữ liệu:**
   - Từ điển, mean/std âm thanh và trọng số lớp chỉ tính từ train.
   - Siêu tham số chỉ chọn trên dev.
   - Không bao giờ chọn mô hình theo điểm test.
5. Giữ nguyên chia train/dev/test của MELD. Không xóa câu thiếu âm thanh, chỉ đánh dấu `has_audio = False`.

---

## 2. Cài đặt

Cần Python 3.10 hoặc 3.11, PyTorch có CUDA, và ffmpeg (cài qua hệ thống).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip

# PyTorch: chọn lệnh đúng phiên bản CUDA tại https://pytorch.org/get-started/locally/
pip install torch torchaudio
pip install -r requirements.txt
pip install -r requirements-demo.txt   # chỉ người làm demo

# kiểm tra
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ffmpeg -version
```

---

## 3. Dữ liệu

Dữ liệu đã xử lý **không có trên git**. Tải tại Google Drive của nhóm:

**https://drive.google.com/drive/folders/19YbO1uZEg659kL_SNQtbme9ztiuwlXEc?usp=sharing**

Tải **cả thư mục** về (khoảng 690 MB) rồi đặt vào `data/meld_processed/`, sao cho có `data/meld_processed/meta.json`, `data/meld_processed/train_dialogues.pkl`, … Nếu để dữ liệu chỗ khác (Colab: mount Drive; Kaggle: upload thành dataset), sửa `data.root` trong config.

Dữ liệu được tạo từ MELD gốc (video + CSV) bằng `notebooks/prepare_data.ipynb`. Mọi thống kê (từ điển, mean/std âm thanh, trọng số lớp) chỉ tính trên train. **Code của nhóm chỉ đọc các file này, không đọc lại video hay CSV gốc của MELD.**

### 3.1. Các file trong thư mục Drive

Ba file chính cho mỗi split (`train`, `dev`, `test`):

| File | Là gì | Dùng để |
| --- | --- | --- |
| `{split}_dialogues.pkl` | Nhãn và văn bản đã mã hóa, mỗi phần tử là một hội thoại (xem 3.2) | Đầu vào chính của `MELDDataset` |
| `{split}_audio.pt` | Log-mel của từng câu, đã chuẩn hóa (xem 3.3) | Đầu vào của audio encoder |
| `{split}_utterances.csv` | Bảng phẳng, mỗi dòng một câu, có cả văn bản gốc, tên người nói, thời gian | Xem dữ liệu, phân tích lỗi, vẽ biểu đồ. **Không dùng để train** |

File dùng chung:

| File | Là gì |
| --- | --- |
| `vocab.json` | Từ điển `{từ: chỉ số}`, 2701 từ. `<pad>` = 0, `<unk>` = 1. Xây từ train, giữ từ xuất hiện ≥ 2 lần |
| `embedding_matrix.npy` | Ma trận 2701 × 300 (float32). Hàng *i* là vector GloVe của từ có chỉ số *i*. Hàng `<pad>` = 0. GloVe phủ 98.6% từ điển |
| `meta.json` | Mọi tham số và thống kê tiền xử lý (xem 3.4) |

File phụ (không cần cho code, chỉ để tham khảo):

| File | Là gì |
| --- | --- |
| `prepare_data.ipynb` | Notebook đã sinh ra dữ liệu (bản giống `notebooks/prepare_data.ipynb` trên git) |
| `data_report.md` | Báo cáo dữ liệu: phân bố nhãn, độ dài, ví dụ một hội thoại hoàn chỉnh |
| `FINAL_DELIVERY_REPORT.md` | Báo cáo bàn giao: kiểm tra tính toàn vẹn, giới hạn đã biết, cách tái tạo |
| `dataset.py` | Bản `MELDDataset` mẫu của người làm dữ liệu. Code chính sẽ nằm ở `src/dataset.py` |
| `audio_review.json` | 5 câu được chọn để nghe kiểm tra bằng tai (chưa duyệt xong) |
| `manifest.json`, `logs/` | Danh sách file, checksum, log từng bước xử lý |

### 3.2. `{split}_dialogues.pkl`

Một `list`, mỗi phần tử là `dict` của một hội thoại. Mọi khóa kiểu list đều dài bằng số câu của hội thoại; phần tử thứ *i* ứng với câu thứ *i*.

| Khóa | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `dialogue_id` | int | Mã hội thoại trong MELD |
| `utt_ids` | list[int] | Mã câu trong hội thoại (0, 1, 2, …) |
| `token_ids` | list[list[int]] | Mỗi câu là list chỉ số từ theo `vocab.json`, tối đa 50 từ, không đệm |
| `speaker_local` | list[int] | Người nói thứ mấy trong hội thoại này, đánh số theo thứ tự xuất hiện (0, 1, 2, …) |
| `speaker_global` | list[int] | Nhân vật: Rachel 0, Ross 1, Monica 2, Chandler 3, Joey 4, Phoebe 5, người khác 6 |
| `emotion_id` | list[int] | Nhãn cảm xúc, 0–6 (xem dưới) |
| `sentiment_id` | list[int] | Nhãn sắc thái: negative 0, neutral 1, positive 2 (nhóm không dùng) |
| `shift` | list[int] | 1 = cảm xúc khác câu trước của cùng người nói; 0 = giống; −1 = câu đầu tiên của người đó (không tính loss) |
| `has_audio` | list[bool] | Câu có âm thanh hay không |
| `n_frames` | list[int] | Số khung log-mel của câu (0 nếu không có âm thanh) |

**Nhãn cảm xúc** (thứ tự cố định): `neutral 0, surprise 1, fear 2, sadness 3, joy 4, disgust 5, anger 6`.

Ví dụ thật, hội thoại đầu tiên của dev:

```python
{'dialogue_id': 0,
 'utt_ids': [0, 1],
 'token_ids': [[16, 30, 82, 2, 46, 9, 404, 10, 3, 46, 9, 262, 404, 10, 3], [19, 7]],
 'speaker_local': [0, 1],       # hai người khác nhau
 'speaker_global': [5, 2],      # Phoebe, Monica
 'emotion_id': [3, 1],          # sadness, surprise
 'sentiment_id': [0, 0],
 'shift': [-1, -1],             # cả hai đều là câu đầu tiên của người nói
 'has_audio': [True, True],
 'n_frames': [280, 134]}
```

### 3.3. `{split}_audio.pt`

Một `dict`: khóa là `f"dia{dialogue_id}_utt{utt_id}"` (ví dụ `"dia0_utt1"`), giá trị là tensor `float16` cỡ (số khung × 80).

- Log-mel 80 dải, 16 kHz, cửa sổ 25 ms, bước nhảy 10 ms, nên 100 khung = 1 giây. Cắt tối đa 1000 khung (10 s).
- Đã chuẩn hóa theo từng dải bằng mean/std của train, nên trên train mỗi dải có mean 0, std 1.
- Câu không có âm thanh vẫn có khóa, nhưng tensor có 0 khung.

Cách đọc an toàn:

```python
audio = torch.load("data/meld_processed/train_audio.pt", weights_only=True, map_location="cpu")
x = audio["dia0_utt0"].float()   # (568, 80)
```

### 3.4. `meta.json`

| Khóa | Nội dung |
| --- | --- |
| `CONFIG` | Tham số tiền xử lý: `SAMPLE_RATE` 16000, `N_MELS` 80, `N_FFT` 512, `WIN_LENGTH` 400, `HOP_LENGTH` 160, `MAX_FRAMES` 1000, `MAX_TOKENS` 50, `MIN_FREQ` 2, … Demo phải dùng **đúng** các giá trị này |
| `emotion_mapping`, `sentiment_mapping`, `speaker_global_mapping` | Bảng tên → số |
| `class_weights` | 7 trọng số lớp (tỉ lệ nghịch với số câu mỗi lớp trong train, tổng = 7), dùng làm α cho focal loss |
| `audio_train_statistics` | `mean`, `std`: mỗi cái 80 số, tính trên train. Demo dùng để chuẩn hóa log-mel mới |
| `tokenizer`, `logmel` | Mô tả cách tách từ và cách tính log-mel |
| `splits` | Số hội thoại, số câu, số câu có âm thanh của mỗi split |

**Chưa có bảng tính cách PELD.** Người D cần thêm khóa `personality` (tên nhân vật → [O, C, E, A, N]) trước khi chạy ablation dòng 9.

### 3.5. Số liệu (đã kiểm tra 25/11/2026)

| | Train | Dev | Test |
| --- | --- | --- | --- |
| Số hội thoại | 1038 | 114 | 280 |
| Số câu | 9989 | 1109 | 2610 |
| Hội thoại dài nhất (câu) | 24 | 23 | 33 |
| Câu không có âm thanh | 1 (`dia125_utt3`) | 1 (`dia110_utt7`) | 0 |
| Tỉ lệ `<unk>` | 2.5% | 4.0% | 4.3% |

Phân bố nhãn khớp số chính thức của MELD:

| | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Train | 4710 | 1205 | 268 | 683 | 1743 | 271 | 1109 |
| Dev | 470 | 150 | 40 | 111 | 163 | 22 | 153 |
| Test | 1256 | 281 | 50 | 208 | 402 | 68 | 345 |

Lưu ý khi viết code:
- Train thiếu dialogue 60. Đây là lỗi đã biết của MELD, không phải lỗi xử lý.
- Câu ngắn nhất có 7 khung âm thanh; không có câu 0 token; `speaker_local` lớn nhất là 8.
- Lớp *neutral* chiếm 47% train, *fear* và *disgust* mỗi lớp chưa tới 3%. Vì vậy dùng weighted-F1 làm chỉ số chính, và dùng focal loss có trọng số lớp.

### Output của `collate_fn`

Ký hiệu: B hội thoại, N câu, T từ, F khung âm thanh.

| Khóa | Shape | Kiểu / giá trị đệm |
| --- | --- | --- |
| `tokens`, `token_mask` | B×N×T | long, bool. Mỗi câu thật có ít nhất 1 token (câu rỗng → `[<unk>]`) |
| `audio`, `audio_mask` | B×N×F×80, B×N×F | float, bool. **F đệm lên bội số của 8** |
| `has_audio` | B×N | bool |
| `speaker` | B×N | long (`speaker_local`, chặn ở 9) |
| `speaker_global` | B×N | long, lấy thẳng từ pkl: Rachel 0, Ross 1, Monica 2, Chandler 3, Joey 4, Phoebe 5, Other 6 |
| `personality`, `has_personality` | B×N×5, B×N | float (Big Five O,C,E,A,N), bool |
| `prev_idx` | B×N | long, chỉ số câu trước của cùng người nói, −1 nếu không có |
| `utt_mask` | B×N | bool, False = câu đệm |
| `emotion` | B×N | long, câu đệm = −100 |
| `shift` | B×N | long, câu đệm hoặc không xác định = −1 |

Tính cách chỉ có cho 6 nhân vật chính, lấy từ PELD. Người nói khác có `has_personality = False`.

---

## 4. Cấu trúc repo và người phụ trách

Hiện repo mới có khung thư mục và `configs/`; các file code dưới đây là kế hoạch, mỗi người tạo file của mình. Mỗi file ghi shape vào/ra trong docstring. Một khi đã thống nhất tên hàm, lớp và shape thì không tự đổi; muốn đổi thì báo cả nhóm trước.

```
configs/
  base.yaml                  # = SECNet đầy đủ (ablation dòng 8)
  ablation/*.yaml            # mỗi dòng ablation một file, chỉ ghi phần khác base
data/meld_processed/         # không commit
notebooks/prepare_data.ipynb # chuẩn bị dữ liệu (đã xong)
src/
  preprocess.py              # D  tokenizer, log-mel, chuẩn hóa: dùng chung cho dataset và demo
  dataset.py                 # D  MELDDataset, collate_fn (+ personality, prev_idx, speaker_global)
  models/
    common.py                #    pack/scatter câu thật, masked_softmax (dùng chung)
    attention.py             # C  MultiHeadAttention tự viết (B dùng cho fusion)
    text_encoder.py          # A  khối 1
    audio_encoder.py         # B  khối 2
    fusion.py                # B  khối 3 (gated, concat, audio-only)
    speaker.py               # C  khối 4, 4b
    context.py               # C  khối 5
    heads.py                 # A  khối 6a, 6b
    secnet.py                # A  ghép toàn mạng
  losses.py                  # A  focal loss, shift BCE
  train.py                   # A  vòng train/eval, early stopping, AMP
  evaluate.py                # A  metric, confusion matrix, --drop_personality
  utils.py                   #    seed, load config (đọc khóa `_base_`), logging
tests/
  conftest.py                #    batch giả dùng chung
  test_shapes.py             # mỗi người thêm/sửa test cho khối của mình
demo/
  pipeline.py                # D  video → câu → SECNet → kết quả
  app.py                     # D  giao diện Gradio
outputs/                     # checkpoint, log, kết quả (không commit checkpoint)
```

---

## 5. Kiến trúc SECNet (tóm tắt)

d = 256 cho mọi biểu diễn câu. Chi tiết và lý do thiết kế xem doc kiến trúc.

| Khối | Cấu tạo | Đầu ra |
| --- | --- | --- |
| 1. Text encoder | Embedding 300 (GloVe, lr 1e-4) → Dropout 0.3 → BiGRU 128×2 (packed) → additive attention pooling | `h_t` B×N×256 |
| 2. Audio encoder | 3 × [Conv2d 3×3 pad 1, BN, ReLU, MaxPool 2×2], kênh 1→32→64→64 → reshape 640 → Linear 640→256 → BiGRU 128×2 | `A` F/8 × 256 mỗi câu |
| 3. Fusion | `h_t` làm query, attention 4 head lên `A` → `a`; `g = σ(W[h_t; a])`; `x = LayerNorm(h_t + g ⊙ a)`; thiếu âm thanh: `g = 0` | `x` B×N×256 |
| 4. Speaker | `x += Embedding(10, 256)[speaker_local]` | |
| 4b. Personality (tùy chọn) | có: `e_p = MLP(5→64→256)`; không có: `e_p = e_unknown` (học được); `x += e_p`; dropout 30% khi train | |
| 5. Context | 2 lớp Transformer tự viết, pre-LN, 4 head, FFN 512 (GELU), dropout 0.1; mặt nạ nhân quả + mặt nạ đệm; score cộng `b_spk[h][same_speaker] + b_dist[h][min(i−j, 10)]`, khởi tạo 0 | `c` B×N×256 |
| 6a. Emotion head | Linear 256→256, ReLU, Dropout 0.3, Linear 256→7 | B×N×7 |
| 6b. Shift head | MLP trên `[c_i; c_prev; |c_i − c_prev|; e_p]` (1024→256→1) | B×N |

**Loss:** `L = Focal(γ=2, α=class_weights) + 0.5 · BCEWithLogits(shift)`, bỏ qua `emotion = −100` và `shift = −1`.

---

## 6. Huấn luyện, đánh giá, ablation

Mặc định:
- AdamW (weight decay 1e-4), lr 1e-3, riêng embedding 1e-4.
- Batch 8 hội thoại, tối đa 30 epoch.
- Early stopping sau 5 epoch không cải thiện weighted-F1 trên dev.
- Clip gradient 1.0, bật AMP.

```bash
python -m src.train --config configs/base.yaml --seed 42
python -m src.train --config configs/ablation/04_gated_fusion.yaml --seed 42
python -m src.evaluate --ckpt outputs/08_full_s42/best.pt --split test
python -m src.evaluate --ckpt outputs/09_personality_big5_s42/best.pt --split test --drop_personality   # dòng 9c
pytest tests/
```

Mỗi run lưu vào `outputs/<tên file config>_s<seed>/`: config đã dùng, log, `best.pt`, kết quả dev/test dạng JSON.

**Báo cáo:**
- weighted-F1 (chỉ số chính), macro-F1, F1 từng lớp.
- Accuracy trên các câu có `shift = 1`.
- Kết quả cuối chạy 5 seed (42–46), ghi trung bình ± độ lệch chuẩn.

### Bảng ablation

Mỗi file ablation khai báo `_base_: ../base.yaml` và chỉ ghi phần khác.

| # | File | text | audio | fusion | speaker | context | bias | shift | loss | personality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `01_text_only` | ✓ | ✗ | – | ✗ | ✗ | ✗ | ✗ | ce | ✗ |
| 2 | `02_audio_only` | ✗ | ✓ | query học được | ✗ | ✗ | ✗ | ✗ | ce | ✗ |
| 3 | `03_concat_fusion` | ✓ | ✓ | concat | ✗ | ✗ | ✗ | ✗ | ce | ✗ |
| 4 | `04_gated_fusion` | ✓ | ✓ | gated | ✗ | ✗ | ✗ | ✗ | ce | ✗ |
| 5 | `05_context` | ✓ | ✓ | gated | ✓ | ✓ | ✗ | ✗ | ce | ✗ |
| 6 | `06_context_bias` | ✓ | ✓ | gated | ✓ | ✓ | ✓ | ✗ | ce | ✗ |
| 7 | `07_shift_head` | ✓ | ✓ | gated | ✓ | ✓ | ✓ | ✓ | ce | ✗ |
| 8 | `08_full` (= base) | ✓ | ✓ | gated | ✓ | ✓ | ✓ | ✓ | focal | ✗ |
| 8b | `08b_ce_unweighted` (tùy chọn) | như 7, CE không trọng số lớp | | | | | | | ce | ✗ |
| 9 | `09_personality_big5` | như 8 | | | | | | | focal | big5 |
| 9b | `09b_personality_onehot` | như 8 | | | | | | | focal | onehot |
| 9c | không có file: chạy `evaluate.py --drop_personality` trên checkpoint dòng 9 | | | | | | | | | |

---

## 7. Quyết định đã chốt (25/11/2026)

Những điểm doc kiến trúc chưa nói rõ, nay đã chốt. Code phải theo đúng các điểm này.

1. **Ablation 2 (chỉ audio).** `model.use_text: false`. Khối 3 dùng một query học được `q0` thay cho `h_t`, giữ nguyên cross-attention 4 head, rồi `x = LayerNorm(a)`. Câu không có âm thanh dùng vector học được `e_noaudio`. Không xóa các câu này khỏi tập đánh giá.
2. **Shift head khi tắt tính cách.** Đầu vào luôn 1024 chiều. Khi `use_personality: false`, `e_p` là vector 0 (buffer, không học). `c_prev` lấy bằng `gather` với `prev_idx.clamp(min=0)`; câu có `prev_idx = -1` đã có `shift = -1` nên không tính loss.
3. **Ablation 9b.** Khi `personality_mode: onehot`, MLP khối 4b nhận one-hot 6 chiều của `speaker_global` (Linear 6→64→256). Người khác dùng `e_unknown`. Dropout và đường vào shift head giữ nguyên.
4. **Cờ bổ sung.**
   - `use_speaker`: tắt ở dòng 1–4.
   - `audio_dropout` (mặc định 0.0) và `personality_dropout` (0.3): chỉ áp dụng khi train, cài trong `secnet.forward` bằng cách đặt `has_audio` / `has_personality = False` cho từng câu.
   - Dòng 9c là cờ của `evaluate.py`.
5. **Loss.**
   - `ce` là CE **có trọng số lớp**, cài bằng hàm focal với `γ = 0`, nên dòng 8 chỉ đo tác dụng của γ.
   - Chuẩn hóa chia cho số câu có nhãn |U|, không chia cho tổng α.
6. **Chi tiết kỹ thuật.**
   - Hệ số chia attention là √d_h = 8 (d_h = 64), cho cả fusion và Transformer.
   - F đệm lên bội số của 8. Sau CNN có F/8 khung, độ dài thật là `ceil(len/8)`. BiGRU âm thanh dùng `pack_padded_sequence`.
   - Softmax có mặt nạ: ép float, `masked_fill(-inf)`, softmax, `nan_to_num(0)`. Viết một hàm `masked_softmax` dùng chung trong `src/models/common.py`, các khối gọi lại hàm đó.
   - Câu 0 token được `collate_fn` thay bằng `[<unk>]`.
   - Fusion concat khi thiếu âm thanh: `mean-pool(A) = 0`.
7. **Dữ liệu.** Đường dẫn là `data.root` trong config. Mean/std (`audio_train_statistics`) và trọng số lớp (`class_weights`) đọc từ `meta.json`, không tính lại. Bảng tính cách sẽ thêm vào `meta.json` (khóa đề xuất: `personality`, map tên nhân vật → [O, C, E, A, N]).

---

## 8. Quy ước code

- Mọi siêu tham số nằm trong `configs/*.yaml`, không viết cứng trong code. Số thực viết dạng `1.0e-3`, không viết `1e-3` (PyYAML đọc thành chuỗi).
- Cố định seed cho `random`, `numpy`, `torch` (kể cả CUDA) ở đầu mỗi lần chạy.
- Mỗi module trong `src/models/` có docstring ghi shape vào/ra, và có test trong `tests/test_shapes.py` dùng batch giả (có câu đệm, câu thiếu âm thanh, câu thiếu tính cách).
- Encoder chỉ chạy trên câu thật: gộp B×N, lọc theo `utt_mask` (thêm `has_audio` cho audio encoder), rồi scatter kết quả về vị trí cũ, dùng hàm chung trong `src/models/common.py`.
- Hàm tiền xử lý (tokenizer, log-mel) chỉ nằm ở `src/preprocess.py`, dùng chung cho dataset và demo.
- Không commit `data/`, checkpoint, file âm thanh/video.
- Tên biến, hàm, comment trong code viết tiếng Anh; báo cáo và tài liệu viết tiếng Việt.

---

## 9. Demo

`demo/pipeline.py`:
1. ffmpeg tách âm thanh (wav 16 kHz mono).
2. Tách người nói và chuyển giọng nói thành chữ.
3. Ghép câu: gộp đoạn ngắn cùng người, cắt đoạn dài hơn ~10 s, bỏ đoạn không có chữ.
4. Tiền xử lý **giống hệt lúc train**: `src/preprocess.py`, `vocab.json`, mean/std trong `meta.json`, `speaker_local` theo thứ tự xuất hiện, `has_personality = False`.
5. Chạy SECNet, ra cảm xúc, xác suất và cờ chuyển cảm xúc cho từng câu.

`demo/app.py` hiển thị:
- Bảng lời thoại tô màu theo cảm xúc.
- Biểu đồ cảm xúc từng người theo thời gian.
- Tỉ lệ cảm xúc từng người.
- Đổi được tên người nói.
- Bấm vào câu để phát lại đoạn video tương ứng.

Hội thoại dài (40–80 câu): nếu kết quả kém, chạy cửa sổ trượt 20–30 câu.

```bash
python -m demo.app
```

---

## 10. Phân công (3 tuần, 4 người)

Theo doc kiến trúc, mục 11. Dữ liệu đã xong, nên:
- **Tuần 1:** bốn người cài song song bốn phần độc lập.
- **Tuần 2:** ghép mạng và chạy ablation.
- **Tuần 3:** chạy kết quả cuối, hoàn thiện demo và báo cáo.

**Ngày đầu tiên (cả nhóm):** thống nhất cấu trúc repo, file config, input/output của từng module (tên hàm, shape tensor; xem mục 3 và mục 7). Đây là điều kiện để làm song song mà cuối tuần ghép được.

| Người | Tuần 1 (23–29/11): cài từng khối | Tuần 2 (30/11–6/12): ghép và ablation | Tuần 3 (7–13/12): kết quả, demo, báo cáo |
| --- | --- | --- | --- |
| **A** (khung huấn luyện) | Khối 1 Text encoder; vòng train/eval, early stopping, tính weighted-F1 và macro-F1; chạy ablation dòng 1 | Ghép toàn mạng; khối 6a, 6b, focal loss, loss shift; chạy dòng 5, 7, 8 | Chạy mô hình đầy đủ và các dòng ablation với 5 seed; tổng hợp bảng kết quả |
| **B** (âm thanh) | Khối 2 Audio encoder, khối 3 Fusion (cả bản concat); kiểm tra bộ nhớ GPU, bật AMP | Chạy dòng 2, 3, 4; chỉnh `MAX_FRAMES`, dropout âm thanh | Phân tích lỗi: ma trận nhầm lẫn, F1 từng lớp, ví dụ đoán sai; viết phần Thí nghiệm trong báo cáo |
| **C** (ngữ cảnh) | Khối 4, 4b và khối 5 Transformer tự viết (bias người nói, bias khoảng cách, mặt nạ); test mặt nạ với batch giả | Chạy dòng 6, 9, 9b, 9c; đo độ chính xác trên các câu có chuyển cảm xúc | Vẽ trọng số attention và giá trị bias đã học; viết phần Phương pháp và Nghiên cứu liên quan |
| **D** (dữ liệu bổ sung + demo) | Thêm bảng tính cách PELD và chỉ số `prev(i)` vào `collate_fn`; demo bước 1–4 (tách âm thanh, tách người nói, chuyển giọng nói thành chữ, ghép câu) | Demo bước 5 (tiền xử lý dùng chung code với phần dữ liệu, gọi mô hình); khung giao diện Gradio/Streamlit | Hoàn thiện giao diện, thử 3–5 video, quay video demo dự phòng; làm slide |

File phụ trách của từng người xem cây thư mục ở mục 4.

### Mốc kiểm tra

- [ ] **Hết tuần 1:**
  - Mỗi khối chạy được với một batch thật, ra đúng shape.
  - Dòng ablation 1 có số đầu tiên.
  - Demo trả ra được danh sách câu từ một video thử.
- [ ] **Hết tuần 2:**
  - Mọi dòng ablation chạy xong ít nhất 1 seed.
  - Demo chạy trọn từ video đến kết quả (giao diện có thể còn thô).
- [ ] **Hết tuần 3:** bảng kết quả 5 seed, báo cáo đủ các phần, demo và video dự phòng, slide.

### Phụ thuộc cần chú ý

- **A cần khối của B và C:** tuần 2 của A cần khối 2–5 của B và C chạy được từ cuối tuần 1. Nếu một khối trễ, A ghép tạm bản đơn giản (concat thay fusion, bỏ khối 5) để không chặn việc chạy thí nghiệm.
- **D cần checkpoint:** demo bước 5 của D cần một checkpoint bất kỳ từ tuần 2; dùng checkpoint tốt nhất ở tuần 3.
- **Báo cáo viết dần từ tuần 2,** không dồn vào cuối: mỗi người viết phần mô tả khối mình đã cài.
- **Chia GPU từ đầu tuần 3:** chạy 5 seed cho nhiều dòng ablation tốn GPU, nên chia Colab, Kaggle, máy cá nhân cho cả nhóm.
