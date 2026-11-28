# Nguyễn Minh Tuấn — mã bàn giao tuần 1

Gói này chỉ gồm **PELD, prev(i) trong collate_fn và demo tiền xử lý bước 1–4**. Huấn luyện và suy luận cảm xúc SECNet thuộc giai đoạn ghép mô hình, không nằm trong gói này.

## 1. Các file cần đưa vào repo nhóm

| File | Vai trò |
|---|---|
| `src/dataset.py`, `src/__init__.py` | Đọc MELD, gắn PELD và tạo batch có `prev_idx` |
| `configs/personality_peld.json` | Bảng Big Five O,C,E,A,N có nguồn |
| `scripts/add_personality.py` | Bổ sung meta có backup; chạy lại không sửa dữ liệu đã đúng |
| `demo/preprocess_video.py`, `demo/__init__.py` | Demo độc lập bước 1–4 |
| `configs/week1.yaml` | Cấu hình video/ASR/tách người nói |
| `scripts/download_demo_models.py` | Tải hai model ONNX chuẩn bị đầu vào |
| `requirements-week1-demo.txt`, `requirements-week1.txt` | Thư viện demo và thư viện kiểm thử dữ liệu |
| `tests/test_week1.py`, `tests/__init__.py`, `pytest.ini` | Kiểm thử tuần 1 |
| `scripts/verify_week1_demo.py` | Chạy kiểm chứng video có lời nói thật |
| `scripts/verify_week1_data.py` | Kiểm chứng PELD/prev_idx trên toàn bộ DataLoader thật |

`verification/` chứa bằng chứng chạy, không cần import vào chương trình. `MANIFEST.json` liệt kê SHA-256 từng file. Có thể giữ nguyên thư mục sau giải nén để chạy; khi ghép repo đang có code A/B/C, so sánh các file cùng tên trước khi chép. Không ghi đè mù quáng `src/dataset.py` nếu nhóm đã thay đổi giao diện.

## 2. Cài và chạy demo bước 1–4 trên máy mới

Mở PowerShell **tại thư mục có README này**, sau khi giải nén ZIP. Bản này đã kiểm thử trên Windows, Python 3.13. Dùng Python 3.13 để sát môi trường đã chạy.

```powershell
py -3.13 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -X utf8 -m pip install -r requirements-week1-demo.txt
& $py -X utf8 scripts/download_demo_models.py
& $py -X utf8 -m demo.preprocess_video 'C:\duong_dan\hoi_thoai.mp4'
```

Thay đường dẫn video bằng file thực tế. Video tiếng Anh, có lời thoại, tối đa 600 giây theo config. Không cần bộ DATA, vocabulary, GloVe, Gradio, GPU hoặc checkpoint SECNet để chạy lệnh này.

Lần đầu cần Internet: script tải model tách người nói; ASR tự tải `Systran/faster-whisper-base.en` khi chạy video lần đầu. Đây là model chuẩn bị đầu vào, không phải mạng cảm xúc do nhóm huấn luyện. FFmpeg dùng bản đi kèm `imageio-ffmpeg`, không cần cài vào PATH.

Sau khi ASR đã tải thành công, dùng `--offline` để tái dùng cache và tránh chờ mạng:

```powershell
& $py -X utf8 -m demo.preprocess_video 'C:\duong_dan\hoi_thoai.mp4' --offline
```

Nếu model ONNX đã ở chỗ khác:

```powershell
& $py -X utf8 -m demo.preprocess_video 'C:\duong_dan\hoi_thoai.mp4' --models-dir 'C:\duong_dan\demo_models' --offline
```

Thư mục đó phải có `segmentation.onnx` và `speaker_embedding.onnx`. Nếu cache ASR chưa có, bỏ `--offline` và tải online một lần. Có thể sửa `asr_model` trong YAML thành đường dẫn thư mục faster-whisper đã tải. Trên CPU việc xử lý video và nạp model lần đầu có thể mất vài phút; chờ lệnh trả về, không chạy lặp nhiều phiên.

## 3. Kết quả cần xem khi trình diễn

Console báo `PREPROCESSING_COMPLETE` và đường dẫn `result.json`. Mỗi lần chạy tạo thư mục riêng `outputs/week1_demo/run_.../`:

1. **Bước 1 — tách âm thanh:** `audio.wav`, mono 16 kHz, giải mã từ video thật.
2. **Bước 2 — tách người nói:** trường `diarization` trong `result.json`, gồm start/end/speaker.
3. **Bước 3 — ASR:** trường `words`, mỗi từ có timestamp và người nói được gán theo phần giao thời gian.
4. **Bước 4 — ghép câu:** trường `utterances` và `transcript.csv`, gồm start/end/speaker/text. Ngắt khi đổi người nói, dấu câu, khoảng nghỉ hoặc vượt giới hạn khoảng 10 giây; ghép các câu ngắn liên tiếp cùng người nói trong giới hạn.

Mở CSV bằng Excel hoặc trình đọc CSV để xem kết quả. `SPEAKER_00`, `SPEAKER_01` là ID trong riêng video, không phải tên nhân vật. `UNKNOWN` nghĩa là chưa gán được. Hai cờ `ambiguous_speaker` và `unassigned_speaker` báo câu cần nghe kiểm tra. Không có cột dự đoán cảm xúc trong demo tuần 1. Với video không nhận dạng được lời nói, JSON vẫn ghi kết quả rỗng và warning; không tự tạo lời thoại.

Không khẳng định ASR/tách người nói đúng 100%. Nhạc nền, tiếng cười và nói chồng có thể gây lỗi; cần nghe kiểm chứng. Các video mẫu đi kèm riêng là waveform từ âm thanh MELD dev thật, không phải cảnh phim gốc.

## 4. PELD và prev(i): ghép vào training của nhóm

Phần này mới cần bộ dữ liệu `meld_processed` và PyTorch. Đặt dữ liệu tại `data/meld_processed`, hoặc dùng đường dẫn thực tế trong lệnh:

```powershell
& $py -X utf8 -m pip install -r requirements-week1.txt
& $py -X utf8 scripts/add_personality.py --data-root 'C:\duong_dan\meld_processed'
& $py -X utf8 scripts/verify_week1_data.py --data-root 'C:\duong_dan\meld_processed'
& $py -X utf8 -m pytest -q
```

Script chỉ bổ sung personality/provenance vào `meta.json`, cập nhật hash meta trong manifest, lưu bản gốc ở `outputs/part_d/original_metadata`. Không thay CSV/PT/PKL.

```python
from torch.utils.data import DataLoader
from src.dataset import MELDDataset, collate_fn

dataset = MELDDataset('train', root='data/meld_processed')
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
batch = next(iter(loader))
print(batch['personality'].shape)      # (B, N, 5), O,C,E,A,N
print(batch['has_personality'].shape)  # (B, N), bool
print(batch['prev_idx'])              # (B, N), long
```

`prev(i)` trong tài liệu được triển khai thành khóa **`prev_idx`**. Với speaker A,B,A,C,B, kết quả là -1,-1,0,-1,1. -1 nếu chưa có câu trước cùng speaker hoặc padding. Tính lịch sử theo ID speaker gốc trước khi clamp embedding về 0..9. PELD chỉ có cho sáu nhân vật; Other dùng vector zero và mask False. Batch còn có tokens/audio/masks/emotion/shift theo hợp đồng dữ liệu repo.

## 5. Chạy lại ba video mẫu

Giải nén gói video mẫu sao cho thư mục `examples/` nằm cạnh README. Sau khi tải các model:

```powershell
& $py -X utf8 scripts/verify_week1_demo.py examples/dev_dialogue_5.mp4 examples/dev_dialogue_6.mp4 examples/dev_dialogue_7.mp4 --offline
```

Kết quả ở `outputs/week1_verification/verification.json` cùng các thư mục `run_...`. Kiểm tra này yêu cầu video có nhận dạng lời nói; dùng các mẫu đã cung cấp hoặc video hội thoại phù hợp. PASS xác nhận pipeline/timestamp/CSV hoạt động, không phải WER/DER đạt 100%.

Nguồn PELD: https://github.com/preke/PELD#personality-emotionline-dataset-peld (commit trong JSON). Diarization: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html . ASR: https://github.com/SYSTRAN/faster-whisper .
