"""Week 1 only: video -> mono audio -> diarized ASR words -> utterances.
No training dataset or emotion checkpoint is required.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg
import soundfile as sf
import yaml

def load_config(path: str | Path = "configs/week1.yaml") -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Demo config must be a mapping")
    return config


def extract_audio(video: str | Path, wav: Path, sample_rate: int, timeout: int, max_seconds: float) -> None:
    """Decode locally, including one extra second to detect oversized inputs."""
    if not Path(video).is_file():
        raise FileNotFoundError(video)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-y", "-i", str(Path(video).resolve()),
               "-t", str(max_seconds + 1), "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(wav)]
    try:
        subprocess.run(command, capture_output=True, check=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Cannot decode input audio: {exc.stderr.decode('utf-8', errors='replace')}") from exc
    info = sf.info(wav)
    if info.frames == 0 or info.duration > max_seconds:
        raise ValueError(f"Input must contain audio and be at most {max_seconds} seconds")


def align_words(words: list[dict], turns: list[dict]) -> list[dict]:
    """Assign by maximum time overlap; ties/overlapping speakers remain visible."""
    result = []
    for word in sorted(words, key=lambda w: (w["start"], w["end"])):
        start, end = float(word["start"]), float(word["end"])
        if not math.isfinite(start + end) or start < 0 or end <= start:
            raise ValueError("Invalid word timestamps")
        text = str(word["text"]).strip()
        if not text:
            continue
        overlap = {}
        for turn in turns:
            amount = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
            if amount:
                overlap[turn["speaker"]] = overlap.get(turn["speaker"], 0.0) + amount
        ranked = sorted(overlap, key=lambda s: (-overlap[s], s))
        result.append({"start": start, "end": end, "text": text,
                       "speaker": ranked[0] if ranked else "UNKNOWN",
                       "ambiguous_speaker": len(ranked) > 1, "unassigned_speaker": not ranked})
    return result


def merge_words(words: list[dict], max_seconds: float, gap_limit: float, short_seconds: float) -> list[dict]:
    """Split at speaker changes, punctuation, pauses and the training duration limit."""
    if max_seconds <= 0 or gap_limit < 0 or short_seconds < 0:
        raise ValueError("Invalid utterance segmentation limits")
    sentences, current = [], []

    def flush():
        if current:
            sentences.append({"start": current[0]["start"], "end": max(w["end"] for w in current),
                              "speaker": current[0]["speaker"], "text": " ".join(w["text"] for w in current),
                              "ambiguous_speaker": any(w["ambiguous_speaker"] for w in current),
                              "unassigned_speaker": any(w["unassigned_speaker"] for w in current)})
            current.clear()

    for word in words:
        if word["end"] - word["start"] > max_seconds:
            raise ValueError("One word exceeds the maximum duration; check ASR timestamps")
        if current and (word["speaker"] != current[-1]["speaker"] or word["start"] - current[-1]["end"] > gap_limit or word["end"] - current[0]["start"] > max_seconds):
            flush()
        current.append(word)
        if word["text"].rstrip().endswith((".", "!", "?")):
            flush()
    flush()
    merged = []
    for sentence in sentences:
        if merged:
            previous = merged[-1]
            if (previous["speaker"] == sentence["speaker"] and previous["end"] - previous["start"] < short_seconds
                    and 0 <= sentence["start"] - previous["end"] <= gap_limit and sentence["end"] - previous["start"] <= max_seconds):
                previous["text"] += " " + sentence["text"]
                previous["end"] = sentence["end"]
                previous["ambiguous_speaker"] |= sentence["ambiguous_speaker"]
                previous["unassigned_speaker"] |= sentence["unassigned_speaker"]
                continue
        merged.append(dict(sentence))
    return merged



class VideoPreprocessor:
    def __init__(self, config: dict):
        self.config = dict(config)
        self._asr = None
        self._diarizer = None

    def _load_backends(self):
        if self._diarizer is None:
            import sherpa_onnx as so
            for name in ["segmentation_model", "embedding_model"]:
                if not Path(self.config[name]).is_file():
                    raise FileNotFoundError(f"Missing {name}; run python scripts/download_demo_models.py")
            c = self.config
            cfg = so.OfflineSpeakerDiarizationConfig(
                segmentation=so.OfflineSpeakerSegmentationModelConfig(
                    pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=c["segmentation_model"]),
                    num_threads=c["num_threads"], provider="cpu"),
                embedding=so.SpeakerEmbeddingExtractorConfig(model=c["embedding_model"], num_threads=c["num_threads"], provider="cpu"),
                clustering=so.FastClusteringConfig(num_clusters=c["num_speakers"], threshold=c["cluster_threshold"]),
                min_duration_on=c["min_duration_on"], min_duration_off=c["min_duration_off"],
            )
            if not cfg.validate():
                raise ValueError("Invalid diarization config")
            self._diarizer = so.OfflineSpeakerDiarization(cfg)
        if self._asr is None:
            from faster_whisper import WhisperModel
            self._asr = WhisperModel(self.config["asr_model"], device=self.config["asr_device"],
                                     compute_type=self.config["asr_compute_type"], cpu_threads=self.config["num_threads"],
                                     local_files_only=self.config.get("asr_local_files_only", False))


    def run(self, video: str | Path, output_dir: str | Path | None = None) -> dict:
        c = self.config
        base = Path(output_dir or c["output_dir"])
        base.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="run_", dir=base))
        wav = run / "audio.wav"
        extract_audio(video, wav, c.get("sample_rate", 16000), c["ffmpeg_timeout"], c["max_video_seconds"])
        self._load_backends()
        signal, sr = sf.read(wav, dtype="float32")
        if sr != self._diarizer.sample_rate:
            raise ValueError("Diarization/training sample rate mismatch")
        turns = [{"start": float(x.start), "end": float(x.end), "speaker": f"SPEAKER_{x.speaker:02d}"}
                 for x in self._diarizer.process(signal).sort_by_start_time()]
        segments, info = self._asr.transcribe(str(wav), language=c["language"], beam_size=c["beam_size"],
                                             word_timestamps=True, vad_filter=True, condition_on_previous_text=False)
        words = [{"start": max(0.0, float(w.start)), "end": min(len(signal) / sr, float(w.end)), "text": w.word}
                 for segment in segments for w in (segment.words or []) if w.end > w.start and w.start < len(signal) / sr]
        aligned = align_words(words, turns)
        max_seconds = c.get("max_utterance_seconds", 9.99)
        utterances = merge_words(aligned, max_seconds, c["word_gap_limit"], c["merge_short_seconds"])
        result = {"status": "PREPROCESSING_COMPLETE", "video": str(Path(video).resolve()),
                  "audio": str(wav.resolve()), "sample_rate": sr, "duration": len(signal) / sr,
                  "language": info.language, "diarization": turns, "words": aligned,
                  "utterances": utterances, "emotion_model_run": False,
                  "warning": None if utterances else "Không nhận dạng được lời thoại; hãy kiểm tra âm thanh.",
                  "result_path": str((run / "result.json").resolve())}
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        with (run / "transcript.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=["start", "end", "speaker", "text", "ambiguous_speaker", "unassigned_speaker"])
            writer.writeheader()
            writer.writerows(utterances)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuần 1: video thành lời thoại theo người nói; không cần DATA/SECNet.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", default="configs/week1.yaml")
    parser.add_argument("--models-dir", type=Path, help="Thư mục chứa segmentation.onnx và speaker_embedding.onnx")
    parser.add_argument("--offline", action="store_true", help="Chỉ dùng ASR đã tải trong cache")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.models_dir:
        config["segmentation_model"] = str(args.models_dir / "segmentation.onnx")
        config["embedding_model"] = str(args.models_dir / "speaker_embedding.onnx")
    if args.offline:
        config["asr_local_files_only"] = True
    result = VideoPreprocessor(config).run(args.video, args.output_dir)
    print(json.dumps({"status": result["status"], "result_path": result["result_path"],
                      "utterances": len(result["utterances"]), "warning": result["warning"]}, ensure_ascii=False, indent=2))
