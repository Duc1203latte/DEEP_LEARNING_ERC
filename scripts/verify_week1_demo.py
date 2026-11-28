"""Run actual ASR/diarization on supplied speech videos without any MELD dataset."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import soundfile as sf
from demo.preprocess_video import VideoPreprocessor, load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--models-dir", type=Path, default=Path("outputs/demo_models"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/week1_verification"))
    args = parser.parse_args()
    config = load_config("configs/week1.yaml")
    config.update(segmentation_model=str(args.models_dir / "segmentation.onnx"),
                  embedding_model=str(args.models_dir / "speaker_embedding.onnx"),
                  asr_local_files_only=args.offline)
    processor = VideoPreprocessor(config)
    records = []
    for video in args.videos:
        start = time.perf_counter()
        print(f"RUN {video}", flush=True)
        result = processor.run(video, args.output)
        audio = sf.info(result["audio"])
        assert audio.samplerate == 16000 and audio.channels == 1
        assert result["status"] == "PREPROCESSING_COMPLETE" and not result["emotion_model_run"]
        assert result["diarization"] and result["words"] and result["utterances"]
        assert " ".join(u["text"] for u in result["utterances"]) == " ".join(w["text"] for w in result["words"])
        assert all(0 <= u["start"] < u["end"] <= result["duration"] for u in result["utterances"])
        assert all(u["end"] - u["start"] <= config["max_utterance_seconds"] for u in result["utterances"])
        assert all("emotion" not in u for u in result["utterances"])
        csv_path = Path(result["result_path"]).with_name("transcript.csv")
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            assert [r["text"] for r in csv.DictReader(stream)] == [u["text"] for u in result["utterances"]]
        assert "src.preprocess" not in sys.modules and "demo.pipeline" not in sys.modules
        records.append({"status": "PASS", "video": str(video), "utterances": len(result["utterances"]),
                        "words": len(result["words"]), "duration": result["duration"],
                        "seconds": time.perf_counter() - start, "result_path": result["result_path"]})
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "verification.json").write_text(json.dumps({"status": "PASS", "scope": "Actual execution, not WER/DER accuracy certification", "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
