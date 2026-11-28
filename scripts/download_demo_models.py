"""Download public input-preparation models, with atomic files and SHA-256 receipts."""
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCES = {
    "segmentation.onnx": "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx",
    "speaker_embedding.onnx": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx",
}


def download_models(root: Path = Path("outputs/demo_models")):
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "sources.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for name, url in SOURCES.items():
        target = root / name
        if target.exists() and name in receipts:
            with target.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() == receipts[name]["sha256"]:
                    continue
        temp = target.with_suffix(".onnx.partial")
        print(f"Downloading {name}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "SECNet-demo-preparation"})
        with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as f:
            digest = hashlib.sha256()
            while block := response.read(1024 * 1024):
                f.write(block)
                digest.update(block)
        if temp.stat().st_size < 1000:
            raise ValueError(f"Invalid model download: {name}")
        temp.replace(target)
        receipts[name] = {"url": url, "bytes": target.stat().st_size, "sha256": digest.hexdigest(),
                          "purpose": "Demo input preparation only; never an emotion encoder"}
        receipt_path.write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    download_models()
