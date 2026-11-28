"""Week-one PELD, same-speaker indices and video segmentation tests."""
import json
import pytest
import torch
from demo.preprocess_video import align_words, merge_words
from src.dataset import collate_fn
from scripts.add_personality import add_personality


def test_week1_import_does_not_load_training_or_secnet():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-X", "utf8", "-c",
                    "import sys; from demo.preprocess_video import VideoPreprocessor; "
                    "VideoPreprocessor({}); assert 'torch' not in sys.modules; "
                    "assert 'src.preprocess' not in sys.modules; assert 'demo.pipeline' not in sys.modules"], check=True)
    from demo.preprocess_video import load_config
    config = load_config()
    assert "data_root" not in config and "checkpoint" not in config
    assert config["sample_rate"] == 16000

def item(speakers=(0, 1, 0), audio=True):
    n = len(speakers)
    return {"dialogue_id": 0, "utt_ids": list(range(n)), "token_ids": [[2, 3]] * n,
            "speaker_local": list(speakers), "speaker_global": [0] * n,
            "audio": [torch.ones(7, 80, dtype=torch.float16) if audio else torch.empty(0, 80, dtype=torch.float16) for _ in speakers],
            "has_audio": [audio] * n, "emotion_id": [0] * n, "shift": [-1] * n,
            "personality": [[0.2] * 5] * n, "has_personality": [True] * n}


def test_contract_padding_and_previous_same_speaker():
    a, b = item(), item((0,), audio=False)
    a["token_ids"][1] = []
    batch = collate_fn([a, b])
    assert batch["prev_idx"].tolist() == [[-1, -1, 0], [-1, -1, -1]]
    assert batch["audio"].shape == (2, 3, 8, 80)
    assert batch["audio"].dtype == torch.float32
    assert batch["audio_mask"].dtype == torch.bool
    assert batch["audio_mask"][0, 0].sum() == 7
    assert not batch["audio_mask"][1].any()
    assert batch["tokens"][0, 1, 0] == 1
    assert batch["token_mask"][0, 1].sum() == 1
    assert batch["emotion"][1, 1:].tolist() == [-100, -100]
    assert not batch["has_personality"][1, 1:].any()


def test_clamping_does_not_merge_previous_speaker_histories():
    batch = collate_fn([item((9, 10, 9, 10))])
    assert batch["speaker"].tolist() == [[9, 9, 9, 9]]
    assert batch["prev_idx"].tolist() == [[-1, -1, 0, 1]]


def test_missing_personality_and_all_missing_audio():
    a = item(audio=False)
    a["has_personality"] = [False] * 3
    a["speaker_global"] = [6] * 3
    out = collate_fn([a])
    assert not out["personality"].any() and not out["has_personality"].any()
    assert out["audio"].shape[2] == 8 and not out["audio_mask"].any()


@pytest.mark.parametrize("fault", ["alignment", "audio", "personality", "global", "negative_speaker"])
def test_invalid_items_rejected(fault):
    a = item()
    if fault == "alignment": a["shift"] = []
    if fault == "audio": a["audio"][0][0, 0] = float("nan")
    if fault == "personality": a["personality"][0] = [1, 2]
    if fault == "global": a["speaker_global"][0] = 99
    if fault == "negative_speaker": a["speaker_local"][0] = -1
    with pytest.raises(ValueError): collate_fn([a])


def test_alignment_overlap_gap_and_blank():
    words = [{"start": 0, "end": 1, "text": " Hello"}, {"start": 2, "end": 3, "text": "world!"}, {"start": 4, "end": 5, "text": " "}]
    turns = [{"start": 0, "end": 1, "speaker": "A"}, {"start": 0.5, "end": 1.5, "speaker": "B"}]
    aligned = align_words(words, turns)
    assert len(aligned) == 2 and aligned[0]["speaker"] == "A" and aligned[0]["ambiguous_speaker"]
    assert aligned[1]["speaker"] == "UNKNOWN" and aligned[1]["unassigned_speaker"]


def test_merge_preserves_every_word_and_duration_limit():
    words = [{"start": i, "end": i + 0.8, "text": str(i), "speaker": "A" if i < 12 else "B", "ambiguous_speaker": False, "unassigned_speaker": False} for i in range(20)]
    merged = merge_words(words, 9.99, 0.7, 1.5)
    assert " ".join(s["text"] for s in merged).split() == [str(i) for i in range(20)]
    assert all(s["end"] - s["start"] <= 9.99 for s in merged)
    assert merged[-1]["speaker"] == "B"


def test_short_same_speaker_sentences_merge():
    words = [{"start": 0, "end": 0.4, "text": "Hi!", "speaker": "A", "ambiguous_speaker": False, "unassigned_speaker": False},
             {"start": 0.5, "end": 1.1, "text": "Hello.", "speaker": "A", "ambiguous_speaker": False, "unassigned_speaker": False}]
    assert len(merge_words(words, 10, 0.7, 1.5)) == 1


def test_personality_update_is_idempotent_and_keeps_original(tmp_path):
    root, backup = tmp_path / "data", tmp_path / "backup"
    root.mkdir()
    original = b'{"unrelated": 123}'
    (root / "meta.json").write_bytes(original)
    (root / "manifest.json").write_text(json.dumps({"artifacts": [{"path": "meta.json", "bytes": len(original), "sha256": "before"}]}))
    from pathlib import Path
    table = Path(__file__).resolve().parents[1] / "configs/personality_peld.json"
    first = add_personality(root, table, backup)
    meta, manifest = (root / "meta.json").read_bytes(), (root / "manifest.json").read_bytes()
    second = add_personality(root, table, backup)
    assert first["status"] == "PASS" and second["status"] == "ALREADY_PRESENT"
    assert (root / "meta.json").read_bytes() == meta
    assert (root / "manifest.json").read_bytes() == manifest
    assert Path(first["backup"]).read_bytes() == original
