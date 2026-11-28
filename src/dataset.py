"""MELD dialogue -> tensor dict: tokens (B,N,T), audio (B,N,F,80), F multiple of 8."""

import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset


class MELDDataset(Dataset):
    """Read trusted locally prepared PKL/PT files; preserve original utterance labels."""

    def __init__(self, split: str = "train", root: str | Path = "data/meld_processed", max_frames: int | None = None):
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be train/dev/test")
        if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0):
            raise ValueError("max_frames must be a positive integer or None")
        self.root, self.split, self.max_frames = Path(root), split, max_frames
        with (self.root / f"{split}_dialogues.pkl").open("rb") as f:
            self.dialogues = pickle.load(f)
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        table = self.meta.get("personality", {})
        self.personality = {}
        for name, values in table.items():
            values = torch.tensor(values, dtype=torch.float32)
            if values.shape != (5,) or not torch.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"Invalid O,C,E,A,N personality for {name}")
            gid = self.meta["speaker_global_mapping"].get(name)
            if gid is not None and 0 <= gid < 6:
                self.personality[gid] = values.tolist()
        self._audio = None

    def __len__(self):
        return len(self.dialogues)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_audio"] = None
        return state

    def __getitem__(self, index: int) -> dict:
        if self._audio is None:
            self._audio = torch.load(self.root / f"{self.split}_audio.pt", map_location="cpu", weights_only=True, mmap=True)
        item = dict(self.dialogues[index])
        item["audio"] = [self._audio[f"dia{item['dialogue_id']}_utt{u}"][:self.max_frames] for u in item["utt_ids"]]
        item["personality"] = [self.personality.get(g, [0.0] * 5) for g in item["speaker_global"]]
        item["has_personality"] = [g in self.personality for g in item["speaker_global"]]
        return item


def collate_fn(items: list[dict], max_speakers: int = 10, frame_multiple: int = 8) -> dict[str, torch.Tensor]:
    """Add Big Five and previous SAME speaker indices; ignore padding and absent audio."""
    if not items or max_speakers <= 0 or frame_multiple <= 0:
        raise ValueError("Nonempty batch and positive padding/speaker limits required")
    lengths = [len(d["utt_ids"]) for d in items]
    if min(lengths) == 0:
        raise ValueError("Empty dialogues are not valid dataset items")
    for d, count in zip(items, lengths, strict=True):
        for key in ["token_ids", "audio", "speaker_local", "speaker_global", "has_audio", "emotion_id", "shift", "personality", "has_personality"]:
            if len(d[key]) != count:
                raise ValueError(f"Unaligned {key}")
        for a, present in zip(d["audio"], d["has_audio"], strict=True):
            if a.ndim != 2 or a.shape[1] != 80 or bool(len(a)) != bool(present):
                raise ValueError("Invalid audio dimensions/has_audio")
            if not torch.isfinite(a).all():
                raise ValueError("Non-finite audio")
        if any(int(s) < 0 for s in d["speaker_local"]):
            raise ValueError("Negative real speaker ID")
        if any(not 0 <= int(g) <= 6 for g in d["speaker_global"]):
            raise ValueError("Invalid global speaker ID")
    b, n = len(items), max(lengths)
    t = max(1, max(len(ids) for d in items for ids in d["token_ids"]))
    raw_f = max(len(a) for d in items for a in d["audio"])
    f = max(frame_multiple, (raw_f + frame_multiple - 1) // frame_multiple * frame_multiple)
    out = {
        "tokens": torch.zeros(b, n, t, dtype=torch.long),
        "token_mask": torch.zeros(b, n, t, dtype=torch.bool),
        "audio": torch.zeros(b, n, f, 80, dtype=torch.float32),
        "audio_mask": torch.zeros(b, n, f, dtype=torch.bool),
        "has_audio": torch.zeros(b, n, dtype=torch.bool),
        "speaker": torch.zeros(b, n, dtype=torch.long),
        "speaker_global": torch.full((b, n), 6, dtype=torch.long),
        "personality": torch.zeros(b, n, 5, dtype=torch.float32),
        "has_personality": torch.zeros(b, n, dtype=torch.bool),
        "prev_idx": torch.full((b, n), -1, dtype=torch.long),
        "utt_mask": torch.zeros(b, n, dtype=torch.bool),
        "emotion": torch.full((b, n), -100, dtype=torch.long),
        "sentiment": torch.full((b, n), -100, dtype=torch.long),
        "shift": torch.full((b, n), -1, dtype=torch.long),
    }
    for i, d in enumerate(items):
        size = lengths[i]
        out["utt_mask"][i, :size] = True
        for dest, source in [("speaker_global", "speaker_global"), ("has_audio", "has_audio"), ("emotion", "emotion_id"), ("shift", "shift")]:
            out[dest][i, :size] = torch.as_tensor(d[source], dtype=out[dest].dtype)
        if "sentiment_id" in d:
            out["sentiment"][i, :size] = torch.tensor(d["sentiment_id"])
        previous = {}
        for j, (ids, a, speaker) in enumerate(zip(d["token_ids"], d["audio"], d["speaker_local"], strict=True)):
            ids = ids or [1]
            if any(int(x) < 0 for x in ids):
                raise ValueError("Negative token ID")
            out["tokens"][i, j, :len(ids)] = torch.as_tensor(ids, dtype=torch.long)
            out["token_mask"][i, j, :len(ids)] = True
            out["audio"][i, j, :len(a)] = a.float()
            out["audio_mask"][i, j, :len(a)] = True
            out["speaker"][i, j] = min(int(speaker), max_speakers - 1)
            # Use original IDs: clamping speaker embeddings must not merge speaker histories.
            out["prev_idx"][i, j] = previous.get(speaker, -1)
            previous[speaker] = j
            if d["has_personality"][j]:
                p = torch.as_tensor(d["personality"][j], dtype=torch.float32)
                if int(d["speaker_global"][j]) == 6 or p.shape != (5,) or not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                    raise ValueError("Invalid known personality")
                out["personality"][i, j] = p
                out["has_personality"][i, j] = True
    return out
