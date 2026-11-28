"""Verify PELD and previous-same-speaker indices in every real MELD batch."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from src.dataset import MELDDataset, collate_fn


def verify(root: Path):
    torch.set_num_threads(1)
    table = json.loads((Path(__file__).resolve().parents[1] / "configs/personality_peld.json").read_text(encoding="utf-8"))["personality"]
    totals = {}
    for split in ("train", "dev", "test"):
        dataset = MELDDataset(split, root)
        names = {v: k for k, v in dataset.meta["speaker_global_mapping"].items()}
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        utterances = 0
        for number, batch in enumerate(loader):
            for b in range(batch["utt_mask"].shape[0]):
                source = dataset[number * 4 + b]
                previous = {}
                for j, speaker in enumerate(source["speaker_local"]):
                    assert int(batch["prev_idx"][b, j]) == previous.get(speaker, -1)
                    previous[speaker] = j
                    gid = source["speaker_global"][j]
                    expected = table.get(names[gid])
                    assert bool(batch["has_personality"][b, j]) == (expected is not None)
                    assert torch.equal(batch["personality"][b, j], torch.tensor(expected or [0.] * 5))
                    utterances += 1
            assert torch.all(batch["prev_idx"][~batch["utt_mask"]] == -1)
            assert not batch["has_personality"][~batch["utt_mask"]].any()
            if number == 0:
                print(split, {key: list(batch[key].shape) for key in ("audio", "personality", "prev_idx")}, flush=True)
        totals[split] = {"dialogues": len(dataset), "utterances": utterances, "batches": len(loader)}
    print(json.dumps({"status": "PASS", "splits": totals}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    verify(parser.parse_args().data_root)
