"""Add cited PELD metadata atomically, retaining original meta/manifest backups."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def add_personality(root: Path, table_path: Path, backup: Path) -> dict:
    table = json.loads(table_path.read_text(encoding="utf-8"))
    values = table["personality"]
    expected = {"Rachel", "Ross", "Monica", "Chandler", "Joey", "Phoebe"}
    if set(values) != expected or table["trait_order"] != ["O", "C", "E", "A", "N"]:
        raise ValueError("Invalid PELD identities/trait order")
    if any(len(p) != 5 or any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in p) for p in values.values()):
        raise ValueError("Invalid PELD values")
    path = root / "meta.json"
    original = path.read_bytes()
    meta = json.loads(original)
    if "personality" in meta and meta["personality"] != values:
        raise ValueError("Existing personality differs; refusing silent replacement")
    if (meta.get("personality") == values and
            meta.get("personality_provenance") == {k: v for k, v in table.items() if k != "personality"}):
        return {"status": "ALREADY_PRESENT", "characters": len(values),
                "meta_sha256": hashlib.sha256(original).hexdigest()}
    backup.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(original).hexdigest()
    saved = backup / f"meta_before_personality_{key}.json"
    if not saved.exists():
        saved.write_bytes(original)
    meta["personality"] = values
    meta["personality_provenance"] = {k: v for k, v in table.items() if k != "personality"}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_manifest = backup / "manifest_before_personality.json"
        if not old_manifest.exists():
            shutil.copy2(manifest_path, old_manifest)
        for entry in manifest.get("artifacts", []):
            if entry["path"] == "meta.json":
                entry.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest["total_output_bytes"] = sum(e["bytes"] for e in manifest.get("artifacts", []))
        manifest["personality_metadata_update"] = {"source": table["source"], "original_meta_sha256": key}
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(manifest_path)
    return {"status": "PASS", "characters": len(values), "original_meta_sha256": key, "meta_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "backup": str(saved)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--table", type=Path, default=Path("configs/personality_peld.json"))
    parser.add_argument("--backup", type=Path, default=Path("outputs/part_d/original_metadata"))
    args = parser.parse_args()
    print(json.dumps(add_personality(args.data_root, args.table, args.backup), ensure_ascii=False, indent=2))
