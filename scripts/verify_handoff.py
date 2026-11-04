"""Verify the public simulation handoff with the Python standard library."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/handoff/artifact-manifest.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path leaves the checkout: {relative}")
    return path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = read_json(MANIFEST)
    expected = {}
    for record in manifest["files"]:
        path = checked_path(record["path"])
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"File size differs: {record['path']}")
        expected[record["path"]] = record["sha256"]

    source_names = set()
    for relative in manifest["protocols"]:
        protocol = read_json(checked_path(relative))
        sources = protocol.get("sources", protocol.get("controller_source_hashes", {}))
        if not sources:
            raise ValueError(f"No frozen source inventory: {relative}")
        for name, digest in sources.items():
            if name in expected and expected[name] != digest:
                raise ValueError(f"Conflicting frozen source hashes: {name}")
            expected[name] = digest
            source_names.add(name)

    for name, digest in expected.items():
        if file_hash(checked_path(name)) != digest:
            raise ValueError(f"SHA256 differs: {name}")

    print(json.dumps({
        "status": "verified",
        "published_files": len(manifest["files"]),
        "frozen_protocols": len(manifest["protocols"]),
        "unique_frozen_sources": len(source_names),
        "scope": "Published artifacts and frozen source bytes only",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
