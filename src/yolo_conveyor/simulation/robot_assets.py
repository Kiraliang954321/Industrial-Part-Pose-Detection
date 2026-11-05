"""Fetch and audit one pinned, licensed Menagerie robot; no executable remote scripts."""

import argparse
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET

from yolo_conveyor.orientation.common import new_directory, sha256, unlinked, utc_now, write_json, dataset_file

PANDA_COMMIT = "a662de603224c47123b0783821c58ad4402ad3e3"
PANDA_BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{PANDA_COMMIT}/franka_emika_panda/"


def fetch_panda(output: str | Path) -> dict:
    out = new_directory(output)
    records = []
    def download(relative):
        if not re.fullmatch(r"(?:assets/)?[A-Za-z0-9_.-]+", relative):
            raise ValueError("unexpected asset-relative path")
        request = urllib.request.Request(PANDA_BASE + relative, headers={"User-Agent": "screw-m4-research/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(30 * 1024 * 1024 + 1)
        if len(payload) > 30 * 1024 * 1024:
            raise ValueError("single asset exceeds download budget")
        target = out / relative
        target.parent.mkdir(exist_ok=True, parents=True)
        with target.open("xb") as stream:
            stream.write(payload)
        records.append({"path": relative, "sha256": sha256(target), "bytes": len(payload)})
        return target
    for name in ("panda.xml", "README.md", "LICENSE"):
        download(name)
    xml = ET.parse(out / "panda.xml")
    files = sorted({element.attrib["file"] for element in xml.findall(".//mesh")})
    for index, name in enumerate(files):
        download(f"assets/{name}")
        if index % 10 == 0:
            print(f"Panda assets: {index + 1}/{len(files)}", flush=True)
    manifest = {"schema_version": 1, "robot": "franka_emika_panda", "commit": PANDA_COMMIT,
                "source": PANDA_BASE, "license": "Apache-2.0", "created_at": utc_now(), "files": records,
                "bytes": sum(r["bytes"] for r in records)}
    write_json(out / "asset-manifest.json", manifest)
    return manifest


def audit_panda(path: str | Path) -> Path:
    root = unlinked(path)
    manifest = json.loads(dataset_file(root, "asset-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("commit") != PANDA_COMMIT or manifest.get("robot") != "franka_emika_panda":
        raise ValueError("unexpected robot asset version")
    for record in manifest["files"]:
        if sha256(dataset_file(root, record["path"])) != record["sha256"]:
            raise ValueError(f"robot asset hash drift: {record['path']}")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = fetch_panda(args.output)
    audit_panda(args.output)
    print(json.dumps({"commit": result["commit"], "files": len(result["files"]), "bytes": result["bytes"]}))
    return 0
