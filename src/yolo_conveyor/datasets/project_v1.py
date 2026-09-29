"""Explicit session-split Project Dataset V1 assembly; never opens the training gate."""

import json
import os
from pathlib import Path
import re
import shutil

from yolo_conveyor.annotation import AnnotationSession, audit_annotations
from yolo_conveyor.annotation.contract import safe_file, safe_source_path, sha256_file
from yolo_conveyor.annotation.store import load_source
from yolo_conveyor.label_contract import FACE_LABELS, canonical_class_name

_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SPLITS = ("train", "val", "test")


def _plan(path: Path, sessions: set[str]) -> dict[str, list[str]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(plan, dict) or set(plan) != {"schema_version", "splits"} or
            type(plan["schema_version"]) is not int or plan["schema_version"] != 1 or
            not isinstance(plan["splits"], dict) or set(plan["splits"]) != set(_SPLITS)):
        raise ValueError("invalid split plan schema")
    splits = plan["splits"]
    assigned: set[str] = set()
    for split in _SPLITS:
        entries = splits[split]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{split} must contain at least one session")
        for session_id in entries:
            if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
                raise ValueError("unsafe session ID in split plan")
            if session_id in assigned:
                raise ValueError(f"duplicate session in split plan: {session_id}")
            if session_id not in sessions:
                raise ValueError(f"unknown session in split plan: {session_id}")
            assigned.add(session_id)
    if assigned != sessions:
        raise ValueError(f"unassigned sessions: {sorted(sessions - assigned)}")
    return {split: list(splits[split]) for split in _SPLITS}


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def build_project_dataset(annotations: list[str | Path], split_plan: str | Path,
                          output: str | Path) -> dict:
    """Preflight all inputs before mkdir; an I/O error during writing leaves visible partial output."""
    output = Path(output)
    # lexists catches even dangling symlinks, which Path.exists misses.
    if os.path.lexists(output):
        raise FileExistsError(output)
    destination = output.resolve()
    if not annotations:
        raise ValueError("at least one annotation workspace is required")
    workspaces: dict[str, tuple[AnnotationSession, Path, dict, dict[str, str]]] = {}
    part_id: str | None = None
    for workspace in annotations:
        store = AnnotationSession(workspace)
        qa = audit_annotations(store.path)
        if qa.errors:
            raise ValueError(f"annotation QA failed for {store.path}: {'; '.join(qa.errors)}")
        meta = store.metadata()
        source = store.source()
        session, expected = load_source(source)
        session_id = session["session_id"]
        if session_id in workspaces:
            raise ValueError(f"duplicate source session ID: {session_id}")
        if part_id is not None and session["part_id"] != part_id:
            raise ValueError("mixed part_id inputs")
        part_id = session["part_id"]
        if any(_overlaps(destination, p.resolve()) for p in (store.path, source)):
            raise ValueError("output overlaps annotation workspace or source session")
        workspaces[session_id] = (store, source, meta, expected)
    splits = _plan(Path(split_plan), set(workspaces))
    assert part_id is not None
    names = {str(index): canonical_class_name(part_id, face) for index, face in enumerate(FACE_LABELS)}
    counts: dict[str, dict] = {}
    excluded_counts: dict[str, int] = {}
    items: list[tuple[Path, Path, Path, str]] = []
    destinations: set[str] = set()
    for split, session_ids in splits.items():
        faces = {face: 0 for face in FACE_LABELS}
        excluded = 0
        for session_id in session_ids:
            store, source, _, expected = workspaces[session_id]
            for raw in sorted(expected):
                record = store.read_record(raw)
                if record["status"] == "excluded":
                    excluded += 1
                    continue
                relative = safe_source_path(raw, record["face"])
                image = safe_file(source, relative)
                # Revalidate again during preflight via the accepted store/contract APIs.
                if sha256_file(image) != record["source_image_sha256"]:
                    raise ValueError(f"source image hash drift: {raw}")
                tail = Path(*relative.parts[1:])
                prefix = Path(split) / session_id / record["face"]
                image_rel = Path("images") / prefix / tail
                label_rel = Path("labels") / prefix / tail.with_suffix(".txt")
                for path in (image_rel, label_rel):
                    key = path.as_posix().casefold()
                    if key in destinations:
                        raise ValueError(f"duplicate output path: {path}")
                    destinations.add(key)
                class_id = FACE_LABELS.index(record["face"])
                coordinates = [format(value, ".17g") for point in record["corners"] for value in point]
                row = " ".join((str(class_id), *coordinates)) + "\n"
                items.append((image, image_rel, label_rel, row))
                faces[record["face"]] += 1
        if any(not faces[face] for face in FACE_LABELS):
            raise ValueError(f"{split} lacks annotated front/back coverage")
        counts[split] = {"images": sum(faces.values()), "faces": faces}
        excluded_counts[split] = excluded
    provenance = []
    for session_id in sorted(workspaces):
        store, source, meta, _ = workspaces[session_id]
        provenance.append({"annotation_workspace": str(store.path.resolve()),
                           "source_session": str(source.resolve()), "source_session_id": session_id,
                           "source_session_sha256": meta["source_session_sha256"],
                           "source_manifest_sha256": meta["source_manifest_sha256"],
                           "annotation_sha256": sha256_file(store.path / "annotation.json"),
                           "record_sha256": {
                               raw: sha256_file(store.record_file(raw))
                               for raw in sorted(workspaces[session_id][3])},
                           "excluded": sum(store.read_record(raw)["status"] == "excluded"
                                           for raw in workspaces[session_id][3])})
    manifest = {"schema_version": 1, "part_id": part_id, "names": names,
                "splits": splits, "counts": counts, "excluded_counts": excluded_counts,
                "inputs": provenance, "training_gate": "closed"}
    # mkdir is deliberately last: failures after this point remain visible, never masquerading as atomic.
    output.mkdir(parents=True, exist_ok=False)
    for image, image_rel, label_rel, row in items:
        target = output / image_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(image, target)
        except OSError:
            shutil.copyfile(image, target)
        label = output / label_rel
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(row, encoding="utf-8")
    yaml = "path: .\n" + "".join(f"{split}: images/{split}\n" for split in _SPLITS)
    yaml += "names:\n" + "".join(f"  {i}: {json.dumps(name)}\n" for i, name in names.items())
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
