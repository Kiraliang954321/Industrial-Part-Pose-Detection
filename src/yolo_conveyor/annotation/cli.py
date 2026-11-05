"""OpenCV review of accepted capture images into a separate annotation sidecar."""

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

from .contract import jpeg_dimensions, safe_file, safe_source_path, sha256_file
from .proposal_review import load_review_proposals
from .store import AnnotationSession, load_source
from .ui import review_image


def open_workspace(source: Path, workspace: Path | None = None) -> tuple[AnnotationSession, dict, dict[str, str]]:
    source = source.resolve(strict=True)
    session, accepted = load_source(source)
    workspace = Path("data/annotations") / session["session_id"] if workspace is None else Path(workspace)
    if workspace.exists() or workspace.is_symlink():
        store = AnnotationSession(workspace)
        meta = store.metadata()
        if (store.path.resolve() == source or store.path.resolve().is_relative_to(source) or
                source.is_relative_to(store.path.resolve()) or
                meta.get("schema_version") != 1 or meta.get("source_session") != str(source) or
                meta.get("source_session_id") != session["session_id"] or
                meta.get("part_id") != session["part_id"] or
                meta.get("class_names") != session["class_names"] or
                meta.get("source_session_sha256") != sha256_file(source / "session.json") or
                meta.get("source_manifest_sha256") != sha256_file(source / "manifest.jsonl") or
                not (store.path / "records").is_dir() or (store.path / "records").is_symlink()):
            raise ValueError("workspace does not match this source session")
    else:
        store = AnnotationSession.create(workspace, source)
    return store, session, accepted


def pending_images(store: AnnotationSession, accepted: dict[str, str], *, revisit: bool = False) -> list[str]:
    pending = []
    for name in accepted:
        path = store.record_file(name)
        if path.exists():
            record = store.read_record(name)  # Malformed records fail closed, not silently overwritten.
            if not revisit:
                continue
        pending.append(name)
    return pending


def run(source: Path, *, workspace: Path | None = None, max_window: int = 1000,
        revisit: bool = False, proposal_dir: Path | None = None) -> int:
    if max_window <= 0:
        raise ValueError("max window must be positive")
    proposals = load_review_proposals(proposal_dir, source) if proposal_dir is not None else None
    if proposals is not None:
        source = source.resolve(strict=True)
        frozen = next(iter(proposals.values()))
        if (sha256_file(source / "session.json") != frozen.source_session_sha256 or
                sha256_file(source / "manifest.jsonl") != frozen.source_manifest_sha256):
            raise ValueError("source metadata changed after candidate validation")
    store, session, accepted = open_workspace(source, workspace)
    pending = pending_images(store, accepted, revisit=revisit)
    source = store.source()
    for index, name in enumerate(pending, 1):
        face = accepted[name]
        image_path = safe_file(source, safe_source_path(name, face))
        if proposals is not None:
            candidate = proposals[name]
            if (sha256_file(source / "session.json") != candidate.source_session_sha256 or
                    sha256_file(source / "manifest.jsonl") != candidate.source_manifest_sha256):
                raise ValueError("source metadata changed during candidate review")
            payload = image_path.read_bytes()
            image_hash = hashlib.sha256(payload).hexdigest()
            if image_hash != candidate.source_image_sha256:
                raise ValueError(f"source image differs from frozen candidate: {name}")
            width, height = candidate.image_width, candidate.image_height
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8),
                                 cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            width, height = jpeg_dimensions(image_path)
            image_hash = sha256_file(image_path)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (height, width):
            raise ValueError(f"cannot decode source image or image dimensions differ: {name}")
        review_options = {"initial_corners": [list(point) for point in proposals[name].corners]} if proposals is not None else {}
        action, corners = review_image(image, session_id=session["session_id"],
                                       part_id=session["part_id"], face=face,
                                       class_name=session["class_names"][face],
                                       index=index, total=len(pending),
                                       existing=store.record_file(name).exists(), max_window=max_window,
                                       **review_options)
        if action == "quit":
            break
        if action == "skip":
            continue
        if action not in ("save", "exclude") or (action == "save" and corners is None):
            raise ValueError("invalid review action")
        if sha256_file(image_path) != image_hash or jpeg_dimensions(image_path) != (width, height):
            raise ValueError(f"source image changed during review: {name}")
        record = dict(schema_version=1, status="annotated" if action == "save" else "excluded",
                      part_id=session["part_id"], face=face,
                      class_name=session["class_names"][face], source_path=name,
                      source_image_sha256=image_hash,
                      image_width=width, image_height=height)
        if action == "save":
            record["corners"] = corners
        else:
            record["reason"] = "operator_excluded"
        store.write_record(record)  # Single persistence and validation path.
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="schema-v2 capture session directory")
    parser.add_argument("--workspace", type=Path, help="sidecar directory (default: data/annotations/<session_id>)")
    parser.add_argument("--max-window", type=int, default=1000, help="maximum image display edge in pixels")
    parser.add_argument("--revisit", action="store_true", help="explicitly revisit existing records; save/exclude replaces them")
    parser.add_argument("--proposals", type=Path,
                        help="verified candidate directory to preload; every image still requires operator confirmation")
    args = parser.parse_args(argv)
    options = {"proposal_dir": args.proposals} if args.proposals is not None else {}
    return run(args.source, workspace=args.workspace, max_window=args.max_window,
               revisit=args.revisit, **options)
