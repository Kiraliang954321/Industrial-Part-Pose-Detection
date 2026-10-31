"""Small, explicit artifact helpers for the simulation development pipeline."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import sys
import site

TASK = "screw_directed_axis_v1"
KEYPOINTS = ["head_axis_end", "tip_axis_end"]
SCOPE = "simulation_development"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unlinked(path: str | Path) -> Path:
    """Reject symlinks and Windows junctions, including existing ancestors."""
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f"linked path is not supported: {part}")
    return path.resolve()


def new_directory(path: str | Path, sources=()) -> Path:
    destination = unlinked(path)
    if os.path.lexists(destination):
        raise FileExistsError(f"use a new output directory: {destination}")
    for source in sources:
        source = unlinked(source)
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError("output and source paths must not overlap")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def dataset_file(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or
            any(s in relative for s in ("\\", ":", "\x00")) or
            any(p in ("", ".", "..") for p in relative.split("/"))):
        raise ValueError("unsafe dataset-relative path")
    path = unlinked(root / relative)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or unsafe dataset file: {relative}")
    return path


def write_json(path: Path, value, *, replace: bool = False) -> None:
    """Only progress files in a newly created run may use replace=True."""
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def environment() -> dict:
    versions = {}
    for name in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python", "mujoco", "glfw",
                 "PyOpenGL", "etils", "zipp", "absl-py", "fsspec", "websockets"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "executable": sys.executable,
            "user_site_enabled": site.ENABLE_USER_SITE, "platform": platform.platform(), "packages": versions}


def source_hashes() -> dict:
    root = Path(__file__).resolve().parents[3]
    files = [*Path(__file__).parent.glob("*.py"),
             *(root / "src/yolo_conveyor/simulation").glob("*.py"),
             root / "src/yolo_conveyor/datasets/screw_orientation.py"]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files) if p.is_file()}
