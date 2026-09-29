"""Convert COCO-style MVTec Screws split JSONs to Ultralytics OBB labels.

MVTec bbox = [row, col, width, height, phi]: row/col are the box
centre in image pixels, and phi is counter-clockwise from the horizontal
image axis (so positive phi decreases image y). No clipping is performed.
"""

import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def _integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} must be an integer")
    return value


def _positive(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{description} must be finite and positive")
    return float(value)


def bbox_corners(bbox: object, image_width: object, image_height: object) -> tuple[float, ...]:
    """Return four normalized (x,y) corners in perimeter order; reject invalid boxes."""
    width_px = _positive(image_width, "image width")
    height_px = _positive(image_height, "image height")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 5:
        raise ValueError("bbox must be [row, col, width, height, phi]")
    row, col, width, height, phi = bbox
    for value, name in ((row, "row"), (col, "col"), (phi, "phi")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    width = _positive(width, "bbox width")
    height = _positive(height, "bbox height")
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    coordinates = []
    for dx, dy in ((-width / 2, -height / 2), (width / 2, -height / 2),
                   (width / 2, height / 2), (-width / 2, height / 2)):
        x = (col + dx * cos_phi + dy * sin_phi) / width_px
        y = (row - dx * sin_phi + dy * cos_phi) / height_px
        if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError("converted corner is non-finite or outside image bounds")
        coordinates.extend((x, y))
    return tuple(coordinates)


def _relative_image(file_name: object) -> Path:
    if not isinstance(file_name, str) or not file_name or "\\" in file_name:
        raise ValueError("image file_name must be a relative POSIX path")
    path = PurePosixPath(file_name)
    if path.is_absolute() or any(part in (".", "..") for part in file_name.split("/")) or ":" in file_name:
        raise ValueError("image file_name must stay within source")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"unsupported image suffix: {file_name}")
    return Path(*path.parts)


def _load_split(source: Path, split: str) -> dict:
    with (source / f"mvtec_screws_{split}.json").open(encoding="utf-8") as stream:
        data = json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON value: {value}")))
    if not isinstance(data, dict) or not all(isinstance(data.get(key), list) for key in ("images", "annotations", "categories")):
        raise ValueError(f"invalid {split} split: expected images, annotations and categories arrays")
    return data


def convert_mvtec_screws(source: Path, output: Path) -> dict:
    """Convert all three splits into a new output directory; never modify source.

    Raises ValueError on invalid annotations and FileExistsError for an occupied
    output directory. Images are hardlinked when possible, otherwise copied.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output directories must not overlap")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    splits = {split: _load_split(source, split) for split in SPLITS}
    categories: dict[int, str] = {}
    plans: dict[str, list[tuple[Path, Path, list[tuple[int, tuple[float, ...]]]]]] = {}
    # Validate the entire input before creating output files.
    for split, data in splits.items():
        for category in data["categories"]:
            if not isinstance(category, dict):
                raise ValueError("invalid category")
            category_id = _integer(category.get("id"), "category id")
            name = category.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("category name must be nonempty")
            if category_id in categories and categories[category_id] != name:
                raise ValueError(f"inconsistent category id: {category_id}")
            categories[category_id] = name
    if not categories:
        raise ValueError("no categories")
    mapping = {category_id: new_id for new_id, category_id in enumerate(sorted(categories))}

    source_root = source.resolve()
    for split, data in splits.items():
        images: dict[int, tuple[Path, Path, float, float]] = {}
        stems: set[str] = set()
        for image in data["images"]:
            if not isinstance(image, dict):
                raise ValueError("invalid image entry")
            image_id = _integer(image.get("id"), "image id")
            relative = _relative_image(image.get("file_name"))
            stem = relative.with_suffix("").as_posix()
            if image_id in images or stem in stems:
                raise ValueError(f"duplicate image id or stem in {split}: {image_id}")
            stems.add(stem)
            image_path = source / relative
            if not image_path.is_file():
                image_path = source / "images" / relative
            if not image_path.is_file() or not image_path.resolve().is_relative_to(source_root):
                raise ValueError(f"missing or unsafe source image: {relative}")
            images[image_id] = (image_path, relative, _positive(image.get("width"), "image width"),
                                _positive(image.get("height"), "image height"))
        annotations: dict[int, list[tuple[int, tuple[float, ...]]]] = {image_id: [] for image_id in images}
        for annotation in data["annotations"]:
            if not isinstance(annotation, dict):
                raise ValueError("invalid annotation entry")
            image_id = _integer(annotation.get("image_id"), "annotation image_id")
            category_id = _integer(annotation.get("category_id"), "annotation category_id")
            if image_id not in images or category_id not in mapping:
                raise ValueError("annotation references unknown image or category")
            _, _, width, height = images[image_id]
            annotations[image_id].append((mapping[category_id], bbox_corners(annotation.get("bbox"), width, height)))
        plans[split] = [(image_path, relative, annotations[image_id])
                        for image_id, (image_path, relative, _, _) in images.items()]

    output.mkdir(parents=True)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
        for image_path, relative, boxes in plans[split]:
            dest_image = output / "images" / split / relative
            dest_label = output / "labels" / split / relative.with_suffix(".txt")
            dest_image.parent.mkdir(parents=True, exist_ok=True)
            dest_label.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(image_path, dest_image)
            except OSError:
                shutil.copyfile(image_path, dest_image)
            dest_label.write_text("".join(f"{category} " + " ".join(f"{value:.16g}" for value in corners) + "\n"
                                          for category, corners in boxes), encoding="utf-8")
    names = [categories[category_id] for category_id in sorted(categories)]
    (output / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" +
                                      "".join(f"  {index}: {json.dumps(name, ensure_ascii=False)}\n"
                                              for index, name in enumerate(names)), encoding="utf-8")
    manifest = {"source": str(source), "splits": {split: {"images": len(plans[split]),
                 "annotations": sum(len(boxes) for _, _, boxes in plans[split])} for split in SPLITS},
                "category_id_map": {str(key): value for key, value in mapping.items()}, "names": names}
    (output / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
