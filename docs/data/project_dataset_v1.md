# Project Dataset V1: session-split OBB assembly

Project Dataset V1 consumes **QA-clean Phase 1D annotation workspaces**, all for one `part_id`. Run `python scripts/qa_annotations.py WORKSPACE` for each workspace first: errors block assembly, while warnings (including near-square `angle_evaluable=false`) do not. The assembler re-runs QA, including current source hashes, JPEG dimensions and record/path validation, before creating output.

Create an explicit JSON split plan assigning **every supplied source session ID exactly once**:

```json
{
  "schema_version": 1,
  "splits": {
    "train": ["session-a"],
    "val": ["session-b"],
    "test": ["session-c"]
  }
}
```

The three arrays must be nonempty; IDs must match `[A-Za-z0-9][A-Za-z0-9_-]*`. Unknown, missing or repeated IDs fail. Capture sessions are indivisible: **no automatic or frame-level random splitting**. Each split must retain at least one *annotated* front and back image after exclusions.

```powershell
python scripts/build_project_dataset.py --annotation data/annotations/session-a --annotation data/annotations/session-b --annotation data/annotations/session-c --split-plan plan.json --output data/project-v1
python scripts/audit_dataset.py data/project-v1 --classes 2
```

The output must be new and disjoint from every source and annotation workspace. On an unexpected write error a partial output may remain visible: inspect/remove it deliberately before retrying; assembly makes no claim of an atomic whole-dataset commit. Sources and sidecars are never changed. Images are hardlinked where possible and copied otherwise.

Output layout:

```text
project-v1/
  data.yaml
  dataset_manifest.json
  images/{train,val,test}/<session_id>/<face>/<original relative tail>
  labels/{train,val,test}/<session_id>/<face>/<same stem>.txt
```

JPEG extensions and nested paths are preserved in images. Each annotated image receives one Ultralytics OBB row: class ID followed by eight validated normalized corner coordinates in recorded perimeter order. Numeric class mapping is assigned **only here**: `0: <part_id>_front`, `1: <part_id>_back` (FACE_LABELS order). Explicitly excluded images have neither output image nor label; they are not negative samples. `data.yaml` declares `path: .`, the three image directories, and the exact class mapping. `dataset_manifest.json` records schema version, part, mapping, split session IDs, split face/image and exclusion counts, source and workspace paths/provenance hashes, and `training_gate: "closed"`.

Successful assembly and a clean dataset syntax audit **do not unlock training**. Training Gate remains CLOSED; visual and project-level approval is separate.
