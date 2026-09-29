# Project Dataset V1: V4.1 four-way session-split OBB assembly

Project Dataset V1 consumes **QA-clean V4.1 Phase 1 OBB annotation workspaces**, all for one `part_id`. Run `python scripts/qa_annotations.py WORKSPACE` for every workspace first: QA errors block assembly, while warnings such as near-square `angle_evaluable=false` do not. The assembler re-runs annotation QA and source-integrity checks before creating output.

This tool implements the V4.1 **four-way whole-session split contract**. It is data assembly infrastructure only: successful assembly does not mean the Experiment Protocol is frozen and does not open the Dataset V1 Gate.

## Split plan schema v2

Create an explicit JSON plan assigning **every supplied source session ID exactly once**:

```json
{
  "schema_version": 2,
  "splits": {
    "train": ["session-a"],
    "val": ["session-b"],
    "test": ["session-c"],
    "final_holdout": ["session-d"]
  }
}
```

All four arrays must be nonempty. Session IDs must match `[A-Za-z0-9][A-Za-z0-9_-]*`; unknown, missing, duplicate or unassigned IDs fail before output creation. Capture sessions are indivisible: **no automatic or frame-level random splitting**. Each split must retain at least one annotated `front` and one annotated `back` image after explicit exclusions.

Intended roles follow `docs/new-plans/EXPERIMENT_PROTOCOL_V1.md`:

- `train`: model fitting or Traditional CV parameter fitting.
- `val`: model/threshold/calibration/Risk Gate workpoint selection.
- `test`: frozen formal evaluation; no tuning.
- `final_holdout`: unseen final validation after Test-error feedback, hard-sample retraining or a later model iteration; no tuning.

If Test errors are fed back into training, the original Test set must not be reused as the new model's unbiased final evaluation. Use the untouched Final Holdout or a newly collected independent session.

## Build and audit

Example:

```powershell
python scripts/build_project_dataset.py `
  --annotation data/annotations/session-a `
  --annotation data/annotations/session-b `
  --annotation data/annotations/session-c `
  --annotation data/annotations/session-d `
  --split-plan plan.json `
  --output data/project-v1

python scripts/audit_dataset.py data/project-v1 --classes 2
```

The output must be new and disjoint from every source capture session and annotation workspace. Sources and sidecars are never modified. Images are hard-linked where possible and copied otherwise. An unexpected write error may leave a visible partial output; inspect and deliberately remove that output before retrying.

Output layout:

```text
project-v1/
  data.yaml
  dataset_manifest.json
  images/{train,val,test,final_holdout}/<session_id>/<face>/<original relative tail>
  labels/{train,val,test,final_holdout}/<session_id>/<face>/<same stem>.txt
```

JPEG extensions and nested paths are preserved. Each annotated image receives one Ultralytics OBB row: numeric class ID followed by eight validated normalized corner coordinates in recorded perimeter order. Numeric class mapping is assigned **only at Dataset V1 assembly**:

```text
0 = <part_id>_front
1 = <part_id>_back
```

Explicitly excluded images have neither output image nor label; exclusions are not synthesized negative samples.

## Training/tuning isolation

`data.yaml` intentionally contains only:

```yaml
path: .
train: images/train
val: images/val
test: images/test
names:
  0: "<part_id>_front"
  1: "<part_id>_back"
```

It deliberately has **no `final_holdout` key**, so the held-out data is not silently exposed through the ordinary Ultralytics train/val/test configuration.

`dataset_manifest.json` uses `schema_version: 2` and records:

- `part_id` and exact class-name mapping;
- all four session assignments;
- per-split image/face counts and exclusion counts;
- source-session and annotation provenance hashes;
- a dedicated `final_holdout` object with image path, label path and `purpose: "untuned final validation"`;
- `training_gate: "closed"`.

A clean syntax audit checks the assembled files, including the physical Final Holdout files. It does **not** authorize their use for tuning.

## V4.1 gate status

Software support for four-way assembly is now implemented, but the project still has **no real Part A four-way Dataset V1**. Phase 1 remains incomplete until the V4.1 protocol requirements are satisfied with real evidence, including Part A / Part B pre-freeze, Front/Back and orientation semantics, Operating Envelope draft, event-level Ground Truth schema, Test Condition Matrix, real capture and visual review, OBB QA, Dataset Audit, distribution/leakage checks, Traditional CV fairness rules, statistical reporting, cross-session repeatability and Final Test no-tuning confirmation.

Therefore:

> **Dataset V1 Gate CLOSED — no formal training yet.**

The authoritative gate and evaluation rules are in [EXPERIMENT_PROTOCOL_V1](../new-plans/EXPERIMENT_PROTOCOL_V1.md); the overall project direction is in the [V4.1 plan](../new-plans/面向柔性流水线的低成本工业零件姿态检测系统_项目方案_V4.1_优化版.md).
