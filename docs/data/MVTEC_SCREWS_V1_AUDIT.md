# MVTec Screws public V1.0 audit (Dataset V1)

## Provenance and scope

- **Authoritative publication and license:** MVTec's official Screws dataset page states **384 images, 13 classes, 4426 OBB annotations**, under **CC BY-NC-SA 4.0 (non-commercial)**. This does not grant commercial usage rights. MVTec HALCON **26.05 release notes** report that some `class_13` orientation labels were wrong and were fixed. These are MVTec statements, not findings that the locally inspected V1.0 mirror incorporates the correction.
- **Local observation only:** Metadata inspected from a public GitHub **V1.0 mirror** is transfer/audit provenance, not the authoritative source for dataset counts, corrected labels, or licensing. It reports **384 images, 4427 annotations** (one annotation more than the official page). This discrepancy remains unresolved; do not silently reconcile it or assume the mirror is corrected/current.
- This is a metadata/conversion and selected-sample audit, **not** a full dataset approval, full label review, model evaluation or license clearance beyond the stated non-commercial restriction. The previously proposed **ICDM 3214/11** source could not be reliably verified and is **excluded**.

## Local V1.0 metadata checks

| Partition | Images | Annotations |
| --- | ---: | ---: |
| Train | 269 | 3119 |
| Val | 55 | 647 |
| Test | 60 | 661 |
| Total | 384 | 4427 |

No split filename overlap was found; the combined split metadata matches the unsplit metadata exactly. These checks do not establish sequence-level independence, corrected orientation labels, or valid geometry for every annotation.

Category **13** accounts for **321 annotations** in V1.0. Given MVTec's `class_13` orientation correction notice, this V1.0 copy is not assumed safe for orientation training. Converted corners in V1.0 metadata cross strict normalized **[0,1]** bounds in **50 annotations**; **22** are also outside the current Ultralytics **[-0.01,1.01]** tolerance. These cases are distributed among categories **1–4**. The converter rejects out-of-bounds corners rather than silently clipping them; tolerance is not a label-quality approval.

## Selected real-data conversion and audit

The safe sample used `train/screws_004` (**16 annotations**), `val/screws_001` (**16**), and `test/screws_022` (**8**). It excludes `class_13` and boundary-crossing boxes. Conversion to Ultralytics OBB labels succeeded; the existing audit reports **3 images, 3 labels, `errors=[]`**. This result applies only to these selected images and does not imply successful full-V1.0 conversion or a train-ready dataset. No raw images or weights are included here.

## Gate and next data work

**Training Gate closed.** Acquire a corrected/current full dataset **or** freeze an explicitly reviewed filtering policy before attempting full-dataset training. Resolve the official/local count discrepancy and verify label geometry, orientation, provenance and split suitability as part of that decision. MVTec Screws has **no `front` / `back` labels**: later small real-part collection, face-label criteria and an independently audited split remain required for the project task. No accuracy or FPS is asserted. The reported **31/31 full project tests PASS** is software verification only, not evidence of model quality.

See [capability matrix](DATASET_CAPABILITY_MATRIX.md) and [Phase 1 execution plan](../exec-plans/active/phase1_execution_plan.md).
