# Label Contract V2 — per-part face names

Each new raw capture session is bound to one explicit `part_id`, a single ASCII token matching `[A-Za-z0-9][A-Za-z0-9_-]*`. The operator selects a **face label** from `front` or `back`; these are not final YOLO class names. For later annotation, the canonical class name is `<part_id>_<face>` (for example, `screw_A-1_front` and `screw_A-1_back`). The reusable rules live in `yolo_conveyor.label_contract` (`FACE_LABELS`, `validate_part_id`, `canonical_class_name`). No numeric class IDs are assigned at capture time.

A new `session.json` has `schema_version: 2`, `part_id`, `face_labels: ["front", "back"]`, `class_names` mapping each face to its canonical name, and `manifest_label_semantics: "face_label"`. The existing `labels` list and each `manifest.jsonl` attempt `label` remain `front`/`back` face labels for compatibility. Each session is for a single part ID; use a separate session for another part. Raw images remain in `front/` and `back/` directories.

This is raw capture only (`state: "raw_capture"`, `obb_labeled: false`, `train_ready: false`), not annotation or training readiness. **UNCERTAIN is runtime-only and is not a training class.** Training remains gated on actual OBB annotation, QA and session-aware split; neither capture nor this naming contract supplies these.
