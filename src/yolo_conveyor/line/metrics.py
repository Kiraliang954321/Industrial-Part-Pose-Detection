"""Event-level diagnostics with explicit GT and uncertainty denominators."""

from collections import Counter
from yolo_conveyor.orientation.evaluation import rate, distribution


def decision_metrics(rows, key):
    for row in rows:
        if row["ground_truth"]["label"] not in ("CONFORMING", "NONCONFORMING", "INDETERMINATE"):
            raise ValueError("invalid GT; UNCERTAIN is a system decision, not ground truth")
        if row.get(key) and row[key]["outcome"] not in ("PASS", "FAIL_ORIENTATION", "UNCERTAIN"):
            raise ValueError("unsupported system decision")
    determinate = [r for r in rows if r["ground_truth"]["label"] in ("CONFORMING", "NONCONFORMING")]
    conforming = [r for r in determinate if r["ground_truth"]["label"] == "CONFORMING"]
    nonconforming = [r for r in determinate if r["ground_truth"]["label"] == "NONCONFORMING"]
    outcome = lambda r: r[key]["outcome"] if r.get(key) else None
    return {"events": len(rows), "determinate_gt": len(determinate), "conforming_gt": len(conforming),
            "nonconforming_gt": len(nonconforming), "indeterminate_gt": len(rows)-len(determinate),
            "outcome_counts": dict(Counter(outcome(r) or "NO_EVENT" for r in rows)),
            "wrong_accept": rate(sum(outcome(r) == "PASS" for r in nonconforming), len(nonconforming)),
            "wrong_reject": rate(sum(outcome(r) != "PASS" for r in conforming), len(conforming)),
            "hard_reject_conforming": rate(sum(outcome(r) == "FAIL_ORIENTATION" for r in conforming), len(conforming)),
            "uncertain_reject_conforming": rate(sum(outcome(r) == "UNCERTAIN" for r in conforming), len(conforming)),
            "coverage_determinate": rate(sum(outcome(r) in ("PASS", "FAIL_ORIENTATION") for r in determinate), len(determinate)),
            "decision_latency_sim_s": distribution([r[key]["latency_sim_s"] for r in rows if r.get(key)])}


def summarize(rows):
    ids = [r["event"]["event_id"] for r in rows if r.get("event")]
    automatic = [r for r in rows if r.get("event") and r["event"]["outcome"] != "UNCERTAIN"]
    halted = [r for r in rows if r.get("event") and r["event"]["outcome"] == "UNCERTAIN"]
    return {"single_frame": decision_metrics(rows, "single_frame"), "multi_frame": decision_metrics(rows, "event"),
            "missing_events": len(rows)-len(ids), "duplicate_events": len(ids)-len(set(ids)),
            "automatic_action_success": rate(sum(r["action"]["success"] for r in automatic), len(automatic)),
            "halted_for_review": len(halted), "safe_halt": rate(sum(r["action"]["safe_hold"] for r in halted), len(halted)),
            "automatic_active_segment_sim_s": distribution([r["elapsed_sim_s"] for r in automatic]),
            "decision_wall_ms_including_rendering_and_logging": distribution([r["decision_wall_ms"] for r in rows]),
            "continuous_line_throughput_parts_per_min": None,
            "throughput_reason": "Independent single-pallet episodes; unloading/return/loading/review recovery are not integrated. No continuous-line throughput claim."}
