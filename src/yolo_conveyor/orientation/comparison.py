"""Select development operating points on Val and publish an explicit comparison."""

import argparse
import json
from pathlib import Path

from .common import TASK, SCOPE, new_directory, sha256, unlinked, utc_now, write_json


def selection_key(report: dict) -> tuple:
    stats = report["overall"]
    median = stats["angle_error_deg_conditional"]["median"]
    return (-stats["correct_images_10deg"]["x"], float("inf") if median is None else median)


def fraction(value: dict) -> str:
    if value["n"] == 0:
        return "0/0 (N/A)"
    low, high = value["wilson95"]
    return f"{value['x']}/{value['n']} ({value['estimate']:.2%}; 95% CI {low:.2%}–{high:.2%})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [unlinked(p) for p in args.reports]
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    for report in reports:
        if (report.get("task") != TASK or report.get("scope") != SCOPE or report.get("split") != "val" or
                report.get("status") != "completed" or report.get("formal_training_gate") != "closed"):
            raise ValueError("comparison requires completed, simulation development Val reports")
    if len({r["dataset_manifest_sha256"] for r in reports}) != 1:
        raise ValueError("all candidate reports must use the identical dataset manifest")
    if len({r["overall"]["images"] for r in reports}) != 1:
        raise ValueError("candidate image counts differ")
    if {r["method"] for r in reports} != {"cv", "pose"}:
        raise ValueError("provide both CV and Pose candidates")
    out = new_directory(args.output, paths)
    selected = {}
    for method in ("cv", "pose"):
        candidates = [(i, r) for i, r in enumerate(reports) if r["method"] == method]
        index, report = min(candidates, key=lambda candidate: (*selection_key(candidate[1]), candidate[0]))
        selected[method] = {"report": str(paths[index]), "report_sha256": sha256(paths[index]),
                            "parameters": report["parameters"], "overall": report["overall"],
                            "checkpoint_sha256": report["checkpoint_sha256"]}
    value = {"task": TASK, "scope": SCOPE, "formal_training_gate": "closed", "created_at": utc_now(),
             "selection_split": "val", "dataset_manifest_sha256": reports[0]["dataset_manifest_sha256"],
             "selection_rule": "maximize correct_images_10deg; tie: lower conditional median angle error; then input order",
             "candidates": [{"report": str(p), "sha256": sha256(p), "method": r["method"],
                             "parameters": r["parameters"], "overall": r["overall"]} for p, r in zip(paths, reports)],
             "selected": selected, "claims": "Val was used for selection; this comparison is not an unbiased Test result"}
    write_json(out / "selection.json", value)
    lines = ["# 螺丝朝向：开发验证集比较", "", "范围：仿真代理、同一数据版本的 Val；真实门禁 CLOSED。", "",
             "Val 同时用于选择参数/权重，因此这些数值不是无偏 Test 成绩。10° 是开发诊断阈值，不是工业容差。", "",
             "| 方法 | 图像成功（10°） | 方向覆盖 / 全 GT | 方向半平面错误 / 可评分 | 角误差 Median / P95 | 漏检 / 额外检测 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for method, item in selected.items():
        s = item["overall"]
        angle = s["angle_error_deg_conditional"]
        errors = "N/A" if angle["n"] == 0 else f"{angle['median']:.3f}° / {angle['p95']:.3f}°"
        lines.append(f"| {method} | {fraction(s['correct_images_10deg'])} | {fraction(s['orientation_coverage_all_gt'])} | "
                     f"{fraction(s['halfplane_errors_conditional'])} | {errors} | {s['missed']} / {s['extra_or_unmatched_predictions']} |")
    lines += ["", "计时数据保留在各报告中；若校准时并行运行了训练，应在空闲设备上重新测量选定配置的延迟，再作速度比较。", "",
              "完整参数、候选记录、来源哈希与逐工况指标见 selection.json 及其引用报告。", ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({method: {"parameters": data["parameters"], "correct_images_10deg": data["overall"]["correct_images_10deg"]}
                      for method, data in selected.items()}, indent=2))
    return 0
