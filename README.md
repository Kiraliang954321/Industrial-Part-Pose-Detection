# FlexPose-YOLO：V4.1 低成本工业零件姿态检测假设验证原型

当前项目权威为 [V4.1 项目方案](docs/new-plans/面向柔性流水线的低成本工业零件姿态检测系统_项目方案_V4.1_优化版.md)与修正其真值/吞吐/重复性规则的 [EXPERIMENT_PROTOCOL_V1](docs/new-plans/EXPERIMENT_PROTOCOL_V1.md)。旧 V2 名称指现有采集契约，不再是项目计划。普通 RGB 相机、消费级 GPU、开源轻量 OBB 构成课程研究原型，**不是生产系统、安全联锁或设备控制**；性能结论必须限定在冻结的 Operating Envelope 内。

## 研究问题与路线

- RQ1：Traditional CV vs YOLO26n-OBB / YOLO26s-OBB，AI 是否值得使用？
- RQ2：Single Frame vs Tracking + Multi-frame + Decision Zone，单件事件决策是否更可靠？
- RQ3：Risk-Coverage-Latency/Validated Event Throughput，UNCERTAIN 降低 Wrong Accept 需要多少自动处理率、延迟和吞吐代价？
- RQ4：Part A → Part B，量化采集、标注、Fine-tune、Recipe、代码改动和 Part A 回归。

六阶段：Phase 1 — Data Foundation + Experiment Protocol Freeze；Phase 2 — Perception Baseline；Phase 3 — Reliable Event Decision；Phase 4 — Risk-Aware Decision；Phase 5 — Flexible Changeover；Phase 6 — Production-Oriented Finalization。**当前仍在 Phase 1，验收为 Experiment Protocol FROZEN + Dataset V1 Gate OPEN，不是组装成功。** Part A、Part B 和 Operating Envelope 的物理值均 TBD，等待用户提供实物/工况证据；虽然源协议写有条件性冻结声明，现状尚未正式 FROZEN。

已有单帧 OBB 脚手架、part-aware 原始采集、OBB sidecar/四点击标注/只读 QA，以及按完整 session 的 schema v2 `train/val/test/final_holdout` 四路 Dataset V1 组装能力。`data.yaml` 仅暴露 train/val/test；Final Holdout 单独物理输出并在 manifest 中记录为 untuned final validation。真实手机/iVCam 硬件验收、真实 Part A GUI 标注可视验收、真实 Part A 数据集、真实四路划分和门禁均未完成；无正式训练、模型指标或 Phase 2–6 运行时证据。预训练/合成烟测与 MVTec Screws 有限审计不证明项目精度；MVTec 无项目正反面标签。

## Phase 1 必须先冻结什么

预冻结 Part A/B、Front/Back/Orientation 与 Operating Envelope draft；定义单件事件 GT `CONFORMING` / `NONCONFORMING` / `INDETERMINATE`、复核与版本记录，建立 nominal/low_light/reflection/background_shift/motion_blur/partial_occlusion 测试矩阵；完成真实采集、OBB 标注/可视复核/QA/Audit、分布与来源/泄漏检查、完整 session 的 train/val/calibration/test/final_holdout 隔离。`UNCERTAIN` 是系统拒绝/决策，不是训练或 GT 类；Wrong Accept = `NONCONFORMING → PASS`，INDETERMINATE 单独分层。Traditional CV/YOLO 必须共享输入、ROI、事件、GT、工况和计时口径；只在 Train/Val/Calibration 调参和选择工作点，**Test/Final Holdout 不调参**。Test 错误回流后须用未见 Final Holdout 或新 session 做无偏最终测试。比例报告 `x/n`、Wilson 95% CI（零错误亦报上界）；报告不同 condition 与独立 session 的重复性、端到端 P95 与有效完成事件/分钟的 Validated Event Throughput，不能用 FPS 代替生产吞吐。具体 gate 见 [范围与验收](docs/product-specs/project_scope.md) 与 [Phase 1 执行计划](docs/exec-plans/active/phase1_execution_plan.md)。

## 本地软件烟测（非模型质量证据）

需要 Python >=3.10；推理可能需要合法获取官方权重及一张有权使用的本地图像。在仓库根目录运行 PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,inference]"
python scripts/check_env.py
python -m pytest -q
# <frame-path> 必须替换为合法、非敏感的真实单帧图像
# 预训练类别不是未来的项目 <part_id>_front / <part_id>_back
yolo-conveyor "<frame-path>" --model yolo26n-obb.pt
```

以上命令仅供操作，未在本次文档迁移中运行模型/硬件测试。运行时记录权重来源、Python/Ultralytics 版本、设备、分辨率、输入和参数；不得把预训练输出或官方 benchmark 写成项目结果。

## 现有 Phase 1 软件操作

- [PHONE_CAPTURE_V1](docs/data/PHONE_CAPTURE_V1.md)：`python scripts/capture_phone.py --part-id <part-id> --source 0 --backend dshow`。每次原始会话绑定一个 part，操作者选 `front`/`back`；采集不自动生成 OBB。
- [OBB Annotation Sidecar V1](docs/data/obb_annotation_v1.md)：`python scripts/annotate_obb.py data/captures/<capture-session-id>`；`python scripts/qa_annotations.py data/annotations/<capture-session-id>`。四点击标注或明确排除；自动 QA 不能代替真实 GUI 人工可视验收。
- [Project Dataset V1 组装指南](docs/data/project_dataset_v1.md)：当前仅支持按完整 session 划分 train/val/test；不能随机逐帧切分。数值类 ID 仅在组装时分配为 Part A 的 `0=<part_id>_front`、`1=<part_id>_back`；`front`/`back` 不可由无方向模 180° OBB 角度推断。门禁仍 CLOSED，不要将组装脚本能力当成四路完成。

设计见 [架构](ARCHITECTURE.md)、[基线设计](docs/design-docs/phase1_obb_baseline.md) 与 [ADR-0001](docs/decisions/adr-0001-single-obb-baseline.md)。95% 面准确率、90% mAP50、30/60 FPS 如被列作目标，仅是未来目标，绝非已测结果。
