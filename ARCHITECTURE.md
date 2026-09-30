# 架构：V4.1 低成本工业零件视觉验证原型

当前项目权威是 [V4.1 项目方案](docs/new-plans/面向柔性流水线的低成本工业零件姿态检测系统_项目方案_V4.1_优化版.md)及其正式修正 [EXPERIMENT_PROTOCOL_V1](docs/new-plans/EXPERIMENT_PROTOCOL_V1.md)；旧 V2 名称只用于现存采集/标注软件契约，不是当前项目计划。范围见 [项目范围](docs/product-specs/project_scope.md)。本项目以普通 RGB 相机、消费级 GPU 与轻量 OBB 探索有边界的生产检测能力；不是安全联锁、PLC/机器人控制或 24/7 生产系统。任何性能结论只能限定在已冻结的 Operating Envelope（相机/ROI、目标尺寸、运动/间距、照明/背景、模糊/遮挡等工况）内，不得外推为通用工业可靠性。Part A/B 实物及 Operating Envelope 具体值仍 TBD，不能预填性能。

## 六阶段与数据流

| 阶段 | 目标（未完成不视为成果） |
| --- | --- |
| Phase 1 — Data Foundation + Experiment Protocol Freeze | 预冻结 Part A/B、工况与实验规则，真实采集/标注、四路 session 划分和 Dataset V1 Gate |
| Phase 2 — Perception Baseline | 公平比较 Traditional CV、YOLO26n-OBB、YOLO26s-OBB，回答 RQ1 |
| Phase 3 — Reliable Event Decision | Tracking + Multi-frame + Decision Zone；单帧与单件事件对照 |
| Phase 4 — Risk-Aware Decision | UNCERTAIN 拒绝及风险/覆盖/端到端延迟/吞吐工作点 |
| Phase 5 — Flexible Changeover | Part A → Part B 的数据、微调、Recipe、代码变更和回归实测 |
| Phase 6 — Production-Oriented Finalization | 反馈、轻量环境监测、事件追溯、演示与报告 |

未来目标路径（**非当前已实现运行时**）：Camera/Video → timestamp/ROI → Perception (Traditional CV / YOLO OBB) → Part + Face + OBB + 2D Angle → Geometry Validity → Tracking → Multi-frame Evidence → Decision Zone → Risk Gate → PASS / FAIL_FACE / FAIL_ORIENTATION / UNCERTAIN → Single-Part Event → Statistics / Evidence Log。1 Track → 1 Final Event；单帧候选不等于物理单件事件，也不是控制指令。当前已有单帧脚手架、part-aware 原始采集、OBB 标注 sidecar/QA 以及按完整会话的 train/val/test 组装工具；尚无真实 Part A 项目数据集、正式训练、实测性能或 Phase 2–6 事件运行时证据。默认 YOLO 起点为官方 `yolo26n-obb.pt`，YOLO26s-OBB 是 Phase 2 对比项，Seg+OBB 仅当基线证据显示实质瓶颈时另行考虑；见 [ADR-0001](docs/decisions/adr-0001-single-obb-baseline.md)。

## Phase 1 协议与数据门禁（当前 CLOSED）

Phase 1 验收为 **Experiment Protocol FROZEN + Dataset V1 Gate OPEN**，不是脚本运行成功。真实 Part A 须有客观 Front/Back 与姿态适用性/参考判据（可为 NOT_APPLICABLE）、人工 OBB 标注/可视复核/QA、来源/分布/泄漏审查、Dataset Audit；Part B 候选须在 Part A 正式训练前预冻结。Operating Envelope draft、事件 GT schema、Test Condition Matrix、公平对照、统计与 Final Test 禁调参规则均须按协议核实。协议文件写有“FROZEN BEFORE FORMAL TRAINING”但其 TBD 尚未填写，**协议尚未正式冻结，Training Gate / Dataset V1 Gate CLOSED**。MVTec Screws 不具备项目面标签，不能替代真实 Part A。

V4.1 要求独立完整 session 分 train、val/calibration、test、final_holdout；禁止把连续帧随机逐帧划分。当前 `scripts/build_project_dataset.py` 已实现 schema v2 四路组装：Final Holdout 会被实体输出并写入 manifest，但不会进入 `data.yaml` 的训练/调参 split。该软件能力仍**不等于**真实四路数据已经建立或 gate 已通过。参见 [Project Dataset V1 组装指南](docs/data/project_dataset_v1.md)与 [OBB 标注指南](docs/data/obb_annotation_v1.md)。Train 用于模型拟合，Val/Calibration 用于选模型/阈值/风险工作点；Test 与 Final Holdout 都不得调参。Test 错误若回流训练，原 Test 不能再充当新模型的无偏最终测试，必须使用未见 Final Holdout 或新 session。

## 真值、输出与证据语义

正式 Ground Truth 以单件事件为单位：`CONFORMING`、`NONCONFORMING`、`INDETERMINATE`，Face 依冻结的实物判据，Angle 优先物理基准/治具，无治具时人工标注须二次复核；争议事件单列并对真值变更版本化。OBB 四角 sidecar 是**训练标注证据**，不是事件 GT。系统输出 `PASS`、`FAIL_FACE`、`FAIL_ORIENTATION`、`UNCERTAIN`；`UNCERTAIN` 仅为系统拒绝状态，既非 GT 也非训练类别。Wrong Accept 严格指 `NONCONFORMING → PASS`；Wrong Reject 是 CONFORMING 未 PASS，需分拆明确失败与 UNCERTAIN；INDETERMINATE 不进入两者分母。

V4.1a 的 `orientation_semantics` 仅为 `AXIAL_OBB | FEATURE_ANGLE | SYMMETRY_AWARE | NOT_APPLICABLE`。AXIAL_OBB 长轴 `θ ∈ [0°,180°)` 是图像水平轴起算的轴向角，`θ` 与 `θ+180°` 等价，不可推断正反面/头尾；需要稳定轴向时使用 `min_aspect_ratio`，近方形/轴不稳定不得强制评分。FEATURE_ANGLE 是相对零件中心的冻结方向特征，必须定义 `feature_definition` 与可复现 GT 方法，圆形外轮廓不自动使它无效。SYMMETRY_AWARE 周期为 `360/symmetry_order`，必须声明角度来源/参考；对称性本身不产生角度，只有依赖稳定 OBB 长轴的来源才启用长轴 guard。近方形不是通用 UNCERTAIN 条件。

NOT_APPLICABLE 的 Angle=N/A/null，按剩余冻结 Face / Identity 等维度判定，不能仅因角度缺失输出 FAIL_ORIENTATION / UNCERTAIN 或 ORIENTATION 失败真值。Event GT 使用 `orientation_gt_status: VALID | INDETERMINATE | NOT_APPLICABLE` 配合语义：VALID 有可用真角度，INDETERMINATE 是适用姿态 GT 建立失败/歧义而非 N/A，NOT_APPLICABLE 必须 null。Angle MAE/Median/P95、姿态失败与 Required-but-Undefined Orientation Rate（必需但未定义姿态率）只以适用且 GT=VALID 的事件为评价集合；缺失预测另报，N/A 不入角度分母但可参与剩余维度的整体合规评价。

至少一个正式 Part A/B 对象具有适用姿态语义与有效正式测试证据，才可声明已验证二维姿态性能；均 NOT_APPLICABLE 时姿态声明范围外。以上是协议/Recipe 设计，**FEATURE_ANGLE / SYMMETRY_AWARE 运行时并未因此实现，须按实物选择后实施**；Training Gate CLOSED 与四路 session/Final Holdout 规则不变。

ROI 须与图像坐标一致，只是候选过滤边界，不是跟踪/事件判定。Traditional CV 与 YOLO 必须共享输入、分辨率、ROI、事件、GT、工况和计时起止。正式比例报告 `x/n`、事件数及 Wilson 95% CI（零错误亦报上界），端到端延迟 n/Median/P95，并分 Overall、condition 与独立 session 报告；Validated Event Throughput 用有效单件事件/有效运行分钟，不能用推理 FPS 冒充。实现接口与实施顺序见 [基线设计](docs/design-docs/phase1_obb_baseline.md)和 [执行计划](docs/exec-plans/active/phase1_execution_plan.md)。
