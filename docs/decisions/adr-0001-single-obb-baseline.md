# ADR-0001：Phase 1 采用单个官方 OBB 基线

状态：已选定 Phase 1 方案（非实测性能结论）。

## 背景

V2 项目定位是用普通 RGB 相机、消费级 GPU 和开源 YOLO26n-OBB 探索低成本、软件定义、可再训练的柔性产线零件可见面与 2D 长轴姿态。主线基线始于 `7b9ae4b`；现有单帧脚手架、Phase 1C 原始采集工具与工作树中的 [V2 按零件契约](../data/LABEL_CONTRACT_V2.md)，但尚无真实项目 OBB 数据集、微调指标或生产就绪证据。Phase 1 是单帧基线及数据审计，而非生产部署。

## 决定

仅以官方 Ultralytics `yolo26n-obb.pt` 为单模型 OBB 预训练基线与未来微调起点，不在 Phase 1 增加 Seg 分支或长时训练。通过单帧输出检查类别、置信度及旋转框；仅在审计数据支持时，后续另行微调并独立评估 `<part_id>_front` / `<part_id>_back`。每会话 `part_id` 下的 `front`/`back` 是人工可见面标签，不是全局最终训练类；不用 OBB 角度推断可见面。将长轴方向约定为图像水平轴起算、模 180° 的 `θ ∈ [0°, 180°)`；近方形或轴不稳定时不强行评分。

## 取舍与后果

单模型 OBB 使输入、标注与输出契约集中，适合先审计可见面判据、旋转框与 ROI 坐标；它**不证明**模型准确、实时或能应对反光/遮挡。Seg+OBB 增加标注和集成范围，仅当单 OBB 基线证据显示实质瓶颈时才作为后续实验；当前不采用。固定传送带 ROI 只过滤观测区域（候选框中心在 ROI 内），不是跟踪或事件判定；跨帧 ID、决策区触发和单件一次决策留待后续集成。不得将烟测成功当成可用于控制的证据。Training Gate CLOSED：真实零件采集及复核、OBB 标注+QA、按会话/序列隔离划分和 Dataset Audit 通过前不得训练；MVTec Screws V1.0 / 公开 OBB 参考数据还须取得修正/当前完整版或冻结明确审查的过滤策略，其无项目面标签。未来路径（尚未实现跟踪/事件运行时）：Camera/Video → ROI → YOLO26n-OBB → Tracking → Multi-frame Aggregation → Confidence/Stability Gate → Decision Zone → PASS/FAIL_FACE/FAIL_ORIENTATION/UNCERTAIN → single-part statistics。`UNCERTAIN` 是运行时拒绝/决策状态，绝非训练类。柔性换型实验先建立 Part A、再加入 Part B，测量新增图像、标注工作量、微调时间、软件/配置变化及 Part A 回归；目前无此结果。95% 面准确率、90% mAP50、30/60 FPS 均非已验证结果。

适用范围及验收见 [项目范围](../product-specs/project_scope.md)，标注/几何契约见 [基线设计](../design-docs/phase1_obb_baseline.md)，实施顺序见 [执行计划](../exec-plans/active/phase1_execution_plan.md)。
