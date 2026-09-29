# Phase 1 — Data Foundation + Experiment Protocol Freeze

状态：**进行中；Experiment Protocol 未正式 FROZEN、Dataset V1 Gate CLOSED、正式训练未开始。**当前项目权威：[V4.1 方案](../../new-plans/面向柔性流水线的低成本工业零件姿态检测系统_项目方案_V4.1_优化版.md)与 [EXPERIMENT_PROTOCOL_V1](../../new-plans/EXPERIMENT_PROTOCOL_V1.md)（其正式修正规则优先）；协议中的 TBD 和 gate checklist 尚未满足。Phase 1 验收必须是 **Experiment Protocol FROZEN + Dataset V1 Gate OPEN**，不能以数据组装/脚本测试通过替代。

## 已有能力与缺口

已实现单帧 OBB 脚手架、part-aware 原始采集、独立 OBB sidecar/store/四点击 OpenCV 标注与只读 QA、QA-clean sidecar 的 schema v2 显式完整 session `train/val/test/final_holdout` 四路组装工具和审计脚本。Final Holdout 会实体输出并记录在 manifest 中，但不进入 `data.yaml` 的训练/调参 split。仅有有限 MVTec Screws 公开参考数据审计与合成空白图像烟测；MVTec 无项目正反面标签。真实手机/iVCam 硬件烟测及 Part A GUI 标注可视验收待完成。尚无真实 Part A 数据集、真实四路 split plan、正式训练、模型指标或 Phase 2–6 运行时证据；当前仍不能签署 Dataset V1 Gate。具体软件操作见 [采集](../../data/PHONE_CAPTURE_V1.md)、[标注](../../data/obb_annotation_v1.md) 与 [组装](../../data/project_dataset_v1.md)。

## 工作顺序与冻结清单

1. **实物与工况预冻结（待用户物理证据）**：确定 Part A、Part B 候选实物及差异/选择理由（须在 Part A 正式训练前选定 Part B，Phase 5 才正式采集 Part B），Front/Back 客观判据、Orientation/对称语义；填写相机、ROI/像素范围、运动、间距、照明、背景、模糊/遮挡的 Operating Envelope draft。Part A、Part B、Operating Envelope 值均 **TBD**；禁止杜撰或宣称已冻结。
2. **冻结实验定义**：建立单件事件 GT schema `CONFORMING / NONCONFORMING / INDETERMINATE`、真值来源与双人/二次复核及版本变更记录；`UNCERTAIN` 仅为系统输出/拒绝，非 GT/训练类。Wrong Accept 严格为 `NONCONFORMING → PASS`；争议样本 INDETERMINATE 单列不进确定真值的风险指标分母。建立 `nominal`、`low_light`、`reflection`、`background_shift`、`motion_blur`、`partial_occlusion` 的 Test Condition Matrix，每事件绑定 condition/session ID。
3. **冻结比较与报告口径**：Traditional CV/YOLO 使用同样输入、分辨率、ROI、Test Events、GT、Operating Envelope 和计时起止；双方在 Train/Val/Calibration 调参并在最终 Test 前冻结。风险工作点在 Val/Calibration 扫描 Pareto 权衡 Wrong Accept、Wrong Reject、Coverage、端到端 P95、Validated Event Throughput；Test/Final Holdout 不调参。核心比例 `x/n`、n、点估计、Wilson 95% CI（0 次错误亦报上界），延迟 n/Median/P95；同时分 Overall/Condition/独立 Session 展示失败与波动。Validated Event Throughput 按有效完成物理单件事件/有效运行分钟计，不能以 FPS 充当；跨 Session 重复性不得把同一录像切片冒充独立会话。
4. **真实 Part A 数据与四路划分**：硬件预览/采集，按 Front/Back、角度/位置及测试工况覆盖收集独立会话；人工可视检查每个接受图像的 OBB/面类或明确排除；逐 sidecar 跑 `scripts/qa_annotations.py`、核对来源/几何/覆盖。按完整 session 建立 train、val/calibration、test、final_holdout（禁随机逐帧分）；当前 schema v2 组装工具已支持四路实体输出，但真实四路 Part A 数据尚未建立。运行 `scripts/audit_dataset.py <output> --classes 2`、核查类别/角度分布、近重复与泄漏、provenance、Test Condition Matrix 和独立 session 覆盖；不过则补拍/重标而非训练。Final Holdout 留作 Hard Sample 回流/二次训练后未见数据验证；Test 错误一旦回流不可继续把原 Test 当新模型的最终无偏测试。
5. **独立复核与 gate**：按协议第 17 节逐项核实 Part A/B、Operating Envelope draft、GT schema、条件矩阵、四路 session split 与 Final Holdout 策略、标注/QA/audit/分布/溯源/泄漏、公平与统计/吞吐/跨 Session 规则和 Final Test 调参禁令。任一项失败则 **Dataset V1 Gate CLOSED** 且协议不正式 FROZEN；全部满足才记录 Experiment Protocol FROZEN + Dataset V1 Gate OPEN 并另行进入 Phase 2，不预填指标。

## 后续阶段（不属于当前实现）

Phase 2 — Perception Baseline：Traditional CV vs YOLO26n-OBB / YOLO26s-OBB；Phase 3 — Reliable Event Decision：Single Frame vs Tracking/Multi-frame/Decision Zone；Phase 4 — Risk-Aware Decision：UNCERTAIN 与 Risk-Coverage-Latency/Throughput；Phase 5 — Flexible Changeover：Part A → Part B，Recipe、成本及 Part A 回归；Phase 6 — Production-Oriented Finalization：反馈、轻量环境监测、事件追溯与最终报告。以上尚无运行时/实测证据。Seg+OBB 非默认基线，仅在证据显示瓶颈时另行考虑。相关边界见 [范围](../../product-specs/project_scope.md)、[设计](../../design-docs/phase1_obb_baseline.md) 和 [ADR-0001](../../decisions/adr-0001-single-obb-baseline.md)。
