# Phase 1 数据与实验协议设计；Phase 2 单 OBB 感知基线

当前权威：[V4.1 方案](../new-plans/面向柔性流水线的低成本工业零件姿态检测系统_项目方案_V4.1_优化版.md)与 [EXPERIMENT_PROTOCOL_V1](../new-plans/EXPERIMENT_PROTOCOL_V1.md)。本文件保留已实现的单帧 OBB 接口约定，不把它误称为 Phase 1 正式感知验收；Phase 1 的验收是 **Experiment Protocol FROZEN + Dataset V1 Gate OPEN**。Part A、Part B 与 Operating Envelope 实测定义仍为 **TBD，等待用户实物/工况证据**；协议尚未正式冻结。

## 输入、输出与单帧软件边界

合法来源单帧图像，记录相机、照明、原图分辨率、ROI 顶点及坐标系。现有 Ultralytics `yolo26n-obb.pt` 预训练烟测脚手架可检查类别、置信度、旋转框四角和可视化；预训练类别不等于 `<part_id>_front` / `<part_id>_back`，不能据此报告项目精度。保存权重来源/校验、运行环境、输入尺寸、参数、ROI、数据版本与 ROI 过滤前原始输出。ROI 可先在完整画面推理后按同坐标系 OBB 中心过滤，裁切/缩放须同步变换 ROI/标注；ROI 不是跟踪/事件判定。

每会话一个 `part_id`，人工选择 `front` / `back`，不能从角度推断可见面。类名 `<part_id>_front` / `<part_id>_back`，数值 ID 在组装时赋予 Part A 的 0/1。OBB 四角沿周界排序并归一化；由像素空间长边计算图像水平轴起算的 `θ ∈ [0°,180°)`，与 `θ+180°` 等价；不能据此推断头尾。该长轴计算仅对应 AXIAL_OBB；需要稳定轴向时冻结 `min_aspect_ratio` / near-square guard，轴不稳定不能强行输出可靠方向，但几何有效的近方形 OBB 仍可作为训练标注。Angle GT 须对应冻结的语义/来源，优先物理角度基准/治具，无治具时采用相应的长轴或特征方向人工标注并二次复核。详见 [标注指南](../data/obb_annotation_v1.md) 与 [数据组装](../data/project_dataset_v1.md)。

## V4.1a 姿态协议（不是新增已实现接口）

`orientation_semantics` 仅为 `AXIAL_OBB | FEATURE_ANGLE | SYMMETRY_AWARE | NOT_APPLICABLE`。FEATURE_ANGLE 用相对零件中心的明确方向特征，必须冻结 `feature_definition` 与可复现 GT 参考方法，圆形外轮廓本身不使它无效。SYMMETRY_AWARE 的等价周期为 `360/symmetry_order`，必须声明角度来源/参考，对称性本身不产生测量；若来源依赖稳定 OBB 长轴仍用长轴 guard，有效非长轴参考则不受近方形几何阻断。近方形绝不是统一 UNCERTAIN 条件。

NOT_APPLICABLE 表示姿态不作合规维度，Angle=N/A/null，仅按 Face / Identity 等剩余冻结维度评价；不能仅因角度 N/A 输出 FAIL_ORIENTATION / UNCERTAIN 或标注 ORIENTATION 失败真值。Event GT 使用 `orientation_gt_status: VALID | INDETERMINATE | NOT_APPLICABLE` 配合语义；VALID 必须有可用真角度，INDETERMINATE 是适用姿态 GT 建立失败/歧义而非 N/A，NOT_APPLICABLE 必须 null。Angle MAE/Median/P95、姿态失败率及 Required-but-Undefined Orientation Rate（必需但未定义姿态率）只以适用且 GT=VALID 事件为评价集合，缺失预测另报，不适用事件不入角度分母但可参与剩余维度的整体合规统计。

至少一个正式 Part A/B 对象须有适用姿态语义与有效正式测试证据才可声称已验证二维姿态性能；均 NOT_APPLICABLE 时检测/正反面仍可验证，姿态声明范围外。这些是协议/Recipe 规则，**FEATURE_ANGLE / SYMMETRY_AWARE 运行时未因本修正实现，须按实物选用后实施**。Training Gate 仍 CLOSED，四路 session / Final Holdout 规则不变。

## Phase 1 数据与评估接口

part-aware 采集、OBB sidecar/四点击 GUI/只读 QA、schema v2 session-aware `train/val/test/final_holdout` 四路组装工具已实现；Final Holdout 会实体输出并从 `data.yaml` 的训练/调参 split 中隔离。真实 Part A 采集/GUI 可视验收、真实项目数据集、真实四路 split、正式训练及模型指标尚不存在，不能把软件支持视为符合 V4.1 gate。Phase 1 要先预冻结 Part A/B、Operating Envelope draft、Front/Back/Orientation 规则和事件 GT schema、Test Condition Matrix，再完成真实采集/标注或明确排除/QA/audit/人工可视复核及完整 session train/val/test/final_holdout、provenance/类别角度分布/泄漏核查。禁止连续帧随机拆分。MVTec Screws 仅公开 OBB 参考数据，无项目面类。软件 QA/组装通过不是协议冻结，也不解锁正式训练。

事件级 GT 为 `CONFORMING` / `NONCONFORMING` / `INDETERMINATE`，含 session/condition、面/方向真值、复核及版本；争议真值单列。系统 `UNCERTAIN` 是决策层拒绝，不是 GT/训练类。Wrong Accept = `NONCONFORMING → PASS`，Wrong Reject 主指标 = CONFORMING 未 PASS，分拆 hard/uncertain reject；INDETERMINATE 不混入两者分母。只在 Train/Val/Calibration 选模型和参数，Test/Final Holdout 不调参；Test 错误回流后原 Test 不再可作为新模型无偏最终测试。Traditional CV 与 YOLO 共享视频、分辨率、ROI、事件、GT、Operating Envelope 与计时起止。报告核心比例 `x/n`、n、Wilson 95% CI（含零错误上界）、Overall/Condition/独立 Session；端到端延迟 n/Median/P95 与有效物理单件事件/有效运行分钟的 Validated Event Throughput，不能用 FPS 代替。结论限于冻结的 Operating Envelope；不把官方 benchmark 当项目性能。

## Phase 2–6 接口（未来，不是本阶段实现）

Phase 2 — Perception Baseline 对比 Traditional CV、YOLO26n-OBB、YOLO26s-OBB；单 OBB 是默认 YOLO 起点，Seg+OBB 仅在实测基线证明瓶颈后考虑。Phase 3 — Reliable Event Decision：Tracking + Multi-frame + Decision Zone，1 Track → 1 Final Event；Phase 4 — Risk-Aware Decision：UNCERTAIN/依稳定长轴来源启用 guard，Val/Calibration 选择风险/覆盖/端到端延迟/吞吐工作点；Phase 5 — Flexible Changeover：Part B 实际采集/训练/Recipe 与 Part A 回归；Phase 6 — Production-Oriented Finalization：反馈、环境监测、追溯与报告。这些运行时能力与指标尚未取得；单帧候选不是生产单件事件或设备控制指令。

范围见 [项目范围](../product-specs/project_scope.md)，步骤见 [执行计划](../exec-plans/active/phase1_execution_plan.md)，单 OBB 取舍见 [ADR-0001](../decisions/adr-0001-single-obb-baseline.md)。
