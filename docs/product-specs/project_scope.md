# 项目范围：V4.1 假设验证

当前权威：[V4.1 项目方案](../new-plans/面向柔性流水线的低成本工业零件姿态检测系统_项目方案_V4.1_优化版.md)及其正式修正 [EXPERIMENT_PROTOCOL_V1](../new-plans/EXPERIMENT_PROTOCOL_V1.md)。旧 V2 命名仅用于现存采集/标注软件契约，并非当前项目方案。本项目是普通 RGB 相机、消费级 GPU、轻量 OBB 的课程研究原型，不是生产系统、安全联锁或 PLC/机器人控制。结论只能限定在已冻结 Operating Envelope 内；不得宣称通用工业可靠性。

## 四个研究问题与六阶段

1. RQ1 Traditional CV vs YOLO26n-OBB / YOLO26s-OBB：AI 在指定工况是否值得使用？
2. RQ2 Single Frame vs Tracking + Multi-frame + Decision Zone：事件决策能否减少错误和抖动？
3. RQ3 Risk-Coverage-Latency/Throughput：UNCERTAIN 降低 Wrong Accept 的代价是什么？
4. RQ4 Part A → Part B：换型的数据、标注、训练、Recipe、代码与回归成本如何？

| 阶段 | 验证主线 |
| --- | --- |
| Phase 1 — Data Foundation + Experiment Protocol Freeze | 冻结实验规则，通过 Dataset V1 Gate |
| Phase 2 — Perception Baseline | Traditional CV 与 YOLO 公平对照 |
| Phase 3 — Reliable Event Decision | 单帧与单件事件对照 |
| Phase 4 — Risk-Aware Decision | UNCERTAIN 工作点与风险/覆盖/延迟/吞吐权衡 |
| Phase 5 — Flexible Changeover | Part A → Part B 换型与回归 |
| Phase 6 — Production-Oriented Finalization | 反馈、轻量环境监测、追溯、最终演示与报告 |

Phase 1 验收为 **Experiment Protocol FROZEN + Dataset V1 Gate OPEN**，并非仅成功组装数据。Phase 1 不做正式长时训练；Phase 2 才进行感知基线对照。Seg+OBB 不在默认主线，仅当基线证明实质瓶颈时考虑。Active Learning 的复杂版本、Unknown/Anomaly、3D、真实执行机构为 Future Work。

## Phase 1 数据与协议门禁（当前 CLOSED）

- Part A 和 Part B 的候选实物及选择理由须在 Part A 正式训练前预冻结，包含客观 Front/Back 与 Orientation 语义；Part B 须有明确差异，不可见结果后挑易成功对象。**Part A、Part B 实物及参数均 TBD，等待用户提供物理证据，不得虚构。**
- Operating Envelope draft 须记录相机/分辨率/FPS/曝光/距离、ROI/目标像素尺寸、运动/间距/遮挡、照明/背景/模糊等范围；具体值 **TBD，等待用户物理证据**。形成事件级 GT schema (`CONFORMING`、`NONCONFORMING`、`INDETERMINATE`)、Face/Angle 真值与复核规则，建立 nominal、low_light、reflection、background_shift、motion_blur、partial_occlusion 的 Test Condition Matrix。
- 真实 Part A 自建会话采集、人工可见面判据、OBB 四角标注与 GUI 可视复核、只读 QA、Dataset Audit、类别/角度/条件分布、来源和近重复/泄漏审查。禁止同一连续视频逐帧随机拆分。须按独立完整会话分 train、val/calibration、test、final_holdout；当前 schema v2 组装工具已支持四路完整 session，其中 Final Holdout 实体输出并从 `data.yaml` 的训练/调参 split 隔离，但尚无真实四路 Part A 数据，不能声称门禁已通过。数据详情见 [组装指南](../data/project_dataset_v1.md) 与 [标注指南](../data/obb_annotation_v1.md)。MVTec Screws 不含项目面类，不可替代 Part A。
- 公平性：Traditional CV 与 YOLO 使用相同视频/分辨率/ROI/Test Events/GT/Operating Envelope/计时起止；两侧参数仅在 Train/Val/Calibration 上拟合和选择，最终 Test 前冻结。Risk Gate 工作点同样只在 Val/Calibration 扫描和冻结；**Test 与 Final Holdout 均不用于模型/规则/阈值调参**。若 Test 错误回流训练，原 Test 不再是新模型的无偏最终测试，使用未见 Final Holdout 或新 Session。
- 统计：核心比例指标给 `x/n`、事件数 n、点估计和 Wilson 95% CI；零错误也须报告上置信界。延迟报告 n、Median、端到端 P95。报告 Overall、每个 Condition 与独立 Session，包含失败样本；不许把同一录像切片冒充跨 Session 重复性。Validated Event Throughput = 完成且可映射真值、恰好一个事件且日志完整的真实单件事件数 / 有效运行分钟，单位 parts/min；与 Wrong Accept/Reject、Coverage、Missed/Duplicate Event、P95 一同报告，而非用 FPS 替代。

任一前提未通过：**Dataset V1 Gate CLOSED — 不正式训练**。源协议标注“FROZEN BEFORE FORMAL TRAINING”是冻结条件，不代表现状：Part A/B/Operating Envelope TBD 仍未解决，协议尚未正式 FROZEN。

## 指标语义与证据边界

GT 仅 `CONFORMING` / `NONCONFORMING` / `INDETERMINATE`；系统输出 `PASS` / `FAIL_FACE` / `FAIL_ORIENTATION` / `UNCERTAIN`。`UNCERTAIN` 是系统拒绝状态，不是训练类别或 GT。Wrong Accept = `NONCONFORMING → PASS` / 所有有效 NONCONFORMING 事件；Wrong Reject = CONFORMING 未 PASS / 所有有效 CONFORMING 事件，须分拆 hard reject 与 uncertain reject。INDETERMINATE 单独分层，不进入两者分母。Coverage 为可评价事件中收到明确 PASS/FAIL 的比例。决策以物理单件事件而非逐帧投票计数；角度真值优先物理基准，无治具时人工标注二次复核，争议真值标记 INDETERMINATE 并版本化。

### V4.1a 圆形/近圆形件与姿态适用性

`orientation_semantics` 仅为 `AXIAL_OBB | FEATURE_ANGLE | SYMMETRY_AWARE | NOT_APPLICABLE`。AXIAL_OBB 使用 OBB 长轴，需要稳定轴向时冻结 `min_aspect_ratio` / near-square guard；FEATURE_ANGLE 使用相对零件中心的冻结方向特征，必需 `feature_definition` 与可复现 GT 参考方法，圆形轮廓本身不能使其失效；SYMMETRY_AWARE 按 `360/symmetry_order` 等价，仍须声明角度来源/参考，对称性不提供测量。近方形只阻断实际依赖稳定长轴的来源，有效特征/对称参考仍可评价，不能统一判 UNCERTAIN。

NOT_APPLICABLE 表示姿态不是合规维度，Angle=N/A/null，按 Face / Identity 等剩余冻结维度判定；角度 N/A 本身不得造成 FAIL_ORIENTATION / UNCERTAIN 或 ORIENTATION 失败真值。Event GT 以 `orientation_gt_status: VALID | INDETERMINATE | NOT_APPLICABLE` 配合语义替代模糊布尔判据；VALID 必须有可用真角度，INDETERMINATE 是适用姿态 GT 建立失败/歧义而非 N/A，NOT_APPLICABLE 必须 null。Angle MAE/Median/P95、姿态失败及 Required-but-Undefined Orientation Rate（必需但未定义姿态率）仅以适用且 GT=VALID 的事件为评价集合；缺失预测另报，不适用不入角度分母，但剩余维度有效时仍进入整体合规统计。

正式 Part A/B 至少一个对象须有适用姿态语义及有效正式实验结果才可声明已验证二维姿态性能；两者均 NOT_APPLICABLE 时仍可验证检测/正反面，但姿态声明范围外。以上是协议/Recipe 语义，**不宣称 FEATURE_ANGLE / SYMMETRY_AWARE 运行时已实现**，仅在实物选用时实施。Training Gate 仍 CLOSED；假设验证、四路 session 隔离与 Final Holdout 规则不变。

**现状（非成果）：**已有单帧脚手架、part-aware 原始采集、OBB sidecar/四点击标注与 QA、按完整 session 的 schema v2 `train/val/test/final_holdout` 四路 Dataset V1 组装能力。无真实 Part A 项目数据集、真实四路 split、正式训练、模型指标、已冻结协议或 Phase 2–6 运行时证据；软件测试或合成烟测不是性能结论。95% 面准确率、90% mAP50、30/60 FPS 仅为未来工程目标，不能预填实测值。

实现约定见 [基线设计](../design-docs/phase1_obb_baseline.md)，工作顺序见 [执行计划](../exec-plans/active/phase1_execution_plan.md)，技术选择见 [ADR-0001](../decisions/adr-0001-single-obb-baseline.md)。
