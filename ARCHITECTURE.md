# 架构：V4.1 低成本工业零件视觉验证原型

既有二维基线权威是 [V4.1 项目方案](docs/new-plans/面向柔性流水线的低成本工业零件姿态检测系统_项目方案_V4.1_优化版.md)及其正式修正 [EXPERIMENT_PROTOCOL_V1](docs/new-plans/EXPERIMENT_PROTOCOL_V1.md)；旧 V2 名称只用于现存采集/标注软件契约，不是当前项目计划。范围见 [项目范围](docs/product-specs/project_scope.md)。本项目以普通 RGB 相机、消费级 GPU 与轻量 OBB 探索有边界的生产检测能力；不是安全联锁、实体 PLC/机器人控制或 24/7 生产系统。任何性能结论只能限定在已冻结的 Operating Envelope（相机/ROI、目标尺寸、运动/间距、照明/背景、模糊/遮挡等工况）内，不得外推为通用工业可靠性。Part A/B 实物及 Operating Envelope 具体值仍 TBD，不能预填性能。

## 工业件朝向与 3D 仿真拓展

后续研究方向按 [ADR-0002](docs/decisions/adr-0002-industrial-orientation-3d-simulation.md)记录，执行见[新计划](docs/exec-plans/active/industrial_orientation_3d_execution_plan.md)：对象只限轴承、锥形齿轮或螺丝，先确认可观察朝向，再按条件扩展完整 6D。历史开发对象不再作为后续训练或换型任务。

拓展链路：工业件 CAD/场景与相机 → 图像及独立监督标签 → 朝向/姿态预测 → 固定标定与高度先验下的平面目标 → 运动学/路径/碰撞检查 → 虚拟抓取/放置记录。当前已在螺丝代理仿真开发域实现：`simulation/` 生成图像与真值，`orientation/` 提供两点有向角；`grasp_geometry.py` 只接收预测与固定标定，`panda_control.py` 使用关节反馈执行，`grasp_evaluation.py` 独立读取真值评分。见[视觉开发](docs/exec-plans/active/screw_orientation_v1_development.md)与 [M4 闭环记录](docs/exec-plans/active/screw_m4_closed_loop.md)。这是一帧视觉引导的 2.5D 抓取，完整 6D、执行中视觉伺服和动作策略学习未实现。

现有面类/OBB 接口不等于 3D 契约，已有对象专用训练与过滤也不等于通用换型。螺丝代理采用独立[仿真开发契约](docs/data/screw_orientation_v1.md)，不改变旧 180° OBB 语义；真实型号、尺度/标定与工艺仍须完成 S0。原真实协议门禁保持 CLOSED，实体设备控制仍在范围外。

## L1 分度式单工位仿真

最新用户用途已按 [ADR-0003](docs/decisions/adr-0003-held-part-inspection-before-assembly.md)转为夹持后检测与装配前复核。[H1](docs/exec-plans/active/held_inspection_h1.md)使用独立 `held/` 模块实现：已知窝位物理抓取 → 固定检测姿态 → 双 RGB 头尾预测/三角化 → 可修正轴向/露出长度检查 → 一次性夹持许可 → 预测端点相对实际 TCP 的补偿 → 导向孔预插入 → 独立物理评分。许可与运动代码只接收图像预测、固定标定及机器人反馈；`evaluation.py` 的对象 GT 不回流控制。`data.py`/`training.py` 使用专门的夹持视图，`protocol.py` 冻结后才能评价新 episode。预插入继续夹持，不模拟真实螺纹锁付，也不估计不可观察的绕轴角。M4/L1 的抓取前模型与成果保留，不能代替 H1 验证。

[H2](docs/exec-plans/active/held_recheck_h2.md)位于独立 `held_recheck/`：H1 模型 → 有界端点残差补偿 → 初检/转运许可 → 孔上方新增双 RGB 复检 → 比较 TCP 坐标系中的相对端点/轴线 → 新鲜证据绑定的插入许可 → 最新预测补偿与物理预插入。`faults.py` 只在实验编排中施加世界坐标外力，`evaluation.py` 仅独立评分；两者均不向视觉许可输入对象真值。两种许可分用途且单次消费，初检证据不能越过复检直接签发插入许可。四相机布局和补偿参数仅在当前代理域验证，不表示真实低成本硬件方案已确定。

[L1](docs/exec-plans/active/screw_line_simulation_v1.md)在独立版本中连接物理索引托盘、8 帧视觉证据、单件事件与 Panda：模拟 PLC job/编码器到位 → 停稳 → `line/perception.py` → `line/events.py` → `line/state.py` → PASS 下游交接 / FAIL_ORIENTATION 拒收分拣 / UNCERTAIN 停机留件。`line/metrics.py` 独立读取 GT 评分；执行后物理终态由仿真评价标注，不冒充真实执行反馈。

事件 ID 来自已声明的单件工位编排，不是视觉多目标跟踪。L1 有仿真开发的多帧拒绝和唯一事件记录，但尚无连续上下料、回程、复核恢复与真实风险工作点标定。M4 冻结文件不变，L1 使用自己的[开发协议](docs/data/screw_line_protocol_v1.json)。

## A1–A5 轴承与锥形齿轮

各阶段成果、可克隆的预览、开发入口及模型材料见[接手指南](docs/handoff/README.md)。

[A5](docs/exec-plans/active/keyed_cell_a5.md)在独立 `keyed_cell/` 将已知圆轴工位扩展为实体索引配合：齿轮内键槽配凸键、轴承外缺口配定位销。复用 A4 视觉模型与有界观察；`scene.py` 增加碰撞几何和腕部力/力矩传感，`control.py` 分离接近、分段插入与退回，闭爪预检使用实际夹爪编码器开度。`guard.py` 仅用机器人传感反馈触发停止；`evaluation.py` 独立读取工件/索引件几何和接触评分。`reference.py` 的理想位姿控制只验证机械配合，不算神经网络成果。录像俯视示意读物理状态，但不进入检测或动作条件。

[A4](docs/exec-plans/active/active_pose_a4.md)在独立 `active_pose/` 增加有限主动观察。复用 A3 冻结的场景、双端面几何、校正与预定位；`policy.py` 用图像质量/单件检测门控观察动作，`control.py` 只用机器人 FK 规划两个小角度观察姿态。`runtime.py` 分开记录观察、校正和插入，并在下游比较视觉估计在工具坐标中的一致性。`diagnostics.py` 的理想种子仅用于离线分析，运行时不导入该诊断入口；`preprocessing.py` 是独立的纯图像/标定变换。新数据用 `grab_id` 绑定同一次抓取的多个观察姿态，不能按视角随机拆分。具体训练与验证进展以 A4 记录为准。

[A3](docs/exec-plans/active/pose_cell_a3.md)在独立 `pose_cell/` 扩展翻转与大倾角：高位料架侧夹 → 上下六路 RGB → 有版本的孔中心/定位结构中心预测 → 双端面圆几何、多视角拟合与已知 CAD 外形边缘校核 → 数值旋转/平移 → 结合实际 TCP 计算相对持件变换 → 分段关节校正 → 新鲜图像复核 → 原竖轴有间隙预定位。每次不确定、不可达或物理中止均留在评价分母。`control.py` 只接收视觉估计和机器人反馈；`profile.py` 只编译静态几何，不读取仿真对象状态。对象 GT 在 `evaluation.py` 和观测后的评分阶段读取，不签发动作许可。具体实测范围以 A3 记录为准，不能由控制入口存在推断已通过独立评价。

最新连续角度分支 [A2](docs/exec-plans/active/datum_pose_a2.md)在独立 `datum_pose/`：固定 ROI → 两对象/五定位点网络 → RGB 局部边缘与孔口椭圆 → 定位结构提供周向基准 → 跨视角几何拟合与重投影优化 → 连续旋转矩阵/倾角/方位/绕轴角 → 可配置目标比较。估计器只接收 RGB 预测、相机标定和已知几何；对象状态只在之后评分。孔口圆参考采用当前窄包络的弱透视近似，不代表完整6D定位或任意翻转可观测。当前独立覆盖55%，必须保留 UNCERTAIN，尚未接自动纠正或新姿态装配。

用户要求两个新对象也判正反并保持简单流程，[A1](docs/exec-plans/active/bearing_bevel_a1.md)在独立 `annular/` 实现：项目自建留孔代理 → 已知窝位物理夹取 → 单 RGB 六类检测（两对象各 front/back/tilted，空抓负样本）→ 图像/夹持有效性与类型/端向门控 → 固定配方套轴预定位 → 独立接触/孔轴/深度评分。`perception.py` 不读取 GT；`control.py` 只读取机器人反馈与已知配方，不能把固定目标动作称为视觉测得孔位或完整 6D 补偿。`data.py` 与 `training.py` 的新对象标签、数据和权重独立，旧螺丝协议不变。

F608ZZ 单侧法兰与锥齿面/轮毂提供结构方向特征，不使用人造色标。目录尺寸仅参考外形，轴承整件刚体、齿轮简化齿廓；1 mm 径向间隙、3 mm 套入用于用途演示，不用于真实压装或啮合。正式协议仍未冻结。

## 既有 V4.1 六阶段与数据流

| 阶段 | 目标（未完成不视为成果） |
| --- | --- |
| Phase 1 — Data Foundation + Experiment Protocol Freeze | 预冻结 Part A/B、工况与实验规则，真实采集/标注、四路 session 划分和 Dataset V1 Gate |
| Phase 2 — Perception Baseline | 公平比较 Traditional CV、YOLO26n-OBB、YOLO26s-OBB，回答 RQ1 |
| Phase 3 — Reliable Event Decision | Tracking + Multi-frame + Decision Zone；单帧与单件事件对照 |
| Phase 4 — Risk-Aware Decision | UNCERTAIN 拒绝及风险/覆盖/端到端延迟/吞吐工作点 |
| Phase 5 — Flexible Changeover | Part A → Part B 的数据、微调、Recipe、代码变更和回归实测 |
| Phase 6 — Production-Oriented Finalization | 反馈、轻量环境监测、事件追溯、演示与报告 |

既有二维目标路径（**完整链路尚未实现，L1 是另行声明的单件仿真分支**）：Camera/Video → timestamp/ROI → Perception (Traditional CV / YOLO OBB) → Part + Face + OBB + 2D Angle → Geometry Validity → Tracking → Multi-frame Evidence → Decision Zone → Risk Gate → PASS / FAIL_FACE / FAIL_ORIENTATION / UNCERTAIN → Single-Part Event → Statistics / Evidence Log。1 Track → 1 Final Event；单帧候选不等于物理单件事件，也不是控制指令。已有单帧、采集、标注/QA、四路组装及[历史实拍开发微调](docs/exec-plans/active/real_cap_80_development.md)，但尚无新工业件正式数据集、正式训练或 Phase 2–6 正式事件验证。既有单 OBB 默认起点与对照见 [ADR-0001](docs/decisions/adr-0001-single-obb-baseline.md)，不自动代表新增 3D 方法已选定。

## Phase 1 协议与数据门禁（当前 CLOSED）

Phase 1 验收为 **Experiment Protocol FROZEN + Dataset V1 Gate OPEN**，不是脚本运行成功。真实 Part A 须有客观 Front/Back 与姿态适用性/参考判据（可为 NOT_APPLICABLE）、人工 OBB 标注/可视复核/QA、来源/分布/泄漏审查、Dataset Audit；Part B 候选须在 Part A 正式训练前预冻结。Operating Envelope draft、事件 GT schema、Test Condition Matrix、公平对照、统计与 Final Test 禁调参规则均须按协议核实。协议文件写有“FROZEN BEFORE FORMAL TRAINING”但其 TBD 尚未填写，**协议尚未正式冻结，Training Gate / Dataset V1 Gate CLOSED**。MVTec Screws 不具备项目面标签，不能替代真实 Part A。

V4.1 要求独立完整 session 分 train、val/calibration、test、final_holdout；禁止把连续帧随机逐帧划分。当前 `scripts/build_project_dataset.py` 已实现 schema v2 四路组装：Final Holdout 会被实体输出并写入 manifest，但不会进入 `data.yaml` 的训练/调参 split。该软件能力仍**不等于**真实四路数据已经建立或 gate 已通过。参见 [Project Dataset V1 组装指南](docs/data/project_dataset_v1.md)与 [OBB 标注指南](docs/data/obb_annotation_v1.md)。Train 用于模型拟合，Val/Calibration 用于选模型/阈值/风险工作点；Test 与 Final Holdout 都不得调参。Test 错误若回流训练，原 Test 不能再充当新模型的无偏最终测试，必须使用未见 Final Holdout 或新 session。

## 真值、输出与证据语义

正式 Ground Truth 以单件事件为单位：`CONFORMING`、`NONCONFORMING`、`INDETERMINATE`，Face 依冻结的实物判据，Angle 优先物理基准/治具，无治具时人工标注须二次复核；争议事件单列并对真值变更版本化。OBB 四角 sidecar 是**训练标注证据**，不是事件 GT。系统输出 `PASS`、`FAIL_FACE`、`FAIL_ORIENTATION`、`UNCERTAIN`；`UNCERTAIN` 仅为系统拒绝状态，既非 GT 也非训练类别。Wrong Accept 严格指 `NONCONFORMING → PASS`；Wrong Reject 是 CONFORMING 未 PASS，需分拆明确失败与 UNCERTAIN；INDETERMINATE 不进入两者分母。

V4.1a 的 `orientation_semantics` 仅为 `AXIAL_OBB | FEATURE_ANGLE | SYMMETRY_AWARE | NOT_APPLICABLE`。AXIAL_OBB 长轴 `θ ∈ [0°,180°)` 是图像水平轴起算的轴向角，`θ` 与 `θ+180°` 等价，不可推断正反面/头尾；需要稳定轴向时使用 `min_aspect_ratio`，近方形/轴不稳定不得强制评分。FEATURE_ANGLE 是相对零件中心的冻结方向特征，必须定义 `feature_definition` 与可复现 GT 方法，圆形外轮廓不自动使它无效。SYMMETRY_AWARE 周期为 `360/symmetry_order`，必须声明角度来源/参考；对称性本身不产生角度，只有依赖稳定 OBB 长轴的来源才启用长轴 guard。近方形不是通用 UNCERTAIN 条件。

NOT_APPLICABLE 的 Angle=N/A/null，按剩余冻结 Face / Identity 等维度判定，不能仅因角度缺失输出 FAIL_ORIENTATION / UNCERTAIN 或 ORIENTATION 失败真值。Event GT 使用 `orientation_gt_status: VALID | INDETERMINATE | NOT_APPLICABLE` 配合语义：VALID 有可用真角度，INDETERMINATE 是适用姿态 GT 建立失败/歧义而非 N/A，NOT_APPLICABLE 必须 null。Angle MAE/Median/P95、姿态失败与 Required-but-Undefined Orientation Rate（必需但未定义姿态率）只以适用且 GT=VALID 的事件为评价集合；缺失预测另报，N/A 不入角度分母但可参与剩余维度的整体合规评价。

至少一个正式 Part A/B 对象具有适用姿态语义与有效正式测试证据，才可声明已验证二维姿态性能；均 NOT_APPLICABLE 时姿态声明范围外。以上是协议/Recipe 设计，**FEATURE_ANGLE / SYMMETRY_AWARE 运行时并未因此实现，须按实物选择后实施**；Training Gate CLOSED 与四路 session/Final Holdout 规则不变。

ROI 须与图像坐标一致，只是候选过滤边界，不是跟踪/事件判定。Traditional CV 与 YOLO 必须共享输入、分辨率、ROI、事件、GT、工况和计时起止。正式比例报告 `x/n`、事件数及 Wilson 95% CI（零错误亦报上界），端到端延迟 n/Median/P95，并分 Overall、condition 与独立 session 报告；Validated Event Throughput 用有效单件事件/有效运行分钟，不能用推理 FPS 冒充。实现接口与实施顺序见 [基线设计](docs/design-docs/phase1_obb_baseline.md)和 [执行计划](docs/exec-plans/active/phase1_execution_plan.md)。
