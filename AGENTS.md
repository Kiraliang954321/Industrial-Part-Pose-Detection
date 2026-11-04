# Project Instructions

## Communication

- 默认使用中文与用户沟通；代码、命令、变量名、路径、API 名称保持英文。
- 结论优先；发现需求、架构或实现方案存在问题时直接指出。

## Workflow

对于新功能、复杂功能和架构修改：先阅读相关代码、文档与实现；明确需求与约束，不立即编码。设计阶段可一次询问一个关键问题，提供 2~3 个方案及优缺点并推荐一个；用户确认设计与 implementation plan 后开始实现。简单、明确、局部修改可简短设计后直接处理。

## 当前项目权威与边界

- 协作者与 Agent 的交接入口为 [成果总览与运行指南](docs/handoff/README.md)，公开预览位于 `docs/previews/`，评价摘要与材料清单位于 `docs/handoff/`。开始继续开发前运行 `python scripts/verify_handoff.py`，并核对所需权重／实验归档是否齐备。历史协议绑定原路径、源码与产物哈希，迁移或新模型使用新开发版本与新协议。源码、协议和公开证据按 `.gitattributes` 保持 LF。
- 用户已确认的 [A5](docs/exec-plans/active/keyed_cell_a5.md)已完成键槽/凸键、外缺口/定位销实体配合与腕部反馈停止、带件退回。独立 `keyed_cell/` 复用 A4 视觉，[冻结协议](docs/data/keyed_cell_a5_protocol_v1.json)后 66 次新抓取中，正常配合 34/36 成功（轴承 16/18、齿轮 18/18），12/12 故意错位工装实际接触停止并保持夹持退回，意外有效进入 0/12。两个正常留件均为轴承端面几何无解。成功深度 3.251–4.168 mm，距 4.2 mm 上限的裕量仍小。机械理想位姿参考独立统计，工装相对配方错位与视觉误差分别归因。完整记录、30 s 原始录像剪辑和腕部轨迹图见 A5 记录。阶段范围为有间隙、闭爪预插入，真实门禁 CLOSED。
- 用户已确认的 [A4](docs/exec-plans/active/active_pose_a4.md)已完成困难姿态诊断、有限主动观察和 10 轮补训。独立 `active_pose/` 在 [A4 协议](docs/data/active_pose_a4_protocol_v1.json)冻结后 54 次新抓取中，正常工况 35/36 完整成功，初检无解的 4/5 经观察恢复并完整成功；黑屏/空抓不观察，无定位结构观察至上限后均留件。补训覆盖收益跨批次不稳定，保留 A3 权重；校准仍有齿轮深度超限，不能说已解决下滑。观察与校正/插入分别计数，同图初检比较不能冒充完整物理 A/B。`pose_cell/` 及更早冻结源码保留；A3 原评价已用于 A4 分析，不是 A4 未见评价。真实门禁 CLOSED。
- 用户进一步选择“覆盖翻转和大倾角，自动摆正到指定朝向，再沿已有轴预定位”。[A3](docs/exec-plans/active/pose_cell_a3.md)在独立 `pose_cell/` 完成侧夹、上下六 RGB、双端面几何与静态外形校核、视觉校正后复拍及原竖轴预定位。[A3 协议](docs/data/pose_cell_a3_protocol_v1.json)冻结后 48 次新抓取为 25 成功 / 20 留件 / 3 齿轮深度超限，正常图像且有结构的总体成功为 25/36；180° 翻转和反向上料各 6/6，90°/120° 各 2/6。保留覆盖与深度边界，不能称任意姿态稳定装配；真值驱动物理参考不能记入预测成功数。实体门禁仍 CLOSED，不默认扩展斜轴工位、压装或啮合。
- 最新用户已明确需要连续空间方向与绕轴角，不能只做正反分类。[A2](docs/exec-plans/active/datum_pose_a2.md)位于 `datum_pose/`，采用定制外圈定位缺口/齿轮键槽、三 RGB、YOLO 定位与孔口椭圆几何融合；[冻结协议](docs/data/datum_pose_a2_protocol_v1.json)后 48 次评价中，有定位结构样本可测 22/40（55%）。必须同时报告覆盖率和条件角度误差，不能声称稳定任意姿态/完整6D。当前输入包络为已知朝上、倾角0–30°；反向/翻转及自动摆正/新姿态装配未验证。机器人展示角和对象 GT 不得进入估计器。A1/螺丝旧代码与成绩保留。
- 用户已确认螺丝阶段足够并要求收尾提交，见[阶段收尾](docs/exec-plans/completed/screw_stage_completion_20261031.md)：H1 为主展示，H2 保留扩展，不再默认细化螺丝精度/滑移。下一阶段转为轴承和锥形齿轮；轴承也须判正反，应选具有明确非对称结构的型号。
- 螺丝阶段已提交 `d9d342b`；最新 [A1 轴承/齿轮](docs/exec-plans/active/bearing_bevel_a1.md)位于 `annular/`，采用 F608ZZ 法兰代理与 SB1-2040 简化锥齿代理，六类单 RGB 方向检测及已知配方套轴预定位。[A1 协议](docs/data/bearing_bevel_a1_protocol_v1.json)冻结后 40 次新抓取为 8 PASS / 16 REJECT / 16 UNCERTAIN，8/8 实际预定位成功。模型判方向，不测完整孔位/6D，不表示压装或齿轮啮合；真实 CAD/实拍与生产门禁仍未通过。不默认追加精细优化。
- 既有二维基线为 [V4.1 项目方案](docs/new-plans/面向柔性流水线的低成本工业零件姿态检测系统_项目方案_V4.1_优化版.md)，实验规则以 [EXPERIMENT_PROTOCOL_V1](docs/new-plans/EXPERIMENT_PROTOCOL_V1.md) 为准；后续对象范围见 [ADR-0002](docs/decisions/adr-0002-industrial-orientation-3d-simulation.md)，最新用途优先按 [ADR-0003](docs/decisions/adr-0003-held-part-inspection-before-assembly.md)与[执行计划](docs/exec-plans/active/industrial_orientation_3d_execution_plan.md)。旧 V2 名称仅是现有软件版本。正式对象及 Operating Envelope 未冻结，协议**尚未正式 FROZEN**，不可用未经物理证据确认的值替代。
- 用户最新主线是“机械臂先夹起零件 → 固定检测姿态相机复核 → 满足条件再装配”，且已选择首轮螺丝对孔预插入。[H1](docs/exec-plans/active/held_inspection_h1.md)已实现独立夹持数据/训练、双 RGB 端点三角化、夹持绑定的单次许可、预测补偿和闭爪预插入；源码位于 `held/`，开发冻结见 [H1 协议](docs/data/screw_held_protocol_v1.json)。M3/M4/L1 旧模型/成绩不能充作 H1 证据，不能默认继续 L1 连续扩线。预插入不是拧紧，代理没有真实螺纹啮合；轴对称绕轴角未估计，不能称为完整 6D。真实门禁仍 CLOSED。
- 最新已完成 [H2](docs/exec-plans/active/held_recheck_h2.md)：`held_recheck/` 增加 Train 拟合的端点残差补偿、孔口双视角复检和独立短时插入许可，仿真共四路相机。[H2 协议](docs/data/screw_held_recheck_protocol_v1.json)冻结后 36 次新抓取中 9/9 实际预插入成功，6 次显著滑移/偏转全部留件；同图 H1/H2 初检均为 24/24，不能声称该独立批次证明拒绝率下降。旧 `held/` / M4 / L1 源码和冻结结果保留。复检后滑移、判定边界、真实标定/配合和连续节拍仍未验证。
- 用户已确认后续只考虑**轴承、锥形齿轮或螺丝**，首轮选择螺丝头尾与有向轴线；M0–M3 视觉开发训练及后续 [M4 Panda 仿真闭环](docs/exec-plans/active/screw_m4_closed_loop.md)已完成。用户明确当前暂无实物/CAD、先完成仿真，不再安排瓶盖训练、换型或机械臂实验。M4 是单次 RGB 观测、固定支撑面的 2.5D 规划与关节反馈执行；完整 6D、执行中视觉伺服和动作策略学习未实现，实体执行机构仍在范围外。真实型号、工况与真值协议仍待定义；不得以仿真真值冒充预测或以仿真成绩替代真实验收。
- 用户进一步选择分度式用途验证，[L1 单工位](docs/exec-plans/active/screw_line_simulation_v1.md)已完成物理索引托盘、8 帧事件判定及放行/分拣/停机留件。冻结后 24 个事件为 8 PASS / 8 FAIL_ORIENTATION / 8 UNCERTAIN，16 次自动动作完成、8 次停机留件，缺/重复事件 0。ID 来自声明的模拟 PLC job，不是视觉跟踪；故障/边界人为加重，不能把总体比例当生产分布。独立 episode 不包含连续上/卸料、回程和复核恢复，连续吞吐必须保持未测。
- V4.1 以四个研究问题验证假设：RQ1 Traditional CV vs YOLO；RQ2 Single Frame vs Event Decision；RQ3 Risk-Coverage-Latency/Validated Event Throughput；RQ4 Part A → Part B 换型。六阶段：Phase 1 Data Foundation + Experiment Protocol Freeze；Phase 2 Perception Baseline；Phase 3 Reliable Event Decision；Phase 4 Risk-Aware Decision；Phase 5 Flexible Changeover；Phase 6 Production-Oriented Finalization。Phase 1 验收是 **Experiment Protocol FROZEN + Dataset V1 Gate OPEN**，不是工具可用或数据组装成功。
- 已有采集、OBB 标注/QA、四路 session-aware 组装和单帧脚手架；历史 8 张人工 GUI 接受及新增 80 张助手复核的实拍开发微调已完成，见[开发记录](docs/exec-plans/active/real_cap_80_development.md)，不等于正式指标。组装 schema v2 支持 `train/val/test/final_holdout`，`data.yaml` 仅暴露 train/val/test。螺丝视觉、M4 与 L1 均属代理仿真开发证据，各自开发协议冻结不等于真实实验协议 FROZEN。尚无真实新工业件正式数据、正式四路 split、正式训练、Phase 2–6 正式事件验证或完整 6D 证据，真实门禁保持 CLOSED。
- Phase 1 须先依据实物预冻结 Part A/B、Front/Back/Orientation 判据，记录 Operating Envelope draft、事件级 Ground Truth schema 与 Test Condition Matrix；再完成真实采集、标注/QA/可视复核、session-aware train/val/test/final_holdout、Dataset Audit/Provenance/分布与泄漏检查，以及 Traditional CV 公平对照、统计报告、跨 Session 重复性和 Final Test 不调参规则。所有门禁通过前不正式训练；MVTec Screws 是独立公开参考数据，不含项目正反面标签。
- Ground Truth：`CONFORMING` / `NONCONFORMING` / `INDETERMINATE`；`UNCERTAIN` 是系统决策、不是 GT 或训练类。Wrong Accept = `NONCONFORMING → PASS`；INDETERMINATE 单独分层，不混入确定 GT 分母。只在 Train/Val/Calibration 调参；Test/Final Holdout 不用于调参，Test 错误回流后不能再把原 Test 当无偏最终测试。对照共享输入、ROI、事件、工况、GT、计时口径；核心比例报告 x/n、Wilson 95% CI（含零错误上界），并分 condition/session；报告端到端 P95 与 Validated Event Throughput，结论限于已冻结 Operating Envelope。
- 已定义面类任务沿用 `<part_id>_front` / `<part_id>_back`；人工可见面不能由无方向的模 180° OBB 长轴推断。新对象的头尾、轴向与 3D 标签另按任务定义，不能强制套用两面分类或现有二维 angle。既有 Phase 2 对照仍为 Traditional CV、YOLO26n-OBB、YOLO26s-OBB；Seg+OBB 仅在证据支持后考虑。L1 已有单件工位事件和保守证据拒绝仿真版；多目标视觉跟踪、真实风险工作点标定及换型尚未实现，不以烟测或官方 benchmark 冒充项目性能。
- 范围见 [project_scope](docs/product-specs/project_scope.md)，当前步骤见[工业件执行计划](docs/exec-plans/active/industrial_orientation_3d_execution_plan.md)，方向决定见 [ADR-0002](docs/decisions/adr-0002-industrial-orientation-3d-simulation.md)。既有二维设计、门禁与技术起点分别见 [phase1_obb_baseline](docs/design-docs/phase1_obb_baseline.md)、[phase1_execution_plan](docs/exec-plans/active/phase1_execution_plan.md)、[ADR-0001](docs/decisions/adr-0001-single-obb-baseline.md)；采集类名、标注和组装能力分别见 [LABEL_CONTRACT_V2](docs/data/LABEL_CONTRACT_V2.md)、[obb_annotation_v1](docs/data/obb_annotation_v1.md)、[project_dataset_v1](docs/data/project_dataset_v1.md)。

## Project Documentation

开发前根据任务需要阅读 `ARCHITECTURE.md`、`docs/product-specs/`、`docs/design-docs/`、`docs/exec-plans/active/`、`docs/decisions/`；不要无目的加载所有文档。

## Implementation

- 遵循现有架构与代码风格；优先修改现有实现，不做无关重构或不必要抽象；保持模块职责单一，避免超大文件与耦合。
- 新增或修改行为时补充对应测试，不扩大当前需求范围。

## Verification

任务完成前检查 diff、运行相关测试、lint、支持时运行 typecheck，检查边界与需求；未验证不声称已完成。

## Git

未经用户明确要求不执行 `git commit`、`git push`、`git rebase`、`git reset --hard` 或修改 Git 历史；提交前展示变更摘要。

## Safety

删除文件或目录、修改 `.env`/密钥证书/CI/CD/生产配置或破坏性数据库迁移，须先获得用户确认。
