# Project Instructions

## Communication

- 默认使用中文与用户沟通；代码、命令、变量名、路径、API 名称保持英文。
- 结论优先；发现需求、架构或实现方案存在问题时直接指出。

## Workflow

对于新功能、复杂功能和架构修改：先阅读相关代码、文档与实现；明确需求与约束，不立即编码。设计阶段可一次询问一个关键问题，提供 2~3 个方案及优缺点并推荐一个；用户确认设计与 implementation plan 后开始实现。简单、明确、局部修改可简短设计后直接处理。

## 当前项目权威与边界

- 当前方案为 [V4.1 项目方案](docs/new-plans/面向柔性流水线的低成本工业零件姿态检测系统_项目方案_V4.1_优化版.md)，实验规则以 [EXPERIMENT_PROTOCOL_V1](docs/new-plans/EXPERIMENT_PROTOCOL_V1.md) 为准；旧 V2 名称的采集契约等仅是现有软件版本，不是当前项目方案。协议文件写有条件性冻结声明，但 Part A、Part B 及 Operating Envelope 均为 TBD，**尚未正式 FROZEN**。不可用未经物理证据确认的值替代。
- V4.1 以四个研究问题验证假设：RQ1 Traditional CV vs YOLO；RQ2 Single Frame vs Event Decision；RQ3 Risk-Coverage-Latency/Validated Event Throughput；RQ4 Part A → Part B 换型。六阶段：Phase 1 Data Foundation + Experiment Protocol Freeze；Phase 2 Perception Baseline；Phase 3 Reliable Event Decision；Phase 4 Risk-Aware Decision；Phase 5 Flexible Changeover；Phase 6 Production-Oriented Finalization。Phase 1 验收是 **Experiment Protocol FROZEN + Dataset V1 Gate OPEN**，不是工具可用或数据组装成功。
- 已有 part-aware 原始采集、OBB sidecar/四点击标注/只读 QA、按完整 session 划分的 Dataset V1 组装软件，以及单帧 OBB 脚手架。当前组装器已支持 schema v2 的 `train/val/test/final_holdout` 四路完整会话划分；`data.yaml` 只暴露 train/val/test，Final Holdout 仅在物理输出与 manifest 中记录为 untuned final validation。无真实 Part A 数据集、真实四路 split、正式训练、模型指标或 Phase 2–6 运行时证据；软件支持本身不表示门禁通过。真实 Part A GUI 可视验收和硬件采集尚待完成。
- Phase 1 须先依据实物预冻结 Part A/B、Front/Back/Orientation 判据，记录 Operating Envelope draft、事件级 Ground Truth schema 与 Test Condition Matrix；再完成真实采集、标注/QA/可视复核、session-aware train/val/test/final_holdout、Dataset Audit/Provenance/分布与泄漏检查，以及 Traditional CV 公平对照、统计报告、跨 Session 重复性和 Final Test 不调参规则。所有门禁通过前不正式训练；MVTec Screws 是独立公开参考数据，不含项目正反面标签。
- Ground Truth：`CONFORMING` / `NONCONFORMING` / `INDETERMINATE`；`UNCERTAIN` 是系统决策、不是 GT 或训练类。Wrong Accept = `NONCONFORMING → PASS`；INDETERMINATE 单独分层，不混入确定 GT 分母。只在 Train/Val/Calibration 调参；Test/Final Holdout 不用于调参，Test 错误回流后不能再把原 Test 当无偏最终测试。对照共享输入、ROI、事件、工况、GT、计时口径；核心比例报告 x/n、Wilson 95% CI（含零错误上界），并分 condition/session；报告端到端 P95 与 Validated Event Throughput，结论限于已冻结 Operating Envelope。
- 未来训练类为 `<part_id>_front` / `<part_id>_back`；采集中的 `front` / `back` 是人工可见面标签，不能由无方向的模 180° OBB 长轴推断。Phase 2 比较 Traditional CV、YOLO26n-OBB 与 YOLO26s-OBB；Seg+OBB 只在基线证明瓶颈后另行考虑，不是默认主线。跟踪、事件、风险拒绝及换型均未实现，不可把单帧烟测或官方 benchmark 当成项目性能。
- 范围见 [project_scope](docs/product-specs/project_scope.md)，设计见 [phase1_obb_baseline](docs/design-docs/phase1_obb_baseline.md)，当前步骤见 [phase1_execution_plan](docs/exec-plans/active/phase1_execution_plan.md)，决定见 [ADR-0001](docs/decisions/adr-0001-single-obb-baseline.md)，采集类名见 [LABEL_CONTRACT_V2](docs/data/LABEL_CONTRACT_V2.md)，标注见 [obb_annotation_v1](docs/data/obb_annotation_v1.md)，组装能力和缺口见 [project_dataset_v1](docs/data/project_dataset_v1.md)。

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
