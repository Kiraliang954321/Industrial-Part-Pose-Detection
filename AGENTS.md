# Project Instructions

## Communication

- 默认使用中文与用户沟通。
- 代码、命令、变量名、路径、API 名称保持英文。
- 结论优先。
- 发现需求、架构或实现方案存在问题时直接指出。

## Workflow

对于新功能、复杂功能和架构修改：

1. 先阅读相关代码、文档和现有实现。
2. 不要立即编码。
3. 先明确需求和约束。
4. 设计阶段可以主动向用户提问。
5. 每次只询问一个关键问题。
6. 提供 2\~3 个实现方案并说明优缺点。
7. 推荐一个方案。
8. 用户确认设计后再制定 implementation plan。
9. 用户确认后开始实现。

对于简单、明确、局部的修改，可以使用简短设计后直接处理。

## Phase 1 项目边界

- V2 定位：普通 RGB 相机、消费级 GPU 与开源 YOLO26n-OBB 支撑的低成本、软件定义、可再训练柔性产线零件可见面 + 2D 姿态原型，不是生产系统或安全联锁。主线基线始于 `7b9ae4b`；现有单帧脚手架、有限公开数据审计和 Phase 1C 原始采集工具；当前工作树已有 V2 按零件采集契约及 Phase 1D 标注 sidecar/store、只读 QA、轻量 OpenCV 四点击标注器。尚未通过 GUI 标注任何真实 Part A 采集图像，真实可视/数据验收待完成；无 Project Dataset V1、微调指标或生产就绪证据。
- Phase 1 仅用官方 Ultralytics `yolo26n-obb.pt` 作单模型 OBB 基线与未来微调起点；不安排长时训练。Seg+OBB 仅在基线证据表明有实质瓶颈时作为后续实验评估，不加入 Phase 1。
- 采集会话绑定 `part_id`，人工选择的 `front` / `back` 是可见面标签；未来训练类名为 `<part_id>_front` / `<part_id>_back`，并非永久全局两类。OBB 长轴姿态无方向、模 180°；不能由角度推断可见面或头尾。`UNCERTAIN` 只用于未来运行时拒绝/决策，不是训练类。
- Training Gate 保持 CLOSED：下一步依次进行真实 Part A 采集与复核、OBB GUI 标注及真实可视验收、只读 QA，再在后续按会话/序列隔离划分并组装 Project Dataset V1、执行 Dataset Audit；软件已实现不等于这些真实数据步骤通过；MVTec Screws V1.0 / 公开 OBB 参考数据的修正/当前完整版或明确审查的过滤策略亦须先核定，不能把其无正反面标签的数据当项目面类训练数据。
- 未来路径（非当前实现）：Camera/Video → ROI → YOLO26n-OBB → Tracking → Multi-frame Aggregation → Confidence/Stability Gate → Decision Zone → PASS/FAIL_FACE/FAIL_ORIENTATION/UNCERTAIN → single-part statistics。跟踪、事件、判定及柔性换型验证均属后续阶段；换型实验先建立 Part A，再加入 Part B，记录新增图像、标注工作量、微调时间、软件/配置变更与回归。
- 需求与验收见 `docs/product-specs/project_scope.md`，设计见 `docs/design-docs/phase1_obb_baseline.md`，计划见 `docs/exec-plans/active/phase1_execution_plan.md`，决策见 `docs/decisions/adr-0001-single-obb-baseline.md`，采集类名见 `docs/data/LABEL_CONTRACT_V2.md`，标注 sidecar、`python scripts/annotate_obb.py data/captures/<capture-session-id>`（四点击标注）及 `python scripts/qa_annotations.py data/annotations/<capture-session-id>`（只读 QA）见 [OBB 标注指南](docs/data/obb_annotation_v1.md)。

## Project Documentation

开发前根据任务需要阅读：

- 项目架构：`ARCHITECTURE.md`
- 产品需求：`docs/product-specs/`
- 功能设计：`docs/design-docs/`
- 当前执行计划：`docs/exec-plans/active/`
- 技术决策：`docs/decisions/`

不要无目的加载所有文档，只读取当前任务相关内容。

## Implementation

- 优先遵循项目现有架构和代码风格。
- 优先修改现有实现，不创建没有必要的新抽象。
- 不进行与当前任务无关的重构。
- 避免超出当前需求的功能。
- 新增或修改行为时，应补充对应测试。
- 保持模块职责单一。
- 避免超大文件和高度耦合模块。

## Verification

任务完成前：

1. 检查本次 diff。
2. 运行相关测试。
3. 运行 lint。
4. 运行 typecheck（如果项目支持）。
5. 检查异常和边界情况。
6. 确认需求全部实现。

不要在没有验证的情况下声称任务已经完成。

## Git

未经用户明确要求：

- 不执行 `git commit`
- 不执行 `git push`
- 不执行 `git rebase`
- 不执行 `git reset --hard`
- 不修改 Git 历史

提交前先展示变更摘要。

## Safety

以下操作必须先获得用户确认：

- 删除文件或目录
- 修改 `.env`
- 修改 token、密钥、证书
- 修改 CI/CD
- 修改生产环境配置
- 数据库破坏性迁移
