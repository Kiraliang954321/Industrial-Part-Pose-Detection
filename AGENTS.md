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

- 本仓库当前是低成本传送带零件正/反面与平面姿态的课堂原型文档/基线脚手架，不是生产系统；不要把规划中的能力写成已经验证的结果。
- Phase 1 只采用官方 Ultralytics `yolo26n-obb.pt` 作为单模型 OBB 基线与后续微调起点，不引入 Seg。当前工作是基线脚手架和数据审计，不安排长时训练。
- OBB 的轴向姿态按无方向的 180° 周期处理；正反面是类别标签，不可由角度推断头尾方向。
- 传送带 ROI 用于限定观测范围；跨帧跟踪和决策区触发属于后续集成，不属于 Phase 1 检测基线。
- 需求与验收见 `docs/product-specs/project_scope.md`，设计见 `docs/design-docs/phase1_obb_baseline.md`，计划见 `docs/exec-plans/active/phase1_execution_plan.md`，决策见 `docs/decisions/adr-0001-single-obb-baseline.md`。

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
