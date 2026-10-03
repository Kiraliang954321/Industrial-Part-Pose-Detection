# Contributing to Industrial-Part-Pose-Detection

感谢你愿意参与这个项目。

本仓库当前采用 **V4.1a evidence-first** 开发方式：先冻结实验问题、数据协议和评价口径，再训练模型和报告结果。贡献应尽量保持可复现、可审查，不以“看起来效果不错”代替证据。

## 可以贡献什么

当前优先方向：

- 真实工业零件采集协议与 Operating Envelope；
- OBB / FEATURE_ANGLE / SYMMETRY_AWARE 姿态方法；
- Traditional CV baseline；
- YOLO OBB 数据、训练和公平实验；
- Tracking / Multi-frame / Decision Zone；
- Risk Gate 与 `UNCERTAIN`；
- Dataset QA、来源审计、重复 / 泄漏检测；
- Recipe Wizard / 新零件引导注册；
- GUI、测试、文档和自动化。

如果改动较大，请先开 Issue 描述问题、预期收益、验证方式和可能影响。

## 开发环境

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,inference]"
python -m pytest -q
```

## 分支和 Pull Request

建议从最新 `main` 创建短生命周期分支，例如：

```text
feat/recipe-wizard
fix/capture-feedback
docs/experiment-protocol
test/dataset-qa
```

Pull Request 请说明：

1. 解决什么问题；
2. 修改了哪些文件 / 模块；
3. 如何验证；
4. 是否改变数据、GT、阈值、实验协议或评价口径；
5. 是否影响 Train / Val / Test / Final Holdout 隔离；
6. 已运行哪些测试。

## 证据规则

请不要：

- 把官方 benchmark 当成本项目实测结果；
- 把预训练模型输出当成项目精度；
- 用 Test / Final Holdout 调参；
- 把同一连续录像随机拆分到多个 split；
- 把 `UNCERTAIN` 当训练类别或 GT；
- 对圆形零件强行输出没有物理意义的 OBB Angle；
- 删除失败样本来美化统计结果。

正式性能结论必须写明：

- Part / Condition / Session；
- 数据与 GT 来源；
- 模型 / 规则版本；
- Operating Envelope；
- 样本数 / Event Count；
- 统计口径。

## 数据、模型与敏感信息

不要提交：

- 未获授权的工业数据；
- 人员隐私数据；
- `data/` 中的本地原始采集；
- `runs/`；
- `*.pt` 权重；
- API Key、Token、密码、`.env`；
- 来源或许可不明确的数据集。

如需提交小型公开测试样本，请在 PR 中说明来源与使用许可。

## 代码质量

提交前至少运行：

```powershell
python -m pytest -q
python -m compileall -q src scripts
git diff --check
```

如果只改文档，也应运行 `git diff --check` 并检查相对链接。

## Issue 建议格式

```text
Problem:
Why it matters:
Proposed approach:
Files / modules:
How to verify:
Data / protocol impact:
```

欢迎对现有设计提出不同意见，但请尽量提供可复现证据或明确的实验方案。
