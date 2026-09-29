# 传送带零件正反面与平面姿态（课堂原型）

目标是用固定相机、低成本照明与单个 OBB 模型探索：在传送带图像中定位零件、识别可见的正/反面，并给出图像平面内的**无方向**姿态（模 180°）。这是非生产原型，不用于设备控制或安全联锁；已完成有限的公开数据转换/审计样本，但尚无获准完整训练的数据集、准确率、帧率或现场可靠性结论。

## 当前阶段与范围

Phase 1 仅做官方 Ultralytics `yolo26n-obb.pt` 单模型基线的可复现运行脚手架、采集/标注规范与数据审计；不安排长时训练，不添加 Seg。仓库已有 `pyproject.toml`、`src/`、`scripts/` 和 `tests/`，包含单帧 OBB 适配器、命令行入口、环境检查、数据审计脚本及自动化测试；Phase 1C 手机/iVCam 原始图像采集器已实现，但真实手机预览/采集硬件烟测待完成，尚未完成项目正反面采集或 OBB 标注。已完成 MVTec Screws V1.0 镜像元数据检查及 3 张真实图像的安全样本转换/审计；完整修正数据集获取、项目正反面采集/标注和训练仍待完成。详情见 [数据能力矩阵](docs/data/DATASET_CAPABILITY_MATRIX.md) 和 [V1.0 审计记录](docs/data/MVTEC_SCREWS_V1_AUDIT.md)。MVTec 官方页面是数据集与许可的权威来源；GitHub 镜像仅用于传输/审计溯源。官方页面列出 384 张图、13 类、4426 个 OBB 标注，CC BY-NC-SA 4.0 **仅限非商业用途**；本地镜像 V1.0 元数据则为 384 张图、4427 个标注，两者计数不一致。MVTec Screws 不含 `front`/`back` 标签。训练门禁保持关闭：必须先获取修正/当前完整数据集，或冻结经明确审查的过滤策略；正反面原始采集还须审查、添加/QA OBB 标注并按会话/序列隔离划分，不能直接训练。模型预训练类别不等于本项目的正反面类别：运行预训练权重只检查环境和推理集成，不作为本任务精度证据。待完整数据训练门禁解除且正反面数据采集/审计通过后，再另行做针对 `front`/`back` 的微调与评估。相机、ROI、跟踪和决策区的职责边界见 [架构](ARCHITECTURE.md)；跟踪、单件事件与决策区触发是**后续集成**，不是本阶段交付。

## 本地准备与最小烟测

需要 Python >=3.10、可联网下载官方权重的环境（或已合法获得的本地权重）；实际图像测试还需要一张**自行采集且有权使用**的单帧图片。GPU 非必需。仓库有项目依赖声明和自动化测试，但没有依赖锁文件或数据集。以下是 PowerShell 示例，在仓库根目录执行；首次推理可能下载权重，具体安装版本请记录，避免把未锁定环境当成复现实验。

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,inference]"
python scripts/check_env.py
python -m pytest -q
# 将 <frame-path> 换成自己的单帧图片路径；确认来源合法且不含敏感内容
# 预训练权重类别不是本项目的 front/back 类别
yolo-conveyor "<frame-path>" --model yolo26n-obb.pt
```

本地 `.venv` 已验证安装 Ultralytics 8.4.165 和 pytest；此前 `compileall` 通过。此前完整项目测试 **31/31 PASS**（历史记录，非本次全量重跑），仅作为软件验证，不作为模型质量证据。项目适配器已加载官方 `yolo26n-obb.pt`，在 RTX 4060/CUDA 上完成一次合成空白图像的烟测推理。这些仅是环境与集成证据，**不是**工业场景准确率、FPS 或正反面识别效果的证据；空白图像烟测不等于对真实零件的验证，独立的 3 图数据审计也不构成模型质量证据。真实帧测试仍需另行记录日期、Python/Ultralytics 版本、权重来源与校验信息、设备、输入分辨率和参数；请勿提交原始图像、权重或数据隐私信息。

## 手机/iVCam 原始采集（Phase 1C）

操作指南见 [PHONE_CAPTURE_V1](docs/data/PHONE_CAPTURE_V1.md)：Windows iVCam 示例在仓库根目录运行 `python scripts/capture_phone.py --source 0 --backend dshow`（实际索引/后端可能不同）。采集器已实现；合并采集测试 **57 PASS**、`--help` **PASS** 仅为软件验证，真实手机/iVCam 预览硬件烟测仍待完成。输出是按人工所选正反面分组的原始 JPEG，不含 OBB 标注，也不具备训练就绪状态。

## 下一步（Phase 1）

1. 按 [范围与验收](docs/product-specs/project_scope.md) 定义正反面可见判据、相机与 ROI；按 [原始采集指南](docs/data/PHONE_CAPTURE_V1.md) 完成真实硬件预览烟测、试采与人工复核，再进行标注；无法从图像判定的面不强行标注。
2. 按 [基线设计](docs/design-docs/phase1_obb_baseline.md) 核查 OBB 四角坐标、标注一致性、按采集序列隔离的数据划分与角度语义；审计不过则先补拍/重标，不训练。
3. 按 [执行计划](docs/exec-plans/active/phase1_execution_plan.md) 继续记录完整数据审计、真实帧烟测与正反面采集；仅在训练门禁解除并另行开展实验时微调、验证并报告实测指标，不得预填结果。

设计选择：[ADR-0001](docs/decisions/adr-0001-single-obb-baseline.md)。已有脚手架、测试与有限样本审计不代表完整数据审计或模型质量通过；未声称 lint 或 typecheck 已运行。
