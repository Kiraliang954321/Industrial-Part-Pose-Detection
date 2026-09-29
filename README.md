# FlexPose-YOLO：低成本柔性产线零件可见面与 2D 姿态原型

V2 项目定位是用普通 RGB 相机、消费级 GPU 与开源 YOLO26n-OBB，探索可通过软件配置与再训练换型的低成本工业检测。识别每种零件的可见正/反面和图像平面内**无方向**长轴姿态（模 180°）。这是课堂原型，不用于设备控制或安全联锁；当前无真实项目 OBB 数据集、微调指标、现场可靠性或生产就绪证据。

## 当前阶段与范围

主线基线始于 `7b9ae4b`。Phase 1 仅做官方 Ultralytics `yolo26n-obb.pt` 单模型基线的可复现运行脚手架、采集/标注规范与数据审计；不安排长时训练，不添加 Seg。Seg+OBB 仅在基线证据证明有实质瓶颈时作为后续实验。仓库已有 `pyproject.toml`、`src/`、`scripts/` 和 `tests/`，包含单帧 OBB 适配器、命令行入口、环境检查、数据审计脚本及自动化测试；Phase 1C 手机/iVCam 原始图像采集器已实现，工作树现有 V2 按零件采集契约；Phase 1D 标注 sidecar/store、只读 QA 和轻量 OpenCV 四点击标注器也已在当前工作树实现。真实手机预览/采集硬件烟测待完成，尚未通过 GUI 标注任何真实 Part A 采集图像，真实图像的可视验收仍待完成。已完成 MVTec Screws V1.0 镜像元数据检查及 3 张真实图像的安全样本转换/审计；完整修正数据集获取、真实零件可见面采集/标注和训练仍待完成。详情见 [数据能力矩阵](docs/data/DATASET_CAPABILITY_MATRIX.md) 和 [V1.0 审计记录](docs/data/MVTEC_SCREWS_V1_AUDIT.md)。MVTec 官方页面是数据集与许可的权威来源；GitHub 镜像仅用于传输/审计溯源。官方页面列出 384 张图、13 类、4426 个 OBB 标注，CC BY-NC-SA 4.0 **仅限非商业用途**；本地镜像 V1.0 元数据则为 384 张图、4427 个标注，两者计数不一致。MVTec Screws 不含 `front`/`back` 标签。模型预训练类别不等于项目面类：运行预训练权重只检查环境和推理集成，不作为本任务精度证据。MVTec Screws V1.0 / 公开 OBB 参考数据的修正/当前完整版或明确审查的过滤策略须先核定；其本身没有项目面标签。真实零件采集并复核、OBB 标注与 QA、按会话/序列隔离划分及 Dataset Audit 全部通过前，**Training Gate CLOSED**。未来微调按零件使用 `<part_id>_front` / `<part_id>_back` 类名：采集器中的 `front` / `back` 仅为人工可见面标签，不可由角度推断。详见 [Label Contract V2](docs/data/LABEL_CONTRACT_V2.md)。相机、ROI、跟踪和决策区的职责边界见 [架构](ARCHITECTURE.md)；跟踪、单件事件与决策区触发是**后续集成**，不是本阶段交付。

## 本地准备与最小烟测

需要 Python >=3.10、可联网下载官方权重的环境（或已合法获得的本地权重）；实际图像测试还需要一张**自行采集且有权使用**的单帧图片。GPU 非必需。仓库有项目依赖声明和自动化测试，但没有依赖锁文件或数据集。以下是 PowerShell 示例，在仓库根目录执行；首次推理可能下载权重，具体安装版本请记录，避免把未锁定环境当成复现实验。

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,inference]"
python scripts/check_env.py
python -m pytest -q
# 将 <frame-path> 换成自己的单帧图片路径；确认来源合法且不含敏感内容
# 预训练权重类别不是未来的 <part_id>_front/<part_id>_back 类别
yolo-conveyor "<frame-path>" --model yolo26n-obb.pt
```

本地 `.venv` 已验证安装 Ultralytics 8.4.165 和 pytest；此前 `compileall` 通过。此前完整项目测试 **31/31 PASS**（历史记录，非本次全量重跑），仅作为软件验证，不作为模型质量证据。项目适配器已加载官方 `yolo26n-obb.pt`，在 RTX 4060/CUDA 上完成一次合成空白图像的烟测推理。这些仅是环境与集成证据，**不是**工业场景准确率、FPS 或正反面识别效果的证据；空白图像烟测不等于对真实零件的验证，独立的 3 图数据审计也不构成模型质量证据。真实帧测试仍需另行记录日期、Python/Ultralytics 版本、权重来源与校验信息、设备、输入分辨率和参数；请勿提交原始图像、权重或数据隐私信息。

## 手机/iVCam 原始采集（Phase 1C）

操作指南见 [PHONE_CAPTURE_V1](docs/data/PHONE_CAPTURE_V1.md)：Windows iVCam 示例在仓库根目录运行 `python scripts/capture_phone.py --part-id screw_A-1 --source 0 --backend dshow`（实际索引/后端可能不同）。采集器已实现；此前合并采集测试 **57 PASS**、`--help` **PASS** 仅为历史软件验证，真实手机/iVCam 预览硬件烟测仍待完成。每个新会话须绑定一个 `part_id`；输出是按人工所选可见面分组的原始 JPEG，不含 OBB 标注，也不具备训练就绪状态。

## Phase 1D OBB 标注与 QA（软件已实现，真实验收待完成）

操作和 sidecar 格式见 [OBB Annotation Sidecar V1](docs/data/obb_annotation_v1.md)。在仓库根目录运行 `python scripts/annotate_obb.py data/captures/<capture-session-id>`，对已接受的 JPEG 用 OpenCV 界面依次点击四角并保存或明确排除；标注写入独立 sidecar，不改原始采集。运行 `python scripts/qa_annotations.py data/annotations/<capture-session-id>` 取得只读 JSON 覆盖率/几何/来源检查报告，错误时退出码非零。软件可用不代表真实 Part A GUI 可视验收已通过，也不代表数据可训练。

## 下一步（Phase 1）

1. 按 [范围与验收](docs/product-specs/project_scope.md) 定义可见面判据、相机与 ROI；按 [原始采集指南](docs/data/PHONE_CAPTURE_V1.md) 完成真实硬件预览烟测、Part A 采集和人工复核；无法判定的面不强行标注。
2. 用上述 OpenCV 工具对真实 Part A 采集图像做 OBB 标注并完成 GUI 可视验收，再运行只读 QA、人工复核四角/面类/角度语义；审计不过则先补拍/重标，不训练。
3. 随后按 [基线设计](docs/design-docs/phase1_obb_baseline.md) 筹备按会话/采集序列隔离的划分及未来 Project Dataset V1 组装与 Dataset Audit；目前尚无 Project Dataset V1，**Training Gate CLOSED**。按 [执行计划](docs/exec-plans/active/phase1_execution_plan.md) 继续记录完整数据审计和真实帧烟测；仅在训练门禁解除并另行开展实验时微调、验证并报告实测指标，不得预填结果。

未来运行路径（**尚未实现跟踪/事件运行时**）：Camera/Video → ROI → YOLO26n-OBB → Tracking → Multi-frame Aggregation → Confidence/Stability Gate → Decision Zone → PASS/FAIL_FACE/FAIL_ORIENTATION/UNCERTAIN → single-part statistics。`UNCERTAIN` 只表示运行时拒绝/决策，绝非训练类。未来柔性换型实验先建立 Part A，再加入 Part B，量化新增图像、标注工作量、微调时间、软件/配置变更及 Part A 回归；现在没有换型结果。95% 面准确率、90% mAP50、30/60 FPS 若作为未来目标均非已达成结果。

设计选择：[ADR-0001](docs/decisions/adr-0001-single-obb-baseline.md)。已有脚手架、测试与有限样本审计不代表完整数据审计或模型质量通过；未声称 lint 或 typecheck 已运行。
