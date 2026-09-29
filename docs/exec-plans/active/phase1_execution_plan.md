# Phase 1 执行计划：单 OBB 基线与数据审计

状态：部分完成（主线基线始于 `7b9ae4b`；脚手架、环境/集成验证、公开 V1.0 元数据检查、有限真实数据转换/审计样本、Phase 1C 原始采集工具、工作树中的 [V2 按零件采集契约](../../data/LABEL_CONTRACT_V2.md) 与 Phase 1D 标注 sidecar/store、只读 QA、轻量 OpenCV 四点击标注器已在当前工作树实现；真实手机预览/采集、真实 Part A GUI 标注与可视/数据验收、完整修正数据获取与训练仍待执行）。以下事实不构成模型质量或生产能力报告。范围见 [项目范围](../../product-specs/project_scope.md)，实现约定见 [基线设计](../../design-docs/phase1_obb_baseline.md)，数据证据见 [V1.0 审计](../../data/MVTEC_SCREWS_V1_AUDIT.md) 与 [能力矩阵](../../data/DATASET_CAPABILITY_MATRIX.md)；采集流程见 [PHONE_CAPTURE_V1](../../data/PHONE_CAPTURE_V1.md)。

## 已完成：脚手架、环境与有限数据核查

- 已建立 `pyproject.toml`、`src/`、`scripts/` 和 `tests/`：单帧 OBB 适配器、命令行入口、环境检查、数据审计脚本与 MVTec Screws 转换入口。项目安装使用 `python -m pip install -e ".[test,inference]"`。
- 本地 `.venv` 已安装 Ultralytics 8.4.165 和 pytest；此前 `compileall` 通过。此前完整项目测试 **31/31 PASS**（历史记录，非本次全量重跑）仅证明软件测试通过，不证明工业准确率、FPS 或模型质量。项目适配器已加载官方 `yolo26n-obb.pt`，在 RTX 4060/CUDA 上对合成空白图像完成烟测推理；不证明 `front`/`back` 识别。
- MVTec 官方 Screws 页面：384 图、13 类、4426 OBB 标注，CC BY-NC-SA 4.0 **非商业**限制。MVTec HALCON 26.05 发布说明指出部分 `class_13` 方向标签错误且已修正。本地检查的公开 GitHub V1.0 镜像元数据为 384 图、4427 标注；train/val/test 分别为 269/55/60 图与 3119/647/661 标注，无跨划分文件名重叠，合并后与未划分元数据完全一致。镜像仅作为传输/审计来源，MVTec 才是数据集与许可的权威来源；计数差异未解决。
- V1.0 的类别 13 有 321 标注；50 个标注角点超出严格 [0,1]，其中 22 个超出当前 Ultralytics 容差 [-0.01,1.01]，分布于类别 1–4。安全样本 `train/screws_004`（16 标注）、`val/screws_001`（16）、`test/screws_022`（8）排除 `class_13` 与跨边界框，已成功转换；现有审计为 3 图 / 3 标签、`errors=[]`。这**不**代表完整 V1.0 可以训练。此前提议的 ICDM 3214/11 来源未获可靠验证，排除在证据之外。
- Phase 1C 原始采集工具 `scripts/capture_phone.py` 已实现：新会话绑定 `part_id`，操作者选择可见面 `front`/`back`，保存会话 JPEG 和尝试日志；未来 OBB 类名为 `<part_id>_front` / `<part_id>_back`，非永久全局两类。不可从角度推断可见面；采集本身不生成 OBB 标注，不表示训练就绪。此前针对性软件验证：合并采集测试 **57 PASS**、`--help` **PASS**（历史记录，非 V2 硬件验证）；真实手机/iVCam 预览及采集的硬件烟测仍**待完成**。
- Phase 1D [OBB 标注指南](../../data/obb_annotation_v1.md)：当前工作树已实现独立于原始采集的标注 sidecar/store、只读 QA 和轻量 OpenCV 四点击标注器。在仓库根目录运行 `python scripts/annotate_obb.py data/captures/<capture-session-id>` 对已接受的 JPEG 保存 OBB 或明确排除；`python scripts/qa_annotations.py data/annotations/<capture-session-id>` 输出只读 JSON 检查报告，错误时退出码非零。软件验证不等于真实 GUI 验收；尚未通过 GUI 标注真实 Part A 采集图像，真实可视/数据验收仍待完成。

## 数据与实测记录：后续顺序

1. **工具已实现；真实 Part A 硬件采集、复核与 GUI 标注/可视验收待完成。** 按 [采集指南](../../data/PHONE_CAPTURE_V1.md) 在手机/iVCam 上验证预览并采集真实 Part A 单件原始图像，记录相机视角、照明、分辨率、合法图像来源、固定传送带 ROI 顶点及版本；明确 `front` / `back` 可见面判据和不可判别处理并人工复核。然后按 [OBB 标注指南](../../data/obb_annotation_v1.md) 用四点击 GUI 标注 OBB 或明确排除，人工可视检查四角、图像坐标系、面类及无方向长轴角度。原始 JPEG 未标注 OBB，MVTec Screws **没有** `front`/`back` 标签，真实 Part A 采集/标注不可省略。
2. **真实数据 QA 与后续组装待完成。** 对上述 sidecar 运行只读 `scripts/qa_annotations.py` 并复核覆盖率、来源与几何错误；不过则补拍/重标，不训练。随后才准备按会话/采集序列隔离划分及 Project Dataset V1 组装、Dataset Audit；目前尚无 Project Dataset V1。
3. **部分完成：公开数据审计；完整数据门禁仍关闭。** 已有上述 V1.0 元数据和 3 图转换/审计样本。接下来获取完整修正/当前数据集，**或**冻结经明确审查的过滤策略，核对类别 13 方向、越界框、数据来源、划分与标注一致性；按会话/采集序列隔离将来的训练/验证/测试，排查近重复、类别及角度覆盖。审计不过不训练；不把有限样本外推成全量合格数据。
4. **部分完成：单帧集成烟测。** 合成空白图像烟测已完成；合法真实测试帧的本地推理运行记录仍待补齐（示例见仓库 `README.md`）。记录权重来源/校验、版本、设备、输入分辨率及命令参数，检查推理可运行及 OBB 输出可读，保留 ROI 过滤前的输出。若无真实帧，标记未运行及原因。不提交原图、权重或私有数据。
5. **待完成：核对范围和训练前条件。** 检查规范、数据审计结论及烟测记录的可追溯性，列出缺口与修正决策。MVTec Screws V1.0 / 公开 OBB 参考数据未取得修正/当前完整版或冻结明确审查的过滤策略，且真实 Part A 原始采集未复核、GUI OBB 标注/可视验收及 QA 未完成、训练/验证/测试未按会话/序列隔离、Dataset Audit 未通过前，**Training Gate CLOSED**。MVTec 不含项目面标签，不能替代真实采集。其后另行规划面向 `<part_id>_front` / `<part_id>_back` 的微调与独立评估；训练仍待执行。单 YOLO26n-OBB 是基线；Seg+OBB 仅在基线证据显示实质瓶颈时作为后续实验，不在 Phase 1 安排长时训练、Seg、跟踪或事件运行时。95% 面准确率、90% mAP50、30/60 FPS 若作为目标均非已达成结果。

通过验证的真实零件采集、OBB 标注与 QA、按会话/序列隔离的数据划分将作为 Phase 2 项目 Dataset V1 的输入；当前尚未形成该项目数据集。

## 停止条件与边界

数据来源、可见面可判别性、OBB 标注/ROI 坐标或序列隔离无法核实，就标记风险并先修复数据；不能声称完整数据通过审计或模型性能达标。角度按模 180° 处理，近方形及重遮挡样本不强报姿态；不能从角度推断正反面。未来路径（**非已实现**）：Camera/Video → ROI → YOLO26n-OBB → Tracking → Multi-frame Aggregation → Confidence/Stability Gate → Decision Zone → PASS/FAIL_FACE/FAIL_ORIENTATION/UNCERTAIN → single-part statistics。`UNCERTAIN` 仅是运行时拒绝/决策状态，不是训练类。跟踪、聚合、门控、单件事件和统计均属后续阶段。未来换型实验先建立 Part A，再加入 Part B，测量新增图像、标注工作量、微调时间、软件/配置变更及 Part A 回归；目前无换型结果。技术选择依据见 [ADR-0001](../../decisions/adr-0001-single-obb-baseline.md)。
