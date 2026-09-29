# Phase 1 单模型 OBB 基线设计

## 输入、输出与实验边界

固定俯视或近俯视相机的合法来源单帧图像，记录相机/照明、原图分辨率和 ROI 顶点（图像像素坐标）。官方 Ultralytics `yolo26n-obb.pt` 是唯一 Phase 1 模型；烟测可用预训练权重检查安装、推理、OBB 四角/类别/置信度输出和可视化是否可读取。预训练类别不对应未来 `<part_id>_front` / `<part_id>_back`，不作项目类别预测性能声明。Phase 1 不做长时训练、不使用 Seg；仅在基线证据揭示实质瓶颈后才可将 Seg+OBB 作为后续实验，不能默认为基线。真实类别能力须先审计真实零件数据，再另行微调和独立评估。

单帧记录应能追溯输入图、权重来源及校验、Ultralytics/Python 版本、设备、输入尺寸、运行参数、ROI 与数据版本；预测时保留未经过 ROI 过滤的原始输出供漏检/边界分析，不把可视化当作定量证据。不得提交未经许可的原图、权重或隐私信息。

## 数据与标注约定（软件已实现，真实数据待审计）

- Phase 1C 原始采集工具及工作树中的 [V2 按零件契约](../data/LABEL_CONTRACT_V2.md) 已存在；当前工作树的 Phase 1D [OBB 标注 sidecar 契约与操作指南](../data/obb_annotation_v1.md) 已实现独立 sidecar/store、只读 QA 和轻量 OpenCV 四点击标注器，但尚无真实项目 OBB 数据。仓库根目录运行 `python scripts/annotate_obb.py data/captures/<capture-session-id>` 对接受的 JPEG 标注或明确排除，再运行 `python scripts/qa_annotations.py data/annotations/<capture-session-id>` 获取只读 JSON QA 报告；sidecar 不修改原始采集。尚未通过 GUI 标注真实 Part A 图像，真实可视/数据验收待完成。每次新会话绑定一个 `part_id`；人工选择 `front` / `back` 作为面标签，未来 OBB 类名为 `<part_id>_front` / `<part_id>_back`，非全局两类；数值类 ID 仅在 Dataset V1 组装时分配，Part A 对应 `0=<part_id>_front`、`1=<part_id>_back`。先制定每种零件的可见面判据并由人工核查；朝向不决定可见面。反光、严重遮挡或其他无法确定可见面的帧标记不确定并复核，不硬分配类别。数据是否真的支持判别必须等采集审计后确认。
- OBB 四角覆盖零件可见轮廓，记录角点顺序/坐标系、可见边界及遮挡规则；对同一长轴的近 180° 旋转保持标注一致。核查角点、宽高和类别；对于近方形或长轴模糊目标将角度标为不可评估。组装输出 Ultralytics OBB 标注时，类别 ID 与归一化四角坐标须按实际数据配置核对，裁切/缩放后同时变换角点和 ROI；sidecar 本身不存数值类 ID。
- 工作树已有 `scripts/build_project_dataset.py` 及 [Project Dataset V1 组装指南](../data/project_dataset_v1.md)，从 QA-clean 标注 sidecar 按显式完整会话 train/val/test 计划组装；尚无真实 Part A 试点标注集、真实 split plan 或已组装的真实 Project Dataset V1，也未微调。真实数据顺序：收集多个 Part A 会话 → 标注或明确排除并 GUI 可视验收 → 对每个 sidecar 运行 `scripts/qa_annotations.py` → 编写显式 split plan → 运行 `scripts/build_project_dataset.py` → 运行 `scripts/audit_dataset.py <output> --classes 2` → Codex/数据复核。**禁止随机逐帧划分**，至少三个会话填充 train/val/test，每个 split 均需有已标注的 front/back。检查相邻帧及近重复泄漏、各面类、角度、位置、光照、遮挡、边界覆盖及缺失；不过则补拍或重标。MVTec Screws V1.0 是独立公开参考数据，仍需修正/当前完整版或明确审查的过滤策略，且没有本项目面类，不得冒充 Project Dataset V1。**Training Gate CLOSED**；软件测试或未来组装成功本身不解锁训练，仅在所有相关门禁通过后另行考虑训练阶段。审计结论不是精度结论。

## 单帧几何与 ROI

基线可先对原图推理，再按同一图像坐标系的固定 ROI 过滤（OBB 中心在 ROI 内），保留过滤前输出；跨 ROI 边缘和遮挡样本人工复核。若日后裁切、缩放，ROI 顶点与标注必须同变换，记录坐标变换，避免将裁切效果误认为模型能力改进。

从 OBB 四角选长边的轴向方向，规范化为相对图像水平轴的 `θ ∈ [0°, 180°)`；记录图像坐标系的轴方向。不同 Ultralytics 输出的角度范围、角点起点或宽高交换不能直接用作头尾朝向。比较角度使用 `min(|θp-θg|, 180°-|θp-θg|)` 度（两角均先归一化）；近方形/严重遮挡的轴不可评估，不强报角度。无世界坐标标定时不声称物理角度。

## 检查与后续接口

Phase 1 检查环境与模型可运行、输出结构可检查、ROI/角度/标注契约和数据审计记录是否完整；没有图像时记录未执行烟测。主线基线始于 `7b9ae4b`；软件/合成烟测不构成真实零件识别、微调指标或生产就绪证据。后续另行开展微调与评估后才可报告按零件面类的实测指标。未来路径（未实现的跟踪/事件运行时）：Camera/Video → ROI → YOLO26n-OBB → Tracking → Multi-frame Aggregation → Confidence/Stability Gate → Decision Zone → PASS/FAIL_FACE/FAIL_ORIENTATION/UNCERTAIN → single-part statistics。`UNCERTAIN` 是运行时拒绝/决策状态，不是训练类；不能把单帧候选描述为稳定事件或设备控制指令。未来换型实验建立 Part A 后加入 Part B，测量新增图像、标注工作、微调时间、软件/配置变更与回归；目前无换型或性能结果。

范围与验收见 [项目范围](../product-specs/project_scope.md)，执行步骤见 [Phase 1 计划](../exec-plans/active/phase1_execution_plan.md)，决策见 [ADR-0001](../decisions/adr-0001-single-obb-baseline.md)。
