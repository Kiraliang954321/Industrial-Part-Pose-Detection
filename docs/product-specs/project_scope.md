# 项目范围：传送带零件正反面与平面姿态

## 目标与现状

V2 定位是使用普通 RGB 相机、消费级 GPU、开源 YOLO26n-OBB 的低成本、软件定义、可再训练柔性产线零件可见面与 2D 姿态原型；先建立 Part A，再实验性增加 Part B，不将当前单件试验限定为永久全局两类。课堂原型，非生产系统；不得用于安全联锁或设备控制。主线基线始于 `7b9ae4b`；已具单帧脚手架、Phase 1C 原始采集工具、工作树中的 [V2 按零件采集契约](../data/LABEL_CONTRACT_V2.md) 和 `scripts/build_project_dataset.py` / [Project Dataset V1 组装指南](../data/project_dataset_v1.md) 所述的从 QA-clean 标注 sidecar 按显式完整会话 train/val/test 划分的组装工具，但无真实 Part A 试点标注集、真实 split plan、已组装的真实 Project Dataset V1、微调指标或生产就绪证据。可见面可辨性、准确率、帧率和现场可靠性**尚未验证**。

## Phase 1 冻结范围

- 仅以官方 Ultralytics `yolo26n-obb.pt` 为**单模型 OBB** 预训练基线及未来微调起点；本阶段建立可复现的运行脚手架、输入/输出约定、采集/标注规范和数据审计，不进行长时训练，不引入 Seg。Seg+OBB 仅在基线证据证明实质瓶颈后作为后续实验。
- 用有权使用的单帧测试图、固定相机与照明配置记录、传送带 ROI 验证环境及 OBB 输出可读性。预训练类别不等同于未来的 `<part_id>_front` / `<part_id>_back`；采集中的 `front` / `back` 是人工可见面标签，非最终全局类。数值类 ID 仅在 Dataset V1 组装时分配，Part A 为 `0=<part_id>_front`、`1=<part_id>_back`。烟测不证明任务精度。
- ROI 只限定观测范围；以同一图像坐标系记录 ROI 顶点及标注，框中心在 ROI 内才视为候选；边界与遮挡样本须人工复核。跟踪、跨帧 ID、决策区触发和单件决策属于后续集成，不是本阶段交付。
- 每个采集会话绑定一个 `part_id`；`front` / `back` 必须由人工可见面判据标注，不能由角度推断；不确定的样本不强行标注。未来运行时 `UNCERTAIN` 是拒绝/决策状态，不是训练类。长轴角度按图像水平轴 `θ ∈ [0°, 180°)` 表示，`θ` 与 `θ + 180°` 等价，不表达头尾方向；近方形或遮挡导致轴不稳定时标记为不可评估。

## 数据审计与验收

1. 形成采集清单：相机视角、照明、分辨率、图像来源/使用权、传送带 ROI 与其版本；覆盖不同可见面、旋转、位置、边界、遮挡等场景，并记录缺口。数据存在与能力均须实际审计，不可凭计划视为已满足。
2. 形成标注规范与审计记录：按可见面证据核查类别、OBB 四角/长轴、模 180° 姿态、无从判别时的处理、ROI 边界样本；明确同一坐标系及裁切/缩放变换。抽查分歧样本并记录修订，而不是把模糊样本算作正确。
3. 真实数据顺序：收集多个 Part A 会话 → 标注或明确排除并可视复核 → 对各 sidecar 运行 `scripts/qa_annotations.py` → 编写显式完整会话 split plan → 运行 `scripts/build_project_dataset.py` → 运行 `scripts/audit_dataset.py <output> --classes 2` → Codex/数据复核。不可随机逐帧划分；当前契约至少需要三个会话填充 train/val/test，且每个 split 均有已标注的 front/back。检查近重复、数据泄漏、各面类/姿态覆盖和难例；审计不通过则补拍或重标，不训练，也不报告任务指标。**Training Gate CLOSED**：软件测试通过或未来数据组装成功均不单独解锁训练；所有相关门禁通过后才可另行考虑训练阶段和独立评估，**不等于模型已达标**。MVTec Screws V1.0 是独立的公开参考数据，还须取得修正/当前完整版或冻结明确审查的过滤策略；其无项目面标签，不能充当 Project Dataset V1。
4. 保留烟测运行记录（权重来源/校验、Ultralytics/Python 版本、设备、输入尺寸、参数、ROI 版本与可视化/输出检查）；若没有帧或环境则明确记录未运行。Phase 1 的验收是规范、记录及基线脚手架可检查，不以预填准确率或实时 FPS 验收。

## 后续 V2 验证（非 Phase 1 交付）

计划路径：Camera/Video → ROI → YOLO26n-OBB → Tracking → Multi-frame Aggregation → Confidence/Stability Gate → Decision Zone → PASS/FAIL_FACE/FAIL_ORIENTATION/UNCERTAIN → single-part statistics；跟踪、事件及统计运行时均未实现。柔性换型实验应先建立 Part A，再加入 Part B，记录新增图像、标注工作量、微调时间、软件/配置变更与 Part A 回归。95% 面准确率、90% mAP50、30/60 FPS 若作为未来目标，均**不是**已取得指标。

实现约定见 [基线设计](../design-docs/phase1_obb_baseline.md)，步骤见 [执行计划](../exec-plans/active/phase1_execution_plan.md)，选择依据见 [ADR-0001](../decisions/adr-0001-single-obb-baseline.md)。
