# ADR-0001：官方 YOLO26n-OBB 作为单模型基线

状态：已选择技术起点；不是性能结论，也不表示 Dataset V1 Gate 已打开。

## 背景

当前权威是 [V4.1 项目方案](../new-plans/面向柔性流水线的低成本工业零件姿态检测系统_项目方案_V4.1_优化版.md)与其正式修正 [EXPERIMENT_PROTOCOL_V1](../new-plans/EXPERIMENT_PROTOCOL_V1.md)。V4.1 的 RQ1 要在 Phase 2 — Perception Baseline 公平比较 Traditional CV、YOLO26n-OBB 与 YOLO26s-OBB，判断 AI 是否在冻结工况内有收益；Phase 1 — Data Foundation + Experiment Protocol Freeze 则先完成真实数据和协议门禁。现存 [V2 按零件采集契约](../data/LABEL_CONTRACT_V2.md) 的版本号只标识软件数据格式，不是当前项目方案。

## 决定

以官方 Ultralytics `yolo26n-obb.pt` 作为默认**单模型 OBB** 预训练基线与未来微调起点；YOLO26s-OBB 是 Phase 2 的比较模型，并非替换此默认起点。Seg+OBB 不进入默认主线，仅当单 OBB 基线实验证据显示 Face 等任务有实质瓶颈时另行考虑。Phase 1 不做正式长时训练；只有 Experiment Protocol 正式 FROZEN、Dataset V1 Gate OPEN 后才可进入后续训练/对照。

现有单帧脚手架可检查预训练类别、置信度和旋转框，但预训练类别并非项目的 `<part_id>_front` / `<part_id>_back`。每会话 `part_id` 的 `front` / `back` 是人工可见面标签，不能由 OBB 角度推断。将 OBB 像素长轴方向规范为图像水平轴起算的 `θ ∈ [0°,180°)`；近方形或轴不稳定时不强制评分。数值类 ID 在 [Dataset V1 组装](../data/project_dataset_v1.md)时才分配。

## 取舍与证据边界

单 OBB 简化输入、标注与输出约定，但不证明准确率、实时性或抗反光/遮挡能力。固定 ROI 仅过滤候选观测区域，不是跟踪或事件判定。Phase 3 起拟实现 Tracking → Multi-frame → Decision Zone → 单件事件，Phase 4 起拟增加 Risk Gate；这些均不是现有单帧烟测的成果。正式 GT 是事件级 `CONFORMING` / `NONCONFORMING` / `INDETERMINATE`；系统 `UNCERTAIN` 是拒绝输出，不是 GT 或训练类。Wrong Accept 仅为 `NONCONFORMING → PASS`。OBB 训练标注也不能代替事件 GT。

当前已有 part-aware 采集、标注 sidecar/QA，以及 schema v2 `train/val/test/final_holdout` 四路 session-aware 组装等基础工具；Final Holdout 被单独实体输出并从 `data.yaml` 的训练/调参 split 中隔离。Part A/B 与 Operating Envelope 值仍 TBD，尚无真实项目数据集、真实四路 split、正式训练或模型实测指标；Training Gate / Dataset V1 Gate CLOSED。结论只能基于已冻结 Operating Envelope 与公平对照，同一输入、ROI、事件 GT、计时规则；选模/调参仅在 Train/Val/Calibration，Test/Final Holdout 均不调参。不得把预训练烟测或官方 benchmark 当成本项目成果。

适用范围见 [项目范围](../product-specs/project_scope.md)，接口见 [基线设计](../design-docs/phase1_obb_baseline.md)，实施顺序见 [执行计划](../exec-plans/active/phase1_execution_plan.md)。
