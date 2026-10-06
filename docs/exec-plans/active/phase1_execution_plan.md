# Phase 1 — Data Foundation + Experiment Protocol Freeze

状态：**进行中；Experiment Protocol 未正式 FROZEN、Dataset V1 Gate CLOSED、正式训练未开始。**当前项目权威：[V4.1 方案](../../new-plans/面向柔性流水线的低成本工业零件姿态检测系统_项目方案_V4.1_优化版.md)与 [EXPERIMENT_PROTOCOL_V1](../../new-plans/EXPERIMENT_PROTOCOL_V1.md)（其正式修正规则优先）；协议中的 TBD 和 gate checklist 尚未满足。Phase 1 验收必须是 **Experiment Protocol FROZEN + Dataset V1 Gate OPEN**，不能以数据组装/脚本测试通过替代。

最近更新：2026-10-06。用户已选择沿项目 **Part A → Part B** 路线推进，当前执行 Phase 1 准备，不再以四类联合开发训练作为主线。新增 80 张 Part A 照片的隔离开发微调已完成，范围见[80 张真实照片开发微调](real_cap_80_development.md)；该开发结果不替代正式 Phase 2 对照。

## 已有能力与缺口

已实现单帧 OBB 脚手架、part-aware 原始采集、独立 OBB sidecar/store/四点击 OpenCV 标注（可选候选预填）与只读 QA、QA-clean sidecar 的 schema v2 显式完整 session `train/val/test/final_holdout` 四路组装工具和审计脚本。Final Holdout 会实体输出并记录在 manifest 中，但不进入 `data.yaml` 的训练/调参 split。另有有限 MVTec Screws 公开参考数据审计和已完成的[独立合成训练演示](synthetic_cap_training_demo.md)；MVTec 无项目正反面标签，合成结果不代表正式项目性能。README 已记录手机/iVCam 真实采集烟测；本批 Part A 8 张静态照片已完成人工 GUI 接受与标注 QA。尚无通过门禁的真实 Part A 数据集、独立四路 split plan、正式训练、正式模型指标或 Phase 2–6 运行时证据；当前仍不能签署 Dataset V1 Gate。具体软件操作见 [采集](../../data/PHONE_CAPTURE_V1.md)、[标注](../../data/obb_annotation_v1.md) 与 [组装](../../data/project_dataset_v1.md)。

## 本轮实拍进度与直接下一步

### 2026-10-06：80 张 Part A 开发微调

用户明确本轮实拍预算为黄色瓶盖 Front 40 / Back 40，共 80 张；此前 200 张是采集建议，不是协议门槛，本轮不追加该数量要求。新图已保真归档于 [20261006 intake manifest](../../../data/real-cap-intake/20261006-v1/intake-manifest.json)，80 / 80 可解码、4096×3072、无 EXIF Orientation、精确字节重复 0。EXIF 时间为 02:37:01–02:42:52，跨度 351 秒，时区未声明，保守归为一个新采集批次。

新 80 张按完整批次用作 Train，旧 2026-10-04 人工接受的 Part A 8 张用作开发 Val / Calibration，见[开发 manifest](../../../data/real-cap-development/20261006-v1/manifest.json)。源图哈希、来源家族和采集组不跨两路；这是开发数据隔离，不是正式四路 Dataset V1。新图 OBB 经助手逐图复核，`human_accepted=false`，不得记作操作者 GUI 接受或正式 GT；近圆 Part A 的角度为 N/A。边缘轻微截断、强反光与偏软样本均保留质量备注。

`runs/real-cap/20261006-finetune-v1` 已从此前合成 YOLO26n-OBB `best.pt` 完成真实照片开发微调：实际 13 epochs，patience 12 早停，best 为 epoch 1，训练 118.882 秒，整体 123.203 秒；batch 8、imgsz 640、seed 42、GPU 0，增强仅用于 Train。旧 8 张开发 Val 的 P=0.954002、R=1、mAP50=0.995、mAP50–95=0.87025，详见 [result summary](../../../runs/real-cap/20261006-finetune-v1/result-summary.json)。本轮不跑 YOLO26s-OBB / Traditional CV 公平对照，这些数值不是正式性能指标。协议未 FROZEN，Dataset V1 / Training Gate CLOSED，Test / Final Holdout、事件 GT 与物理动态性能证据仍缺。具体复现命令和产物见[开发记录](real_cap_80_development.md)。

用户授权的临时 `1/` 目录自动清理当时被审批拦截，返回 `blocked by policy`，没有详细原因，命令未执行、已删除 0 个文件，见历史[清理记录](../../../data/real-cap-intake/20261006-v1/temporary-source-cleanup.json)。用户现已手动移除该目录，实测当前不存在；80 份归档原图、80 份采集副本 SHA-256 及 last 权重哈希已再次核对一致，归档完整保留，见新增[清理确认](../../../data/real-cap-intake/20261006-v1/temporary-source-cleanup-confirmation.json)。

开发推理设置另存 [operating-selection.json](../../../runs/real-cap/20261006-finetune-v1/operating-selection.json)：使用 epoch 13 的 last 权重与瓶盖 OBB 内黄色 HSV 像素占比≥0.1 的过滤，confidence 0.25、NMS IoU 0.7、imgsz 640；best 的原始 AP 选择记录保持不变。仅用旧 8 张开发 Val 比较组合，过滤后两权重均为 8 / 8 单个正确框、0 额外，按显式 `last,best` 并列顺序选择 last。组合回放为已见 Train 80 / 80、旧 Val 8 / 8 单个正确框、0 额外；旧 Val 有 3 个背景框被颜色过滤，见[组合核验](../../../runs/real-cap/20261006-finetune-v1/operating-point-verification.json)。last 原始 AP mAP50=0.97、mAP50–95=0.85029；不得把 best 的 0.87025 归给 last，或把组合计数称为纯 YOLO 性能 / `UNCERTAIN` / 风险门控。

静态照片预测回放 [photo-replay-h264.mp4](../../../runs/real-cap/20261006-finetune-v1/photo-replay-h264.mp4) 已完成：22 秒、640×480、20 FPS、440 / 440 帧解码通过，含 80 张已见 Train 和 8 张开发 Val，不是物理运动录像；视频与预览来自初版批量报告，来源哈希见[回放核验](../../../runs/real-cap/20261006-finetune-v1/photo-replay-verification.json)。推理入口代码 / 安全审查通过；实际 CLI 的 1280 导出 Train 80、Val 8 与原始 4096×3072 Train 80 均 exit 0，逐图按开发标注核对面类与 polygon IoU≥0.5 后全部单个正确框、0 额外保留框，见[交付核验](../../../runs/real-cap/20261006-finetune-v1/delivery-verification.json)与本轮最新[准备状态](../../../data/phase1-real-caps/20261006-v1/phase1-readiness.json)。实际 CLI Val 过滤 2 个原始框，初版批量报告过滤 3 个；两者分别保留，不声称原始检测 / 四角逐位相同。临时 `1/` 现已由用户手动移除，当前清理确认另存。

### 2026-10-05：原 14 张素材与人工接受记录

实物判据和缺口见 [Part A / B 实拍预冻结草案](../../data/part_ab_photo_prefreeze_v1.md)。原图归档于 `data/real-cap-intake/20261005-v1`：Part A `cap_a` 8 张（Front 4 / Back 4），Part B 候选 `cap_b` 6 张（Front 3 / Back 3）。两者有可见的外观、纹理和面结构差异；尺寸与工况实测值尚缺。EXIF 时间为 13:14:21–13:15:38，时区未声明，同一背景下保守归为一个物理采集批次，不能按两种零件或四种可见面制造独立 Session。

| 工作项 | 当前证据 / 状态 |
| --- | --- |
| 原图来源、面类分组 | 14 张原图已归档并保留来源；Part A 8 张面类已人工接受，Part B 6 张仍为助手可视初判 |
| Part A / B 预冻结草案 | 已记录照片支持的判据；仍为 `PREFREEZE DRAFT`，未替代核心协议 TBD |
| 兼容采集契约的照片导入 | 已完成；Part A 开发目录 8 张、Part B 候选参考目录 6 张，共用同一 `capture_group_id`；14 张原图 SHA-256 保真，见 [import-report.json](../../../data/phase1-real-caps/20261005-v1/import-report.json) |
| OBB 候选与叠框预览 | 已完成；Part A `cap_a_v2` 8 个、Part B 6 个候选，各 0 reject / 0 error；均为候选状态，不是接受标注 |
| Part A 人工 GUI 复核、接受标注、只读 QA | 已完成：expected 8、annotated 8、excluded 0、missing 0、errors 0，Front 4 / Back 4；见 [标注 QA](../../../data/phase1-real-caps/20261005-v1/part-a-annotation-qa.json) 和 [操作者接受记录](../../../data/phase1-real-caps/20261005-v1/part-a-operator-acceptance.json) |
| Part B 人工标注 / 完整 Dataset Audit | Part B 6 张仅候选参考，未人工标注；完整 Dataset Audit 尚未完成 |
| 独立真实 Train / Val / Test / Final Holdout | 未建立；独立真实 Val / Test / Final Holdout Session 均为 0，见 [phase1-readiness.json](../../../data/phase1-real-caps/20261005-v1/phase1-readiness.json) |

本轮来源哈希、采集契约和候选凸几何检查通过，4 个新增 Python 文件通过编译检查；采集 / 标注 / 数据组装 / 审计相关现有 focused tests **96 passed in 25.96s**，代码与安全审查通过。候选辅助 GUI 已通过静态审查、8 个候选的纯读加载 / 显示几何检查与 float32 边界舍入烟测；该改动对应的既有 CLI / UI / store / contract focused tests **30 passed in 3.57s**，两轮范围有重叠，分别记录而不累加。仅在 `.venv` 安装 GUI 版 `opencv-python==4.11.0.86`，已确认 `WIN32UI`；NumPy 仍为 `1.26.4`，PyTorch 未改动。没有打开摄像头；已完成的是本批静态照片的人工 GUI 接受与标注 QA，不扩展为相机 / 动态工作站验收，不触发训练。

操作者于 2026-10-05 通过用户消息“确认完了”声明完成，实际 workspace 与只读 QA 已核对。QA 的 3 个近方形警告对应 Front `IMG_20261004_131515.jpg`、Front `IMG_20261004_131533.jpg`、Back `IMG_20261004_131503.jpg`；`angle_evaluable=5` / `angle_unevaluable=3` 仅是 `AXIAL_OBB` 几何判定。Part A 为 `NOT_APPLICABLE`，Angle=N/A，面类 / OBB 标注 QA 可通过，但不产生有效方向 GT 或角度指标。

本批 8 张无需重复确认。如需复查已有记录，可用显式 `--revisit`。可选 `--proposals` 会先校验完整候选集合、session / manifest 与原图 SHA-256、JPEG 尺寸、身份及几何，再预填四角；候选模式按 JPEG 编码像素忽略 EXIF 旋转显示，以匹配候选坐标。操作者确认后按 **Enter/Space** 保存；**R** 重画、**X** 排除、**Q/Esc** 退出。未传该参数时保留四点击流程。以下命令用于复查与重新执行只读 QA，详见[标注说明](../../data/obb_annotation_v1.md#optional-candidate-prefill---proposals)：

```powershell
.\.venv\Scripts\python.exe scripts/annotate_obb.py data/phase1-real-caps/20261005-v1/captures/cap_a_20261004_development01 --workspace data/annotations/cap_a_20261004_development01 --proposals data/phase1-real-caps/20261005-v1/obb-proposals/cap_a_v2 --max-window 850 --revisit
.\.venv\Scripts\python.exe scripts/qa_annotations.py data/annotations/cap_a_20261004_development01
```

当前已使用提供的 80 张完成微调与开发 Val 核验。正式门禁仍缺独立 Part A 的 Val / Calibration、Test（跨多个 Session）和 Final Holdout，以及实物尺寸、相机距离、照明与工况、Operating Envelope draft、事件 GT schema / 条件和 Recipe 合格规则的冻结记录。Part B 当前仅用于 Phase 1 候选预冻结 / reference_only，方向语义仍 TBD；Phase 5 才正式采集、微调并记录换型成本与 Part A 回归。每个 Part 使用自己的 Front / Back 两类模型映射；不建立四类联合模型。照片派生素材可辅助 Train，派生视频只作演示，不能补足真实 Test / Final Holdout 或 Phase 3 / 4 的物理运动事件证据。

## 工作顺序与冻结清单

1. **实物与工况预冻结（照片草案已建立，仍待实测与确认）**：依据本轮实拍记录 Part A、Part B 候选实物及差异/选择理由（须在 Part A 正式训练前冻结 Part B 候选，Phase 5 才正式采集 Part B）、Front/Back 客观判据、Orientation/对称语义。Part A 沿用 `NOT_APPLICABLE` 草案；Part B 方向要求、来源和 GT 待确认，不凭圆形轮廓或径向筋指定语义。填写相机、ROI/像素范围、运动、间距、照明、背景、模糊/遮挡的 Operating Envelope draft。核心协议字段仍 **TBD**，未知数值保留 TBD；不把照片草案宣称为已冻结。至少一个正式 Part A/B 对象具备适用姿态语义及有效正式 GT / 测试证据后，才可声明二维姿态性能；两者均不适用则该声明范围外。
2. **冻结实验定义**：建立单件事件 GT schema `CONFORMING / NONCONFORMING / INDETERMINATE`、真值来源与双人/二次复核及版本变更记录；`UNCERTAIN` 仅为系统输出/拒绝，非 GT/训练类。Wrong Accept 严格为 `NONCONFORMING → PASS`；争议样本 INDETERMINATE 单列不进确定真值的风险指标分母。建立 `nominal`、`low_light`、`reflection`、`background_shift`、`motion_blur`、`partial_occlusion` 的 Test Condition Matrix，每事件绑定 condition/session ID。
3. **冻结比较与报告口径**：Traditional CV/YOLO 使用同样输入、分辨率、ROI、Test Events、GT、Operating Envelope 和计时起止；双方在 Train/Val/Calibration 调参并在最终 Test 前冻结。风险工作点在 Val/Calibration 扫描 Pareto 权衡 Wrong Accept、Wrong Reject、Coverage、端到端 P95、Validated Event Throughput；Test/Final Holdout 不调参。核心比例 `x/n`、n、点估计、Wilson 95% CI（0 次错误亦报上界），延迟 n/Median/P95；同时分 Overall/Condition/独立 Session 展示失败与波动。Validated Event Throughput 按有效完成物理单件事件/有效运行分钟计，不能以 FPS 充当；跨 Session 重复性不得把同一录像切片冒充独立会话。
4. **真实 Part A 数据与四路划分**：硬件预览/采集，按 Front/Back、角度/位置及测试工况覆盖收集独立会话；人工可视检查每个接受图像的 OBB/面类或明确排除；逐 sidecar 跑 `scripts/qa_annotations.py`、核对来源/几何/覆盖。按完整 session 建立 train、val/calibration、test、final_holdout（禁随机逐帧分）；当前 schema v2 组装工具已支持四路实体输出，但真实四路 Part A 数据尚未建立。运行 `scripts/audit_dataset.py <output> --classes 2`、核查类别/角度分布、近重复与泄漏、provenance、Test Condition Matrix 和独立 session 覆盖；不过则补拍/重标而非训练。Final Holdout 留作 Hard Sample 回流/二次训练后未见数据验证；Test 错误一旦回流不可继续把原 Test 当新模型的最终无偏测试。
5. **独立复核与 gate**：按协议第 17 节逐项核实 Part A/B、Operating Envelope draft、GT schema、条件矩阵、四路 session split 与 Final Holdout 策略、标注/QA/audit/分布/溯源/泄漏、公平与统计/吞吐/跨 Session 规则和 Final Test 调参禁令。任一项失败则 **Dataset V1 Gate CLOSED** 且协议不正式 FROZEN；全部满足才记录 Experiment Protocol FROZEN + Dataset V1 Gate OPEN 并另行进入 Phase 2，不预填指标。

## 后续阶段（不属于当前实现）

Phase 2 — Perception Baseline：Traditional CV vs YOLO26n-OBB / YOLO26s-OBB；Phase 3 — Reliable Event Decision：Single Frame vs Tracking/Multi-frame/Decision Zone；Phase 4 — Risk-Aware Decision：UNCERTAIN 与 Risk-Coverage-Latency/Throughput；Phase 5 — Flexible Changeover：Part A → Part B，Recipe、成本及 Part A 回归；Phase 6 — Production-Oriented Finalization：反馈、轻量环境监测、事件追溯与最终报告。以上尚无运行时/实测证据。Seg+OBB 非默认基线，仅在证据显示瓶颈时另行考虑。相关边界见 [范围](../../product-specs/project_scope.md)、[设计](../../design-docs/phase1_obb_baseline.md) 和 [ADR-0001](../../decisions/adr-0001-single-obb-baseline.md)。
