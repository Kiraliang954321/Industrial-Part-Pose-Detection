# Part A / B 实拍预冻结草案 V1

最近更新：2026-10-06。状态：**PREFREEZE DRAFT — 尚未正式 FROZEN；Dataset V1 / Training Gate CLOSED。**新增 80 张 Part A 真实照片开发微调已完成，独立记录于[开发微调说明](../exec-plans/active/real_cap_80_development.md)，不改变本草案的正式状态。

用户已选择按 [V4.1 项目方案](../new-plans/面向柔性流水线的低成本工业零件姿态检测系统_项目方案_V4.1_优化版.md) 的 Part A → Part B 路线推进。本记录依据原 14 张实拍照片及新增 80 张 Part A 照片整理 Phase 1 对象候选和开发进度，不改写 [EXPERIMENT_PROTOCOL_V1](../new-plans/EXPERIMENT_PROTOCOL_V1.md) 第 2、3、17 节的 TBD 字段或门禁勾选。路线已获选择，当前执行 Phase 1 准备；正式训练仍需全部门禁通过。

## 1. 实物与可见面候选判据

| 字段 | Part A | Part B 候选 |
| --- | --- | --- |
| `part_id` | `cap_a` | `cap_b` |
| 实物外观 | 黄色塑料瓶盖，外表面红色印字 | 灰白塑料盖，外表面浮雕文字 |
| Front 判据 | 闭合外表面及红色印字可见 | 闭合外表面及浮雕文字可见 |
| Back 判据 | 内凹开口、内部环形结构可见；部分照片有印码 | 内凹开口、中央套筒及径向加强筋可见 |
| 已接收实拍数量 | 原 8 张 Front 4 / Back 4；新增 80 张 Front 40 / Back 40，两批次分开记录 | 原 6 张：Front 3 / Back 3 |
| 物理尺寸（mm） | TBD | TBD |
| 当前用途 | 初始感知 / 事件决策对象的 Phase 1 数据准备 | Phase 1 候选实物与差异记录；Phase 5 正式换型对象仍待预冻结 |

两者在外形结构、表面纹理和正反面结构上均存在实拍可见差异：Part A 为较浅的带外缘纹理瓶盖，Part B 有较高侧壁、浮雕字和内部套筒 / 加强筋。照片未提供可靠物理尺度，不能据画面大小推断尺寸差异；Part B 是否处于正式 Operating Envelope 内仍待实测。对象选择理由是这些可操作的外观差异，不依据模型分数挑选；正式预冻结后不得为提高成绩替换 Part B。

面类来自可见结构与人工判据，不能从 OBB 长轴或角度推断。Part A 的 8 张照片已完成操作者 GUI 面类与 OBB 接受复核；Part B 的 6 张仍为助手可视初判和候选参考，尚未人工标注。Front / Back 也不等于 PASS / FAIL；Recipe 的允许面及合规规则仍 TBD。

## 2. 姿态适用性与声明范围

- **Part A：**延续当前 `orientation_semantics = NOT_APPLICABLE` 草案，Angle=N/A/null；不把圆形 OBB 长轴作为印字方向。正式确认后，角度来源、参考、特征、GT 方法、对称阶数及长轴阈值按协议记 null。N/A 本身不触发 `FAIL_ORIENTATION` 或 `UNCERTAIN`。
- **Part B：**`orientation_semantics = TBD`。照片可以辨认文字与径向结构，但尚无工业方向要求、稳定特征来源、零度轴、旋转正方向、坐标系、周期或可复现 GT。不能凭圆形轮廓断言方向不适用，也不能凭径向筋数量断言具有某种对称角度语义。
- **二维姿态性能：**至少一个正式 Part A/B 对象采用适用姿态语义并取得有效 GT 和正式测试证据后才可声明已验证。若最终两者均为 `NOT_APPLICABLE`，检测、正反面和换型仍可验证，二维方向性能声明保持范围外，不产生角度指标。

## 3. 来源与物理批次

### 2026-10-06 新增 Part A 80 张

用户本轮提供黄色瓶盖 Front 40 / Back 40 共 80 张，并限定本轮实拍预算为 80 张。原图保真归档与来源信息见 [20261006 intake manifest](../../data/real-cap-intake/20261006-v1/intake-manifest.json)：80 / 80 可解码、4096×3072、无 EXIF Orientation、精确字节重复 0。EXIF 时间为 02:37:01–02:42:52，跨度 351 秒，时区未声明；保守记为一个新批次 `observed-20261006-023701-024252`，不按 Front / Back 拆批。

导入目录为 `data/phase1-real-caps/20261006-v1/captures/cap_a_20261006_development01`。新 80 张按完整批次作 Train，旧人工接受的 Part A 8 张作开发 Val；两路来源家族、原图哈希与采集组隔离，见[开发 manifest](../../data/real-cap-development/20261006-v1/manifest.json)。旧 Val 已被开发过程查看，不能当作未见正式 Test。新图经助手逐图可视复核的开发标注不是操作者接受记录；`human_accepted=false`，正式 GT 仍未建立。

原始候选报告保留 78 个候选与 2 个角点越界错误，不回写成通过。另行生成的[开发 OBB 标注](../../data/phase1-real-caps/20261006-v1/development-labels/cap_a/development-obb-labels.json)以黄色 HSV 定位并围住可见轮廓，用合法的轴对齐 OBB 表示近圆零件；几何角度 0 不代表工业朝向。`IMG_20261006_024040.jpg` 有轻微边缘截断，`IMG_20261006_023757.jpg` 有强反光，`IMG_20261006_024131.jpg` / `IMG_20261006_024248.jpg` 偏软，均保留备注。Part A Angle=N/A，Part B 6 张参考及 TBD 语义不变。

200 张为此前采集建议，不是项目门禁。当前已使用提供的 80 张完成开发微调；AP 最佳 best 权重的旧 8 张开发 Val mAP50–95 为 0.87025，开发推理另使用 last + 黄色过滤组合，指标与组合回放计数分开，详见[开发记录](../exec-plans/active/real_cap_80_development.md)及[本轮准备状态](../../data/phase1-real-caps/20261006-v1/phase1-readiness.json)。增强只作用于 Train，不增加独立真实批次、正式 Test 或 Final Holdout。当时临时 `1/` 自动清理被审批拦截、未执行，见历史[清理记录](../../data/real-cap-intake/20261006-v1/temporary-source-cleanup.json)；用户现已手动移除，实测目录不存在，归档完整保留，见新增[清理确认](../../data/real-cap-intake/20261006-v1/temporary-source-cleanup-confirmation.json)。

### 2026-10-04 原 14 张及其接受记录

原始照片副本、SHA-256、面类初判与 EXIF 记录见 [intake-manifest.json](../../data/real-cap-intake/20261005-v1/intake-manifest.json)。原文件保留于用户提供的 `E:/Download/vivo办公套件/pictureRelay/`，项目归档于 `data/real-cap-intake/20261005-v1/originals/`，未修改原图。

本轮照片分辨率为 3072×4096 或 4096×3072。EXIF 时间为 **2026-10-04 13:14:21–13:15:38，跨度 77 秒，时区未声明**；背景相同，保守视为一个物理采集批次 `observed-20261004-131421-131538`。EXIF 时间不冒充已确认的本地时间。按 part 分目录是采集契约要求，不产生独立物理 Session；两个 part 的导入目录必须保留同一 `capture_group_id` 以防误拆。

这批照片保留为开发来源。没有独立真实 Test / Final Holdout；同一批次不能按颜色、面类、文件先后或合成扩充量拆成四路独立数据。原图及其增强、AI 派生素材、视频须共享来源家族和批次归属。

部分照片有可见强反光、失焦、斜视或细节偏软，详见 [intake-review.md](../../data/real-cap-intake/20261005-v1/intake-review.md)。该文件记录的是路线选择前的历史接收方案，后续以本草案和 [Phase 1 执行计划](../exec-plans/active/phase1_execution_plan.md) 为准。视觉质量备注不等于冻结的条件等级、曝光值或模糊门槛。

## 4. Operating Envelope 待测表

| 项目 | 当前证据 | 正式 draft 尚需记录 |
| --- | --- | --- |
| 相机与静态图像 | 用户提供手机照片及上述像素分辨率 | 设备、镜头、曝光策略、相机距离 mm、正式采集分辨率与 FPS：TBD |
| ROI / 目标尺寸 | Part A 开发批次 8 张完整瓶盖边界已人工接受；Part B 仍为候选 | 正式 ROI、允许目标像素尺寸范围：TBD |
| 运动 / 间距 | 本轮为静态照片，无物理运动视频 | 模式、速度范围、最小间距、重叠与遮挡规则：TBD |
| 光照 | 部分照片有反光 / 亮度变化 | nominal / low_light / reflection 条件定义、实测照度 Lux 与允许范围：TBD |
| 背景 | 同一暗色工作台背景，包含刻度及设备边缘 | nominal、allowed_shift 的可复现定义：TBD |
| 干扰 | 有失焦、反光、斜视样本 | motion_blur / partial_occlusion 等条件等级与范围：TBD |

照片分辨率不能替代正式相机设置；静态失焦不能当作已验证运动模糊。生成的视频属于派生模拟素材，可用于演示与软件检查，不能填写物理速度、事件 GT 或动态性能验收。

## 5. 本轮软件准备与审核状态

导入工具 [import_photo_intake.py](../../scripts/import_photo_intake.py) 已将 `intake-manifest.json` 及原图导入到与 [LABEL_CONTRACT_V2](LABEL_CONTRACT_V2.md) 兼容的单 part 原始采集目录，并保留两 part 共享的 `capture_group_id`、来源哈希和初判状态。实跑根目录为 `data/phase1-real-caps/20261005-v1`，详见 [import-report.json](../../data/phase1-real-caps/20261005-v1/import-report.json)：

| Part | 已导入采集目录 | 实跑数量 / 角色 |
| --- | --- | --- |
| A | `captures/cap_a_20261004_development01` | 8 张，4 Front / 4 Back；`part_a_development` |
| B 候选 | `captures/cap_b_20261004_reference01` | 6 张，3 Front / 3 Back；`part_b_candidate_reference_only` |

全部 14 张图片及 EXIF 字节保留，源 / 副本 SHA-256 一致。两个目录共享一个物理采集组，未形成正式四路划分；导入不写 Ground Truth 或接受 OBB 标注。

[propose_photo_obb.py](../../scripts/propose_photo_obb.py) 已生成候选四角与叠框预览，供人工检查；不写入接受标注、不生成训练 GT、不自动打开门禁。实跑结果如下：

| Part | 候选输出与预览 | 数量 / 状态 |
| --- | --- | --- |
| A | [cap_a_v2 contact-sheet](../../data/phase1-real-caps/20261005-v1/obb-proposals/cap_a_v2/contact-sheet.jpg) · [proposal-report](../../data/phase1-real-caps/20261005-v1/obb-proposals/cap_a_v2/proposal-report.json) | 8 candidate，0 reject / 0 error |
| B 候选 | [cap_b contact-sheet](../../data/phase1-real-caps/20261005-v1/obb-proposals/cap_b/contact-sheet.jpg) · [proposal-report](../../data/phase1-real-caps/20261005-v1/obb-proposals/cap_b/proposal-report.json) | 6 candidate，0 reject / 0 error |

Part A 初版在斜拍 `IMG_20261004_131533.jpg` 上遗漏部分暗色侧壁，`cap_a_v2` 通过新增 `--foreground-color yellow`（灰度 Otsu 与 HSV 颜色掩码并集）改善了该候选的完整瓶盖覆盖；这次叠框检查不是操作者 GUI 接受验收。两份报告均为 `state=candidate`、`operator_review_complete=false`、`formal_gt=false`，候选报告不记录后续人工接受数量。人工接受标注使用 [OBB sidecar / GUI 工作流](obb_annotation_v1.md#optional-candidate-prefill---proposals)：现已支持可选 `--proposals` 预填，仍须操作者逐张确认后保存；接受 / 排除 / 缺失数量以 workspace 记录和只读 QA 报告为准。OBB sidecar 本身也不是事件级 GT。

工具入口已经可用。下面从项目根目录使用 `.venv`，导入与候选工具均拒绝覆盖已有输出；若需重跑，选择新的输出目录：

```powershell
.\.venv\Scripts\python.exe scripts/import_photo_intake.py --help
.\.venv\Scripts\python.exe scripts/propose_photo_obb.py --help
.\.venv\Scripts\python.exe scripts/propose_photo_obb.py --capture data/phase1-real-caps/20261005-v1/captures/cap_a_20261004_development01 --rois data/real-cap-intake/20261005-v1/annotation-rois-cap_a-v1.json --output data/phase1-real-caps/20261005-v1/obb-proposals/cap_a_new --foreground-color yellow
```

软件检查已通过：14 张原图哈希保真、采集契约与候选凸几何核验；4 个新增 Python 文件编译；相关既有采集 / 标注 / 数据组装 / 审计 focused tests **96 passed in 25.96s**；代码与安全审查通过。新增候选辅助复核通过静态审查，8 个 Part A 候选通过纯读加载 / 显示几何核验及 float32 图像边界舍入烟测；相关既有 CLI / UI / store / contract focused tests **30 passed in 3.57s**。这两轮测试是各自的验证记录，存在范围重叠，不累加为新总数；软件验证与下述人工标注 / QA 分别记录，均不表示完整 Dataset Audit 已通过。

GUI 环境仅在 `.venv` 安装 `opencv-python==4.11.0.86`，实际 `cv2` 构建包含 `WIN32UI`；NumPy 保持 `1.26.4`，PyTorch 未改动。未打开摄像头；本轮 Part A 静态照片的 GUI 逐图复核已完成，记录与只读 QA 核验见下文，不扩展为相机 / 动态工作站验收。门禁和本轮环境记录见 [phase1-readiness.json](../../data/phase1-real-caps/20261005-v1/phase1-readiness.json)，其中独立真实 Val / Test / Final Holdout Session 均为 0，协议未冻结、正式数据集未组装、正式训练未启动。

### Part A 人工接受与标注 QA 结果

2026-10-05，操作者通过用户消息“确认完了”声明复核完成；随后核对 `data/annotations/cap_a_20261004_development01`：**expected 8、annotated 8、excluded 0、missing 0、errors 0**，Front 4 / Back 4。只读结果见 [part-a-annotation-qa.json](../../data/phase1-real-caps/20261005-v1/part-a-annotation-qa.json)，操作者声明与实际标注记录、来源 / metadata SHA-256 快照见 [part-a-operator-acceptance.json](../../data/phase1-real-caps/20261005-v1/part-a-operator-acceptance.json)。候选报告保持原始 candidate 状态，完成数量以接受标注及这两份核验记录为准。

QA 另报 3 个近方形 OBB 警告：Front `IMG_20261004_131515.jpg`、Front `IMG_20261004_131533.jpg`、Back `IMG_20261004_131503.jpg`。报告中的 `angle_evaluable=5` / `angle_unevaluable=3` 是 `AXIAL_OBB` 的几何判定；Part A 使用 `NOT_APPLICABLE`，Angle=N/A，因此不影响本批面类 / OBB 标注 QA，也不能解释为 5 张有效方向 GT。Part B 6 张仍仅为候选参考，未完成人工标注；完整 Dataset Audit、独立真实四路数据和正式门禁均未完成。

### 可选候选辅助复核

`--proposals` 只读加载完整候选集合，校验 session / manifest SHA-256、原图 provenance / SHA-256、JPEG 尺寸、part / face / class 与四角几何；不完整或过期的候选会在打开 workspace 前拒绝。复核过程继续核对冻结的来源绑定，保存前检查原图未变化。GUI 在候选模式按 JPEG 编码像素解码并忽略 EXIF 旋转，以保持显示与候选坐标一致；原始 JPEG / EXIF 不改写。来源快照校验不是正式协议冻结，也不会启动训练。

窗口预填四角并标记 `CANDIDATE`。操作者核对面类与完整轮廓后按 **Enter/Space** 接受，无需为 8 张图重新点击 32 个角；不合适时按 **R** 清空并重画四角、**X** 排除、**S** 暂跳或 **Q/Esc** 退出。候选载入本身不写接受标注。不加 `--proposals` 时保留默认四点击流程；已有复核记录默认跳过，只有显式 `--revisit` 才再次打开。Part B 仍为 `part_b_candidate_reference_only`，其方向语义保持 TBD。

Phase 2 先以 Part A 比较 Traditional CV、YOLO26n-OBB 和 YOLO26s-OBB；未来该 part 的两个 YOLO 类别为 `0: cap_a_front`、`1: cap_a_back`。Phase 5 Part B 使用自己的 `0: cap_b_front`、`1: cap_b_back`，记录采集 / 标注 / 训练 / Recipe / 核心代码成本并复核 Part A 回归。这是后续映射计划，采集阶段不分配 numeric class ID；当前不建立四类联合模型，也不把已接收的 Part B 候选照片当作 Phase 5 正式采集或成本成果。

## 6. 正式门禁缺口与下一步

| 门禁项 | 当前状态 / 下一步 |
| --- | --- |
| Part A / B 与 Front / Back 正式冻结 | 照片支持候选判据，待人工确认与冻结记录 |
| Part B 姿态要求与 A/B 适用性范围 | Part A N/A 草案；Part B TBD；适用时补来源、参考和 GT 方法 |
| Operating Envelope draft | 实测字段及可复现条件定义 TBD |
| 事件 GT schema、允许面与条件矩阵 | `CONFORMING / NONCONFORMING / INDETERMINATE`、真值来源、复核和版本规则待冻结；`UNCERTAIN` 不作为 GT / 训练类 |
| 人工标注 / GUI 可视复核 / QA / Dataset Audit | Part A 本批 8 张人工接受与标注 QA 已完成；Part B 尚未人工标注，完整 Dataset Audit 未完成 |
| 独立真实四路划分与 Final Holdout | 待补独立 Part A 会话；照片增强不能补成真实 Test / Final Holdout |
| 分布、provenance、近重复与泄漏审核 | 待正式数据建立后检查，按物理 Session 保持来源隔离 |
| 公平性、统计、吞吐、跨 Session 和禁调参规则 | 待依据协议完成冻结；共享输入 / ROI / 事件 / GT / 计时，报告 x/n 与 Wilson 95% CI、端到端 P95 和有效事件吞吐 |

正式门禁尚需以下材料；当前已完成授权的[80 张开发微调](../exec-plans/active/real_cap_80_development.md)，不把以下缺口改写为本轮追加拍照要求：

1. 补采独立真实 Part A 会话，建立真实 Val / Calibration、跨多个 Session 的 Test 和 Final Holdout。当前批次保留为开发来源；重新开始、重新摆放或不同采集时段并留下条件记录，才有资格审查独立性。不得把同一批次切片或合成帧充当独立 Session。
2. 实测并填写物理尺寸 mm、相机距离 / 正式相机配置、照明与工况，以及 Recipe 的合格规则和事件 GT schema；冻结 Operating Envelope draft、GT / 条件与合格规则。确认 Part B 是否要求以正面文字等特征评价方向或设为 N/A，再依选择补角度参考与 GT 方法。Part B 语义当前保持 TBD。

本批 Part A 无需重复逐图确认。若需复查已接受记录，保留以下复现命令，使用显式 `--revisit`；只读 QA 可单独重跑：

```powershell
.\.venv\Scripts\python.exe scripts/annotate_obb.py data/phase1-real-caps/20261005-v1/captures/cap_a_20261004_development01 --workspace data/annotations/cap_a_20261004_development01 --proposals data/phase1-real-caps/20261005-v1/obb-proposals/cap_a_v2 --max-window 850 --revisit
.\.venv\Scripts\python.exe scripts/qa_annotations.py data/annotations/cap_a_20261004_development01
```

随后按完整物理 Session 组装四路数据，完成真实 QA / Audit、分布与来源 / 泄漏审查和协议第 17 节门禁复核。只在 Train / Val / Calibration 调参；Test / Final Holdout 不调参，Test 错误回流后须使用未见 Final Holdout 或新 Session 验证。所有门禁通过后才正式冻结并进入 Phase 2。

Phase 3 / 4 的跟踪、单件事件、风险、端到端延迟和 Validated Event Throughput 验收仍需真实物理运动与事件证据，不能仅靠照片或照片驱动视频完成。当前不预填正式训练结果、模型指标或换型成本。
