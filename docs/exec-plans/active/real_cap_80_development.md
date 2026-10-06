# Part A — 80 张真实照片开发微调

最近更新：2026-10-06。状态：**数据整理、助手可视复核、开发微调、开发推理配置、实际 CLI 与交付核验、照片回放产物完成；临时 `1/` 已由用户手动移除。**证据范围为 `real_photo_development`，正式 Experiment Protocol 未 FROZEN，Dataset V1 / Training Gate **CLOSED**。

本轮使用用户提供的黄色瓶盖 Front 40 / Back 40，共 80 张，沿已选择的 Part A → Part B 路线完成 Part A 开发微调。80 张为用户授权的本轮实拍预算，先前 200 张建议不是协议门槛。Part B 6 张仍仅作候选参考，方向语义保持 TBD。

## 数据与来源

| 材料 | 数量 / 来源 | 角色与验收范围 |
| --- | --- | --- |
| 新 Part A 实拍 | Front 40 / Back 40，共 80 张；2026-10-06 观察批次 | 完整批次用于 Train；开发标注由助手逐图可视复核，非操作者 GUI 接受 |
| 旧 Part A 实拍 | Front 4 / Back 4，共 8 张；2026-10-04 观察批次 | 人工接受与标注 QA 已完成；本轮仅为开发 Val / Calibration |
| Part B 参考 | Front 3 / Back 3，共 6 张；2026-10-04 原批次 | 本轮不进入训练或验证，保留 Phase 5 换型候选 |

新 80 张原图保真归档于 `data/real-cap-intake/20261006-v1/originals/`，见 [intake manifest](../../../data/real-cap-intake/20261006-v1/intake-manifest.json)。80 / 80 可解码，尺寸均为 4096×3072，无 EXIF Orientation，精确字节重复 0。EXIF 时间为 **2026-10-06 02:37:01–02:42:52，跨度 351 秒，时区未声明**；时间仅为元数据观察，不冒充确认的本地采集时刻。保守记一个新采集组 `observed-20261006-023701-024252`，不按面类或文件顺序拆成独立 Session。

导入结果位于 `data/phase1-real-caps/20261006-v1/captures/cap_a_20261006_development01`，见 [import report](../../../data/phase1-real-caps/20261006-v1/import-report.json)。原始候选输出 `obb-proposals/cap_a_v1` 保留 78 个候选、2 个角点越界错误。候选报告不回写成接受标注，也不作为正式门禁通过的证据。

后续单独生成 [development-obb-labels.json](../../../data/phase1-real-caps/20261006-v1/development-labels/cap_a/development-obb-labels.json)：黄色 HSV 定位可见轮廓，使用合法轴对齐 OBB；助手通过四页 crop review 逐图检查，页图哈希与质量备注见 [assistant-visual-review.json](../../../data/phase1-real-caps/20261006-v1/development-labels/cap_a/assistant-visual-review.json)。每份记录保留原图 SHA-256、面类、采集组、来源家族和 `label_author=assistant_visual_reviewed_proposal` / `human_accepted=false`。这批没有新增操作者 GUI 接受声明，不能并入旧批次的人工验收数字。

Part A `orientation_semantics=NOT_APPLICABLE`，Angle=N/A；OBB 仅作定位和面类训练。轴对齐几何角度 0 不是有效工业朝向 GT。开发数据保留以下质量备注：

- `back/IMG_20261006_024040.jpg`：边缘轻微截断，作为明确记录的开发边界样本保留。
- `front/IMG_20261006_023757.jpg`：强反光。
- `back/IMG_20261006_024131.jpg`、`back/IMG_20261006_024248.jpg`：运动或失焦偏软；尚无正式模糊等级或物理运动证据。

用户授权在整理完成后删除临时 `1/` 来源目录。当时自动删除被审批拦截，仅返回 `blocked by policy`，未提供详细原因，命令未执行、已删除 0 个文件，见历史 [temporary-source-cleanup.json](../../../data/real-cap-intake/20261006-v1/temporary-source-cleanup.json)。用户随后声明“我已经删除了1文件夹”，当前实测目录不存在；80 份归档原图和 80 份采集副本的 SHA-256 已再次核对一致，last 权重哈希亦匹配，见新增 [temporary-source-cleanup-confirmation.json](../../../data/real-cap-intake/20261006-v1/temporary-source-cleanup-confirmation.json)。历史 [delivery-verification.json](../../../runs/real-cap/20261006-finetune-v1/delivery-verification.json)保留当时清理被拦截的快照与哈希绑定，不回写；当前清理确认另存，归档原图与导入目录完整保留。

## 开发划分与增强

[开发 manifest](../../../data/real-cap-development/20261006-v1/manifest.json)只含两个完整批次的 Train / Val：

| Split | 采集组 | Front / Back | 标注来源 |
| --- | --- | --- | --- |
| Train | `observed-20261006-023701-024252` | 40 / 40 | 助手逐图可视复核的开发标注 |
| Val | `observed-20261004-131421-131538` | 4 / 4 | 旧批次人工接受 sidecar，QA 8 / 8、0 errors |

没有跨 Split 的来源家族、原图 SHA-256 或导出图 SHA-256；Train / Val 使用不同采集组。预检会再次核验这些隔离、图片字节、尺寸、OBB 几何与两类映射。旧 Val 已被之前的开发过程查看，只用于开发校准与检查，不能当作无偏正式 Test。这是两路开发 bundle，`formal_dataset_v1=false`，不触发正式四路 Dataset V1 验收。

导出 JPEG 最长边缩至 1280，移除导出副本的 EXIF，原图保留；归一化四角保持不变。训练入口再复制来源绑定的数据与初始权重快照，使用快照 YAML，避免运行中引用漂移。类别固定为 `0: cap_a_front`、`1: cap_a_back`。

随机增强仅用于 Train：平面旋转最多 180°、平移 0.1、scale 0.15、HSV 0.015 / 0.3 / 0.2、mosaic 0.2；mixup / copy_paste 0。Val 不使用随机预测增强。增强扩展外观变化，不增加独立真实图像、采集批次或未见验证证据；Train 与 Val 不混合。

## 训练配置与结果

当前运行目录为 `runs/real-cap/20261006-finetune-v1`，从本地 [合成 YOLO26n-OBB best.pt](../../../runs/synthetic-cap/20261003-training-v2/yolo26n-obb/weights/best.pt) 初始化。原始合成权重保留，类别由 `synthetic_cap_a_front/back` 映射为 `cap_a_front/back`。

| 配置 | 本轮值 |
| --- | --- |
| 模型 | YOLO26n-OBB；仅跑 n 模型 |
| 数据 | Train 80 / development Val 8 |
| epochs / patience | 最多 40 / 12 |
| imgsz / batch | 640 / 8 |
| device / workers / CPU threads | GPU 0 / 0 / 8 |
| seed | 42，deterministic=true |
| 选择规则 | 开发 Val mAP50–95 最大；并列保留命令权重顺序 |
| Test / Final Holdout | 未创建、未评估、不用于选择 |

训练入口 [scripts/train_real_caps.py](../../../scripts/train_real_caps.py) 与实现 [real_photo_training.py](../../../src/yolo_conveyor/real_photo_training.py) 已通过代码 / 安全审查、CLI / AST 检查及实际 manifest 预检；本轮没有新增或修改测试。实际训练完成 13 epochs：设定上限 40、patience 12 早停，best 为 epoch 1；训练耗时 118.882 秒、含校验流程的整体耗时 123.203 秒。训练后的 best 模型已重新用于非增强开发 Val 评估，类别为 `cap_a_front/back`。

开发 Val 共 8 张，每面 4 张；**AP 最佳 best 权重**的 Ultralytics 返回指标如下，P / R 是评估汇总值，不直接等同于固定置信度下的逐图面类正确率：

| 类别 | GT 实例 | Precision | Recall | mAP50 | mAP50–95 |
| --- | --- | --- | --- | --- | --- |
| Overall | 8 | 0.954002 | 1.000000 | 0.995000 | 0.870250 |
| `cap_a_front` | 4 | 0.997010 | 1.000000 | 0.995000 | 0.920000 |
| `cap_a_back` | 4 | 0.910995 | 1.000000 | 0.995000 | 0.820500 |

训练记录见 [training-progress.json](../../../runs/real-cap/20261006-finetune-v1/training-progress.json)，输入、版本、权重哈希和训练设置见 [training-config.json](../../../runs/real-cap/20261006-finetune-v1/training-config.json)，完整指标见 [result-summary.json](../../../runs/real-cap/20261006-finetune-v1/result-summary.json)，选择规则与绑定哈希见 [selection.json](../../../runs/real-cap/20261006-finetune-v1/selection.json)。[best.pt](../../../runs/real-cap/20261006-finetune-v1/model-01/weights/best.pt) 与 [last.pt](../../../runs/real-cap/20261006-finetune-v1/model-01/weights/last.pt) 已保存；best SHA-256 为 `d8635ae41d41eb0ed8ae6693dec370d82a90623bbc570b541fc032f2080f51ad`。开发 Val 用于 best 选择，未评估 Test / Final Holdout。

`data/`、`runs/` 和权重受 Git 忽略；这些链接用于同一工作区复核，不表示实验原图或模型已提交仓库。

## 固定阈值预测复核

训练后对保存的 best 模型进行 `confidence=0.25`、NMS IoU=0.7 的固定设置回放，正确匹配要求面类正确且 polygon IoU≥0.5。记录见 [prediction-verification.json](../../../runs/real-cap/20261006-finetune-v1/prediction-verification.json)，旧 8 张叠框见 [development-val-predictions.jpg](../../../runs/real-cap/20261006-finetune-v1/development-val-predictions.jpg)：

| 输入 / 模型 | 至少有一个面类与定位正确的匹配 | 恰好只有一个正确框 | 未匹配额外框 |
| --- | --- | --- | --- |
| 旧开发 Val / 初始合成模型 | 8 / 8 | 5 / 8 | 3 |
| 已见 Train / 微调 best | 80 / 80 | 73 / 80 | 7 |
| 旧开发 Val / 微调 best | 8 / 8 | 6 / 8 | 2 |

部分图中同一瓶盖同时出现 front / back 报框，另有画面边缘背景误检。因此“都有正确匹配”不能写成“80 / 80 或 8 / 8 无误识别”，AP 指标也不能代替固定阈值下的额外框审核。Train 回放为已见数据，旧 Val 为已使用的开发数据，均不扩展为正式性能。

上述原始复核记录保留，不回写成无额外框。另存的 last 原始 `confidence=0.25` 复核为旧 Val 5 / 8 单个正确框、3 个额外框，见 [last-checkpoint-development-predictions.json](../../../runs/real-cap/20261006-finetune-v1/last-checkpoint-development-predictions.json)。

## 开发推理组合与选择

[operating-selection.json](../../../runs/real-cap/20261006-finetune-v1/operating-selection.json)另行冻结开发推理组合 `YOLO26n_OBB_plus_cap_a_color_filter`，不修改原始 [AP best selection](../../../runs/real-cap/20261006-finetune-v1/selection.json)。本组合使用 **last.pt**，SHA-256 为 `f7349c1e9cf47b65477e4650fa46e8535a43e38ec515e458c28153950a2f033e`；`confidence=0.25`、NMS IoU=0.7、imgsz=640。

颜色规则为：在预测 OBB 内计算 OpenCV HSV 范围 `[12,70,35]` 至 `[40,255,255]` 的黄色像素占比，低于 0.1 的框过滤。该规则显式依赖 Part A 的黄色外观，是当前开发感知组合的一部分；不是已实现的 `UNCERTAIN`、风险门控或事件系统，也不直接用于灰白 Part B。黄色支撑只筛除不符合颜色条件的框，不能解决同一瓶盖同时报 front / back 的竞争检测，也不能单独证明面类正确。

校准仅使用旧 8 张开发 Val：best / last 经颜色过滤后均为 8 / 8 单个正确框、0 额外框，按记录的并列顺序 `last,best` 选择 last，保留第 13 epoch 的适配权重。last 的原始 AP（[results.csv](../../../runs/real-cap/20261006-finetune-v1/model-01/results.csv) 第 13 epoch）为 mAP50=0.97、mAP50–95=0.85029，不能将 best 的 0.87025 归给 last。

**初版批量预测**的组合固定阈值核验见 [operating-point-verification.json](../../../runs/real-cap/20261006-finetune-v1/operating-point-verification.json)；该记录是下节预览 / 视频的来源，后续实际 CLI 单独记录：

| 输入 | 恰好只有一个面类与定位正确的框 | 未匹配额外框 | 被颜色规则过滤的框 |
| --- | --- | --- | --- |
| 已见 Train | 80 / 80 | 0 | 0 |
| 旧开发 Val | 8 / 8 | 0 | 3 |

这些是**YOLO + 黄色过滤组合**的开发回放计数，与上节纯模型 AP 分开报告。该 Val 同时用于配置选择与核验，量小且已被查看，不能扩展为未见测试正确率。开发预览与照片回放已完成；当前不声明实际工作站部署效果。

## 预测入口与照片回放产物

[scripts/predict_real_caps.py](../../../scripts/predict_real_caps.py) 已通过代码与安全审查；1280 最长边导出的 Train / Val 与 4096×3072 原始 80 图均已通过实际 CLI 跑批，退出码均为 0。结果分别保留：

- [operating-train-predictions/predictions.json](../../../runs/real-cap/20261006-finetune-v1/operating-train-predictions/predictions.json)：80 张 Train 导出副本。
- [operating-val-predictions/predictions.json](../../../runs/real-cap/20261006-finetune-v1/operating-val-predictions/predictions.json)：旧 8 张开发 Val 导出副本。
- [operating-native80-predictions/predictions.json](../../../runs/real-cap/20261006-finetune-v1/operating-native80-predictions/predictions.json)：80 张原始尺寸图像。

三份输出均绑定上述 last 权重与 operating selection SHA-256，逐图预览位于各自的 `previews/`。依据现有开发标注独立核对面类与 polygon IoU≥0.5 的结果见 [delivery-verification.json](../../../runs/real-cap/20261006-finetune-v1/delivery-verification.json)：

| 实际 CLI 输入 | 恰好只有一个面类与定位正确的框 | 未匹配保留框 | 被颜色规则过滤的原始框 | 最小 polygon IoU |
| --- | --- | --- | --- | --- |
| Train 1280 导出副本 | 80 / 80 | 0 | 0 | 0.72014 |
| 旧开发 Val 1280 导出副本 | 8 / 8 | 0 | 2 | 0.70107 |
| 同一 Train 的 4096×3072 原图 | 80 / 80 | 0 | 0 | 0.72265 |

初版批量报告的旧 Val 有 3 个过滤框，实际 CLI 旧 Val 有 2 个，两份记录均保留。这里确认相同的最终单件面类 / 定位结果，不声称原始检测或四角逐位相同，也不猜测差异原因。原图与导出图是同一批 80 张的两种输入表示，不计作额外独立样本。交付报告还绑定模型、数据、代码、媒体与清理记录的 SHA-256；本轮没有新增或修改测试。

已完成产物：

- [calibrated-development-val.jpg](../../../runs/real-cap/20261006-finetune-v1/calibrated-development-val.jpg)：冻结的 last + 黄色过滤组合在旧 8 张开发 Val 上的叠框预览，来自初版批量报告。
- [photo-replay-h264.mp4](../../../runs/real-cap/20261006-finetune-v1/photo-replay-h264.mp4)：依据初版批量报告制作的 88 张静态照片及预测叠框回放，包含已见 Train 80 张和开发 Val 8 张；H.264、22 秒、640×480、20 FPS，共 440 帧。
- [photo-replay-verification.json](../../../runs/real-cap/20261006-finetune-v1/photo-replay-verification.json)：视频 / 预览 SHA-256、operating selection、来源预测报告 / 哈希绑定及 440 / 440 帧解码通过的记录；后续实际 CLI 的独立结果保存于上列目录。
- [phase1-readiness.json](../../../data/phase1-real-caps/20261006-v1/phase1-readiness.json)：本轮开发完成、正式门禁 CLOSED、原 8 张人工接受 / 新 80 张助手复核分开记录；临时目录当前清理状态为 `completed_by_user`，目录不再保留，绑定新增清理确认。

该视频是静态照片预测回放，不是物理运动录像。440 帧是回放编码帧数，不是 440 个独立样本或单件事件；不产生物理速度、时间稳定性、端到端延迟或 Validated Event Throughput 证据。

## 复现

在项目根目录使用 PowerShell 与现有 `.venv`。入口拒绝覆盖输出，重跑时必须使用新的运行目录；以下示例选择 `20261006-finetune-reproduction01`：

```powershell
.\.venv\Scripts\python.exe scripts/train_real_caps.py --help
.\.venv\Scripts\python.exe scripts/train_real_caps.py `
  --manifest data/real-cap-development/20261006-v1/manifest.json `
  --initial-weight runs/synthetic-cap/20261003-training-v2/yolo26n-obb/weights/best.pt `
  --output runs/real-cap/20261006-finetune-reproduction01 `
  --epochs 40 --patience 12 --imgsz 640 --batch 8 --device 0 --workers 0 --cpu-threads 8 --seed 42
```

命令会重新进行实际训练并消耗 GPU 时间；当前运行结果不需要重跑才能查看。输入 manifest 内记录本机归档绝对路径，移动工作区后须依据真实归档重建来源绑定，再执行预检，不能仅修改哈希值或略过来源校验。

仅复现开发推理时，使用 [scripts/predict_real_caps.py](../../../scripts/predict_real_caps.py) 加载冻结的 operating selection，无需重新训练。入口接受一个 JPEG 或一个直接包含 JPEG 的平铺目录，拒绝覆盖已有输出；例如对旧开发 Val 的导出副本重新生成叠框：

```powershell
.\.venv\Scripts\python.exe scripts/predict_real_caps.py --help
.\.venv\Scripts\python.exe scripts/predict_real_caps.py `
  --selection runs/real-cap/20261006-finetune-v1/operating-selection.json `
  --source data/real-cap-development/20261006-v1/dataset/images/val `
  --output runs/real-cap/20261006-inference-reproduction01 --device 0
```

输出为带来源 / 模型 / selection 哈希的 `predictions.json` 与 `previews/*.jpg`，分别记录原始检测、保留框和颜色过滤框。推理本身不读取面类 GT，不产生事件 GT 或正式指标。预测坐标基于 JPEG 编码像素，忽略 EXIF Orientation；原图不修改。

## 正式声明边界

本轮已取得“80 张真实照片开发微调完成”和“旧 8 张开发 Val 的观察结果”的证据。单批新图加一个已查看的开发 Val 无法证明完整 Operating Envelope、跨多个 Session 重复性或无偏最终性能。当前未实施 Traditional CV / YOLO26n-OBB / YOLO26s-OBB 公平对照，不签署 RQ1 结果。

正式协议、四路 Train / Val / Test / Final Holdout、事件 GT、条件矩阵、物理动态与风险、端到端延迟 / Validated Event Throughput 证据仍缺。照片增强和照片驱动模拟视频不能填写这些物理证据；Part B 换型未开始。具体缺口见 [Phase 1 执行计划](phase1_execution_plan.md)、[Part A / B 预冻结草案](../../data/part_ab_photo_prefreeze_v1.md) 和 [EXPERIMENT_PROTOCOL_V1](../../new-plans/EXPERIMENT_PROTOCOL_V1.md)。
