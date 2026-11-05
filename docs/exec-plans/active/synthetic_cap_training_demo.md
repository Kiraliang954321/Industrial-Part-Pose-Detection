# 独立合成瓶盖训练演示

**最后更新：2026-10-04。状态：合成素材、双模型训练、合成 Test 评估与逐帧预测演示完成；Final Holdout 未评估，正式 Training Gate CLOSED。**

本路线用于交付可复现的合成瓶盖素材、OBB 标注、模型权重及带预测的视频，验证本地训练和推理流程。用户于 2026-10-03 批准路线，并要求代码修改先建新分支；实施分支为 `codex/synthetic-cap-training`。

本演示的所有数据与结果均标记为 `synthetic_only`。它使用独立的演示前置检查，不构成正式 [EXPERIMENT_PROTOCOL_V1](../../new-plans/EXPERIMENT_PROTOCOL_V1.md) 或 [Phase 1](phase1_execution_plan.md) 的验收证据；正式 Experiment Protocol 未 FROZEN、Dataset V1 / Training Gate 仍为 CLOSED。

## 已批准范围与语义

- 自行制作黄色圆形瓶盖的 Front / Back 合成图像和输送带运动视频，保留来源、生成参数、随机种子和 Session 信息。
- Front 使用黄色外表面及红色图案，Back 使用内凹面、内部环形结构及深色方形图案。面类别由生成过程明确记录，不能从模 180° 的 OBB 长轴推断。
- 本次 `part_id = synthetic_cap_a`，类名为 `synthetic_cap_a_front` / `synthetic_cap_a_back`，遵循现有标签契约。合成对象不替代真实 Part A 的物理证据。
- `orientation_semantics = NOT_APPLICABLE`，角度评价为 N/A。圆形瓶盖的定位 OBB 仍可用于检测，OBB 的数值角度不表示已验证的工业方向。
- 训练 YOLO26n-OBB 与 YOLO26s-OBB；选择演示模型后生成预测视频和运行记录。
- 不把面标签当作事件级 `CONFORMING` / `NONCONFORMING` / `INDETERMINATE` GT，也不引入 `UNCERTAIN` 训练类。

## 素材、划分与独立性边界

首版已生成 12 个 Session，每个 Session 120 张 JPEG 和对应 OBB 标签，Front / Back 各 60 张，共 1,440 张。每个 Session 使用明确的场景参数和独立随机种子，覆盖位置、尺度、光照、背景、模糊及轻度遮挡。另有每 Session 一段 640×480、12 秒、30 FPS、360 帧的运动视频，共 12 段。静态训练图像是独立采样的程序渲染，未从这些视频逐帧抽取。

| Split | Session 数 | 图像数 | Front / Back | 用途 |
| --- | ---: | ---: | ---: | --- |
| Train | 6 | 720 | 360 / 360 | 训练模型 |
| Val | 2 | 240 | 120 / 120 | 选模型、阈值和工作点 |
| Test | 2 | 240 | 120 / 120 | 冻结配置后的合成评估 |
| Final Holdout | 2 | 240 | 120 / 120 | 保留的未调参验证数据，默认不使用 |

上述数量已从 manifest 和完整 `validate_bundle` 前置检查核实。以完整 Session 分配 split；若后续从视频抽帧，一段视频的所有抽帧及其增强变体必须留在同一路，禁止随机逐帧跨 split。Final Holdout 在物理输出和 manifest 中保留，训练用 `data.yaml` 只暴露 Train / Val / Test。

生成 Session 是不同的合成场景，不能视为不同真实拍摄 Session。不同 split 仍共享瓶盖几何和外观生成模板，因此它们存在共同的生成域。即使背景和随机种子不同，也不能用此结果推断真实瓶盖、相机、照明或输送带的泛化能力；后续必须通过实拍数据检验 domain gap。

完整 `validate_bundle` 和现有 `audit_dataset.py` 已通过；JPEG SHA-256 与解码像素 SHA-256 的跨 split 完全重复对均为 0。近重复检查使用 64-bit full-frame dHash、Hamming distance ≤ 4，共报告 **5,446 个候选对**，仅保留部分示例；这些是待可视复核的相似图像，不是已经确认的数据泄漏，也不能写成近重复检查零命中。

JPEG hash 只能发现字节相同的素材，解码像素 hash 只能发现像素相同的素材。全帧 dHash 易受相似背景和小目标占比影响，可能漏掉裁剪、旋转或光照变化后的同源内容，也可能误报不同瓶盖。Exact 检查通过不等于消除了共享模板或语义相似性；当前自动审计不构成完整的独立性证明。

## 调参与报告规则

所有参数调整只使用 Train / Val。当前入口按 **Val mAP50-95 最高**选择模型，同分保持 CLI 中的候选顺序；视频工作点预设 confidence `0.25`、NMS IoU `0.7`。在查看 Test 前写入 `selection.json`，冻结选择、权重 hash、数据 hash、图像尺寸和工作点，再评价两个候选模型的 Test。Test 结果不能用于重新选优或继续调参。若根据 Test 错误修改素材、参数或模型，原 Test 不再作为该新版本的未见最终验证，应记录新版本及新的未见验证安排。

默认流程不评价 Final Holdout，也不把它用于模型选择。若后续需要启用，先明确使用目的、冻结模型及配置，再单独保存最终验证结果。

检测指标和面分类结果只描述合成域。视频预测是逐帧 OBB 检测演示；当前不具备真实事件 GT、Tracking / 多帧事件决策、Risk Gate 或 Part B 换型验收。单帧耗时、模型 FPS 或视频处理速度不能写成端到端工业 P95、Validated Event Throughput、Wrong Accept 风险或生产性能结论。

## Windows 环境

在项目根目录使用 PowerShell；以下命令显式调用 `.venv`，无需修改终端激活策略。运行需要 NVIDIA CUDA GPU、足够磁盘空间及模型首次下载的网络连接。

本次已安装的本地环境：

| 项目 | 已核实版本 / 设备 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8 GB |
| PyTorch | `2.11.0+cu128` |
| torchvision | `0.26.0+cu128` |
| Ultralytics | `8.4.171`，安装在项目 `.venv` |
| Polars / polars-runtime-32 | 均为 `1.35.2`，用于修复本机导入兼容问题 |
| imageio-ffmpeg | `0.6.0`，独立可选依赖，用于输出 H.264 演示视频 |
| 项目包 | `yolo-conveyor 0.1.0`，已在 `.venv` 中 editable 安装 |
| 实际 OpenCV 模块 | `cv2.__version__ = 4.11.0` |

`.venv` 复用本机已有 CUDA PyTorch；本次没有升级全局 PyTorch、NumPy 或 OpenCV。其他机器应先安装与自身驱动匹配的 CUDA PyTorch，再安装项目与兼容 Ultralytics，不能直接照搬本机的 CUDA 版本。复现时应保存实际 Python / CUDA / 库版本和 GPU 信息。

初次安装的 Polars `1.44.2` 在本机导入时报 `RuntimeError: unknown feature flag: 'sse3'`，阻断首轮 checkpoint 保存。因此仅在项目 `.venv` 将 `polars` 与 `polars-runtime-32` 同时固定为 `1.35.2`；重启训练已成功写入两个模型的 `best.pt` / `last.pt`，兼容问题已在本机实际恢复。可选 `imageio-ffmpeg` 不加入 `pyproject.toml`。

本机复用环境同时存在 `opencv-python 4.8.1.78`、`opencv-python-headless 4.11.0.86` 与 `opencv-contrib-python 4.11.0.86` 的分发 metadata；实际导入的是 `cv2 4.11.0`。来源记录已分别保存 metadata 与模块版本，不把旧 `opencv-python` metadata 误写成运行版本。迁移机器时应核实实际导入模块与编解码能力。

```powershell
.\.venv\Scripts\python.exe -m pip install "polars==1.35.2" "polars-runtime-32==1.35.2" "imageio-ffmpeg==0.6.0"
```

```powershell
.\.venv\Scripts\python.exe scripts/check_env.py
```

`data/`、`runs/`、`.venv/` 和 `*.pt` 已被 Git 忽略。不要把数据、模型权重或下载缓存混入源码提交。重复运行应使用新的版本化输出目录；不要通过删除或覆盖已有实验来消除失败记录。

环境、官方预训练权重 SHA-256、数据前置检查及代码快照保存在 `runs/synthetic-cap/environment/`。官方 `yolo26n-obb.pt` / `yolo26s-obb.pt` 均已下载并实际加载为 `task=obb`。训练进程使用自身的 Ultralytics settings 目录，局部 `api_key` 为空、清除该进程的 Platform API key 环境项并禁用外部 logger，不写入用户全局凭据。

## 复现命令与产物

生成及训练 CLI 均已用 `--help` 核实，以下命令均已实际执行成功。数据和训练输出目录已经存在，复跑必须替换为新的版本化目录，入口不会覆盖已有数据或实验。只重训相同 bundle 时可保留 `--bundle` 路径，替换 `--output`；重新生成数据时须同时更新 bundle 和训练输出路径。生成 seed `20261003` 与训练 seed `42` 是两个不同用途的随机种子。

制作素材、OBB 标注和四路 bundle：

```powershell
.\.venv\Scripts\python.exe scripts/generate_synthetic_caps.py --output data/synthetic-cap-demo/20261003-v1 --seed 20261003 --images-per-session 120 --width 640 --height 480 --video-seconds 12 --fps 30
```

结构与标签语法审计：

```powershell
.\.venv\Scripts\python.exe scripts/audit_dataset.py data/synthetic-cap-demo/20261003-v1/dataset --classes 2
```

训练入口会先完整检查 bundle 的 provenance、来源文件、标注、视频 / GT hash、四路数量与 exact 重复记录，再创建独立训练目录。首次运行的失败目录为 `runs/synthetic-cap/20261003-training`；完成的运行使用以下目录，未覆盖旧记录：

```powershell
.\.venv\Scripts\python.exe scripts/train_synthetic_caps.py --bundle data/synthetic-cap-demo/20261003-v1 --output runs/synthetic-cap/20261003-training-v2 --models n s --epochs 60 --batch 8 --imgsz 640 --device 0 --workers 0 --cpu-threads 8 --seed 42 --patience 15
```

`--workers 0` 用于 Windows 本地首版；`--batch 8`、`--imgsz 640` 是本次配置，显存不足时只能在查看 Test 前调整并记录。`--epochs 60` 是最大训练轮数，本次 early stopping 后 n / s 实际完成 32 / 27 轮。

已生成的数据与训练产物：

- `data/synthetic-cap-demo/20261003-v1`：`captures/`、`annotations/`、`dataset/`、`videos/`，以及 `synthetic_provenance.json`、`split_plan.json`、`leakage_report.json`、`contact_sheet.jpg`。
- `runs/synthetic-cap/20261003-training-v2`：训练配置 `run-config.json`、两个模型的 `weights/best.pt` / `weights/last.pt`、`training-progress.json`、`selection.json`、`result-summary.json`、`prediction-video.mp4`、逐帧 `frame-predictions.jsonl`、`video-summary.json` 和推理可视预览。
- `runs/synthetic-cap/20261003-training/failure-summary.json` 与相邻的 `20261003-training.log`：首次 epoch-1 checkpoint 保存失败的记录；未生成 checkpoint，也未评价 Test / Final Holdout。
- Final Holdout：保留的数据及来源，不预填未运行的指标。

先前 ImageGen 正反面参考预览保存在 `data/synthetic-cap-preview/20261003/bottle-cap-front-back.png`，提示词保存在同目录 `generation-prompt.txt`。生成 provenance 明确 `reference_image_used_as_training_texture = false`：它只作为参考，训练图像来自程序渲染，不是该图的裁切、变换或实物照片。

## 已完成的合成结果

以下结果取自 [result-summary.json](../../../runs/synthetic-cap/20261003-training-v2/result-summary.json)，均为 `synthetic_only`。Val 和 Test 各 240 张图像，每类 120 个 GT instance。

| 模型 | 实际轮数 | 训练时长 | Val mAP50-95 | Test Precision / Recall | Test mAP50 | Test mAP50-95 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| YOLO26n-OBB | 32 | 1,461.09 秒 / 24.35 分钟 | 0.995000 | 1.000 / 1.000 | 0.995000 | 0.995000 |
| YOLO26s-OBB | 27 | 2,789.85 秒 / 46.50 分钟 | 0.995000 | 1.000 / 1.000 | 0.995000 | 0.994475 |

整条训练、Val、Test 和视频推理流程耗时 4,289.69 秒，约 71.49 分钟。两个候选的 Val mAP50-95 相同，按预先声明的 CLI 顺序 `n s` 选择 **YOLO26n-OBB**；[selection.json](../../../runs/synthetic-cap/20261003-training-v2/selection.json) 在 Test 评估前落盘。选择没有使用 Test 指标；没有评价 Final Holdout。

表中的 Precision / Recall 来自 Ultralytics AP 验证曲线及其默认低 confidence 评价过程，不能当作视频 `confidence=0.25` 的固定工作点统计。合成域达到很高指标，也不能消除生成器共享模板与真实相机之间的 domain gap。

最终模型预测的是一个 Test Session 的 12 秒视频，360 帧分为 133 个完整可见帧、69 个空帧及 158 个局部可见帧。定位匹配使用 OBB IoU ≥ 0.5；confidence `0.25`、NMS IoU `0.7`、imgsz `640` 与冻结记录一致。局部可见帧单独记数，不混入完整可见帧的成功率分母。

| 视频逐帧统计 | x / n | 比例 | 描述性 Wilson 95% CI |
| --- | --- | ---: | --- |
| 完整可见帧检测覆盖 | 133 / 133 | 100% | 97.19%–100% |
| 完整可见帧漏检 | 0 / 133 | 0% | 0%–2.81% |
| 已定位帧的面分类正确 | 133 / 133 | 100% | 97.19%–100% |
| 完整可见帧同时定位及面正确 | 133 / 133 | 100% | 97.19%–100% |
| 空帧出现检测 | 0 / 69 | 0% | 0%–5.27% |
| 局部可见帧 | 158 帧，单独分层 | 不评上述成功率 | 不报告 |

这些连续帧相互相关，Wilson 区间仅作合成逐帧描述，不能作为独立物理事件的置信区间、事件级 Wrong Accept 风险或工业接受率。局部可见帧的完整行为及逐帧框保存在 [frame-predictions.jsonl](../../../runs/synthetic-cap/20261003-training-v2/frame-predictions.jsonl)，不是全部局部帧通过的声明。

[video-summary.json](../../../runs/synthetic-cap/20261003-training-v2/video-summary.json) 中的单帧阶段 P95 为 **23.56 ms**，计时只覆盖 Ultralytics preprocess / inference / postprocess，排除视频解码、渲染、写盘、事件决策及端到端路径。不能由此报告端到端 P95 或 Validated Event Throughput。

## 可用模型与查看入口

下列相对链接指向本次工作区产物；这些数据、权重与运行文件被 Git 忽略，源码仓库发布后不自动包含它们。演示默认使用按 Val 选定的 n 模型 `best.pt`。

| 产物 | 本地入口 |
| --- | --- |
| 最终演示模型 / n 最佳权重 | [yolo26n-obb/weights/best.pt](../../../runs/synthetic-cap/20261003-training-v2/yolo26n-obb/weights/best.pt) |
| n 最后权重 | [yolo26n-obb/weights/last.pt](../../../runs/synthetic-cap/20261003-training-v2/yolo26n-obb/weights/last.pt) |
| s 最佳权重 | [yolo26s-obb/weights/best.pt](../../../runs/synthetic-cap/20261003-training-v2/yolo26s-obb/weights/best.pt) |
| s 最后权重 | [yolo26s-obb/weights/last.pt](../../../runs/synthetic-cap/20261003-training-v2/yolo26s-obb/weights/last.pt) |
| 原始预测视频 | [prediction-video.mp4](../../../runs/synthetic-cap/20261003-training-v2/prediction-video.mp4) |
| H.264 查看版视频 | [prediction-video-h264.mp4](../../../runs/synthetic-cap/20261003-training-v2/prediction-video-h264.mp4)，已核实 H.264、640×480、30 FPS、360 个可解码帧、12 秒 |
| 视频推理预览 | [video-inference-contact-sheet.jpg](../../../runs/synthetic-cap/20261003-training-v2/video-inference-contact-sheet.jpg) |
| n / s Val 推理预览 | [n 预览](../../../runs/synthetic-cap/20261003-training-v2/yolo26n-obb/val-inference-contact-sheet.jpg) / [s 预览](../../../runs/synthetic-cap/20261003-training-v2/yolo26s-obb/val-inference-contact-sheet.jpg) |
| 生成素材预览 / 来源 / 重复审计 | [contact_sheet.jpg](../../../data/synthetic-cap-demo/20261003-v1/contact_sheet.jpg) / [synthetic_provenance.json](../../../data/synthetic-cap-demo/20261003-v1/synthetic_provenance.json) / [leakage_report.json](../../../data/synthetic-cap-demo/20261003-v1/leakage_report.json) |
| 环境 / 数据前置检查 / 代码快照 | [runtime.json](../../../runs/synthetic-cap/environment/runtime.json) / [dataset-preflight.json](../../../runs/synthetic-cap/environment/dataset-preflight.json) / [code-snapshot.json](../../../runs/synthetic-cap/environment/code-snapshot.json) |
| 交付 hash / 视频核验记录 | [delivery-verification.json](../../../runs/synthetic-cap/20261003-training-v2/delivery-verification.json)：四份权重、冻结数据与代码 hash、模型任务 / 类名及 H.264 视频解码检查 |

项目已安装，可复用既有单图入口运行所选权重。例如在项目根目录：

```powershell
$env:YOLO_CONFIG_DIR = (Resolve-Path 'runs/synthetic-cap/20261003-training-v2/ultralytics-config').Path
$env:ULTRALYTICS_API_KEY = ''
.\.venv\Scripts\yolo-conveyor.exe data/synthetic-cap-demo/20261003-v1/dataset/images/test/synthetic-cap-09-28304485/front/cap-00000.jpg --model runs/synthetic-cap/20261003-training-v2/yolo26n-obb/weights/best.pt --confidence 0.25
```

此命令已实际运行成功，示例返回 Front（`class_index=0`）。单图输出的 `class_index` 0 / 1 对应 Front / Back；既有通用 CLI 的 `angle_deg` 是几何长轴数值，在本瓶盖任务中应按 `NOT_APPLICABLE` 解读，不当作方向判断。此命令只用于复现推理；不得根据查看的 Test 样本继续调参后仍将同一 Test 当作未见最终评估。

## 本次实施记录

| 项目 | 状态与证据 |
| --- | --- |
| 路线批准与新分支 | 已批准；`codex/synthetic-cap-training` |
| `.venv` / CUDA 环境 | 已准备，版本见上表 |
| 生成入口与训练入口 | 已实现，两个 `--help` 均通过 |
| 1,440 张素材与视频 | 完成；12 个 Session，12 段视频，数量见上表 |
| 标注 QA、分布、来源与 exact 重复审计 | 完整 bundle 前置检查与已有数据审计通过；跨 split JPEG / 像素完全重复为 0 |
| 近重复候选复核 | dHash 5,446 个候选对；自动候选不是完整独立性证明，可视复核待完成 |
| 相关既有测试 | 新 `.venv` 中 76 项通过；未新增测试 |
| 官方 OBB 预训练权重 | n / s 均下载并实际加载，hash 保存在环境记录 |
| 首次训练 | epoch-1 保存 checkpoint 时失败，记录保留，未生成权重 |
| Polars 兼容调整 | `.venv` 固定 1.35.2；两个模型的最佳 / 最后 checkpoint 实际保存成功 |
| YOLO26n-OBB 训练 | 完成，32 轮，best / last 均可用 |
| YOLO26s-OBB 训练 | 完成，27 轮，best / last 均可用 |
| Val 选模与冻结记录 | 完成；Val 同分按候选顺序选 n，Test 前写入 `selection.json` |
| 合成 Test 评估 | 完成；两个模型各 240 图，指标见结果表 |
| 逐帧预测视频 | 完成；360 帧及逐帧记录，133 个完整可见帧定位与面均正确，69 空帧无误检，158 局部帧单列 |
| H.264 查看版视频核验 | 完成；codec / 360 帧解码 / 30 FPS / 12 秒 / 640×480 与 hash 均留记录 |
| 交付模型与数据一致性 | 四份权重 hash 与冻结记录一致，两个最佳模型实际加载为 OBB、类名匹配；数据 hash 与冻结选择一致 |
| Final Holdout 评估 | 未运行，未用于选择或调参 |
| 真实数据验证 / 正式门禁 / 工业事件性能 | 未完成，本演示不替代 |

本表依据实际运行证据更新；不以软件入口可用、数据组装成功或合成指标签署正式训练门禁。真实相机、Traditional CV 公平对照、事件决策、风险拒绝及换型仍需按正式协议另行完成。
