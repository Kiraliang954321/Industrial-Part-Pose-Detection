# 工业件仿真成果与接手指南

交付日期为 2026-11-04。当前主线是机械臂夹起零件，在固定检测位置用 RGB 估计方向与姿态，必要时主动观察、摆正，再完成有间隙预插入。最新 A5 增加齿轮键槽／凸键和轴承外缺口／定位销配合，通过腕部反馈在受阻时停止并带件退回。证据覆盖项目自建几何代理与 MuJoCo 仿真，真实门禁为 `CLOSED`。

协作者可以先打开[预览目录](../previews/README.md)，再读本页的成果表、运行方法和接手约束。Agent 从根目录 [AGENTS.md](../../AGENTS.md)进入后也按此顺序阅读。

## 成果总览

| 阶段 | 完成的能力 | 已记录结果及分母 | 详细记录 |
| --- | --- | --- | --- |
| 二维基础与历史瓶盖开发 | 采集、OBB 标注／QA、session-aware 四路组装、合成训练、实拍开发微调 | 80 张 Train 与 8 张开发 Val，真实正式协议与数据门禁待完成 | [历史开发](../exec-plans/active/real_cap_80_development.md) |
| M0–M3 螺丝视觉 | 头尾端点、有向轴线与 Traditional CV 对照 | 完成代理数据与视觉开发，后续物理闭环使用其权重 | [视觉记录](../exec-plans/active/screw_orientation_v1_development.md) |
| M4 视觉抓取 | 固定支撑面上的 2.5D 抓取与放置 | 24 个配对场景，视觉 24/24，理想真值参考另记 24/24 | [M4](../exec-plans/active/screw_m4_closed_loop.md) |
| L1 分度式工位 | 8 帧事件判定、放行／分拣／停机留件 | 24 事件，8 PASS、8 FAIL_ORIENTATION、8 UNCERTAIN，16 次自动动作完成 | [L1](../exec-plans/active/screw_line_simulation_v1.md) |
| H1 螺丝夹持检测 | 双 RGB 复核、预测补偿、闭爪预插入 | 24 次新抓取，11 PASS，其中 3 次许可过期留件，实际插入 8/8 | [H1](../exec-plans/active/held_inspection_h1.md) |
| H2 螺丝复检 | 转运后孔口复检、短时插入许可 | 36 次新抓取，实际插入 9/9，显著滑移／偏转 6/6 留件 | [H2](../exec-plans/active/held_recheck_h2.md) |
| A1 轴承／锥齿轮 | 六类 RGB 方向门控、固定配方套轴 | 40 次新抓取，8 PASS、16 REJECT、16 UNCERTAIN，实际预定位 8/8 | [A1](../exec-plans/active/bearing_bevel_a1.md) |
| A2 连续姿态 | 三 RGB、连续轴向与绕轴角估计 | 48 次评价，有结构的 40 次中 22 次可测，覆盖率 55% | [A2](../exec-plans/active/datum_pose_a2.md) |
| A3 自动摆正 | 六 RGB、双端面几何、翻转／大倾角校正后复拍 | 48 次新抓取，正常完整成功 25/36，另有 3 次齿轮深度超限 | [A3](../exec-plans/active/pose_cell_a3.md) |
| A4 主动观察 | 最多两次观察、持件一致性检查、针对性补训 | 54 次新抓取，正常完整成功 35/36，初检无解后恢复且成功 4/5 | [A4](../exec-plans/active/active_pose_a4.md) |
| A5 键／销配合 | 碰撞配合、分段下降、腕部反馈停止和退回 | 66 次新抓取，正常完整成功 34/36，工装错位接触停止并退回 12/12 | [A5](../exec-plans/active/keyed_cell_a5.md) |

各阶段使用各自的任务、分母和冻结批次。A5 正常完整成功 34/36 的 Wilson 95% 区间为 81.9%–98.5%，错位保护 12/12 为 75.8%–100%。原始汇总及分层区间位于 [evidence](evidence/)，A5 的 [66 次逐例结果](evidence/a5-episodes.json)包含全部成功、留件、空抓和受阻结果。

A5 的两次正常留件来自轴承大倾角端面几何冲突。成功深度为 3.251–4.168 mm，当前上限为 4.2 mm。补训覆盖收益跨批次不稳定，小角度机械容差、深度裕量、真实传感噪声、松爪交接和连续上下料仍待验证。A5 故意错位工装相对配方偏转 30°／90°，视觉仍对准配方，之后由实际接触触发腕部保护。

## 克隆后直接预览

从 fork 获取本次分支，在 PowerShell 中执行。

```powershell
git clone --branch codex/bearing-bevel-prototype https://github.com/yunluoxincheng/Industrial-Part-Pose-Detection.git
Set-Location Industrial-Part-Pose-Detection
Start-Process .\docs\previews\index.html
```

预览页使用仓库内九段 MP4 和配图，可以直接双击 HTML 打开。浏览器或播放器需要支持 H.264。A3 动作段为 3 倍速经一致性核验的重放，A4 动作段为 2 倍速原始录像，A5 使用原录像速度并附静态结果卡。播放录像只需要本地浏览器。

也可以在根目录启动只监听本机的静态服务，再打开 `http://127.0.0.1:8768/previews/`，退出服务使用 `Ctrl+C`。

```powershell
py -3 -m http.server 8768 --bind 127.0.0.1 --directory docs
```

## 校验交付内容

本次独立检出、环境安装、浏览器预览和运行结果见[交付校验记录](verification.md)。

以下入口仅使用 Python 标准库，检查公开材料及九份协议关联的 160 个独立源码文件。成功输出 `status=verified`。[产物清单](artifact-manifest.json)

```powershell
py -3 scripts/verify_handoff.py
```

`.gitattributes` 将 Python、协议 JSON、Markdown 和 HTML 固定为 LF 换行，项目文本使用 UTF-8。导入的原始评价 JSON 与 Panda 许可按原始字节保留。这样 Windows 检出时也能保留冻结源码与证据的字节哈希。哈希异常时先核对检出版本和换行格式。

## 运行机械配合参考

此入口无需视觉权重，可以检查键／销几何、机器人运动和腕部反馈。它使用理想工件位姿驱动动作，结果归入机械参考。Windows 是已使用的开发平台，建议 Python 3.11，并具备可用的 OpenGL 图形环境。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[simulation]"
$env:PYTHONPATH='src'
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe scripts/prepare_panda_assets.py --output data/robot-assets/panda-a662de6
.\.venv\Scripts\python.exe scripts/reference_keyed_cell.py --assets data/robot-assets/panda-a662de6 --output runs/keyed-cell/handoff-reference --seed 261400
```

资产准备脚本下载固定提交的 Panda 资产，约 34 MB，首次需要联网。所有 `--output` 使用尚不存在的目录。十个机械参考包括两个正确配合，以及两种零件各自的 30°／90°／180° 错位和无定位结构。检查 `results.json` 中的 `physical.keyed_success`、`blocked`、`physical.retracted_and_held`、接触穿透与失败原因。

如果资产下载遇到网络中断，保留未完成目录，换一个新的输出目录重试，并将参考命令的 `--assets` 指向完成后的目录。当前入口逐文件下载，重试需要从头准备。也可交接已有资产目录，通过 `audit_panda` 校验其固定提交与每个文件的哈希。

## 继续视觉开发与训练

A3–A5 当前推理入口使用 CUDA `device='0'`，训练使用 CUDA 和 BF16，需要兼容的 NVIDIA GPU。历史环境为 Python 3.11.9、PyTorch 2.11.0+cu128、Ultralytics 8.4.171、MuJoCo 3.17.0。完整版本与六份选定权重的相对路径、字节数、SHA256 已记录。[环境及权重清单](runtime-environment.json)

仓库提供生成、训练、校准、冻结与运行入口。视觉权重、训练图像、Panda 资产和完整 `runs/` 归档通过单独材料交接获得，其中 Panda 可由上述固定来源下载。开始视觉运行前按清单取得对应 `.pt` 并核对 SHA256。安装推理依赖时使用已记录的 Ultralytics 版本，并按设备安装兼容 CUDA 的 PyTorch。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[simulation]" "ultralytics==8.4.171"
.\.venv\Scripts\python.exe scripts/generate_pose_cell.py --help
.\.venv\Scripts\python.exe scripts/prepare_pose_cell_centers.py --help
.\.venv\Scripts\python.exe scripts/train_pose_cell.py --help
.\.venv\Scripts\python.exe scripts/generate_active_pose.py --help
.\.venv\Scripts\python.exe scripts/train_active_pose.py --help
.\.venv\Scripts\python.exe scripts/run_keyed_cell.py --help
```

| 要继续的工作 | 入口与前置材料 |
| --- | --- |
| A1 单 RGB 面类 | `generate_annular.py` → `train_annular.py` → `run_annular.py` → `freeze_annular.py`，初始化为 YOLO26n 检测权重 |
| A2 连续角度 | `generate_datum_pose.py` → `crop_datum_pose.py` → `train_datum_pose.py` → `validate_datum_pose.py`，目标配置位于 `docs/data/datum_pose_targets_v1.json` |
| A3 大倾角与翻转 | `generate_pose_cell.py` → `prepare_pose_cell_centers.py` → `train_pose_cell.py` → `validate_pose_cell.py`，注意 `centers_v2` 语义与历史初始化链 |
| A4 主动观察 | `generate_active_pose.py --source <A3-center-dataset>` → `train_active_pose.py --initial-weights <A3-center-best.pt>` → `validate_active_pose.py`，需要 A3 中心数据及权重 |
| A5 工装／反馈 | `reference_keyed_cell.py` 检查机械行为，`run_keyed_cell.py` 使用指定 A4 权重，`freeze_keyed_cell.py` 绑定机械参考与视觉校准 |

历史协议绑定了原始 Windows 绝对路径、数据／训练／校准清单、源码、软件包和模型哈希，完整历史评价重放需要对应归档及路径条件。重新下载资产会生成新的带时间戳清单，新训练也会生成新的模型身份。迁移到另一台机器或改变方法后，应建立新开发版本、完整校准和新协议，再分配未用于调参的新评价批次。当前 A4/A5 入口固定了模型哈希，新权重的接入需要在新的开发版本中明确更新身份约束。

## 下一位协作者与 Agent 的阅读顺序

1. [AGENTS.md](../../AGENTS.md)说明授权范围、当前事实与实验约束。本页提供公开材料和环境入口。
2. [A5 记录](../exec-plans/active/keyed_cell_a5.md)给出最近一次交付，[架构](../../ARCHITECTURE.md)说明各阶段复用关系。
3. [ADR-0003](../decisions/adr-0003-held-part-inspection-before-assembly.md)解释夹持检测用途，[工业件计划](../exec-plans/active/industrial_orientation_3d_execution_plan.md)保留阶段演进。
4. 阅读 `keyed_cell/runtime.py` 和 `inspection.py` 了解执行编排，再看 `control.py`、`guard.py`、`evaluation.py`。视觉求解复用 `pose_cell/predictor.py`、`metrology.py` 与 `active_pose/policy.py`。
5. 继续工作前记录分支、`git status` 和手头材料，运行交付校验。新增实验用新目录保存失败样本、参数和环境，执行当前任务所需的最小验证。

运行时输入关系为六路 RGB → 定位／关键点 → 多视角几何 → 持件一致性和目标误差 → 机器人校正 → 新鲜图像确认 → 机械预插入。腕部力／力矩和夹爪编码器负责插入保护。对象真值用于生成监督、机械参考、独立评分及标注的示意图，视觉动作许可依赖预测、标定和机器人反馈。

数据按完整抓取分组，所有视角与主动观察图像随同一抓取进入同一个 split。A3 原评价已用于 A4 分析，后续使用时按已见数据处理。评价错误分析若用于修改模型或阈值，应另设未见评价批次。报告保留覆盖率、条件误差、实际动作分母与置信区间。

后续可立项的问题包括轴承大倾角可测性、插入深度裕量、真实材料与传感条件、松爪交接和连续工位。下一阶段按具体任务目标确认后实施，当前交付边界为 A5 闭爪预插入。

## PR 关系与来源

本分支基于 `0d31b58` 的历史瓶盖开发提交和 `d9d342b` 的螺丝阶段提交，随后加入 A1–A5。上游 PR #2 对应 `0d31b58`，本次向 `main` 提交的比较会包含这项祖先变更。若上游保留该提交合并，GitHub 会按共同祖先更新差异；采用 squash 合并时，需要重新核对两条 PR 的重叠变更。[上游 PR #2](https://github.com/Kiraliang954321/Industrial-Part-Pose-Detection/pull/2)

预览均为项目生成的仿真画面。机器人来自 MuJoCo Menagerie 的固定提交 `a662de603224c47123b0783821c58ad4402ad3e3`，Panda 资产许可随预览附带。模型和资产的使用遵循各自来源许可。[预览来源与许可](../previews/README.md)
