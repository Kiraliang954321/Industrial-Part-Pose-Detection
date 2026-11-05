# Screw Orientation V1：仿真开发数据与预测契约

日期：2026-10-31。适用任务 `screw_directed_axis_v1`，域 `simulation_development`。用户已确认首轮 M0–M3；本契约固定软件与仿真开发语义，**不表示真实实验协议 FROZEN 或真实 Dataset Gate OPEN**。

## 资产与观测

- 首轮资产 `screw_proxy_v0` 是项目自建代理，不对应已确认实物、厂商型号或标准 M 规格。
- 对象坐标 +X 指向头部；总长 0.043 m，杆径 0.006 m，头径 0.012 m，头长 0.006 m。
- 杆体和头部是同一个刚体的圆柱几何；17 个同心薄环仅模拟螺纹外观，不是螺旋螺纹。碰撞使用平滑杆体/头部，环不参与碰撞，密度和摩擦是仿真设定值，未经实物测量。
- 每个场景随机初始化位置和绕竖直轴方向，以 MuJoCo 3.17.0、0.002 s 时间步下降/沉降 250 步；记录最终位姿和剩余广义速度，不据此声称接触模型真实或机械臂抓取成功。
- 固定外部相机位于世界 `(0,0,0.21)` m，fovy 42°，640×640。保存实际 K、CV 相机外参与无畸变假设；这不是实际相机标定值。
- 本阶段图中最多一枚螺丝。每 60 张的最后 6 张为空背景，覆盖每个工况；生成 2000/400 张时正目标数为 1802/364，空背景数为 198/36。

## 两个语义点与有向角

| 索引 | 名称 | 对象坐标锚点（m） |
| --- | --- | --- |
| 0 | `head_axis_end` | `(0.023,0,0)`，头端轴心 |
| 1 | `tip_axis_end` | `(-0.020,0,0)`，尾端轴心 |

图像坐标 x 向右、y 向下。`heading_deg = atan2(head_y-tip_y, head_x-tip_x) % 360°`，表示尾端 → 头端，图像顺时针增加。误差使用 `abs(((pred-gt+180) % 360)-180)`，180° 头尾颠倒不被当作等价方向。

角度无效时输出 null 并给拒绝原因；漏检、低关键点置信度、短投影或头尾模糊不从总目标数里消失。原 OBB `[0°,180°)` 几何与 Front/Back 标签契约保持独立；两个点不输出完整 6D 或绕轴旋转。

关键点是轴上几何端点，可能被自身表面或遮挡物遮住。可见性通过目标 ID/depth 与锚点投影比较：`2` 为可见，`1` 为仿真位置已知但不可见，`0` 为未知/画外。当前生成器对画外/退化场景显式重采样并计数，因此最终正样本只含 1/2。可见性检查使用 1 mm 仿真深度容差，它是渲染标注规则，不是工业精度目标。

Ultralytics 的关键点有效性目标使用非零标记，`1` 和 `2` 都会参与已知坐标监督。模型输出的点置信度不能直接解释为真实的可见概率。头尾反射增强保持索引 `[0,1]`，不交换点的身份。

## 工况与限制

六个生成工况均匀轮换：`nominal`、`low_light`、`reflection`、`background_shift`、`motion_blur`、`partial_occlusion`。位置、方向与颜色/光照参数独立采样，并保存每场景的 seed sequence 和参数。

- 低光是灯光强度变化，仍保留环境光；反光是材质高光变化，均不代表已覆盖真实金属反射分布。
- 背景是颜色/光照变化的平面，没有声称覆盖复杂真实纹理。
- 模糊为 RGB 上固定 5 px 水平核；不等价于真实曝光与流水线速度。
- 遮挡物是视觉障碍，不参与接触；其阴影和遮挡保留在 RGB 中，GT 位姿只用于标签与评价。
- 训练原图没有标签点、箭头、坐标轴或文字；`previews/` 是单独的覆盖图，不进入训练目录。

## 文件与来源

每次生成写入不存在的新目录，拒绝覆盖或通过符号链接/Windows reparse point 访问数据。

| 文件 | 内容 |
| --- | --- |
| `dataset.json` | 任务/范围、计数、生成版本、资产、文件哈希、代码哈希、重采样计数、耗时 |
| `asset.json` / `scene.xml` | 尺度、锚点、简化声明与实际 MuJoCo 模型 |
| `records.jsonl` | 图像/标签哈希、场景/来源家族、split、工况、seed、相机、位姿、点/框/可见性 |
| `split-plan.json` | Train/Val 场景清单；Test/Final Holdout 未生成的显式记录 |
| `data.yaml` | 以 JSON 子集写出的 YAML，仅包含本地 Train/Val 和两点映射，无下载脚本 |
| `images/{train,val}/` / `labels/{train,val}/` | RGB PNG 与 YOLO Pose 标签；负样本为空标签 |
| `audit.json` / `progress.json` / `environment.json` | 审计、运行状态与环境版本 |

`scene_id == source_family`，一场景一快照，seed 使用 `[root_seed, split_index, image_index, resample_attempt]`。相邻视频帧未作为独立样本使用；同一代理几何跨 Train/Val 是“已知对象、独立场景”的开发设定，不能称为跨型号泛化。

审计验证哈希、完整文件清单、精确重复、标签数值、点顺序、有向角、刚体变换、3D→2D 重投影与两路计数。此审计建立软件/仿真一致性，不等价于真实 GT 或视觉可观察性的人工验收。

## 训练与评价

训练入口仅接受记录过官方 SHA-256 的 `yolo26n-pose.pt` 初始化，审计后在新目录记录输入快照、环境、配置、权重哈希、进度、耗时和显存。最大 30 epochs / 60 分钟，时间上限在 batch 边界请求停止，随后仍需保存与验证；若预算触发，最后一个记录的 epoch 可能未跑完整。支持已验证 CUDA BF16 和 FP32；本轮不升级原 Torch/Ultralytics。

评价入口只允许 Train/Val。CV 与 Pose 都只读相同 RGB；标注仅交给独立 scorer。当前最多一个 GT，匹配规则是预测框最高 IoU ≥ 0.5；其余预测计作额外检测，负样本上的所有检测均保留。

主报告包含全部目标、匹配、漏检、额外检测、方向拒绝和覆盖率，以及条件角误差 Median/P95、端点像素误差与按 GT 轴长归一化误差。方向半平面错误定义为误差 >90°，不称为精确的 180° 交换率。比例给 x/n 与 Wilson 95% CI，零错误保留上界。

开发选模诊断 `correct_images_10deg`：正图恰好一个匹配且有方向的预测、误差 ≤10°；负图没有预测。10° 是预先声明的开发比较口径，不是实际工艺容差。它与 5/10/15/30° 的全 GT 覆盖结果同时报告。只在 Val 上比较候选；不能将这些开发数值写成无偏 Test 成绩。

计时为模型预热后，从图像读取/解码到同步完成预测/后处理；模型加载、GT 评分和绘制不计入。报告计数、Median/P95，单位 ms/image；不是动态工位的事件延迟或 Validated Event Throughput。

## 运行入口

在仓库根目录使用 PowerShell 和现有 `.venv`。输出目录必须不存在；下列名称是示例，已有产物时换新目录。

```powershell
.\.venv\Scripts\python.exe scripts/generate_screw_orientation.py --output data/screw-orientation/my-development --train-count 2000 --val-count 400 --size 640 --seed 20261031
.\.venv\Scripts\python.exe scripts/train_screw_orientation.py --dataset data/screw-orientation/my-development --output runs/screw-orientation/my-training --initial-weights weights/screw-orientation/yolo26n-pose.pt --epochs 30 --batch 4 --precision bf16 --max-minutes 60
.\.venv\Scripts\python.exe scripts/evaluate_screw_orientation.py --dataset data/screw-orientation/my-development --output runs/screw-orientation/my-cv-val --method cv
.\.venv\Scripts\python.exe scripts/evaluate_screw_orientation.py --dataset data/screw-orientation/my-development --output runs/screw-orientation/my-pose-val --method pose --weights runs/screw-orientation/my-training/fit/weights/best.pt
.\.venv\Scripts\python.exe scripts/predict_screw_orientation.py --input data/screw-orientation/my-development/images/val --output runs/screw-orientation/my-replay --method pose --weights runs/screw-orientation/my-training/fit/weights/best.pt
```

回放是静态预测图片组成的视频，程序检查视频解码帧数；不是实物运动或机械臂动作视频。实际本轮产物与结果见[实施方案](../exec-plans/active/screw_orientation_v1_implementation_plan.md)。
