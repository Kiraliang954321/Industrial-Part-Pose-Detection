# 螺丝头尾与有向轴线：首轮技术调研

调研日期：2026-10-31。本文保留启动前的选型依据；当天用户随后确认实施方案，已完成首轮仿真开发与训练，实际结果见[开发记录](../exec-plans/active/screw_orientation_v1_development.md)。本文中的“建议/待确认”描述调研时的决策状态，不替代该运行记录。

依据 [ADR-0002](../decisions/adr-0002-industrial-orientation-3d-simulation.md)与[工业件研究计划](../exec-plans/active/industrial_orientation_3d_execution_plan.md)。具体螺丝型号、实物、CAD 和真实工位参数仍未确认。本文是面向当前设备的首轮方案，不代表穷尽相关文献或完成真实验收。

## 1. 结论与推荐路线

推荐以 **RGB 螺丝检测 + 两个语义关键点 + 360° 有向轴线**作为第一任务，保留同输入的 Traditional CV 对照；本机仿真优先验证 **MuJoCo**，需要更真实的材质和照明时再引入 BlenderProc。第一批交付为可复现的仿真域数据、开发训练、预测可视化及独立场景评估，随后接入预测驱动的虚拟机械臂。

这项推荐是结合本机环境与以下来源得到的工程判断，不是已测出的算法优胜结论：

- OBB 在旋转 180° 后仍是同一个框，单靠长轴不能区分螺丝头尾；既有 `geometry.py` 也明确将角度归一化到 `[0°,180°)`。[R1]
- Ultralytics Pose 支持自定义关键点数量、坐标与可见性，可将同一螺丝上的两个关键点固定为头端和尾端，而不是把两端当成独立检测类别。[R2][R3]
- 两个轴上关键点只提供有限的方向信息，不能解出完整 6D，也不能唯一确定绕轴旋转。PnP 另需已知 3D 对应点、相机内参与非退化几何；增加共线中点不会解决这一问题。[R4]
- 目前只有合成数据准备路线，尚无当前型号的真实标注。因此首轮最多建立仿真域证据；真实迁移必须另取独立实拍数据。

## 2. 本机与仓库核查

以下来自本轮只读命令及一次微小 CUDA 张量运算，不是训练/渲染性能测量。

| 项目 | 2026-10-31 核查结果 | 含义 |
| --- | --- | --- |
| Shell / Python | PowerShell 7.6.9；项目 `.venv` Python 3.11.9 | 使用 Windows 原生命令及现有 Python 版本 |
| CPU / RAM | i7-13650HX，14 cores / 20 threads；约 31.8 GiB RAM | 可先做小场景；未测试生成吞吐 |
| GPU | RTX 4060 Laptop；8188 MiB；驱动 591.86 | 训练批量需实测显存，不能据型号承诺速度 |
| Torch | 2.11.0+cu128；torchvision 0.26.0+cu128 | CUDA 可用，GPU 小张量求和得到预期值 10 |
| 视觉依赖 | Ultralytics 8.4.171；OpenCV 4.11.0.86；NumPy 1.26.4 | 已有 `yolo26-pose.yaml`，本轮不需先升级训练栈 |
| 仿真依赖 | 项目 `.venv` 未安装 PyBullet / MuJoCo / BlenderProc | 引擎安装和本机渲染仍待实施 |
| 磁盘 | E: 当时可用约 103.7 GiB | 产物设大小上限，不能无限生成数据 |
| 权重 / CAD | 项目根目录及 `weights/` 未见 Pose 权重；本轮 `rg --files` 可见项目文件中未见 CAD/URDF/mesh | 不是对用户其他目录或所有被忽略文件的全盘盘点 |

已有实现的复用边界：

- `src/yolo_conveyor/real_photo_training.py` 绑定 `cap_a_front/cap_a_back` 与 OBB；`scripts/predict_real_caps.py` 另有黄色过滤。不能改个类别名就视作螺丝训练与预测。
- `src/yolo_conveyor/datasets/project_v1.py` 的四路完整 session 和来源哈希理念可复用，但它实际绑定面类与 OBB 标注，不直接承载关键点数据。
- `src/yolo_conveyor/types.py` 的 `Detection` 只有类别、置信度和四角；新增有向关键点契约独立实现，不改变旧 `angle` 的 180° 语义。
- 旧采集/标注、历史瓶盖资产和成绩保留；新数据、权重、报告使用螺丝专用路径与任务版本。

## 3. 方法比较

| 方法 | 首轮作用 | 前提与局限 | 建议 |
| --- | --- | --- | --- |
| Traditional CV：前景/轮廓、轴向、两端宽度或形状差异 | 可解释对照，验证简单方法是否已经足够 | 头部显著宽于杆体时有用；反光、遮挡和复杂背景可能失败，参数仅由 Train/Val 选择 | 必做同任务对照，不读取 GT mask |
| YOLO26n-Pose：头/尾两个关键点 | 检测位置与头尾有向角，复用现有训练栈 | 需新的关键点监督；遮挡、端视和太小目标须拒绝或分层 | 首选学习基线 [R2][R3] |
| YOLO26s-Pose | 检查更大模型是否值得额外算力 | 必须先有 nano 基线与显存/速度记录 | 有必要再做，首轮不默认双模型长训 |
| YOLO26 OBB + 头尾判别分支 | OBB 定位后再消除 180° 歧义 | 仍需独立头尾证据，工程环节更多 | 备选；既有 V4.1 OBB 对照要求不被本方案删除 |
| 多关键点 + PnP | 后续单目 6D 候选 | 可信尺度 CAD、相机标定、足够可见且不共线的对应点；需处理多解/对称性 | S4 条件满足后评估 [R4] |
| FoundationPose | 后续 RGB-D 6D 参考 | 官方 demo 使用 RGB、depth、K、对象 mesh 与初始 mask；并非仅替换普通 RGB 模型即可使用 | 不作为当前第一批训练路线 [R5] |

YOLO 的人体关键点预训练权重仅作为初始化来源；自定义螺丝关键点头需要训练。COCO 的官方 mAP、延迟及默认 OKS 容差不等于本项目精度。安装版对非人体关键点会采用通用 OKS 设置，因此主结果使用明确的头尾错误、像素误差、角度误差和拒绝覆盖率，Pose AP 只作辅助诊断。

## 4. 仿真平台与 Windows 安装条件

| 平台 | 已核实能力/条件 | 本轮取舍 |
| --- | --- | --- |
| MuJoCo | 原生 Python 包包含引擎与 viewer；官方 PyPI 当前 3.17.0 有 `cp311-cp311-win_amd64` wheel [R6][R7] | 推荐先验证，版本候选 3.17.0；wheel 存在不代表本机渲染已通过 |
| PyBullet | 支持动力学、运动学、碰撞及 CPU/OpenGL 渲染；官方 PyPI 当前 3.2.7 发布文件没有 Windows wheel，返回的历史文件索引也未找到 CPython 3.11 Windows wheel [R8][R9] | 保留备选；需评估源码构建或另一隔离环境，不能承诺本机一条 pip 命令即可安装 |
| BlenderProc | 离线材质、灯光、相机随机化，RGB/depth/segmentation 与 BOP 输出；运行于 Blender Python [R10] | 需要更真实外观时加入独立生成进程；不直接当作机器人策略训练平台 |
| Isaac Sim | 核查的 `latest` 要求页列最低 32GB RAM / 16GB VRAM，参考 GPU 为 RTX 4080 [R11] | 本机 8GB 显存低于该表，不作为当前默认平台；不据此声称所有旧版本或小场景绝对不能运行 |

因此，本轮建议把 ADR-0002 中“PyBullet 优先评估、MuJoCo 备选”的调研顺序调整为 **本机先验证 MuJoCo，保留 PyBullet**。这是待确认的实施建议，尚未修改已确认方向或安装引擎。

虚拟机械臂候选为 MuJoCo Menagerie 的 Franka Emika Panda：官方包提供关节与夹爪模型，说明模型采用 Apache-2.0。[R12] 实施时固定资产 commit、文件哈希与许可；螺丝尺寸、夹爪接触面、碰撞几何、摩擦和时间步必须匹配，不能为演示成功随意放大零件或使用隐藏吸附。

## 5. 数据来源和标注可用性

| 来源 | 可用于什么 | 尚缺什么 |
| --- | --- | --- |
| 自建参数化螺丝代理 | 验证投影、标注、训练、推理和受约束仿真链路；来源与尺寸可完整记录 | 不对应已确认实物；简化螺纹、材质和接触模型会造成域差异 |
| 具体型号的合法 CAD + 实物 | 定义真实尺度、轴与头尾；后续生成与真实验证 | 型号、文件、许可、实测尺寸和采集会话尚待取得 |
| MVTec Screws | 公开 OBB 检测参考 | 官方公布 384 张、13 类、4426 个 OBB，未提供本任务两个语义关键点或 6D 真值；CC BY-NC-SA 4.0 [R13] |
| 历史瓶盖数据/权重 | 追溯既有原型 | 不加入新螺丝训练或冒充工业件验证 |

仓库 [MVTec V1 审计](../data/MVTEC_SCREWS_V1_AUDIT.md)还记录了镜像数量差异、方向修正和越界问题；本轮没有重新下载全量数据或解决这些问题。不能把公共 OBB 四角解释成头尾关键点，也不能因本次选择螺丝就自动打开旧数据训练门禁。

合成数据应从几何投影得到标签并记录可见性；RGB 中不绘制彩色头尾点、坐标轴、类别文字或 GT 轮廓。可视化覆盖层单独输出，防止模型学习人工标签标记。关键点被遮挡但由仿真已知时记录“已知、不可见”，不能假装是人工可见的可靠真实真值。

## 6. 评价与闭环的关键约束

1. 图像坐标 `x` 向右、`y` 向下；有向轴固定为 **尾端 → 头端**。角度以右向为零，在图像中顺时针增加，范围 `[0°,360°)`。错误 180° 必须保留，不使用绝对点积或模 180° 误差掩盖头尾颠倒。
2. 预测不可用、漏检和拒绝要进入计数报告，不能仅在成功检测的容易样本上报告漂亮角度。GT 不可判断与模型拒绝分开。
3. 同 episode、同源帧和增强图归同一 split；分别报告同分布独立场景与未见工况，不能仅换随机 seed 就宣称跨域泛化。
4. Test / Final Holdout 不用于选模型或阈值；首次开发训练只使用 Train / Val。比例给 `x/n` 与 Wilson 95% CI；有关联帧时以独立 episode 为统计单位或给出适当聚类统计。
5. 首轮机械臂只能在明确的固定相机、支撑面和已知几何约束下做 2.5D 抓取。图像两点不能直接输出不受约束的空间姿态。
6. 预测驱动规划器只读预测与已冻结标定/几何先验，GT 独立供参考组与评估器使用。两组共享场景、初始状态、控制器和预算。
7. 抓取必须评价接触、提起保持、搬运、释放与落点；仅末端到达不算成功。先确认 GT 驱动组可行，再分析视觉误差传播。
8. 6D 扩展需要声明对象对称性；可参考 BOP 的 VSD / MSSD / MSPD，而非强制对不可观察的绕轴角评分。[R14]
9. MuJoCo/OpenGL 与 OpenCV 的相机方向、四元数顺序、米/毫米均须显式转换并做投影复核。BlenderProc 文档同样明确区分 OpenGL 与 OpenCV 相机坐标。[R15]

## 7. 许可与实施前仍需核实的事项

引擎、机器人资产、零件 CAD、纹理、数据与权重分别记录许可，不以“开源平台”推断所有资产均可自由分发。MuJoCo 代码为 Apache-2.0；PyBullet 代码主要为 zlib 且存在文件级例外；Ultralytics 提供 AGPL-3.0 和 Enterprise 许可路线。[R8][R16][R17] 本轮只记录来源，不改变仓库许可或作商业授权结论。

下一步确认[首轮实施方案](../exec-plans/active/screw_orientation_v1_implementation_plan.md)。需要补齐的真实域输入是：具体螺丝型号与尺寸、实物/CAD 可得性、目标动作、相机方式及真实工艺容差；缺少时可以先做明确标记的仿真域开发，真实门禁仍 CLOSED。

## 8. 一手来源

以下链接均在 2026-10-31 核查；动态 `latest` 页面与包版本须在安装时再次固定。

- [R1 — Ultralytics OBB](https://docs.ultralytics.com/tasks/obb/)
- [R2 — Ultralytics Pose](https://docs.ultralytics.com/tasks/pose/)
- [R3 — 自定义 Pose 数据格式](https://docs.ultralytics.com/datasets/pose/)
- [R4 — OpenCV PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
- [R5 — FoundationPose 官方 demo](https://github.com/NVlabs/FoundationPose/blob/main/run_demo.py)
- [R6 — MuJoCo Python 文档](https://mujoco.readthedocs.io/en/stable/python.html)
- [R7 — MuJoCo 官方 PyPI 元数据](https://pypi.org/pypi/mujoco/json)
- [R8 — Bullet / PyBullet 官方仓库](https://github.com/bulletphysics/bullet3)
- [R9 — PyBullet 官方 PyPI 元数据](https://pypi.org/pypi/pybullet/json)
- [R10 — BlenderProc 文档](https://dlr-rm.github.io/BlenderProc/)
- [R11 — Isaac Sim 系统要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [R12 — Menagerie Panda 说明与许可入口](https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/README.md)
- [R13 — MVTec Screws 官方页面](https://www.mvtec.com/research-teaching/datasets/mvtec-screws)
- [R14 — BOP 任务与对称性评价](https://bop.felk.cvut.cz/tasks/)
- [R15 — BlenderProc 相机坐标](https://dlr-rm.github.io/BlenderProc/docs/tutorials/camera.html)
- [R16 — MuJoCo LICENSE](https://github.com/google-deepmind/mujoco/blob/main/LICENSE)
- [R17 — Ultralytics 许可说明](https://www.ultralytics.com/license)
