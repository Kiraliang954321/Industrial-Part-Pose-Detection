# M4：螺丝代理的 Panda 仿真闭环

日期：2026-10-31。用户在 M0–M3 交付后要求继续完成，并确认“暂无实物/CAD，先完成仿真闭环”。本轮按既定 M4 路线实施；M5 真实验证不在本轮补造数据或指标。

状态：**M4 仿真开发闭环已完成：6 个控制开发场景 + 冻结后 24 个新场景，共 48 次配对评价执行；GT 与视觉组均 24/24 成功。**所有数字是仿真开发设定或该设置下的测量，不是实际工艺容差或实体机械臂验收。

## 固定设计

- MuJoCo 3.17.0，沿用 `screw_proxy_v0` 的原始尺度与几何。
- Panda 候选来自 MuJoCo Menagerie，固定 commit `a662de603224c47123b0783821c58ad4402ad3e3`；保存原始文件、SHA-256 与 Apache-2.0 许可。
- 固定外部 RGB 相机，复用已训练 `best.pt`。学习模型不再调参；原 M3 Train/Val 不参与新的控制器验收计数。
- 对照为 `gt_xy_heading_reference` 与 `vision_prediction`：前者获得真实平面中心/方向，后者仅从 RGB 的两个关键点和固定相机/支撑面/资产高度先验计算。两者使用相同固定抓取高度、IK、轨迹、夹爪和初始状态。
- 世界/机器人基座/相机变换显式记录。两点不变成完整 6D；对称夹爪允许等价抓取分支，但放置后的螺丝头端仍须指向目标有向角。
- 路线为预抓取、下降、闭合、提起、保持、搬运、下降、释放、撤离与落稳；通过关节执行器控制，只有 episode 初始化允许设置对象位姿。
- 6 mm 杆径相对原 Panda 指垫较小。可添加明确记录的薄指尖与调整夹爪执行器接触预载；不缩放零件、不隐藏关闭碰撞、不加吸附/weld、不在执行中直接修改对象位姿。
- 运动学预检检查关节限位、机器人与固定支撑面/自身碰撞；预测组不能通过读取对象真值来修正抓取目标。实际对象接触与成功判定只由物理引擎和独立评估器提供。

## 校准与评价顺序

1. 使用单列的控制开发场景验证 IK、薄指尖接触和 GT 驱动可行性。
2. 在控制开发场景确认视觉投影及模型输入，排查相机到基座的符号/单位问题。
3. 保存实际控制配置、资产/权重/代码哈希后，再运行新的配对场景；同一场景两组初始化状态一致。
4. 首轮目标为 24 个新场景、每场景两个分支。不同朝向、位置与有限照明/背景变化均记录；本轮不把 M3 的视觉悬浮遮挡物当成已实现的实体避障场景。
5. 提起/保持、双侧夹持接触、放置误差、头尾方向、掉落、非法碰撞、不可达和拒绝分别计数。成功不能仅由末端到达位置判断。
6. 比例报告 x/n 与 Wilson 95% CI，给出配对结果、失败归因和物理引擎实际积分过程的视频。所有新场景都保持 `simulation_closed_loop_development`，不开放真实正式门禁。

## 已实现模块

- `simulation/robot_assets.py`：固定来源、许可与本地资产审计。
- `simulation/panda_scene.py`：原 Panda 模型与螺丝场景组合、相机和明确的指尖适配。
- `simulation/grasp_geometry.py`：像素到支撑面坐标转换，明确的有向抓取/放置目标。
- `simulation/panda_control.py`：限位约束 IK、轨迹预检、执行器跟踪。
- `simulation/closed_loop.py`：配对运行、独立物理结果评分与视频/报告。
- `simulation/grasp_evaluation.py`：只读物理结果判定、逐阶段统计与轨迹记录。
- `simulation/m4_protocol.py`：绑定校准证据、控制代码/资产/权重哈希和独立场景种子。
- CLI 入口、必要文档与 M4 产物索引。遵循既有测试政策，优先编译、聚焦边界检查与真实仿真烟测，不额外扩张软件测试套件。

## 最终控制与接触设置

| 项目 | 冻结设置 |
| --- | --- |
| 机器人基座 | 世界坐标 `(-0.45,0,0)` m，基座轴与世界轴平行 |
| 感知输入 | 固定 640×640 RGB，相机 K/外参显式保存；单次观测后执行 |
| 高度先验 | 头端 6 mm、尾端 3 mm、抓取 TCP 4.5 mm，均为固定代理/支撑面先验 |
| 末端/指尖 | TCP 位于 hand 的 `(0,0,0.115)` m；新增每侧 16×4×4 mm 薄指尖 |
| 夹爪 | tendon 位置控制，kp 1500、kv 10、力限 ±10 N；开闭使用平滑斜坡，闭合前增加稳定段 |
| 重力 | 对象正常受重力；仅机器人模型启用重力补偿，XML 编译时明确配置 |
| 物理积分 | 1 ms，implicitfast；Newton，elliptic friction cone，impratio 10，tolerance 1e-10 |
| 接触 | solref `0.003 1`；指尖摩擦 `1 0.005 0.0001`；未启用 NoSlip 求解器 |
| 动作 | 预抓取→下降→稳定→闭合→提起→保持→搬运→下降→释放→撤离→落稳 |
| 目标 | 世界 XY `(0.13,0.04)` m，螺丝头端朝世界 +X；释放 TCP 高度 10 mm |
| 规划 | 限位约束阻尼最小二乘 IK，候选关节路径采样检查，最大关节速度设定 1.2 rad/s |

早期控制开发记录保留在 `runs/screw-orientation/20261031-m4-gt-probe-v1` 至 `v6`：定位并修复了仅修改运行时数组却未编译启用重力补偿的问题；改进薄指尖/闭合预载与斜坡；针对软接触滑移采用官方建议的 elliptic cone + impratio；随后加入落台穿透质量约束和闭合前稳定段。中途一次结果 JSON 的 NumPy 布尔序列化错误已修复，未把该次缺少结果文件的运行计入成功率。

软接触滑移的设置依据见 [MuJoCo 3.17 建模说明](https://mujoco.readthedocs.io/en/3.17.0/modeling.html)。这些设置没有引入吸附、对象 weld 或执行中的对象位姿重置；物理摩擦/刚度和适配指尖仍未经实物标定，不能视为原厂 Panda 夹爪已获真实验证。模型来源与许可见 [Menagerie Panda](https://github.com/google-deepmind/mujoco_menagerie/blob/a662de603224c47123b0783821c58ad4402ad3e3/franka_emika_panda/README.md)。

## 冻结、数据隔离与成功条件

控制开发使用 seeds `9001–9006`，GT 和视觉各 6/6 成功。随后建立[冻结的 M4 开发协议](../../data/screw_m4_protocol_v1.json)，状态仅为 `frozen_development_only`，不是项目真实实验协议 FROZEN。

新的配对评价使用 seeds `2026103104–2026103127`，与控制开发不重叠；四种有限工况 nominal / low_light / reflection / background_shift 各 6 个场景。每个场景 GT 与视觉分支恢复完全相同的积分状态，状态哈希逐对一致。权重保持 M3 best 的 `0a29b8c5660d62cb19b84bf2cc0805e1c3b3ed2fb2e7478dc80813ab2f1e155a`，没有再训练或在这些结果上调参。

成功同时要求：闭合末段双侧接触覆盖 ≥80%；提起至少 40 mm；保持和搬运期间持续高于该高度且双侧接触覆盖 ≥80%；释放后落在支撑面且不接触夹爪；落点误差 ≤15 mm；有向头尾角误差 ≤10°；速度降到设定范围；无非法碰撞；最大软接触穿透 ≤1 mm。正常物体/支撑面/夹爪接触允许并记录；浅的原生指间闭合限位接触单列为允许接触，不将其混为机械臂碰台或非预期自碰撞。

这些门槛在新场景运行前已冻结。视觉组的规划实现只读取预测观测与机器人关节状态；IK 工作副本从模型初始状态构建，不复制当前对象真值。对象真值用于 GT 参考观测、误差分析与只读结果评分。当前是**单次视觉观测 + 关节反馈执行 + 结果判定**，不是运动中的视觉伺服，也没有训练动作策略。

## 新场景评价结果

| 指标 | GT 平面中心/方向参考 | 视觉预测驱动 |
| --- | --- | --- |
| 完整任务成功 | 24/24，100% | 24/24，100% |
| Wilson 95% CI | 86.20%–100% | 86.20%–100% |
| 规划/感知拒绝 | 0/24 | 0/24 |
| 双侧夹持、提起、保持、搬运、释放、落稳 | 各 24/24 | 各 24/24 |
| 非法碰撞 | 0/24 | 0/24 |
| 落点 XY 误差 Median / P95 | 0.504 / 0.860 mm | 0.643 / 0.983 mm |
| 放置有向角误差 Median / P95 | 0.492° / 0.849° | 0.495° / 0.851° |
| 最大软接触穿透 | 0.512 mm | 0.434 mm |

配对结果：两组均成功 24，对照独有成功 0，视觉独有成功 0，两组均失败 0。每种工况两组均为 6/6；子组样本量小，不单独外推稳定性。

视觉观测相对独立真值的平面中心误差 Median/P95 为 0.308/0.744 mm，有向角误差为 0.291°/1.054°。两组都能完成这一受约束代理任务；不能据此认定真实感知误差对抓取没有影响。成功率的样本区间较宽，且所有数据都来自同一个代理、固定相机和既定接触模型。

评价实际用时 141.23 秒（包含部分录像渲染）；每次动作的仿真时间约 10.70–16.877 秒。这不是实机节拍或 Validated Event Throughput。逐步记录的最大对象位置变化约 0.360 mm/step，没有执行中位置跳变。

## 产物与核验

- [完整配对报告](../../../runs/screw-orientation/20261031-m4-paired-v1/report.json)、[摘要](../../../runs/screw-orientation/20261031-m4-paired-v1/summary.json)、[运行清单](../../../runs/screw-orientation/20261031-m4-paired-v1/run-manifest.json)。
- [控制开发报告](../../../runs/screw-orientation/20261031-m4-calibration-v1/report.json)、[冻结协议](../../data/screw_m4_protocol_v1.json)。
- [相机/世界/机器人基座变换](../../../runs/screw-orientation/20261031-m4-paired-v1/calibration-transforms.json)：IK 实际使用世界坐标，额外给出等价的基座矩阵供复核。
- [视觉驱动 H.264 视频](../../../runs/screw-orientation/20261031-m4-paired-v1/episode-2026103104/vision-h264.mp4)、[同场景 GT 参考视频](../../../runs/screw-orientation/20261031-m4-paired-v1/episode-2026103104/gt-h264.mp4)。这是实际积分过程的仿真录像；首两个场景的两个分支共 4 个视频，预先由 `record_first=2` 选定，不是从结果中挑选成功案例。
- [动作过程截图](../../../runs/screw-orientation/20261031-m4-paired-v1/vision-contact-sheet.jpg)、[视频解码记录](../../../runs/screw-orientation/20261031-m4-paired-v1/video-verification.json)。4 个 H.264 视频分别 147/147/196/196 帧，全部解码通过。
- [物理证据复核](../../../runs/screw-orientation/20261031-m4-paired-v1/physical-verification.json)：48 次轨迹、配对初始状态、输入图像哈希、物体重力、碰撞几何和约束类型均已检查；唯一 equality 是两个手指关节的耦合，mocap body 为 0，NoSlip 为 0。

每个 `episode-{seed}/` 下保存原始传感图、模型覆盖图、输入观测、分支规划/评分、GT/vision 各自的 JSONL 轨迹和终态图。原始资产另存 `data/robot-assets/panda-a662de6`，共 70 个源文件、34320232 bytes，另有本地清单，并保留 README/许可。大型模型、资产与运行产物遵循现有忽略规则留在本机。

原 M3 场景的一个已记录样本按原 seed 重新渲染，PNG 哈希完全一致；新增可选组合场景与初始化 hook 未改变该默认生成路径。像素→支撑面投影、旋转右手性和 180° 有向区别的定向检查通过，冻结协议在评价后重新验证通过。未新增无关测试套件，未提交或推送 Git。

## 复现入口

使用现有已审计资产和权重，指定不存在的新输出目录：

```powershell
.\.venv\Scripts\python.exe scripts/run_screw_closed_loop.py --assets data/robot-assets/panda-a662de6 --weights runs/screw-orientation/20261031-development-train-v2/fit/weights/best.pt --output runs/screw-orientation/m4-reproduction --role paired_evaluation --protocol docs/data/screw_m4_protocol_v1.json --seed 2026103104 --episodes 24 --record-first 2
```

如修改控制代码、接触参数或模型，当前协议会拒绝漂移；应重新做控制开发并生成新版本协议/新评价种子，不改写本轮成绩。`prepare_panda_assets.py` 用于获取固定版本资产，`freeze_screw_closed_loop.py` 用于有合格校准证据后的协议冻结。

## 保留边界

M5 按用户“暂无实物/CAD，先完成仿真”的选择保留待办。尚无真实螺丝数据/相机标定、完整 6D 视觉估计、执行中视觉重规划、动作策略训练或实体执行器接入。当前 2.5D 结果和接触参数不替代真实 Operating Envelope、制造合规 GT 或正式训练/验收门禁。
