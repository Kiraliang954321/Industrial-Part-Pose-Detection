# Industrial-Part-Pose-Detection

**FlexPose-YOLO — 面向柔性流水线的低成本工业零件正反面与姿态检测研究原型**

当前主线已推进到 **A5 视觉摆正与键／销实体配合仿真**。首次接手请从[成果总览与运行指南](docs/handoff/README.md)开始，查看每阶段完成内容、指标分母、环境、模型材料和下一位 Agent 的阅读顺序。

**克隆后即可预览。** 仓库附带[九段演示目录](docs/previews/README.md)、可离线打开的[预览页](docs/previews/index.html)及[原始评价汇总](docs/handoff/evidence/)。在根目录运行 `py -3 scripts/verify_handoff.py` 可校验公开材料和冻结源码。完整视觉重放所需的权重与实验归档见接手指南。

**A5 实体键／销配合已完成（2026-11-03）。** 齿轮键槽配带键轴，轴承外缺口配定位销，连接视觉摆正、分段预插入、腕部力／力矩反馈停止与带件退回。冻结后 66 次新抓取中，正常工况 **34/36 完整成功**（齿轮 18/18、轴承 16/18），故意错位工装 **12/12 实际接触停止并退回**，意外有效进入 0/12。两个正常留件均为轴承几何无解，成功插入深度的上沿仍接近 4.2 mm 限值。A5 复用已有视觉权重，能力范围为有间隙、保持夹持的代理仿真。真实门禁 CLOSED。完整分母、置信区间与物理证据见 [A5 记录](docs/exec-plans/active/keyed_cell_a5.md)。

A5 本地成果包含 [30 s 原始运行剪辑](runs/keyed-cell/20261101-demo-v1/a5-keyed-mating.mp4)与[腕部反馈对照图](runs/keyed-cell/20261101-force-plot-v1/wrist-contact-comparison.png)。四段依次展示轴承正确／工装错位、齿轮正确／工装错位，片尾结果卡和右上物理俯视示意已标注用途。

**阶段收尾（2026-10-31）：**螺丝阶段已按用户要求收尾，H1 简单流程作为主展示，H2 作为扩展保留；后续转向可判正反的非对称轴承和锥形齿轮。见[阶段交付清单](docs/exec-plans/completed/screw_stage_completion_20261031.md)。

**轴承/齿轮 A1 已完成首轮仿真开发：**以 F608ZZ 单侧法兰与 SB1-2040 锥齿面/轮毂为方向特征，训练单相机六类 YOLO 模型，连接夹取、判定和有间隙套轴预定位。冻结后两对象各 20 次新抓取，共 8 次放行预定位全部成功，16 次反向/倾斜拒绝，16 次空抓/黑屏留件；真实门禁仍 CLOSED。见 [A1 记录](docs/exec-plans/active/bearing_bevel_a1.md)。41.83 s 视频位于 `runs/annular/20261031-demo-v1/bearing-bevel-inspection.mp4`。

**A2 连续角度初版（2026-11-01）：**用户要求指定空间方向及绕轴角后，新增定制外圈缺口/齿轮键槽代理、三 RGB 视角、关键点与孔口几何融合，可输出旋转与目标误差。冻结后 48 次评价中，有定位结构的 40 次里 22 次可测（55%）；可测样本平均轴向/绕轴误差约 1.80°/1.19°，但尚未达到稳定全覆盖，也未连接自动摆正。见 [A2 记录](docs/exec-plans/active/datum_pose_a2.md)及[目标配置](docs/data/datum_pose_targets_v1.json)。

**A3 大倾角/翻转自动摆正（2026-11-01）：**已实现侧夹、六 RGB、中心关键点训练、双端面几何与外形校核，连接“预测摆正 → 复拍 → 原竖轴预定位”。[冻结协议](docs/data/pose_cell_a3_protocol_v1.json)后 48 次新抓取为 **25 完整成功 / 20 留件 / 3 齿轮深度超限**；正常图像、有定位结构的总体成功为 **25/36（69.4%）**，实际预定位为 25/28。180° 翻转和反向上料各 6/6 完整成功，90°/120° 各 2/6，不能称任意姿态稳定覆盖。数据、模型、分母、置信区间和限制见 [A3 记录](docs/exec-plans/active/pose_cell_a3.md)。仍是定制代理、有间隙预定位，真实门禁 CLOSED。

A3 本地演示：`runs/pose-cell/20261101-demo-export-v1/a3-auto-orientation.mp4`（30.46 s；两段通过检测 RGB/物理终态一致性校验的 3 倍速动作回放，另含原始留件图像证据）。

**A4 有限主动观察（2026-11-01）：**已完成困难姿态诊断、2,328/780 图的分组数据与 10 轮补训，增加最多两个观察姿态和持件一致性检查。[A4 冻结协议](docs/data/active_pose_a4_protocol_v1.json)后 54 次新抓取中，正常工况 **35/36 完整成功**；初检无解的 **4/5** 通过观察恢复并完成预定位，90° 为 5/6、120° 为 6/6。黑屏/空抓未观察转腕，无定位结构观察至上限后均留件。补训初检覆盖收益在不同批次不稳定，不能把本轮成绩全部归功于新模型，详见 [A4 记录](docs/exec-plans/active/active_pose_a4.md)。29.46 s 原始运行视频：`runs/active-pose/20261101-demo-v1/a4-active-inspection.mp4`。仍为代理仿真，真实门禁 CLOSED。

本项目研究如何使用普通 RGB 相机、消费级 GPU、Traditional CV 与轻量 YOLO OBB，在低成本条件下完成工业零件的检测、正反面判定、可适用的二维朝向判断，以及跨零件换型验证。

后续研究方向已更新为**轴承、锥形齿轮或螺丝的朝向检测，以及条件具备时的 3D/6D 姿态与虚拟机械臂验证**，见 [ADR-0002](docs/decisions/adr-0002-industrial-orientation-3d-simulation.md)。先明确可观察方向，再生成带真值的仿真数据并验证视觉引导动作；历史原型不代表这些工业件的已验证能力。

**H1 历史阶段为夹持后检测与螺丝对孔预插入。** 已按用户选择实现 [H1](docs/exec-plans/active/held_inspection_h1.md)：物理夹起 → 固定检测位双 RGB 复核 → 单次许可与预测补偿 → 对孔预插入并保持夹持。独立夹持模型已完成开发微调；数据、训练、冻结协议及评价证据见 H1 记录。它不等于真实锁付或投产验收，现有来料分拣成绩也不计入 H1。工业依据见[工艺调研](docs/design-docs/held_part_inspection_assembly_research_20261031.md)与 [ADR-0003](docs/decisions/adr-0003-held-part-inspection-before-assembly.md)。

H1 冻结后 24 次新抓取：11 视觉 PASS / 6 REJECT / 7 UNCERTAIN；3 次 PASS 因许可过期留件，实际预插入 **8/8 成功**，非合规件误放行 **0/9**，另有 1 次普通正常件被保守拒绝。样本小且故障比例人为加重，不表示生产可靠性。51.67 s 三分支演示位于 `runs/screw-orientation/20261031-h1-demo-v1/h1-held-inspection-preinsert.mp4`；原始数据和视频为本地产物，未提交到 Git。

**H2 历史阶段为[插入前滑移复检](docs/exec-plans/active/held_recheck_h2.md)**：在孔上方增加独立 RGB 复检及短时插入许可，使用 Train 拟合的有界端点补偿。冻结后 36 次新抓取中，正常流程预插入 **9/9**；6 次仍夹持但实际位姿已变的场景全部在插入前留件。新批同图初检对照 H1/H2 均为 24/24，尚不能宣称独立评价证明拒绝率降低。62.42 s 演示：`runs/screw-orientation/20261031-h2-demo-v1/h2-recheck-preinsert.mp4`。四路固定相机与宽松代理孔仍是仿真条件。

> 既有 **V4.1a / Phase 1** 的正式协议与数据门禁尚未通过；**螺丝代理视觉训练与 Panda 2.5D 抓取仿真闭环已完成**，真实型号/CAD 和工况仍待 S0 定义。

已新增 [L1 分度式单工位用途验证](docs/exec-plans/active/screw_line_simulation_v1.md)：物理托盘到位、多帧检测、合格放行、方向不合格分拣和无法判断时停机留件。24 个新事件产生 8 PASS / 8 FAIL_ORIENTATION / 8 UNCERTAIN，含可播放 3D 演示；目前是独立单件工位，不是无间断整线。
> 这是研究与课程原型，不是生产安全联锁或实体设备控制系统。任何精度、延迟和吞吐结论都必须来自冻结协议下的正式实验，不能用预训练模型输出或软件烟测代替。

## 当前状态

项目权威文档：

- [V4.1 项目方案](docs/new-plans/面向柔性流水线的低成本工业零件姿态检测系统_项目方案_V4.1_优化版.md)
- [EXPERIMENT_PROTOCOL_V1](docs/new-plans/EXPERIMENT_PROTOCOL_V1.md)
- [项目范围与验收](docs/product-specs/project_scope.md)
- [Phase 1 执行计划](docs/exec-plans/active/phase1_execution_plan.md)
- [工业件朝向与 3D 研究执行计划](docs/exec-plans/active/industrial_orientation_3d_execution_plan.md)：首轮螺丝代理仿真开发已完成，真实对象和正式数据/验收协议仍待定义。
- [螺丝朝向 V1 开发记录](docs/exec-plans/active/screw_orientation_v1_development.md)：MuJoCo 数据生成、Traditional CV / YOLO Pose、有向轴评价与开发训练的实际状态；使用见[数据契约](docs/data/screw_orientation_v1.md)。
- [M4 Panda 仿真闭环](docs/exec-plans/active/screw_m4_closed_loop.md)：冻结后 24 个配对场景、48 次物理执行，GT/视觉各 24/24 成功，含物理结果与视频；不代表真实机械臂验收或完整 6D。

当前已完成的软件基础：

- part-aware 手机 / iVCam 原始图像采集；
- Front / Back 标签契约；
- OBB sidecar 标注工作流；
- OpenCV 四点击标注工具；
- 标注 QA 与来源完整性检查；
- 按完整 Session 的四路 Dataset V1 组装：
  - Train
  - Val / Calibration
  - Test
  - Final Holdout
- 圆形 / 近圆形零件的 V4.1a 姿态语义协议；
- 手工采集时明确显示 `SAVED` / `REJECTED`，避免重复过滤看起来像按键失效。

历史二维开发对象为 `cap_a`（黄色圆形瓶盖），仅保留为原型证据：

- Front：黄色外表面、红色文字 / 图案；
- Back：内凹结构面、内部环形结构 / 二维码；
- `orientation_semantics = NOT_APPLICABLE`，因为该零件外轮廓近圆且没有稳定的工业方向特征；
- 已使用手机 + iVCam 做真实采集验证；
- 历史采集验证使用 iVCam `source=1`、DirectShow、640×480、请求 30 FPS；
- 本批实拍 8 张（Front 4 / Back 4）已完成人工 GUI 接受与标注 QA：8 接受、0 排除、0 缺失、0 错误；Part B 6 张仍为候选参考，完整 Dataset Audit 尚未完成；
- 2026-10-06 新增黄色瓶盖实拍 80 张（Front 40 / Back 40），已保真归档并完成助手逐图可视复核；这批作为 Train，原人工接受的 8 张作为开发 Val，已完成真实照片开发微调：13 epochs，AP 最佳 best 权重的开发 Val mAP50–95 为 0.87025，见[80 张真实照片开发微调](docs/exec-plans/active/real_cap_80_development.md)；
- 开发推理组合使用 last 权重与显式黄色像素过滤；其固定阈值回放为已见 Train 80 / 80、开发 Val 8 / 8 单个正确框、0 额外框，是组合感知的开发观察，不是纯 YOLO AP、风险拒绝或无偏测试结果；
- 正式四路 Session 尚未建立，因此 **Training Gate 仍为 CLOSED**。

后续正式对象/型号与完整 Operating Envelope 仍需冻结；不再把灰白瓶盖候选作为下一次换型对象。历史开发微调已完成，正式训练和 Phase 2 公平性能对比尚未开始。80 张是该轮实拍预算，先前 200 张建议不作为门槛；数据增强仅作用于 Train，不增加独立真实批次。

## 研究问题

1. **Traditional CV vs YOLO26n-OBB / YOLO26s-OBB**
   AI 是否真的比传统视觉更适合当前工业姿态任务？

2. **Single Frame vs Event Decision**
   Tracking + 多帧 + Decision Zone 是否能降低单帧抖动和错误决策？

3. **Risk-Coverage-Latency-Throughput**
   `UNCERTAIN` 能降低多少 Wrong Accept？代价是多少 Coverage、延迟和有效吞吐？

4. **Part A → Part B 柔性换型**
   换一个零件究竟需要多少新图像、标注、训练时间、Recipe 改动和代码改动？

## V4.1a 姿态语义

不同工业零件不能强制使用同一种角度定义。当前协议支持：

| 语义 | 适用对象 | 角度来源 |
| --- | --- | --- |
| `AXIAL_OBB` | 长条、矩形等具有稳定长轴的零件 | OBB 长轴 |
| `FEATURE_ANGLE` | 圆形但具有 Logo、缺口、定位特征的零件 | 中心 → 冻结特征 |
| `SYMMETRY_AWARE` | 法兰、齿轮等周期对称零件 | 按声明的物理对称周期评价 |
| `NOT_APPLICABLE` | 工艺未要求方向的具体圆形件，不按“轴承”等类别一概指定 | N/A |

`NOT_APPLICABLE` 不等于 `UNCERTAIN`。如果一个零件本来就不要求旋转方向，缺少 Angle 不得单独触发 `FAIL_ORIENTATION` 或 `UNCERTAIN`。

以上是既有二维语义。3D 平移、旋转、坐标系和对称性感知评价需另行定义，不能由 OBB 图像角直接推导；虚拟机械臂数据生成、规划验证和动作策略学习也须分别记录，详见 ADR-0002。

## Dataset V1 原则

正式数据按**完整独立 Session**划分：

```text
Train Session(s)
Val / Calibration Session(s)
Test Session(s)
Final Holdout Session(s)
```

禁止把一段连续录像随机逐帧拆到多个 split 中。

- Train：训练 YOLO 或拟合 Traditional CV 参数；
- Val / Calibration：模型、阈值、Risk Gate 工作点选择；
- Test：冻结后的正式评估，不调参；
- Final Holdout：Hard Sample 回流或二次训练后的最终未见数据验证，不调参。

`UNCERTAIN` 是运行时决策，不是训练类别，也不是 Ground Truth 类别。

## 本地开发

要求 Python >= 3.10。

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,inference]"
python -m pytest -q
```

采集：

```powershell
python scripts/capture_phone.py --part-id cap_a --source 1 --backend dshow --width 640 --height 480 --fps 30
```

标注与 QA：

```powershell
python scripts/annotate_obb.py data/captures/<capture-session-id>
python scripts/qa_annotations.py data/annotations/<capture-session-id>
```

Dataset V1：

```powershell
python scripts/build_project_dataset.py --help
python scripts/audit_dataset.py --help
```

更多说明：

- [Part A / B 实拍预冻结草案](docs/data/part_ab_photo_prefreeze_v1.md)：保留原 14 张素材及 Part A 8 张人工接受记录，并记录新 80 张开发来源；正式门禁还缺独立真实四路数据及 Operating Envelope、GT 与合格规则的冻结。
- [80 张真实照片开发微调](docs/exec-plans/active/real_cap_80_development.md)：新增 80 张 Part A 实拍作为 Train，旧 8 张作为开发 Val，按完整批次隔离；记录助手标注来源、数据增强与已完成的开发微调结果，正式门禁保持 CLOSED。
- [独立合成瓶盖训练演示](docs/exec-plans/active/synthetic_cap_training_demo.md)：已完成 1,440 张合成图像、12 段视频、双 OBB 模型训练与预测演示；结果限定为 `synthetic_only`，不改变正式 Training Gate。
- [PHONE_CAPTURE_V1](docs/data/PHONE_CAPTURE_V1.md)
- [OBB Annotation Sidecar V1](docs/data/obb_annotation_v1.md)
- [Project Dataset V1](docs/data/project_dataset_v1.md)
- [Architecture](ARCHITECTURE.md)

## 想一起做这个项目？

欢迎参与。当前特别需要帮助的方向包括：

- 工业零件数据采集与可复现实验设计；
- OBB / 圆形件 / 对称零件姿态语义；
- Traditional CV baseline；
- YOLO OBB 训练与公平对比；
- Event Decision、Tracking、多帧稳定性；
- Dataset / QA / 数据泄漏检查；
- GUI / Recipe Wizard / 新零件引导注册；
- 测试、文档和可复现性改进。

开始之前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，优先通过 Issue 讨论较大的设计变更。欢迎提交 Issue 和 Pull Request。

## 数据与隐私

仓库默认**不提交**：

- `data/` 原始采集数据；
- `runs/` 实验输出；
- `*.pt` 模型权重；
- `.venv/`；
- 压缩数据集；
- 本地密钥 / `.env` 文件。

需要共享实验数据时，请只提交明确获准公开、已经审查来源和隐私的材料。

## 重要声明

当前项目没有已验证的生产级精度、FPS、吞吐或安全结论。
95% 面准确率、90% mAP50、30/60 FPS 等数字如果出现在设计文档中，只能视为未来目标或实验门槛，不能作为已取得成果。
