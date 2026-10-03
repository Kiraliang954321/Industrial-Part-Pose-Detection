# Industrial-Part-Pose-Detection

**FlexPose-YOLO — 面向柔性流水线的低成本工业零件正反面与姿态检测研究原型**

本项目研究如何使用普通 RGB 相机、消费级 GPU、Traditional CV 与轻量 YOLO OBB，在低成本条件下完成工业零件的检测、正反面判定、可适用的二维朝向判断，以及跨零件换型验证。

> 当前处于 **V4.1a / Phase 1 — Data Foundation + Experiment Protocol Freeze**。
> 这是研究与课程原型，不是生产安全联锁或设备控制系统。任何精度、延迟和吞吐结论都必须来自冻结协议下的正式实验，不能用预训练模型输出或软件烟测代替。

## 当前状态

项目权威文档：

- [V4.1 项目方案](docs/new-plans/面向柔性流水线的低成本工业零件姿态检测系统_项目方案_V4.1_优化版.md)
- [EXPERIMENT_PROTOCOL_V1](docs/new-plans/EXPERIMENT_PROTOCOL_V1.md)
- [项目范围与验收](docs/product-specs/project_scope.md)
- [Phase 1 执行计划](docs/exec-plans/active/phase1_execution_plan.md)

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

当前真实 Part A 已选择为 `cap_a`（黄色圆形瓶盖）：

- Front：黄色外表面、红色文字 / 图案；
- Back：内凹结构面、内部环形结构 / 二维码；
- `orientation_semantics = NOT_APPLICABLE`，因为该零件外轮廓近圆且没有稳定的工业方向特征；
- 已使用手机 + iVCam 做真实采集验证；
- 当前正式采集使用 iVCam `source=1`、DirectShow、640×480、请求 30 FPS；
- 四路正式 Session 仍在建立中，因此 **Training Gate 仍为 CLOSED**。

Part B 与完整 Operating Envelope 仍需后续冻结；正式训练和 Phase 2 性能对比尚未开始。

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
| `NOT_APPLICABLE` | 轴承、垫圈、无方向语义圆形件 | N/A |

`NOT_APPLICABLE` 不等于 `UNCERTAIN`。如果一个零件本来就不要求旋转方向，缺少 Angle 不得单独触发 `FAIL_ORIENTATION` 或 `UNCERTAIN`。

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
