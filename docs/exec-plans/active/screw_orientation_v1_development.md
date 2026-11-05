# 螺丝朝向 V1：开发实施与运行记录

日期：2026-10-31。范围：用户确认的 M0–M3，`simulation_development`。

**本页记录 M0–M3 交付：仿真视觉开发、30 轮训练、Val 对照及预测回放已完成。随后完成的机械臂仿真闭环另见 [M4 记录](screw_m4_closed_loop.md)。真实协议未 FROZEN，正式 Training Gate CLOSED；完整 6D 尚未实现。**

## 已完成的开发

- MuJoCo 3.17.0 本机 Windows 渲染、刚体沉降、相机投影、GT 分离及 PNG 输出。
- 参数化 `screw_proxy_v0`，43 mm 总长、6 mm 杆径、12 mm 头径；这是程序设定的代理，尚无对应实物或厂商 CAD。
- 单类、两个语义点、有向 360° 角、拒绝原因的独立接口；旧面类、OBB 与瓶盖训练代码没有被复用为新任务入口。
- 场景来源、随机种子、配置/代码/图像/标签哈希、可见性、重投影、重复与 split 审计。
- RGB-only Traditional CV 与 YOLO26n-Pose 推理、Train/Val 评价、候选比较和图片预测回放入口。
- 官方初始化权重已按 GitHub release 公布的 SHA-256 验证：`eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9`。

实现与用法见[数据/预测契约](../../data/screw_orientation_v1.md)和[实施方案](screw_orientation_v1_implementation_plan.md)。

## 数据与环境

| 项目 | 实测/记录 |
| --- | --- |
| 数据目录 | `data/screw-orientation/20261031-development-v1` |
| Train | 2000 张：1802 张有螺丝、198 张空背景 |
| Val | 400 张：364 张有螺丝、36 张空背景 |
| 数据生成 | 123.36 秒，不含末尾审计；原图 359566171 bytes，约 342.91 MiB；重采样计数为空 |
| 工况 | nominal / low_light / reflection / background_shift / motion_blur / partial_occlusion |
| 划分 | 同一已知代理；每场景一快照，Train/Val 使用独立来源组与 seed sequence |
| Test / Final Holdout | 未生成、未使用，待仿真实验协议冻结 |
| 设备 | RTX 4060 Laptop 8188 MiB；i7-13650HX；约 31.8 GiB RAM |
| 核心依赖 | Python 3.11.9、Torch 2.11.0+cu128、Ultralytics 8.4.171、NumPy 1.26.4、OpenCV 4.11.0.86 |
| 新增依赖 | MuJoCo 3.17.0 及其 glfw / PyOpenGL / etils / zipp 依赖；未升级核心训练依赖 |

已有 `.venv` 启用了 `include-system-site-packages`，部分已有依赖来自用户/系统 site-packages；运行清单记录实际版本。这不是新建的完全隔离环境。

## 训练记录

| Run | 用途与状态 |
| --- | --- |
| `20261031-smoke-train-v1` | 24/12 张小数据，FP32、batch 4、2 epochs，完成；用于训练/保存/加载接口核查，不作为性能结果 |
| `20261031-smoke-bf16-v1` | 同小数据，BF16、batch 4、1 epoch，完成；验证混合精度数值和权重输出 |
| `20261031-development-train-v1` | batch 4 吞吐试运行；第一轮未完成即停止，保留日志，不进入结果选择 |
| `20261031-development-train-v2` | 从相同官方权重重新初始化，BF16、batch 16、imgsz 640、workers 0、seed 42；实际 30 epochs，未触发 55 分钟预算或停止请求；完成 |

batch 调整发生在开发阶段；原始试运行显存约 0.73 GiB，随后改为 batch 16，并将后续预算缩短为 55 分钟。v2 训练入口记录 2826.45 秒（约 47.11 分钟，含训练后验证/审计，不含启动前审计）；CSV 训练/逐轮验证计时 2694.11 秒。峰值 CUDA allocated 2479515648 bytes（约 2.31 GiB），reserved 3149922304 bytes（约 2.93 GiB）。

每个训练 run 保存输入数据清单/标签记录、初始权重、配置、环境和代码哈希；v2 另保存实际代码副本。`STOP_REQUESTED` 文件可请求在 batch 边界停止，后续仍执行保存和验证；不会删除历史产物。

## 传统视觉开发校准

候选阈值在完整 Val 上比较，width ratio 固定 1.2。选择规则为优先最大化 `correct_images_10deg`，再比较条件角误差 Median，最后按候选顺序。

| 阈值 | 图像成功 / 400 | 漏检 | 额外检测 | 匹配后方向拒绝 | 半平面错误 / 可评分方向 |
| --- | --- | --- | --- | --- | --- |
| 25 | 356 | 20 | 17 | 8 | 5 / 336 |
| 35（选定） | 357 | 20 | 19 | 5 | 6 / 339 |
| 50 | 353 | 26 | 36 | 4 | 7 / 334 |

选定阈值的 43 张失败全部出现在当前定义的局部遮挡工况，另外五类简化工况均成功。这说明当前代理数据上的常规方向判断用简单方法已经有效，不能把总体差异解释为 AI 在所有工况均占优。

这些候选报告在训练并行期间生成，校准只使用正确性；训练完成后已单独重测选定 CV 配置，正确性结果一致。最终延迟使用该重测报告。

## 最终开发比较与回放

best/last 均使用相同 Val、检测阈值 0.25、关键点阈值 0.25、IoU 0.5 和有向角口径评价。两者在全部 400 张图上的预测记录完全一致，按事先定义的并列顺序选择 **best.pt**；未根据测速差异更换选定模型。

| 指标 | Traditional CV（threshold 35） | YOLO26n-Pose（best） |
| --- | --- | --- |
| 图像诊断成功（10°） | 357/400，89.25%；Wilson 95% CI 85.83%–91.92% | 400/400，100%；Wilson 95% CI 99.05%–100% |
| 方向覆盖 / 全部有效 GT | 339/364，93.13%；CI 90.06%–95.30% | 364/364，100%；CI 98.96%–100% |
| 方向半平面错误（>90°）/ 可评分 | 6/339；CI 0.81%–3.81% | 0/364；CI 上界 1.04% |
| 漏检 / 额外检测 / 匹配后方向拒绝 | 20 / 19 / 5 | 0 / 0 / 0 |
| 条件角误差 Median / P95 | 1.283° / 9.183°（n=339） | 0.383° / 1.126°（n=364） |
| 单图读取至同步预测 Median / P95 | 10.831 / 12.610 ms（n=400） | 29.914 / 41.055 ms（n=400） |

两种方法的空背景误检均为 0/36；零错误的 Wilson 95% 上界仍约 9.64%。10° 是开发诊断阈值，不是实际工艺容差。角误差只在可评分预测上统计，必须连同覆盖、漏检和拒绝阅读。Pose 的端点平均像素误差为 1.645 px，按 GT 轴长归一化后平均约 0.00945。

同一个预测结果的 last 重测延迟 Median/P95 为 27.464/30.774 ms，说明运行顺序/设备状态会影响单次测速；不将最小延迟挑作 best 的成绩，也不据此声称真实流水线吞吐。主结论是：**在这个固定代理与已定义开发集合上，学习方法改善了局部遮挡下的目标关联与方向输出；其它五类简化工况中传统方法已能完成诊断。**

Val 已参与选择参数和权重，结果不是无偏 Test 成绩，也不是实际工业精度。没有打开真实正式门禁。

### 可访问产物

- [训练摘要](../../../runs/screw-orientation/20261031-development-train-v2/training-summary.json)与[完整运行清单](../../../runs/screw-orientation/20261031-development-train-v2/run-manifest.json)。
- [选定 best.pt](../../../runs/screw-orientation/20261031-development-train-v2/fit/weights/best.pt)，SHA-256：`0a29b8c5660d62cb19b84bf2cc0805e1c3b3ed2fb2e7478dc80813ab2f1e155a`。
- [Pose 完整 Val 报告](../../../runs/screw-orientation/20261031-val-pose-best/report.json)、[CV 最终 Val 报告](../../../runs/screw-orientation/20261031-val-cv-35/report.json)、[候选选择记录](../../../runs/screw-orientation/20261031-comparison-v1/selection.json)。
- [对比报告](../../../runs/screw-orientation/20261031-comparison-v1/comparison.md)、[训练曲线](../../../runs/screw-orientation/20261031-comparison-v1/training-curves.png)、[三例对比图](../../../runs/screw-orientation/20261031-comparison-v1/comparison-preview.png)。遮挡图是有意选取的 CV 失败案例，不是随机抽样；选择规则另存 `preview-selection.json`。
- [H.264 预测回放](../../../runs/screw-orientation/20261031-selected-replay-v1/replay-h264.mp4)：400 帧、4 FPS、100 秒、4050020 bytes，解码 400/400，通过；这是静态 Val 图片的预测回放，不是实物运动或机械臂录像。
- [回放编码核验](../../../runs/screw-orientation/20261031-selected-replay-v1/h264-verification.json)，视频 SHA-256：`cff7d23051d55a575a70e07e8c50fed19f2e95c56452d3702d2d9e57c4341948`。

数据、权重与运行产物按现有 `.gitignore` 留在本机，未进入版本库；代码和文档提供路径、配置与复现方式。

## 已进行的软件与数据检查

- 新模块编译通过；既有 `tests/test_geometry.py` 为 3 passed，确认旧无向角契约未破坏。
- 定向边界核查通过：四个正交方向、跨 0°、头尾互换 180°、重合/非有限点、漏检/负样本分母、Wilson 零错误上界。
- 路径越界被拒绝，已有输出目录不覆盖；对新增模块做了代码与输入路径检查。
- 全量数据审计错误 0；几何锚点重投影一致；查看了六工况覆盖图。
- 单个已记录场景按相同 seed 重渲染，PNG SHA-256 完全一致；仅表示该本机样本复现，不宣称所有硬件/版本位级一致。
- 训练原图与 GT 覆盖图分开，预测 API 只接收 RGB 图像，GT 只进入生成器与 scorer。
- best/last 重新加载和两点契约检查通过；两者 400 张逐图预测一致；H.264 完整解码与对比图、曲线可视检查通过。
- 修正了进度回调将额外最终验证显示成第 31 轮的问题；本次实际训练轮次由 CSV 与摘要核实为 30，未产生额外优化轮次。

## 剩余边界

尚无真实螺丝型号/CAD、真实采集/标定和正式四路数据；策略学习与完整 6D 未实现。M0–M3 交付时未含机械臂，后续已完成的受约束规划抓取见 [M4](screw_m4_closed_loop.md)。金属反射、背景纹理、遮挡与模糊均是有限代理工况，不能据此填写真实 Operating Envelope 或生产容差。

未执行 Git commit / push；代码、文档与本地开发产物留在工作区。
