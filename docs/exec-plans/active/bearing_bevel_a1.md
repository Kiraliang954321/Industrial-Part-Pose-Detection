# A1：法兰轴承与锥形齿轮的夹持检测、套轴预定位

日期：2026-10-31。螺丝阶段已提交为 `d9d342b`。用户要求开始轴承与齿轮，并明确轴承也需区分正反，采用具有明确非对称结构的型号。沿用简单用途验证，不继续螺丝 H2 的精细化路线。

状态：**首轮开发、训练、校准、冻结后独立仿真评价已完成**。A1 新代码在 `codex/bearing-bevel-prototype`，尚未另行提交或推送；此前螺丝阶段提交保持独立。

## 对象与来源

- 法兰轴承参考 **F608ZZ**。JTEKT 官方目录列出内径 8 mm、外径 22 mm、宽 7 mm、法兰外径 25 mm、法兰宽 1.5 mm；这些尺寸用于项目自建代理的外轮廓与通孔，单侧法兰提供真实几何非对称性。[官方规格](https://koyo.jtekt.co.jp/en/products/detail/?pno=F608+ZZ)
- 锥形齿轮参考 **KHK SB1-2040**。官方规格为模数 1、20 齿、孔径 6 mm、轮毂直径 16 mm、外径 22.41 mm、总长 13.78 mm、轮毂宽 7 mm。项目自建简化锥齿/轮毂代理，不复刻真实齿廓或用于啮合计算。[官方规格](https://catalog.khkgears.us/item/bevel-gears/steel-bevel-gears-sb/sb1-2040)

没有取得实物或厂商 CAD。目录尺寸参考不等于实物匹配、摩擦/惯量验证或真实 Operating Envelope 冻结；真实门禁仍 CLOSED。

## 首轮契约与实施计划

1. 新模块置于 `annular/`，保留螺丝冻结源码/协议。使用现有 MuJoCo 与 Panda 资产，不升级依赖。
2. 轴承 `front` 定义为单侧法兰向上，齿轮 `front` 定义为锥齿面向上、轮毂在下。反向分别为 `back`；这是本工位的方向约定，不声称行业通用正面。明显倾斜另设 `tilted` 开发类，不估计无意义的唯一绕轴角。
3. 物理夹取已知窝位，保持至固定检测位置，一路 RGB 判断类型/端向/明显姿态；图像不可用、空抓或证据不足留件。GT 只标注和评分，不用于视觉组判定或对象位置修正。
4. 先核验物理夹持与观察视野，再生成按完整抓取 episode 隔离的 Train/Val，开发六类 YOLO 检测模型（两对象各 front/back/tilted），空抓为负样本。首轮训练有时间/轮数上限。
5. 正确端向通过后，以已知窝位、工具和固定装配配方进行有间隙套轴预定位；使用真实留孔的碰撞几何，继续保持夹持。不是压装、键槽对齐、啮合或最终装配验收。
6. 校准后冻结再评价新种子。分别报告两对象的放行/拒绝/留件、误放行和物理预定位结果，保留失败分母和实际视频；不继承螺丝模型指标。

## 已实现的开发版本

`annular/` 提供两对象代理、Panda 工位、已知窝位夹取、图像标注/审计、六类模型训练、单相机判定、物理套轴与独立评价。四个入口分别为 `generate_annular.py`、`train_annular.py`、`run_annular.py`、`freeze_annular.py`。螺丝各冻结源码保持不变。

机器人使用更小的外缘夹持接触垫，半尺寸 3×2×1.5 mm；没有对象焊接、吸附或执行中位置重置。检测 TCP `(0,0,0.22)` m，一路斜视 RGB 参与判定；其他相机仅用于录像。轴承/齿轮按明确作业配方夹取，图像用于类型、端向和明显姿态的放行，不声称预测完整 3D 孔位或连续视觉伺服。

装配轴顶高 0.12 m：轴承配 6 mm 轴、齿轮配 4 mm 轴，均保留约 1 mm 径向间隙，目标套入 3 mm 后仍夹持。真实型号通常需要另行确定配合、公差和施力方法，本版不是实际轴承压装配方。通孔由分段环形凸体组成；射线检查中孔中心穿过零件直到台面，偏离孔中心的射线命中实体环面。

轴承按整件刚体处理，没有内外圈/滚珠相对运动；齿轮齿形为视觉示意，碰撞主体为简化轮毂/锥体，不用于齿轮啮合或强度计算。正常近水平与 30–45° 明显倾斜分别采样，15° 附近的分界尚未验证。

## 数据与训练记录

- 数据：`data/annular/20261031-development-v1`。Train **144 次物理抓取 / 575 张图**，Val **48 次 / 96 张图**。同次抓取的光照变体保持在同一 split；一张完全重复的空抓变体被省略并保留 episode 变体清单。
- 每对象的 front/back/tilted：Train 各 18 次抓取 × 4 外观图，Val 各 6 次 × 2 图。空抓为无目标标签，不是训练类别；实际夹持与端向 GT 独立测量，不能按生成场景名称强填标签。图像、标注、分组、文件清单及哈希审计通过，六类预览已可视复核。
- 初始化为 Ultralytics 官方 `yolo26n.pt` v8.4.0 发布资产，源地址保存在 `data/annular-assets/yolo26n-v8.4.0/origin.json`，下载后记录 SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。未沿用螺丝类别头或权重作为当前任务成绩。
- 训练：`runs/annular/20261031-train-v1`，12 个完整 epoch，每轮 36 batch，640 px / batch 16 / BF16 / AdamW，约 **250.3 s**。开发 Val mAP50–95 为 **0.995**；该指标不是物理装配成功率。
- 选定 `fit/weights/best.pt` 的 SHA-256 为 `c281106e782be9b86378d7337ab5663c17f296d845741f9647f45a992040e518`。没有新增或升级 Python 依赖。

```powershell
.\.venv\Scripts\python.exe scripts/generate_annular.py --assets data/robot-assets/panda-a662de6 --output data/annular/<new-dataset>
.\.venv\Scripts\python.exe scripts/train_annular.py --dataset data/annular/<dataset> --initial-weights data/annular-assets/yolo26n-v8.4.0/yolo26n.pt --output runs/annular/<new-training> --epochs 12
.\.venv\Scripts\python.exe scripts/run_annular.py --assets data/robot-assets/panda-a662de6 --weights runs/annular/<training>/fit/weights/best.pt --output runs/annular/<new-calibration> --seed 121000 --episodes-per-part 20
.\.venv\Scripts\python.exe scripts/freeze_annular.py --calibration runs/annular/<calibration> --training runs/annular/<training> --output docs/data/<new-protocol>.json --seed 131000 --episodes-per-part 20
.\.venv\Scripts\python.exe scripts/run_annular.py --assets data/robot-assets/panda-a662de6 --weights runs/annular/<training>/fit/weights/best.pt --output runs/annular/<new-evaluation> --mode evaluate --protocol docs/data/<protocol>.json --seed 131000 --episodes-per-part 20 --record-indices 0,1,20,21
```

输出必须为新目录。Train：101000/102000 两对象种子段；Val：111000/112000；Calibration：121000/121100；冻结后评价另用 131000/131100。新评价不可回用于当前报告调参。预测置信度门槛 0.80，仅是当前开发工作点；不表示经过真实工业风险标定。

## 校准及冻结后的结果

校准目录 `runs/annular/20261031-calibration-v1`，共 40 次。两对象分别 4 PASS / 8 REJECT / 8 UNCERTAIN。齿轮实际预定位 4/4；轴承严格物理成功 3/4，另一次位置、姿态和夹持符合要求，但最大软接触穿透 **0.5016 mm** 略超 **0.5 mm** 上限，仍记失败。没有抬高阈值或改写记录。

冻结协议：[bearing_bevel_a1_protocol_v1.json](../../data/bearing_bevel_a1_protocol_v1.json)，SHA-256 `cf1233036c56c3585831edfa5b83258ad4523b07287bf1f814d7f666bcaadaaf`。随后以新种子 131000–131019、131100–131119 完成 `runs/annular/20261031-evaluation-v1`；评价结果未用于调参。

| 每个对象的独立场景 | 轴承 | 锥形齿轮 |
| --- | --- | --- |
| 正向 | 4 PASS，4 次物理预定位成功 | 4 PASS，4 次物理预定位成功 |
| 反向 | 4 REJECT | 4 REJECT |
| 明显倾斜 | 4 REJECT | 4 REJECT |
| 空抓 | 4 UNCERTAIN | 4 UNCERTAIN |
| 相机黑屏 | 4 UNCERTAIN | 4 UNCERTAIN |

- 共 40 次：**8 PASS / 16 REJECT / 16 UNCERTAIN**，实际预定位 **8/8**，Wilson 95% CI **[67.56%, 100%]**；每对象 4/4 的区间均为 **[51.01%, 100%]**。各条件人为均衡，不代表生产分布。
- Wrong Accept **0/24，95% CI [0%, 13.80%]**；每对象 **0/12，[0%, 24.25%]**。黑屏时物理对象仍合规，属于单独的输入故障；合规未放行 8/16 均来自这 8 次黑屏，不隐去该分母。
- 轴承实际套入深度 **3.000–3.045 mm**，孔轴偏差 **0.120–0.379 mm**，轴线倾斜 **0.007–0.024°**；齿轮套入深度 **2.835–3.002 mm**，孔轴偏差 **0.295–0.635 mm**，倾斜 **0.008–0.013°**。均保持夹持。独立评价最大软接触穿透 **0.457 mm**，没有已定义的机器人自碰撞/台面/夹具非法碰撞。
- 单图推理 P95 **115.8 ms**；包含仿真、记录及部分录像的端到端本机墙钟 P95 **7.53 s**；仿真时钟 P95 **12.763 s**。未实现连续供料、回程及恢复循环，不给出生产吞吐。

## 演示与验证

- 视频：`runs/annular/20261031-demo-v1/bearing-bevel-inspection.mp4`，H.264 / 1280×768 / 12 fps / **41.83 s**。依次展示轴承正确/反向、齿轮正确/反向；字幕说明整段结果，每段末尾加 1 s 定格，原始物理录像保留。
- 正反对照图：同目录 `front-back-evidence.jpg`，均来自独立评价的真实 RGB 与预测结果，不是生成式示意图。
- 288 个评价产物的哈希复核通过，40 个种子唯一，冻结协议复核通过。视频逐帧解码 **502/502**，已检查抽帧与正反预测图。新模块和四个 CLI 编译通过；未新增测试文件或运行无关全量套件。
- 详细复核记录：`runs/annular/20261031-verification-v1/verification.json`。旧 M4/L1/H1/H2 源码及协议继续保留。

## 范围与限制

本轮证明的是当前代理、固定窝位/夹具和单相机条件下的方向门控与套轴预定位。没有真实 CAD/实拍迁移、压力装配、键槽定位、齿轮啮合或最终装配质量证据。明显倾斜通过机器人展示姿态变化产生，尚未验证夹爪姿态正常而零件单独歪斜的情况。维持这一级别的原型交付，不因模型分数较高继续追加亚毫米优化或复杂复检。
