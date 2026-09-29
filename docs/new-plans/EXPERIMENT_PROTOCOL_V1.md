# EXPERIMENT_PROTOCOL_V1

> 项目：面向柔性流水线的低成本工业零件正反面与二维姿态检测系统  
> 对应方案：V4.1 — 生产检测验证与实验协议冻结版  
> 项目目录：`G:\Project\Yolo`  
> 协议状态：**FROZEN BEFORE FORMAL TRAINING**  
> 日期：2026-09-29  
> 目的：在正式数据采集、训练和 Final Test 之前，冻结实验对象、真值定义、数据划分、对照公平性、统计口径与工作点选择规则，防止结果导向调参和实验口径漂移。

---

## 1. 协议作用与冻结原则

本协议服务于 V4.1 的四个核心研究问题：

1. Traditional CV 与 YOLO 相比，AI 是否在指定复杂条件下具有明确收益；
2. Event Decision 是否优于逐帧直接判断；
3. UNCERTAIN 是否能以可接受的 Coverage、Latency 与 Throughput 代价降低生产风险；
4. Part A → Part B 是否能够主要通过 Dataset + Fine-tune + Recipe 完成换型，而无需重写核心系统。

本协议一旦进入 `FROZEN` 状态：

- Final Test 不允许用于模型、阈值或规则调参；
- Test 失败不能通过临时修改 Ground Truth 定义来规避；
- 核心指标定义不得在看到正式结果后改写；
- Part B 不得因为 Part A 结果不理想而临时替换为更容易的零件；
- 新功能默认进入 Future Work，除非核心实验被明确阻塞；
- 所有正式实验必须记录 Git / Dataset / Model / Recipe / Threshold / Ground Truth / Test Set 身份。

---

## 2. 实验对象冻结

### 2.1 Part A

Part A 是感知与事件决策基线零件。

冻结前必须填写：

```yaml
part_a:
  id: TBD
  physical_object: TBD
  front_definition: TBD
  back_definition: TBD
  orientation_semantics: TBD
  symmetry_order: TBD
  expected_aspect_ratio_range: TBD
```

选择原则：

- Front / Back 判据必须可由实物客观定义；
- 应适合二维平面姿态任务；
- 优先选择 OBB 长轴稳定、能建立可靠 baseline 的零件；
- 不允许为了追求高分而只选择极端容易识别的样本。

### 2.2 Part B

Part B 用于验证柔性换型。

冻结前必须填写：

```yaml
part_b:
  id: TBD
  physical_object: TBD
  difference_from_part_a: TBD
  front_definition: TBD
  back_definition: TBD
  orientation_semantics: TBD
  symmetry_order: TBD
```

Part B 要求：

- 与 Part A 同属二维平面姿态检测问题；
- 在外形、纹理、尺寸或正反面差异上至少存在一种明显变化；
- 仍位于本项目 Operating Envelope 内；
- Part A 正式训练前冻结候选实物与选择理由；
- Phase 5 前不因为结果方便而替换。

---

## 3. Operating Envelope

所有正式性能结论必须绑定到被冻结的工况。

至少记录：

```yaml
camera:
  device: TBD
  resolution: TBD
  fps: TBD
  exposure_mode: TBD
  camera_distance_mm: TBD

field_of_view:
  roi: TBD
  object_pixel_size_range: TBD

motion:
  mode: TBD
  speed_range: TBD

spacing:
  min_part_spacing: TBD
  overlap_allowed: false

lighting:
  nominal: TBD
  low_light: TBD
  reflection: TBD

background:
  nominal: TBD
  allowed_shift: TBD

disturbance:
  motion_blur_levels: TBD
  occlusion_levels: TBD
```

最终结论必须使用：

> 在冻结的 Operating Envelope 内，本系统达到……

禁止写成无边界的“适用于工业流水线”。

---

## 4. Ground Truth Protocol

### 4.1 Ground Truth 与系统输出严格分离

V4.1 的系统输出仍为：

```text
PASS
FAIL_FACE
FAIL_ORIENTATION
UNCERTAIN
```

但 Ground Truth 不使用 `UNCERTAIN` 作为普通真实类别。

正式 Ground Truth 状态定义为：

```text
CONFORMING
NONCONFORMING
INDETERMINATE
```

含义：

- `CONFORMING`：根据冻结的 Face / Orientation 规则，零件客观满足当前 Recipe；
- `NONCONFORMING`：根据冻结规则，零件客观不满足当前 Recipe；
- `INDETERMINATE`：样本本身无法可靠建立确定真值，例如严重遮挡、角度基准不可恢复、实物状态存在争议或采集证据不足。

### 4.2 Wrong Accept 的正式定义

**Wrong Accept 只定义为：**

```text
Ground Truth = NONCONFORMING
System Decision = PASS
```

公式：

```text
Wrong Accept Rate
= NONCONFORMING events predicted PASS
  / all valid NONCONFORMING events
```

不再把“应该 UNCERTAIN”写入 Ground Truth 定义。

原因：

> `UNCERTAIN` 是系统的 Risk Gate 输出状态，而不是物理世界的普通真实类别。

### 4.3 Wrong Reject 的正式定义

主指标定义：

```text
Ground Truth = CONFORMING
System Decision = FAIL_FACE / FAIL_ORIENTATION / UNCERTAIN
```

公式：

```text
Wrong Reject Rate
= CONFORMING events not released as PASS
  / all valid CONFORMING events
```

为避免混淆，同时分拆报告：

- `Hard Reject Rate`：CONFORMING → FAIL_FACE / FAIL_ORIENTATION；
- `Uncertain Reject Rate`：CONFORMING → UNCERTAIN。

### 4.4 INDETERMINATE 的处理

`INDETERMINATE` 不进入主 Wrong Accept / Wrong Reject 分母。

但必须单独报告：

- INDETERMINATE Event Count；
- INDETERMINATE → PASS；
- INDETERMINATE → FAIL；
- INDETERMINATE → UNCERTAIN。

建议附加指标：

```text
Ambiguous Accept Rate
= INDETERMINATE events predicted PASS
  / all INDETERMINATE events
```

该指标只用于风险观察，不与确定 Ground Truth 的 Wrong Accept 混合。

### 4.5 Event Ground Truth Schema

每个正式测试 Event 至少记录：

```yaml
event_gt_id: string
part_id: string
true_face: front | back | not_applicable
true_orientation_deg: float | null
orientation_valid: bool
conformance_gt: CONFORMING | NONCONFORMING | INDETERMINATE
expected_failure_reason: FACE | ORIENTATION | BOTH | NONE | INDETERMINATE
session_id: string
condition_id: string
reviewer: string
review_status: PENDING | VERIFIED | DISPUTED
ground_truth_version: string
```

规则：

1. Face GT 必须由实物正反面定义产生，并在采集前冻结；
2. Angle GT 优先由物理角度基准板或固定治具产生；
3. 无治具时可使用人工 OBB / 方向标注，但必须二次复核；
4. Event GT 以单个物理零件事件为统计单位，不以逐帧多数投票生成；
5. 无法可靠确定的样本必须标为 `INDETERMINATE`，不得强行指定；
6. Ground Truth 修改必须版本化并留下变更记录。

---

## 5. Test Condition Matrix

正式测试至少包含：

```text
nominal
low_light
reflection
background_shift
motion_blur
partial_occlusion
```

每个 Event 必须绑定 `condition_id`。

最终结果同时报告：

- Overall；
- Nominal；
- 每个 Stress Condition。

禁止只报告 Overall，以免平均值掩盖特定工况失效。

---

## 6. Dataset Split 与 Final Holdout

采用 Session-aware Split：

```text
Train Session(s)
Val / Calibration Session(s)
Test Session(s)
Final Holdout Session(s)
```

用途：

- Train：训练 YOLO 或拟合 Traditional CV 参数；
- Val / Calibration：模型选择、阈值选择、Risk Gate 工作点选择；
- Test：冻结后的正式评估；
- Final Holdout：Hard Sample 回流或二次训练后的最终未见数据验证。

禁止：

```text
同一连续视频随机逐帧拆分到 Train / Val / Test
```

如果 Test 错误样本进入 Hard Sample 再训练：

> 原 Test 不得继续作为新模型的最终无偏测试集。

必须改用未见过的 Final Holdout 或新采集 Session。

---

## 7. Traditional CV Fairness Protocol

Traditional CV 与 YOLO 必须共享：

- 相同 Camera / Video 输入；
- 相同分辨率；
- 相同 ROI；
- 相同 Test Events；
- 相同 Operating Envelope；
- 相同 Ground Truth；
- 相同计时起点与终点定义。

调参规则：

- Traditional CV 参数仅在 Train / Val / Calibration 上调整；
- YOLO confidence / NMS / Risk Gate 同样只在 Train / Val / Calibration 上调整；
- Final Test 前两侧参数都必须冻结；
- 不允许在 Test 上为任何一侧单独优化。

比较目标：

> 不是证明 AI 一定更好，而是确定不同方法在什么工况和成本约束下更合适。

---

## 8. Event Decision Protocol

Event Pipeline：

```text
Detection
↓
Tracking
↓
Multi-frame Evidence
↓
Decision Zone
↓
Risk Gate
↓
Single-Part Event
```

状态机：

```text
NEW
↓
OBSERVING
↓
STABLE
↓
ENTER_DECISION_ZONE
↓
DECIDED
↓
COMPLETED
```

异常路径：

```text
OBSERVING
↓
UNCERTAIN
↓
REVIEW / REJECT
```

原则：

> 1 Track → 1 Final Event。

正式比较：

```text
Single Frame
vs
Tracking + Multi-frame + Decision Zone
```

必须报告：

- Wrong Accept；
- Wrong Reject；
- Face Flip Rate；
- Angle Jitter；
- Duplicate Event Rate；
- Missed Event Rate；
- End-to-End P95。

---

## 9. Risk-Coverage-Latency Protocol

调节参数：

- confidence；
- min_stable_frames；
- face agreement；
- max_angle_std_deg。

每个候选工作点至少记录：

```text
(
  Wrong Accept,
  Wrong Reject,
  Coverage,
  End-to-End P95
)
```

Coverage：

```text
Coverage
= events receiving definitive PASS / FAIL
  / all evaluable events
```

工作点选择：

1. 只允许在 Val / Calibration 上扫描；
2. 优先观察 Pareto-efficient 工作点；
3. 不允许只追求单一最低 Wrong Accept 而忽略 Coverage / Latency；
4. 最终阈值配置冻结为 `threshold_profile`；
5. Final Test 只运行冻结工作点，不再次寻找更有利阈值。

---

## 10. 新增工业指标：Validated Event Throughput

### 10.1 目的

Inference FPS 不能直接代表流水线实际处理能力。

因此正式增加：

> **Validated Event Throughput**

单位：

```text
parts / minute
```

### 10.2 定义

在冻结 Operating Envelope、冻结 Risk Gate 工作点和稳定连续运行窗口下：

```text
Validated Event Throughput
= valid completed physical-part events
  / effective runtime minutes
```

其中 valid completed event 必须满足：

- 对应一个真实物理零件；
- Track 生命周期完整；
- 产生且仅产生一个 Final Event；
- 非重复计数；
- Ground Truth 可映射；
- Event Log 完整。

### 10.3 同时报告

Throughput 必须和以下指标一起报告：

- Wrong Accept；
- Wrong Reject；
- Coverage；
- Missed Event；
- Duplicate Event；
- End-to-End P95；
- Test Condition；
- Event Count。

禁止只追求高 Throughput 而忽略风险。

### 10.4 推荐比较

不同 `min_stable_frames` 或 Risk Gate 工作点可形成：

| Workpoint | Wrong Accept | Coverage | Event P95 | Validated Throughput |
| --- | ---: | ---: | ---: | ---: |
| WP-A | 实测 | 实测 | 实测 | 实测 |
| WP-B | 实测 | 实测 | 实测 | 实测 |
| WP-C | 实测 | 实测 | 实测 | 实测 |

最终讨论：

> 可靠性提升需要牺牲多少自动处理率、事件延迟与实际吞吐量。

---

## 11. Statistical Reporting Protocol

所有核心比例指标至少报告：

- numerator / denominator；
- Event Count `n`；
- point estimate；
- 95% confidence interval。

默认比例指标使用 Wilson 95% CI。

若出现 0 次错误：

- 禁止只写 `0%`；
- 必须写 `0/n`；
- 必须报告 95% 上置信界；
- 可使用 rule of three 提供直观解释：0 次错误时，95% 上界约为 `3/n`。

示例：

```text
Wrong Accept = 0 / 1000
Point estimate = 0%
95% upper bound ≈ 0.3%  (rule-of-three approximation)
```

Latency 至少报告：

- n；
- Median；
- P95；
- 条件允许时报告 P95 bootstrap CI。

---

## 12. 新增要求：Cross-Session Repeatability

### 12.1 原因

连续 1000 个 Event 来自一次稳定拍摄，不等同于 1000 个完全独立工况。

因此正式 Test 必须包含多个独立 Session。

### 12.2 Session 定义

独立 Session 应尽量包含真实重新开始条件，例如：

- 程序重新启动；
- Camera / iVCam 重新连接；
- 零件重新布置；
- 独立采集时段；
- 条件允许时重新建立相机位置或曝光；
- Lighting / Background 按协议恢复到指定 Condition。

禁止通过把同一连续录像机械切成多个文件伪造独立 Session。

### 12.3 最低报告要求

正式报告至少分层展示：

```text
Overall
Session A
Session B
Session C
```

如果资源允许，建议正式 Final Test 至少包含 3 个独立 Session。

每个 Session 至少报告：

- Event Count；
- Wrong Accept；
- Wrong Reject；
- Coverage；
- Missed Event；
- Duplicate Event；
- End-to-End P95；
- Validated Event Throughput。

### 12.4 重复性结论

禁止只写：

> Overall Accuracy / Wrong Accept 很高。

应同时说明：

> 指标是否在多个独立 Session 中保持一致，是否存在明显 Session-to-Session 波动。

如某 Session 显著恶化，应保留并分析，不得从 Overall 中删除。

---

## 13. Changeover Cost Protocol

Part A → Part B 全流程记录：

- New Images；
- Valid OBB Labels；
- Capture Minutes；
- Annotation Minutes；
- Fine-tune Minutes；
- Recipe Changes；
- Core Code Changes；
- 达标迭代次数；
- Part B Face Accuracy；
- Part B Wrong Accept；
- Part A Regression；
- Total Changeover Time。

核心目标：

```text
不修改核心代码
↓
更换 Dataset / Model / Recipe
```

若发生核心代码修改，必须真实记录修改原因与行数 / 文件数，不得仍声称“零代码换型”。

---

## 14. 正式实验身份

每个正式 Experiment 必须保存：

```yaml
experiment_id: string
git_commit: string
dataset_version: string
dataset_hash: string
model_name: string
model_hash: string
recipe_version: string
recipe_hash: string
training_seed: int
runtime: string
precision: string
operating_envelope_version: string
ground_truth_version: string
test_set_hash: string
test_event_count: int
session_ids: []
threshold_profile: string
timestamp: string
```

Event Log 至少关联：

- model_version；
- recipe_version；
- dataset_version；
- git_commit；
- session_id；
- condition_id；
- evidence_frame。

---

## 15. 四个核心实验的正式口径

### Experiment 1 — Traditional CV vs YOLO

比较：

```text
Traditional CV
vs
YOLO26n-OBB
vs
YOLO26s-OBB
```

报告：

- Face Accuracy；
- Wrong Accept；
- Wrong Reject；
- Angle MAE；
- mAP（YOLO）；
- P95 Latency；
- FPS；
- Validated Event Throughput；
- Light Shift；
- Background Shift；
- Cross-Session Repeatability。

### Experiment 2 — Single Frame vs Event Decision

比较：

```text
Single Frame
vs
Tracking + Multi-frame + Decision Zone
```

报告：

- Wrong Accept；
- Wrong Reject；
- Face Flip；
- Angle Jitter；
- Duplicate Event；
- Missed Event；
- Event P95；
- Throughput。

### Experiment 3 — Risk-Coverage-Latency-Throughput

扫描：

- confidence；
- stable frames；
- face agreement；
- max angle std。

报告：

```text
Wrong Accept
Wrong Reject
Coverage
End-to-End P95
Validated Event Throughput
```

工作点只在 Val / Calibration 冻结。

### Experiment 4 — Part A → Part B Changeover

报告：

- Data；
- Annotation；
- Training；
- Configuration；
- Core Code Changes；
- Accuracy；
- Wrong Accept；
- Regression；
- Total Changeover Time。

---

## 16. 正式报告规则

四个核心实验统一遵守：

1. 先冻结实验配置，再运行 Final Test；
2. 同一比较使用相同 Event 集合；
3. 报告失败样本，不只展示成功 Demo；
4. 比例指标报告 `x/n + 95% CI`；
5. 同时报告 Overall、各 Condition、各 Session；
6. Ground Truth 的 `INDETERMINATE` 单独分层，不混入确定标签；
7. Wrong Accept 严格使用 `NONCONFORMING → PASS`；
8. 报告 Validated Event Throughput，不以 FPS 冒充产线吞吐量；
9. 不把官方 benchmark 当成本项目结果；
10. 不把 Validation 结果写成 Final Test；
11. 实验假设未被支持时按真实结果记录；
12. 不删除表现差的 Session 或 Stress Condition。

---

## 17. Dataset V1 Gate

正式训练前以下全部满足：

- [ ] Part A 已冻结；
- [ ] Part B 候选实物与选择原则已冻结；
- [ ] Front / Back 判据已冻结；
- [ ] Orientation 语义已冻结；
- [ ] Operating Envelope Draft 已完成；
- [ ] Ground Truth Schema 已冻结；
- [ ] `CONFORMING / NONCONFORMING / INDETERMINATE` 定义已确认；
- [ ] Test Condition Matrix 已建立；
- [ ] Session-aware Split 已建立；
- [ ] Final Holdout 策略已建立；
- [ ] OBB QA 通过；
- [ ] Dataset Audit 通过；
- [ ] 类别分布可接受；
- [ ] 角度分布可接受；
- [ ] Provenance 有记录；
- [ ] 无明显数据泄漏；
- [ ] Traditional CV Fairness Protocol 已冻结；
- [ ] Statistical Reporting Protocol 已冻结；
- [ ] Validated Event Throughput 定义已冻结；
- [ ] Cross-Session Repeatability 规则已冻结；
- [ ] Final Test 调参禁令已确认。

任何一项失败：

> **Dataset V1 Gate CLOSED — 不开始正式训练。**

全部通过：

> **Experiment Protocol FROZEN + Dataset V1 Gate OPEN。**

---

## 18. 本协议相对 V4.1 的三个正式修正

### 修正 1 — Ground Truth 语义

旧风险：

```text
“实际应该 FAIL / UNCERTAIN”可能把系统 UNCERTAIN 输出与真实世界标签混为一谈。
```

V1 正式规则：

```text
GT = CONFORMING / NONCONFORMING / INDETERMINATE
Wrong Accept = NONCONFORMING → PASS
UNCERTAIN = System Decision，不是普通 Ground Truth 类别
```

### 修正 2 — Validated Event Throughput

新增：

```text
Validated Event Throughput = valid completed events / effective runtime minute
```

用来补足：

```text
FPS ≠ 真实单件事件处理能力
```

### 修正 3 — Cross-Session Repeatability

新增要求：

```text
Final Test 必须分 Session 报告。
建议至少 Session A / B / C。
同一连续录像切片不能冒充多个独立 Session。
```

目的：

> 防止单次稳定采集产生大量高度相关 Event，使统计样本量看起来很大但对真实重复性证明不足。

---

# Protocol Freeze Statement

当本文件中的 TBD 字段被填写，并且第 17 节 Dataset V1 Gate 全部通过后：

> **EXPERIMENT_PROTOCOL_V1 正式冻结。**

冻结后允许修复软件 Bug，但：

- 不允许根据 Final Test 结果修改核心指标定义；
- 不允许在 Final Test 上继续选阈值；
- 不允许改变 Ground Truth 规则以改善指标；
- 不允许隐去失败 Session；
- 不允许用新增功能替代原本失败的核心假设验证。

后续项目进步应主要来自：

> **真实数据、真实实验结果、失败分析与可复现证据。**
