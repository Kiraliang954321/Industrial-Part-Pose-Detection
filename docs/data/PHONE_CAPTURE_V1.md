# Phase 1C 手机/iVCam 原始图像采集 V1

本工具是操作员控制的 `front` / `back` **原始 JPEG** 采集器，不推断正反面，也不生成 OBB 框或训练标签。V2 起每次新会话必须显式绑定一个 `part_id`；`front` / `back` 是面标签，不是最终 YOLO 类名。真实手机/iVCam 预览与采集的硬件烟测**尚未完成**。下列流程用于首次现场试采，不能据此宣称模型准确率或 FPS。

## 启动与按键

在仓库根目录运行；先安装项目依赖（例如 `python -m pip install -e ".[test,inference]"`），启动手机的 iVCam 并在 Windows 中确认其虚拟摄像头可用。PowerShell 示例：

```powershell
python scripts/capture_phone.py --help
python scripts/capture_phone.py --part-id screw_A-1 --source 0 --backend dshow
```

`--part-id` 必填，须是匹配 `[A-Za-z0-9][A-Za-z0-9_-]*` 的安全单个 ASCII token（如 `screw_A-1`）；缺少或非法时不会创建会话。每个会话只对应一个零件 ID，换零件须创建新会话。`--source 0` 是摄像头索引、`--backend dshow` 指 Windows DirectShow；实际设备的索引或适用后端可能不同，可尝试其他非负索引和 `--backend auto`。`--source` 也接受视频文件路径/URL；`dshow` 仅对摄像头索引生效。可按设备需要设置 `--width`、`--height`、`--fps`，设备可能不接受请求的值；用 `--output-root` 可改变默认输出根目录。

在预览窗口中操作（启动时选中 `front`，**不会自动识别面**）：

| 按键 | 操作 |
| --- | --- |
| `F` | 选择 `front`（正面） |
| `B` | 选择 `back`（反面） |
| `Space` | 手动尝试保存当前帧 |
| `A` | 仅在 `--auto-interval > 0` 时切换自动定时采集开/关；默认 `0` 时不起作用 |
| `Q` / `Esc` | 退出并关闭预览 |

自动采集需显式给出正数秒数，例如 `python scripts/capture_phone.py --part-id screw_A-1 --source 0 --backend dshow --auto-interval 2`，随后按 `A` 启动；即使自动采集，操作者仍须先选择正确的面。Space 和定时器都是“尝试采集”，不保证每次生成图像。

## 文件与元数据

默认每次启动创建独立会话，布局为：

```text
data/captures/<session>/
├── front/            # 已接受的正面 JPEG，如 000001.jpg
├── back/             # 已接受的反面 JPEG
├── session.json
└── manifest.jsonl
```

新 `session.json` 为 `schema_version: 2`，记录 `part_id`、来源、后端和参数；`face_labels: ["front", "back"]`、`class_names`（例如 `{"front": "screw_A-1_front", "back": "screw_A-1_back"}`）和 `manifest_label_semantics: "face_label"` 明确面标签与以后规范类名 `<part_id>_<face>` 的对应关系；保留 `labels: ["front", "back"]` 兼容字段，不分配数字类 ID。详情见 [Label Contract V2](LABEL_CONTRACT_V2.md)。会话仍是 `state: "raw_capture"`、`obb_labeled: false`、`train_ready: false`。`manifest.jsonl` 的 `label` 值仍为 `front` / `back` 面标签，**不是最终 YOLO 类名**；按尝试追加记录所选面、手动/自动、是否接受、相对保存路径（拒绝时为空）、模糊/亮度指标、警告、拒绝原因及哈希等；仅已接受的帧写入对应面的 JPEG。面目录是人工选择的原始图像分组，**没有伪造 OBB 标注**，不可直接用于 OBB 训练。不要把原始图像或含敏感信息的数据提交仓库。

## 质量校准与近重复

默认 `--quality-policy warn`：质量阈值 `--min-blur-score 0`、`--min-brightness 0`、`--max-brightness 255` 实际不筛掉正常 uint8 图像，尚未为具体镜头/零件校准。预览会显示模糊分数（灰度图 Laplacian 方差，越大通常越清晰）和亮度（灰度均值）；每次尝试也记录指标。先在固定环境中观察清晰/模糊、正常/过暗/过亮样本的分布并人工复核；再视本设备和光照条件设定阈值，在 `warn` 下检查警告与误报，**校准后**才考虑 `--quality-policy reject`。没有通用有效阈值。以下 PowerShell 示例交互读取操作者校准后的数值，不预设推荐阈值：

```powershell
$blur = [double](Read-Host "Calibrated minimum blur score")
$dark = [double](Read-Host "Calibrated minimum brightness (0-255)")
$bright = [double](Read-Host "Calibrated maximum brightness (0-255)")
python scripts/capture_phone.py --part-id screw_A-1 --source 0 --backend dshow --min-blur-score $blur --min-brightness $dark --max-brightness $bright --quality-policy warn
# 人工检查并校准后，才可将末尾改为 --quality-policy reject
```

质量警告在 `warn` 模式不阻止保存，在 `reject` 模式阈值不合格会拒绝。**近重复检查独立于质量 warn/reject**：工具对每帧计算灰度 dHash，与当前所选面的已接受哈希比较；默认 Hamming 距离 `<= 5` 时拒绝近重复。`front` 和 `back` 分别保存最多 50 条已接受哈希的有界会话历史（可通过 `--duplicate-distance`、`--duplicate-history` 调整）；拒绝帧和写入失败帧不加入历史。哈希筛重只减少明显近重复，不能替代人工核查或按采集序列隔离数据。

## 现场采集与后续门禁

1. 固定手机支架、拍摄角度和距离；保持照明、背景稳定，记录条件和合法图像来源。先拍只有**一个清楚可见零件**的试采集，不可辨别的面不要强行归类。
2. 拍每帧前用 `F` / `B` 人工选对可见面；缓慢旋转或挪动零件以覆盖角度与位置，拍摄瞬间让手离开画面，检查清晰度、曝光与面标签。一个可选的试采起点是每面约 **100–300 张已接受图像**，只是工程规划目标，**不是要求达到的最低数量**或数据合格证明。
3. 改变光照、背景、角度或拍摄批次时用独立会话并记录条件，避免相近帧跨训练/验证/测试造成泄漏；复核 manifest、图像和拒绝记录，剔除误选面、遮挡和近重复。

**UNCERTAIN 仅为运行时状态，不是训练类。Training Gate 保持关闭**：必须先审查原始采集、添加并 QA OBB 四角标注、按会话/采集序列隔离训练/验证/测试，并解决此前[公开 OBB 数据质量门禁](DATASET_CAPABILITY_MATRIX.md)（获取修正/当前完整数据集，或冻结经明确审查的过滤策略），再另行决定训练。手机采集未验证真实硬件，也未产出可直接训练的 OBB 数据。
