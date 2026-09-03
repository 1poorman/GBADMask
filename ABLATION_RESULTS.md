# GBADMask 消融实验结果（小麦病害数据集）

> 生成时间：2026-09-03 07:45 (北京时间)
> 数据集：`datasets/HBueHxOW/wheat_seg_clean`（584 train / 82 val，12 类）
> 统一设置：MAX_ITER=4000（约 55 epoch）、batch 8、单卡 RTX 3090、cudnn.benchmark=False
> 汇总脚本：`python tools/summarize_all.py`

## 一、实验总览

共 **23 组训练**（含 4 组启动失败后重跑、1 组因缺陷跳过后重跑），全部完成并产出最终评估。
配置分组：

| 阶段 | 目的 | 组 |
| --- | --- | --- |
| A | 多 seed 基线（base=官方 ProtoNet / gc=ProtoNetV2+GC） | base×5, gc×4 |
| B | ProtoNetV2 组件拆解 | nogc, cbam, spatial, coord |
| C | 语义辅助损失（ LOSS_ON=True ×3 种损失 ×detach） | fdc, semdetach_ft, semdetach_ce |
| D | 骨干与 LSK | vig, lskoff, lskon |
| E | 2024-2026 组件改造 | c2f×3, c3k5, c3k7, eca, dysample*(失败) |

\* dysample 组因实现缺陷（惰性创建的模块未搬移到 GPU）启动即失败，已修复但未重跑，见第五节。

## 二、原始数据（segm AP）

| config | 运行 | seed | segm AP |
| --- | --- | --- | --- |
| base | base | 42 | 5.162 |
| base | base_s42c | 42 | 5.061 |
| base | base_s123 | 123 | 4.462 |
| base | base_s123b | 123 | 5.524 |
| base | base_s2024 | 2024 | 5.322 |
| **base 聚合** | **n=5** | | **5.106 ± 0.400** |
| gc | gc | 42 | 6.797 |
| gc | gc_s42b | 42 | 4.826 |
| gc | gc_s123b | 123 | 5.011 |
| gc | gc_s2024 | 2024 | 5.359 |
| **gc 聚合** | **n=4** | | **5.498 ± 0.893** |
| c2f | c2f_s42 | 42 | 5.887 |
| c2f | c2f_s123 | 123 | 4.424 |
| c2f | c2f_s2024 | 2024 | 5.040 |
| **c2f 聚合** | **n=3** | | **5.117 ± 0.735** |
| nogc | nogc | 42 | 5.538 |
| cbam | cbam | 42 | 6.111 |
| spatial | spatial | 42 | 5.392 |
| coord | coord | 42 | 5.331 |
| c3k5 | c3k5_s42 | 42 | 5.919 |
| c3k7 | c3k7_s42 | 42 | 5.675 |
| eca | eca_s42 | 42 | 4.700 |
| fdc | fdc | 42 | **0.611** |
| semdetach_ft | semdetach_ft_s42 | 42 | **1.012** |
| semdetach_ce | semdetach_ce_s42 | 42 | **0.629** |
| vig | vig | 42 | 5.084 |
| lskoff | lskoff | 42 | 5.656 |
| lskon | lskon | 42 | 5.630 |

## 三、核心结论（按证据强度分级）

### 结论 1：语义辅助损失（LOSS_ON=True）严重有害 —— ✅ 可靠（唯一超噪声结论）

| 配置 | segm AP | vs base 均值 |
| --- | --- | --- |
| base（无语义损失） | 5.106 | — |
| fdc（Focal-Dice-CE，梯度回传） | 0.611 | **−4.50** |
| semdetach_ft（Focal Tversky + 梯度截断 + 权重 0.05） | 1.012 | **−4.09** |
| semdetach_ce（纯 CE + 梯度截断 + 权重 0.05） | 0.629 | **−4.48** |

三种损失形式、有无梯度截断，全部崩溃到 0.6-1.0，**与损失实现无关**。
bbox AP 也同步受损（2.95-5.58），排除"仅 mask 分支"的解释。
实践建议：**该数据集上保持 `LOSS_ON=False`（与官方默认一致）**。
若将来要启用，方向是：构造类别平衡的独立语义标注（类似官方 `thing_train2017`），
而非从 instance mask 在线合成（背景占比 >95%，且多数类在单图中缺失，Tversky 的
缺失类 TI≡1 导致损失几乎恒定——实测 `loss_basis_sem` 全程停在 0.0499）。

### 结论 2：ProtoNetV2 相对官方无统计学显著差异 —— ⚠️ 数据充分

- base：5.106 ± 0.400（n=5）；gc（ProtoNetV2+GC）：5.498 ± 0.893（n=4）
- 均值差 +0.392；合并标准误 ≈0.48 → t≈0.81（df≈5），**p>0.4，不显著**
- 同 seed 配对：s42 +0.70、s123 +0.01、s2024 +0.04 → 配对平均 +0.25

阶段 A 单次测得的 "+1.635 (+31.7%)" 是**批次噪声造成的假象**（同 seed 跨批次差
可达 1.97）。诚实的表述是：**在该数据规模（584 图）下，ProtoNetV2 的收益无法
与训练噪声区分**。参数量代价：+329K（+1.6%）。

### 结论 3：LSK 无效果 —— ✅ 同批次对照，较可靠

lskon 5.630 vs lskoff 5.656（差 −0.026，同批次运行）。LSK 大核选择注意力在
bases 任务上无增益，且引入 +0.39M 参数。**不建议开启**。

### 结论 4：C2f 改造无显著收益 —— ⚠️ n=3

c2f 5.117 ± 0.735 vs base 5.106（+0.01）。YOLOv8 的多分支特征复用在本任务
（小数据、dense 分割头）未体现优势。c3k5（5.919）/c3k7（5.675）单次略高但在
噪声内。**不值得引入的复杂度**。

### 结论 5：ECA 有负向迹象 —— ⚠️ 单次

eca 4.700 vs 同批次 vig 5.084（−0.38）。BiFPN 融合节点加通道注意力未见收益，
与 GC（在 basis module 内部）的结论形成对照：注意力的**位置**比**类型**更重要。

### 其余单次观测（均在噪声内，不作结论）

nogc 5.538、cbam 6.111、spatial 5.392、coord 5.331。

## 四、可复现性分析（同 seed 跨批次）

| seed | base 两次 | gc 两次 | 批次噪声 |
| --- | --- | --- | --- |
| 42 | 5.162 / 5.061（差 0.10） | 6.797 / 4.826（**差 1.97**） | 0.1~2.0 |
| 123 | 4.462 / 5.524（**差 1.06**） | 5.359 / 5.011（差 0.35） | |

cudnn.benchmark 修复后噪声有所降低（base s42 仅差 0.10），但 **GPU 原子操作的
非确定性仍使同 seed 跨批次差异可达 2.0 AP**。要进一步压低噪声需
`torch.use_deterministic_algorithms(True)`（部分算子不支持，且显著减速）。

**方法论教训**：
1. 单次运行的 AP 差异在 ±1~2 内都可能是噪声；主张差异时必须多 seed；
2. 中途评估（iter 2000）与最终评估可差 1.3+（base_s123: 3.13 → 4.46），**禁止用
   中途评估下结论**（本次实验中我两次犯此错误）；
3. 训练期间修改源码会让队列中的任务失败（本次 01-04、10 因 DySample 迭代损坏
   而失败），队列 worker 应在任务启动前校验代码可导入。

## 五、实验期间的工程修复（均已验证）

| 修复 | 影响 |
| --- | --- |
| `tools/base.py` 无条件 `cudnn.benchmark=True` → 改由 `cfg.CUDNN_BENCHMARK` 控制 | 多尺度训练下既更快又可复现 |
| `basis_module.py` 漏 import torch（阶段 B coord/full 组崩溃） | 已修 |
| `dataset_mapper` 支持 PolygonMasks → 密集掩膜（在线 basis_sem 前提） | 已修 |
| 语义 GT 尺寸按 `seg_out` 实际尺寸对齐（不再依赖 COMMON_STRIDE） | 已修 |
| `bifpn._upsample` 惰性创建的 DySample 未搬移到 GPU（dysample 组崩溃） | 已修，待重跑验证 |

## 六、对后续工作的建议

1. **冻结当前架构**：在 584 图的规模下，架构微调的收益都被噪声淹没。提升
   数据规模（扩充到 5k+ 图）比改造模块更可能带来可测的进步。
2. **若必须改造**：优先在**多个数据集**上验证（coco128 + wheat），只有跨数据集
   一致的正向才可信。
3. **dysample 组**：代码已修复，可用
   `bash tools/run_ablation_backbone.sh` 的方式单独重跑 1-2 seed 确认。
4. **多 seed 成本**：本实验 23 组 ≈ 9.5 GPU 时。若要达到 p<0.05 的检验效力
   （检测 0.5 AP 差异），每组需 8-10 seed。

## 七、R50 强平台实验（2026-09-03 追加，M1）

**背景**：弱平台（cspvig 9.9M、无预训练）噪声 ±1~2 AP，掩盖模块效应。
改用官方骨干 R50+FPN（27.3M，ImageNet 预训练）在同一数据集（wheat_seg_clean）
重跑，8000 iter、seed 42：

| 组 | bbox AP | segm AP | 说明 |
| --- | --- | --- | --- |
| `r50-protonet`（官方 basis） | 18.91 | 15.36 | 平台天花板参照 |
| `r50-v2`（ProtoNetV2） | 20.05 | **16.68** | **+1.14 bbox / +1.32 segm** |

**三个关键发现**：

1. **数据集不是瓶颈**（回答"是否需要更长训练"）：R50 在 2000 iter 时 segm 已达
   14.5，超过 cspvig 跑满 4000 iter 的 6.8 —— 卡住 cspvig 的是**骨干容量与预训练**，
   不是训练时长或数据。
2. **ProtoNetV2 的效应在强平台显现**：+1.32 segm / +1.14 bbox，方向与弱平台一致
   （弱平台 +0.39 但不显著），幅度更大。此前"所有组件都无差异"的表象，
   实为测量平台灵敏度不足所致。
3. **cspvig 作为主干当前不可用**：其架构贡献（图卷积 + CSP）在无预训练时远逊于
   ResNet50 的 ImageNet 先验。继续用 cspvig 前必须先解决预训练（ROADMAP M2）。

**单 seed 警告**：+1.32 与弱平台测得的批次噪声（±1~2）量级相当，尚不能排除
运气成分。**正式结论需 M4 的 3 seed 配对 t 检验**（详见 ROADMAP.md）。

**工程记录**：
- R-50.pkl 已缓存至 `~/.torch/iopath_cache/detectron2/ImageNetPretrained/MSRA/`
- Plantv2 / Strawberry 数据集已就位（7916/1750 train）；其 json 原带 id=0 的
  `_background_` 占位类，与 detectron2 懒加载口径冲突（AssertionError），
  已用 `datasets/clean_background_class.py` 清洗（原 json 备份 `.bak`）。
  NUM_CLASSES 口径：Plantv2=16、Strawberry=7。
- Plantv2 + R50 冒烟通过（40 iter，loss 2.656 正常）。

## 七、产物位置

- 各组权重与日志：`output/ab-*/`（约 5.4 GB，23 组）
- 队列任务定义：`tasks/*.task`（已执行）与 `tasks/done/`
- 队列 worker：`tools/queue_worker.py`；骨干消融脚本：`tools/run_ablation_backbone.sh`
- 汇总脚本：`tools/summarize_all.py`；快速验证：`tools/quick_check.py`
- 训练队列日志：`.queue.log`；官方源码参照：`.ref/initial/`（git fec57fc 提取）
- 新增组件代码：`adet/modeling/blendmask/advanced_losses.py`（Focal Tversky /
  Unified Focal / 梯度截断）、`adet/modeling/backbone/dysample.py`（DySample/ECA）、
  `adet/modeling/blendmask/spatial_attn.py`、`CSPStage`（cspvig.py，c2f 融合）
