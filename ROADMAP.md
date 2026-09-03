# GBADMask 研发蓝图（ROADMAP）

> 版本：v1.1（2026-09-03，M1 完成后更新）
> 依据：23 组弱平台消融 + R50 强平台实验的实证结论
> 定位：本文档是可执行的研发计划。每个里程碑独立可交付、自带测试方案，
> 可直接交给自动化脚本按序执行。

---

## 0. 背景与已确认的事实

### 0.1 已确认的结论（详见 ABLATION_RESULTS.md）

| # | 结论 | 证据强度 |
| --- | --- | --- |
| F1 | **瓶颈是骨干，不是数据集**：R50（27M，预训练）仅 2000 iter 即达 segm 14.5、最终 15.4；cspvig（9.9M，无预训练）跑满 4000 iter 仅 6.8 | ✅ 差 2.3 倍 |
| F2 | **语义辅助损失有害**：三种损失 × 梯度截断全部崩（0.6~1.0 vs base 5.1） | ✅ 差 −4.1~−4.5 |
| F3 | **弱平台上无法测量模块效应**：单次噪声 ±1~2 AP | ✅ 方差分析 |
| F4 | **强平台上 ProtoNetV2 效应显现**：R50 上 v2 − protonet = **+1.32**（16.68 vs 15.36），方向与弱平台一致且幅度更大 | ⚠️ 单 seed，待 M4 确认 |
| F5 | GPU 训练存在固有非确定性：同 seed 跨批次差 0.1~2.0 AP | ✅ 已验证 |
| F6 | 当前 ProtoNet 与官方 BlendMask 数值等价（state_dict 偏差 0） | ✅ 已验证 |
| F7 | LSK / C2f / ECA / DySample / 坐标编码在弱平台上均无增益 | ⚠️ 弱平台上不可下结论（M4 复测） |

### 0.2 核心推断

> **ProtoNetV2 的效应真实存在，弱平台（9.9M、无预训练、584 图）测不出来。**
> R50 强平台上 +1.32 的信号方向明确。M4 的任务是在强平台 + 大数据上把它做成
> 统计显著（3 seed 配对 t 检验）。

### 0.3 数据集清单（全部就绪，2026-09-03）

| 数据集 | train/val | 类别 | 说明 |
| --- | --- | --- | --- |
| `wheat_seg_clean` | 584 / 82 | 12 | 小麦病害；弱平台消融用 |
| `Plantv2` | **7916** / 2024 | **16** | PlantVillage 系作物病害，test2017 另有独立集 |
| `Strawberry` | 1750 / 750 | **7** | 草莓病害 |
| `coco128-seg` | 96 / 32 | 71 | 冒烟/链路验证用 |

> ⚠️ **`_background_` 口径坑（已解决）**：Plantv2/Strawberry 的 json 原带 id=0 的
> `_background_` 占位类（无任何实例引用）。它导致注册口径（过滤背景）与
> detectron2 懒加载口径（全量）冲突，训练启动即 AssertionError。
> 已用 `datasets/clean_background_class.py` 清洗（原 json 备份为 `.bak`）。
> **NUM_CLASSES 口径**：Plantv2=16、Strawberry=7、wheat=12。

### 0.4 环境与工具（已就绪）

| 资源 | 位置 |
| --- | --- |
| conda 环境 | `gbadmask`（torch 2.0.1+cu117，GPU 1 可用） |
| 训练入口 | `tools/train_bl+.py`（`--dataset <名>` 自动注册并改写 cfg.DATASETS） |
| 数据集注册 | `tools/register_datasets.py`（自动过滤 `_background_`） |
| 队列 worker | `tools/queue_worker.py`（动态追加任务、断点续跑） |
| 统计汇总 | `tools/summarize_all.py`（多 seed 均值±std、配对比较） |
| 配置快检 | `tools/quick_check.py`（CPU 前向/反向验证，无需数据集） |
| 公共配置 | `configs/run-wheat.yaml` / `run-wheat-r50.yaml`（R50 强平台） |
| 预训练权重 | `~/.torch/iopath_cache/detectron2/ImageNetPretrained/MSRA/R-50.pkl` |

**统一实验规范**（所有里程碑必须遵守）：

```bash
# 换数据集 = 改 GBADMASK_DATA_ROOT 为该数据集目录（train 脚本自动扫描同级目录注册）
export GBADMASK_DATA_ROOT=$PWD/datasets/Plantv2      # 例
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1
# SEED 必须 ≥3 个；评估只用最终 iter（EVAL_PERIOD 触发的最后一次）
# 不在训练期间修改 adet/ 下的源码（会污染队列中等待的任务）
```

---

## M1 强平台基线（R50 + FPN）　✅ 已完成

### 目标

回答两个问题：① 小麦数据集的 AP 天花板；② **强平台上 ProtoNetV2 的效应是否可测**。

### 步骤

| # | 步骤 | 状态 |
| --- | --- | --- |
| 1.1 | 下载 R-50.pkl ImageNet 预训练到 iopath 缓存 | ✅ 完成 |
| 1.2 | `configs/run-wheat-r50.yaml`（R50+FPN，其余与 run-wheat 一致） | ✅ 完成 |
| 1.3 | 训练 `r50-protonet`（官方 basis，8000 iter） | ✅ 完成 |
| 1.4 | 训练 `r50-v2`（ProtoNetV2，8000 iter） | ✅ 完成 |

### 结果（seed 42，8000 iter）

| 组 | bbox AP | segm AP | 说明 |
| --- | --- | --- | --- |
| `r50-protonet`（官方 basis） | 18.91 | 15.36 | 数据集天花板参照 |
| `r50-v2`（ProtoNetV2） | 20.05 | **16.68** | **+1.14 bbox / +1.32 segm** |

（对照：cspvig 弱平台同 seed 跑满 4000 iter 仅 5.1~6.8；R50 在 **2000 iter 时
segm 已达 14.5**，超过 cspvig 的最终值）

### 判定（按预设标准）

- `r50-protonet` segm AP = 15.36 **≥ 15** → ✅ 数据集可用，M3 的数据扩充有意义
- `r50-v2 − r50-protonet` = **+1.32 > 0.8** → ✅ **效应在强平台上可测**
  - 且方向与弱平台一致（弱平台 +0.39 不显著，强平台 +1.32）——
    **证实"效应真实存在、弱平台测不出"的推断（F3/F4）**
- 单 seed 下 +1.32 仍可能与批次噪声（±1~2）混淆 → **M4 必须做 3 seed 统计检验**

### 后续动作

进入 M3（数据已就绪）与 M2（cspvig 预训练）并行；M4 的主平台确定为 R50+FPN。

---

## M2 cspvig 预训练权重（关键路径）　待启动

### 目标

消除"从零训练"这一最大混淆变量，让 cspvig 与 R50 站在同一起跑线。

### 步骤

| # | 步骤 | 说明 |
| --- | --- | --- |
| 2.1 | 获取 MobileViG 官方 ImageNet 权重 | 源：[GitHub waltonfuture/MobileViG](https://github.com/waltonfuture/MobileViG)（MIT 协议，`MobileViG_M_80_7.pth.tar` 等三个规模）。若 GitHub 不通，从 HF 镜像 `hf-mirror.com` 找同款 |
| 2.2 | **key 映射转换脚本** `tools/convert_mobilevig_weights.py` | 官方权重是 timm 风格的 `blocks.N.0.xxx` 序列；`cspvig.py` 用 `local_backbone1/2/3`、`backbone`、`stem` 命名且经 CSPStage 封装，必须逐层映射。转换后保存为 detectron2 兼容的 `.pkl`（键加 `backbone.` 前缀） |
| 2.3 | **加载正确性验证**（关键！） | 写 `tools/verify_pretrain.py`：① 加载后打印 missing/unexpected keys，missing 必须只含 `cls_logits`/检测头等新头；② 随机切 1000 张 ImageNet val 图（或用 wheat val 图近似）算 top-1，应显著高于随机（MobileViG-M 应 ≥75%） |
| 2.4 | 冻结式微调试验（可选但推荐） | 前 500 iter 冻结 backbone 只训头，loss 应显著低于从零训练的同期 loss —— 这是最快的"预训练是否生效"信号 |
| 2.5 | 训练 `cspvig-pre_s{42,123,2024}` | 3 seed × 8000 iter，配置 = `run-wheat.yaml` + `MODEL.WEIGHTS converted.pkl` |

### 测试方案

```bash
# 2.2/2.3 的自动化验证
python tools/convert_mobilevig_weights.py --ckpt MobileViG_M_80_7.pth.tar --out pretrain/cspvig_M.pkl
python tools/verify_pretrain.py --pkl pretrain/cspvig_M.pkl
# 期望输出：missing keys 仅含非 backbone 部分；probe acc >> 1/1000 类均匀分布

# 2.4 冻结微调信号（500 iter 对比）
# 从零：total_loss ~2.6（见既有日志）；预训练：应明显更低
```

**判定标准**：

- verify 通过 + 冻结试验 loss 更低 → 预训练生效，跑 2.5
- 权重无法获得 → **降级方案**：M4 全部在 R50 强平台上做（M2 跳过），cspvig 的
  架构价值存疑另议

### 预计耗时

转换+验证 2 小时（人工核对 key 映射）＋ 3 seed × 67 分钟 ≈ 5.5 小时

### 风险

- MobileViG 原始权重的 BN 统计量与我们的 BN 层对不上（我们 BN 是 fused Conv-BN）
  → 转换脚本需做 `conv.weight + BN` 折叠，或把 cspvig 的 Conv-BN 拆开匹配。**推荐后者**（改 key 映射不改结构，保持与已跑实验可比）

---

## M3 数据扩充：Plantv2 / Strawberry + 100 epoch　✅ 数据已就位

### 目标

用更大规模数据把测量信噪比提上来，并检验 ProtoNetV2 在大样本上的效应。

### 数据现状（2026-09-03 已就位）

| 数据集 | train/val | 前景类 | 100 epoch iter（batch 8） | 预估单组耗时* |
| --- | --- | --- | --- | --- |
| `Strawberry` | 1750 / 750 | 7 | **21 900** | ~3 h |
| `Plantv2` | **7916** / 2024 | 16 | **98 950** | ~14 h |

\* 按 R50 实测 0.5 s/iter 估算。Plantv2 单组即 14 h，**3 seed × 2 配置不现实**，
需按下述执行策略裁剪。

### 步骤

| # | 步骤 | 状态 |
| --- | --- | --- |
| 3.1 | 数据就位（COCO 结构，`train_bl+.py` 自动扫描注册） | ✅ 完成 |
| 3.2 | `_background_` 占位类清洗（口径统一，json 已备份 `.bak`） | ✅ 完成 |
| 3.3 | Plantv2 冒烟（R50 + 40 iter，loss 2.656 正常下降） | ✅ 完成 |
| 3.4 | 人工抽查 20 张标注可视化（polygon 贴合度、类别正确性） | ⏳ 待做（写 `tools/vis_anns.py`） |
| 3.5 | 100 epoch 训练（按下面裁剪策略执行） | ⏳ 待做 |

### 100 epoch 执行策略（考虑单卡时间预算）

```bash
# Strawberry（3h/组，可全量执行）
#   100 epoch = 21 900 iter
export GBADMASK_DATA_ROOT=$PWD/datasets/Strawberry
#   配置关键段：DATASETS=("Strawberry_train"/"Strawberry_val") NUM_CLASSES=7
#   MAX_ITER=21900 STEPS=(14500,19500) EVAL_PERIOD=7000

# Plantv2（14h/组，建议二选一）
#   a) 50 epoch = 49 500 iter ≈ 7 h/组，先跑 base/gc 各 1 seed 看方向
#   b) 100 epoch 但只跑 ProtoNetV2（用户指定的主配置），1 seed
#   若 GPU 0 可用，双卡并行减半
```

配置模板（`configs/run-plantv2-r50.yaml`，R50 强平台 + Plantv2）：

```yaml
_BASE_: "run-wheat-r50.yaml"     # 继承 R50+FPN 与超参骨架
MODEL:
  BASIS_MODULE: {NUM_CLASSES: 16}
  FCOS: {NUM_CLASSES: 16}
DATASETS:
  TRAIN: ("Plantv2_train",)
  TEST: ("Plantv2_val",)
SOLVER:
  MAX_ITER: 98950                 # 100 epoch × 7916 / 8
  STEPS: (65300, 88000)           # 0.66 / 0.89 × MAX_ITER
  CHECKPOINT_PERIOD: 20000
TEST: {EVAL_PERIOD: 20000}
OUTPUT_DIR: "output/plantv2-r50-v2"
```

### 测试方案

```bash
# 3.4 标注可视化（人工抽查 20 张）
python tools/vis_anns.py --dataset Plantv2_train --num 20 --out output/vis_anns/

# 3.5 训练后
python tools/summarize_all.py
# 期望：Plantv2 上 segm AP 显著高于 wheat（数据 13 倍）
# 期望：r50-v2 − r50-protonet 的差值方向与 wheat 上一致（+）
```

**判定标准**：

- Plantv2 base 的 segm AP ≥ 25（数据 13 倍，天花板应明显抬升）
- ProtoNetV2 相对官方的增益在 Plantv2 上**方向为正** → 与 wheat R50 结果
  互相印证，写入论文
- 若 Plantv2 上方向为负 → 单数据集结论不可靠，M4 改为三数据集平均

### 风险

- Plantv2 图片为 PlantVillage 室内单叶图（背景统一、病斑居中），**分割难度可能
  偏低**，AP 天花板高不代表田间泛化好 —— 论文表述需谨慎
- Strawberry 与 Plantv2 的类别体系不同（7 vs 16），**不可混训**，分开跑

---

## M4 强平台正式消融　依赖 M1（✅）/ M3（数据已就位）

### 目标

在测量灵敏度足够的平台（**R50+FPN，M1 已验证**）上重新测定 basis module 各组件，
得到**可写进论文**的结论。

### 步骤

| # | 内容 | 组数 |
| --- | --- | --- |
| 4.1 | 主平台：**R50+FPN**（M1 已验证效应可测）；cspvig+预训练作为第二平台（若 M2 完成） | — |
| 4.2 | 数据：wheat（快）+ Strawberry（中）+ Plantv2（慢，可 50 epoch） | — |
| 4.3 | 组件矩阵（3 seed × 3 数据集）：`base` / `gc` / `nogc` / `cbam` / `spatial` / `coord` | 理论 54 组 |
| 4.4 | 统计检验：配对 t 检验（同 seed 配对），报告均值差 ± 95% CI | — |

**裁剪建议**（单卡时间预算）：

| 数据集 | 组数 | iter/组 | 小计 |
| --- | --- | --- | --- |
| wheat（4000 iter） | 6 配置 × 3 seed = 18 | 4000 | ~18 h |
| Strawberry（100 ep = 21900） | base/gc 核心配置 × 3 seed = 6 | 21900 | ~18 h |
| Plantv2（50 ep = 49500） | base/gc × 3 seed = 6 | 49500 | ~42 h |
| **合计** | 30 组 | | **~78 h ≈ 3.3 天**（单卡） |

优先级：wheat 全矩阵（快）→ Strawberry 核心对 → Plantv2 核心对。
GPU 0 若可用，双卡分工减半。

### 测试方案

```bash
# 4.3 任务生成（示例，写入 tasks/ 交给 queue_worker）
for ds in wheat plantv2; do
  for cfg in base gc nogc cbam spatial coord; do
    for s in 42 123 2024; do
      echo "${cfg}_${ds}_s${s}\tDATASETS.TRAIN (\"${ds}_train\")\tDATASETS.TEST (\"${ds}_val\")\t... " \
        > tasks/${ds}_${cfg}_s${s}.task
    done
  done
done
nohup python tools/queue_worker.py tasks/ > .queue2.log 2>&1 &

# 4.4 统计检验（扩展 summarize_all.py 加 scipy.stats.ttest_rel）
python tools/summarize_all.py --ttest
```

**判定标准**（论文口径）：

- `gc vs base`：3 seed 同号、配对 t 检验 p<0.05 → **可宣称有效**
- 其余组件同理；任何 p≥0.05 的组件一律标注"无显著差异"

### 预计耗时

36 组 × 67 分钟 ≈ 40 GPU 时 → 双卡分工 20 小时，或只跑 R50 平台减半

---

## M5 后续方向（按 M4 结果分叉）

| M4 结果 | 走向 |
| --- | --- |
| ProtoNetV2 统计显著 | 写论文主线：多数据集验证 + CAM 可视化（`--cam-target mask`）佐证；补做 M2 的 cspvig 预训练构成完整故事 |
| 所有组件都不显著 | basis module 架构敏感性低成立；转向 **头部结构**（blender/attention）或 **损失**（构造类别平衡的独立语义标注后再试） |
| 部分显著 | 只保留显著组件做最小化模型，重新走 M4 |

**已废弃方向**（弱平台负证据，M4 可选择性复测确认）：LSK、C2f、ECA、DySample、
坐标编码、语义辅助损失（当前标注方式）。

---

## 附录 A：自动化操作手册

### A.1 队列任务格式

```
# tasks/<序号>.task  —— 每行：tag<TAB>配置覆盖（KEY VALUE 空格分隔，多组用 TAB 分隔）
gc_s42	MODEL.BASIS_MODULE.NAME ProtoNetV2	MODEL.BASIS_MODULE.ATTN gc	SEED 42
```

### A.2 启动队列

```bash
export GBADMASK_DATA_ROOT=$PWD/datasets/HBueHxOW/wheat_seg_clean   # 按数据集切换
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1
nohup /home/huachenghao/.conda/envs/gbadmask/bin/python tools/queue_worker.py tasks/ > .queue.log 2>&1 &
# worker 空闲 40 分钟自动退出；运行期间禁止修改 adet/ 源码
```

### A.3 结果汇总

```bash
python tools/summarize_all.py            # 全部
python tools/summarize_all.py output/ab-  # 指定前缀
```

### A.4 快速验证新配置（不占 GPU、不需要数据）

```bash
python tools/quick_check.py "MODEL.VIG.CSP_STYLE c2f" "MODEL.BiFPN.ATTN eca"
```

### A.5 数据集切换

`train_bl+.py` 自动扫描 `GBADMASK_DATA_ROOT` 父目录下所有 COCO 结构目录。
换数据集只需：① 数据放好；② 配置里改 `DATASETS.TRAIN/TEST` 与 `NUM_CLASSES`。

---

## 附录 B：里程碑依赖关系

```
M1 (强平台基线) ✅ ──┬──> M4 (正式消融，主平台=R50) ──> M5 (分叉)
M2 (cspvig 预训练) ⏳─┤
M3 (数据扩充) ✅数据就位─┘
```

- M1 ✅ 已完成（R50 效应可测，+1.32）
- M3 ✅ 数据就位（含 _background_ 清洗）；剩标注可视化抽查 + 训练
- M2 独立可并行（转换脚本无依赖），完成后 cspvig 可作为 M4 的第二平台
- M4 主平台已定为 R50+FPN，**现在即可启动**（wheat 全矩阵先行）
- 当前资源：GPU 1 单卡；若 GPU 0 可用，M3 的 Plantv2 长训练与 M4 的 wheat 矩阵并行
