# GBADMask 研发蓝图（ROADMAP）

> 版本：v3.1（2026-09-03 修订：验收口径改为**绝对 +5 AP**）
> 总目标：**小数据农作物病害实例分割上，模型大小 ≤ 1.5× 官方 BlendMask(R50)，
> 性能稳定提升 ≥ +5 AP（绝对）**
> 核心资产：**cspvigv2 新骨干**（MobileViGv2 + C3K2 融合，官方 ImageNet 预训练
> 1:1 映射，已完成正确性验证）+ 前期 23 组弱平台消融与 R50 强平台实验的实证结论
> 定位：可执行的研发计划。每个里程碑独立可交付、自带测试方案与验收指标。

---

## 0. 目标定义与验收口径

### 0.1 一句话目标

在小麦（584 图 / 12 类）、Plantv2（7916 图 / 16 类）、Strawberry（1750 图 / 7 类）
三个农作物病害实例分割数据集上，交付一个模型：

- **大小约束**：总参数量 ≤ **53.06M**（= 官方 BlendMask R50 基准 35.37M × 1.5）
- **性能约束**：segm AP 相对官方基准 **稳定提升 ≥ +5.0 AP（绝对）**（冲刺目标 ≥ +8.0 AP）

### 0.2 模型大小约束（2026-09-03 实测，`tests/count_params.py`）

| 配置 | 参数量 | 占基准比 | 预算判定 |
| --- | --- | --- | --- |
| **r50-protonet（官方基准 = 大小参照）** | **35.37M** | 1.00× | — |
| r50-v2（ProtoNetV2） | 35.70M | 1.01× | ✅ |
| r101-v2 | 54.64M | 1.54× | ❌ **超预算出局** |
| **vigv2-m + C3K2（旗舰骨干首选）** | **25.64M** | 0.72× | ✅ 余量 27.4M |
| vigv2-s + C3K2（轻量对照） | 17.11M | 0.48× | ✅ |
| vigv2-ti + C3K2 | 14.83M | 0.42× | ✅ |
| cspvig（旧骨干，无预训练） | 14.34M | 0.41× | （已被 F1 证伪） |

> 口径：完整检测模型（骨干 + Neck + FCOS 头 + basis 模块）。vigv2 系列配
> BiFPN（repeats=3, 160ch），r50 配官方 FPN，与各自实际使用的配置一致。

**结论**：vigv2-m 在预算内留 27.4M 余量，可在 basis/Neck 侧叠加新组件而不触顶；
R101 路线（原蓝图 A2）直接出局；DCNv2（仅适用 ResNet 支线）降为备选。

### 0.3 性能验收口径（"稳定 +5 AP（绝对）"的操作化定义）

| 项 | 定义 |
| --- | --- |
| 基准 | 官方 BlendMask R50 + 官方 ProtoNet（`r50-protonet`），同数据集同 iter |
| 指标 | **segm AP**（主），bbox AP（辅） |
| 提升幅度 | 3 seed 均值绝对 AP 相对基准 **≥ +5.0 AP**；冲刺目标 **≥ +8.0 AP**（绝对） |
| 稳定性 | ① 3 个 seed（42/123/2024）**全部为正增益**；② 与基准 3 seed **配对 t 检验 p<0.05**；③ 跨数据集（≥2/3）方向一致 |
| wheat 锚点 | 基准 15.36 → **+5.0 = 20.36**，**+8.0 = 23.36**（冲刺）；r50-v2 单 seed 16.68 仅达相对 +8.6%，距绝对目标仍差 3.7 AP |
| Strawberry 锚点 | 基准 16.68 → **+5.0 = 21.68**，**+8.0 = 24.68**（冲刺） |
| Plantv2 锚点 | 基准（r50-protonet，M6.3 跑）= 待填 → **+5.0 AP** 为验收线 |

> **口径已确认（2026-09-03）**：用户明确采用**绝对 +5 AP**，验收线按上表绝对口径
> 执行（不再使用相对 +5% 解释）。冲刺目标上调至 +8.0 AP。

### 0.4 已确认的事实（前期实验，详见 ABLATION_RESULTS.md）

| # | 结论 | 证据 |
| --- | --- | --- |
| F1 | 瓶颈是骨干与预训练，不是数据：R50 预训练 2000 iter 即超 cspvig 从零训练的最终值 | ✅ 2.3 倍差 |
| F2 | 语义辅助损失有害（三种损失全崩） | ✅ −4.1~−4.5 |
| F3 | 弱平台（9.9M 无预训练）测不出模块效应（噪声 ±1~2 AP） | ✅ |
| F4 | 强平台上 ProtoNetV2 效应显现：+1.32 segm（16.68 vs 15.36） | ⚠️ 单 seed |
| F5 | 任何"有效"宣称必须 3 seed + 配对 t 检验 | ✅ 方法论 |
| F6 | 注意力位置 > 类型：basis 内部 GC 有效，BiFPN 节点 ECA 无效 | ⚠️ 间接 |
| F7 | 禁止清单：语义辅助损失、LSK、C2f、BiFPN 侧 ECA、坐标编码 | ✅/⚠️ |
| **F8** | **cspvigv2 权重映射 1:1 完整**（ti/s/m 均 loaded 全部骨干键、mismatched=0、skipped 仅 11 个分类头键）；**C3K2 恒等初始化成立**（加载预训练后 C3K2 开/关输出 max\|diff\|=0） | ✅ 已实测（`tests/verify_cspvigv2.py`） |

### 0.5 数据与环境（全部就绪）

| 项 | 状态 |
| --- | --- |
| 数据集 | wheat 584/82；Plantv2 7916/2024；Strawberry 1750/750（`_background_` 已清洗） |
| 预训练权重 | `weights/MobileViG_V2_{Ti,S,M}_Class.pth`（B 变体代码已支持，权重需另下载） |
| 环境 | conda `gbadmask`，GPU 1（RTX 3090）；入口 `tools/train_bl+.py --dataset <名>` |
| 工具 | `queue_worker.py`（队列）、`summarize_all.py`（统计）、`quick_check.py`（CPU 快检）、`tests/count_params.py`（大小验收）、`tests/verify_cspvigv2.py`（映射验收） |

**统一实验规范**：`export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1`；wheat 统一
MAX_ITER=8000（与既有 r50 锚点可比）；评估只取最终 iter；队列运行期间不改 `adet/` 源码。

---

## 总体路线

```
M6.0 cspvigv2 验证 ──✅ 主体完成──> M6.1 骨干对决（wheat 快平台选旗舰骨干）
                                        │
              ┌──── 晋级：vigv2-m-pre ──┴── 未晋级：回退 R50 平台 ────┐
              ▼                                                       ▼
        M6.2 组件筛选（L1/L2 漏斗）                    R50 平台组件筛选（原方案）
              └──────────────────────┬──────────────────────────────┘
                                     ▼
                     M6.3 旗舰总装 + 三数据集验收（0.3 节口径）
                                     ▼
                     M6.4 消融成表 + 可视化（论文交付）
```

**决策点只有 M6.1 一个**：预训练新骨干是否打过 R50。打过则围绕 vigv2-m 建旗舰；
打不过则 R50 + 组件路线（两者共享 M6.2 的组件池与漏斗）。

---

## M6.0 cspvigv2 骨干验证　✅ 主体完成（剩 GPU 冒烟）

### 已完成（2026-09-03）

| # | 验证项 | 结果 |
| --- | --- | --- |
| 0.1 | 权重映射完整性（ti/s/m） | ✅ loaded=533/782/782，mismatched=0，skipped 仅分类头 |
| 0.2 | C3K2 恒等初始化（加载预训练后开/关输出一致） | ✅ p3~p7 max\|diff\| = 0.00e+00 |
| 0.3 | 参数量验收（≤53.06M） | ✅ m=25.64M / s=17.11M / ti=14.83M |
| 0.4 | 配置键注册（`VIG.VERSION/PRETRAINED/DROP_PATH/USE_C3K2`） | ✅ |

复现：`python tests/verify_cspvigv2.py && python tests/count_params.py`

### 待做

| # | 步骤 | 验收 |
| --- | --- | --- |
| 0.5 | GPU 冒烟：wheat + vigv2-m-pre，40 iter | loss 正常下降、无 NaN；记录 s/iter 与显存（供 6.5 排期） |
| 0.6 | 冻结微调信号（可选但强烈推荐）：前 500 iter 冻结 backbone 只训头 | total_loss 显著低于从零训练同期（参照既有从零日志 ~2.6）→ 预训练在检测链路上"生效"的直接证据 |

冒烟命令（与 0.5/0.6 对应）：

```bash
export GBADMASK_DATA_ROOT=$PWD/datasets/HBueHxOW/wheat_seg_clean
python tools/train_bl+.py --dataset wheat_seg_clean \
    --config-file configs/run-vigv2.yaml --num-gpus 1 \
    SOLVER.MAX_ITER 40 TEST.EVAL_PERIOD 100000
```

---

## M6.1 骨干对决：vigv2-m/s-pre vs R50（wheat 快平台，1 seed 筛选）

### 目的

回答唯一的决策问题：**带 ImageNet 预训练的 MobileViGv2+C3K2，是否在小麦数据集上
打过 R50 预训练基线？**

### 实验组（统一 wheat / 8000 iter / seed 42 / ProtoNetV2+GC basis）

| 组 | 骨干 | 预训练 | 参数量 | 锚点 segm |
| --- | --- | --- | --- | --- |
| `r50-protonet` | R50+FPN | ImageNet | 35.37M | 15.36（已有） |
| `r50-v2` | R50+FPN | ImageNet | 35.70M | 16.68（已有） |
| `vigv2s-pre` | vigv2-s+BiFPN | MobileViGv2-S | 17.11M | 待跑 |
| `vigv2m-pre` | vigv2-m+BiFPN | MobileViGv2-M | 25.64M | 待跑 |
| `vigv2m-scratch`（可选对照） | vigv2-m+BiFPN | 无 | 25.64M | 待跑 |

配置文件 `configs/run-vigv2.yaml`（新建，M6.1 唯一需要的工程动作）：

```yaml
_BASE_: "run-wheat.yaml"        # 继承 BiFPN/FCOS/basis/数据/超参
MODEL:
  BACKBONE: {NAME: "build_fcos_mobilevigv2_csp_bifpn_backbone"}
  VIG:
    VERSION: "m"                                  # 或 "s"
    PRETRAINED: "weights/MobileViG_V2_M_Class.pth" # 或 S；空串 = 从零对照
    USE_C3K2: True
    DROP_PATH: 0.1
SOLVER:
  MAX_ITER: 8000                                   # 与 r50 锚点对齐
OUTPUT_DIR: "output/vigv2m-pre"
```

### 判定（预设，防事后解释）

| 结果 | 决策 |
| --- | --- |
| `vigv2m-pre` segm ≥ r50-protonet **+1.0**（≥16.4） | ✅ **旗舰骨干 = vigv2-m**，进 M6.2 |
| `vigv2s-pre` ≥ r50-protonet +1.0 但 m 不达标 | 旗舰骨干 = vigv2-s（更小，余量更大） |
| 两者都 < 16.4 但 ≥ r50-v2 − 1.0（≥15.7） | 灰区：vigv2-m 进 M6.2 但 R50 支线并行保留 |
| 两者都 < 15.7 | 回退 R50 平台（原组件路线），cspvigv2 价值存疑，查 0.6 信号找原因 |

### 预计耗时

vigv2-m 算力约为 R50 的 1/3~1/2（MRConv 有 roll 开销），预估 40~70 min/组 ×
2~3 组 ≈ **1.5~3 h**，当晚可出决策。

---

## M6.2 组件筛选漏斗（L1 → L2）

> 平台 = M6.1 决出的旗舰骨干。所有组件的参数量增量都远小于 27.4M 余量，
> 大小约束在本阶段不构成筛选条件，但仍逐组记录。

**代码就绪状态（2026-09-03）**：A1（Copy-Paste + LSJ）、B1（PSA/C2PSA）、B2（EMA）已实现并通过
`tests/test_m62_components.py` 单元验证。L1/L2 训练队列**排在 M6.1 之后**——当前
仅 GPU 1 空闲且正跑 M6.1，待其释放即按 L1 协议挂 A1 / B1 / B2 / C1~C3 单 seed 快检。

### 候选池（按优先级）

| ID | 组件 | 参数增量 | 插入点 | 预期增益 | 成本 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | **Copy-Paste + 大尺度抖动（LSJ）** | 0 | dataset_mapper | +1~3 AP（小数据最敏感） | 中（新代码） | **P0** |
| B1 | **PSA / C2PSA**（YOLO11 引入、YOLO26 沿用的位置敏感自注意力） | ~+0.4M | basis tower（`ATTN` 注册表新分支） | +0.5~1.5 | 中（纯 PyTorch） | **P0** |
| B2 | **EMA** 高效多尺度注意力（ICASSP'23） | ~+0.06M | 同上 | +0.3~1.0 | 低（~60 行） | P1 |
| C1 | **NUM_BASES 4→8**（bases 容量，零代码） | 输出头 +128 通道 | 仅配置 | 未知 | 零 | P1 |
| C2 | **DROP_PATH 消融**（0 / 0.05 / 0.1，小数据+预训练下 0.1 可能过强） | 0 | 仅配置 | 未知 | 零 | P1 |
| C3 | **C3K2 消融**（USE_C3K2 True/False，骨干自身贡献拆解，论文必需） | 0 | 仅配置 | — | 零 | P1 |
| D1 | DCNv2 入骨干 | — | **仅 R50 支线**（vigv2 是图卷积结构，不适用） | +0.5~1.5 | 零代码 | 支线 |
| D2 | R101 | +19M | 超预算出局 | — | — | ❌ |

**插入点纪律**（F6）：新注意力只进 basis tower / 低层分支，**不进 BiFPN**；
禁止清单（F7）全程有效。

### 漏斗协议

```
L0 快检（CPU，分钟级）   quick_check 前向/反向 + count_params 记录
        │
L1 筛选（seed42 × 8000 iter，wheat）
        │                晋级线：Δsegm ≥ +0.8 vs 旗舰骨干基线（M6.1 结果）
        │                每批队列携带同批次基线锚点，排除批次噪声
        │
L2 确认（seeds {42,123,2024}）
        │                晋级线：3 seed 同号 + 配对 t 检验 p<0.05
        ▼
L3 总装（M6.3）
```

### 实施要点

**A1 Copy-Paste + LSJ**（最高预期收益，最先做）—— ✅ **已实现（2026-09-03）**

```bash
# 新增代码：
#   adet/data/copypaste.py        —— DonorPool（惰性建池）+ apply_copy_paste（mask 合成粘贴）
#   adet/data/augmentation.py     —— RandomScaleCrop（LSJ：随机缩放 + 固定尺寸裁剪/填充）
#   adet/data/dataset_mapper.py   —— __call__ 增广前插入 Copy-Paste（粘贴实例一并参与增广）
#   adet/data/detection_utils.py  —— build_augmentation 增加 LSJ 分支
#   adet/config/defaults.py       —— INPUT.COPYPASTE.* / INPUT.LSJ.* 配置键
#
# 启用（命令行 --opts，或在 YAML 中设 True）：
#   Copy-Paste : INPUT.COPYPASTE.ENABLED True   （PROB 0.5 / MAX_DONORS 8 / NUM_SAMPLES 2000）
#   LSJ        : INPUT.LSJ.ENABLED True          （SCALE_RANGE (0.3,1.5) / CROP_SIZE (640,640)，
#                                                小图保守区间；原版 [0.1,2.0] 会过度破坏小目标）
#
# 验收：tests/tmp_smoke_m62.py 已验证 PSA/配置/LSJ/Copy-Paste 逻辑全通过。
# 风险：病斑贴到不真实背景可能掉点 → 先不做旋转/形变；负增益则 PROB=0.25 重试一次。
# 人工抽查：tools/vis_anns.py 抽 20 张增广样本（粘贴边缘、标注一致性）后再挂 L1。
```

**B1 PSA（C2PSA 核心）** —— ✅ **已实现（2026-09-03）**

```bash
# 新增代码：
#   adet/modeling/blendmask/psa.py —— PSA（C2PSA 风格）+ Attention（标准缩放点积 MHA）
#   adet/modeling/blendmask/basis_module.py —— build_attention() 注册 "psa"
#
# 接口：与 GCNet/CBAM 同签名 build_attention(name, channels) -> (B,C,H,W)->(B,C,H,W)，
#       要求 c1==c2（PSA 严格满足）；num_heads 自动取可整除值（128→8，24→4）。
# 启用：MODEL.BASIS_MODULE.ATTN psa   （只动 basis tower / 低层分支，不进 BiFPN，符合 F6）
#
# 验证：tests/tmp_smoke_m62.py 已确认 PSA 前向通道守恒、build_attention('psa',128) 返回 PSA。
# 注意：晋级参照是 ATTN=gc 的旗舰基线 —— PSA 必须赢过已被 F4 验证的 GC 才算数（见 L1 协议）。
```

**B2 EMA 注意力（ICASSP'23 跨空间学习）** —— ✅ **已实现（2026-09-03）**

```bash
# 新增代码：
#   adet/modeling/blendmask/ema.py —— EMA（Efficient Multi-Scale Attention，ICASSP 2023）
#   adet/modeling/blendmask/basis_module.py —— build_attention() 注册 "ema"
#
# 结构：通道分组 → 水平/垂直 1D 池化 + 全局池化 → 1×1 跨空间卷积 + 3×3 局部分支
#       交叉注意力（全局上下文软分配）→ sigmoid 门控回乘原特征。输出通道守恒。
# factor 自适应：低层分支 LOW_LEVEL_DIM=24 → groups=6（每组 4 通道），tower CONVS_DIM=128 → groups=32。
# 启用：MODEL.BASIS_MODULE.ATTN ema   （只动 basis tower / 低层分支，不进 BiFPN，符合 F6）
#
# 验证：tests/test_m62_components.py 已确认 EMA 在 channels=24/128/256 下前向通道守恒、
#       build_attention('ema',128) 返回 EMA，且分组自适应正确（24→6，128→32）。
# 注意：与 B1(PSA) 同属 attention，L1 中二者各跑单 seed；若都晋级，L2 阶段再做组合交互测试。
```

### 预计耗时

| 阶段 | 组数 | 单组 | 小计 |
| --- | --- | --- | --- |
| A1 实现 + 可视化人工验收 | — | 0.5 天人工 | — |
| B1/B2 实现 + L0 | — | 0.5 天人工 | — |
| L1（A1/B1/B2/C1/C2/C3 共 6 组） | 6 | 40~70 min | ~6 h |
| L2（按 3 项晋级 × 3 seed，基线补 2 seed） | 11 | 40~70 min | ~10 h |

---

## M6.3 旗舰总装与三数据集验收　🎯 最终里程碑

### 旗舰配置

`configs/run-flagship.yaml` = M6.1 旗舰骨干 + ProtoNetV2 + 全部 L2 通过项。
总装前若 L2 通过项 ≥ 2，先做 1 组两两组合 L1 确认无负交互
（组合增益 ≥ 单项增益之和的 50%）。

### 验收矩阵（三数据集，3 seeds {42,123,2024}）

| 数据集 | iter | 需跑的组 | 验收线（segm AP 绝对 ≥ r50-protonet 基线 + 5.0 AP） |
| --- | --- | --- | --- |
| wheat | 8000 | 旗舰 ×3 + r50-protonet 补 2 seed | 均值 **≥ 20.36**（16.13→），3 seed 全正，p<0.05 |
| Strawberry | 21900（100 ep） | 旗舰 ×3 + r50-protonet ×3 | 均值 **≥ 21.68**，3 seed 全正，p<0.05 |
| Plantv2 | 49500（50 ep） | 旗舰 ×3 + r50-protonet ×3（或 ×1 方向确认） | 均值 **≥ 基线 + 5.0 AP**，方向一致 |

**大小与效率复核**（与性能同等级别的硬指标）：

| 项 | 验收线 | 工具 |
| --- | --- | --- |
| 总参数量 | ≤ 53.06M（预期 ~26M） | `tests/count_params.py` |
| 推理速度 | ≥ 10 FPS（3090，batch=1，512 输入） | demo 推理计时脚本 |
| 训练耗时 | ≤ r50-v2 的 1.6× / iter（Copy-Paste 的 CPU 开销含内） | 训练日志 |

### 判定

- 三数据集全部达标（绝对 +5 AP 且 p<0.05）→ **项目目标达成**，进 M6.4
- 2/3 达标且未达标项差距 < +2.0 AP → 补 seed 或加训 1.5× iter 重测一次
- wheat 都不达标 → 回 M6.2 检查 L2 假阳性；必要时启动 B 变体权重下载或 R50 支线

### 预计耗时

| 内容 | 组数 | 单组 | 小计 |
| --- | --- | --- | --- |
| wheat（旗舰 3 + 基线 2） | 5 | ~50 min | ~4 h |
| Strawberry（旗舰 3 + 基线 3） | 6 | ~2~3 h | ~15 h |
| Plantv2（旗舰 3 + 基线 3，50ep） | 6 | ~5~7 h | ~36 h |
| **合计** | 17 | | **~55 GPU 时 ≈ 2.3 天**（双卡可减半） |

---

## M6.4 消融成表与交付

| # | 内容 |
| --- | --- |
| 1 | 消融总表：基准 → +预训练骨干 → +各 L2 组件 → 旗舰，三数据集 × 3 seed（均值±std、配对 t 检验 p 值），`summarize_all.py --ttest` |
| 2 | 骨干拆解：vigv2 从零 / 预训练 / 预训练+C3K2（M6.1 与 C3 数据复用） |
| 3 | 可视化：basis CAM（`--cam-target mask`）+ 旗舰 vs 基准的 mask 预测对比图（每数据集 6 张） |
| 4 | 大小/速度表：参数量、FPS、训练耗时，对照官方基准 |
| 5 | 更新 ABLATION_RESULTS.md 与 MODEL_ZOO.md（旗舰权重与配置链接） |

---

## 风险与对策

| 风险 | 概率 | 对策 |
| --- | --- | --- |
| vigv2 预训练增益不及预期（M6.1 灰区/失败） | 中 | 查 0.6 冻结微调信号定位（权重没生效 vs 生效但骨干不适配）；R50 支线并行兜底 |
| DROP_PATH=0.1 在小数据上过强 | 中 | C2 消融零代码先行，L1 即可发现 |
| MRConv roll 算子拖慢训练/推理 | 低 | 0.5 冒烟记录 s/iter；推理不达标则 RepCPE 重参数化导出 |
| Copy-Paste 产生不真实样本掉点 | 中 | PROB 0.5→0.25 重试一次；仍负则放弃，不影响其他线 |
| PSA 在 1/8 分辨率上收益被 GC 覆盖 | 中 | L1 与 gc 正面对比，输了就弃（保住 GC 即可） |
| 多数据集类别体系不同（7/12/16） | 已知 | 分开训练，禁止混训 |
| B 变体权重缺失 | 低 | 仅在 M6.3 不达标时从官方仓库下载 `MobileViG_V2_B_Class.pth` |

---

## 附录 A：自动化操作手册

### A.1 队列任务格式

```
# tasks/<序号>.task  —— 每行：tag<TAB>配置覆盖（KEY VALUE 空格分隔，多组 TAB 分隔）
vigv2m_pre_s42	MODEL.VIG.VERSION m	MODEL.VIG.PRETRAINED weights/MobileViG_V2_M_Class.pth	SEED 42
```

### A.2 启动队列

```bash
export GBADMASK_DATA_ROOT=$PWD/datasets/HBueHxOW/wheat_seg_clean   # 按数据集切换
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1
nohup /home/huachenghao/.conda/envs/gbadmask/bin/python tools/queue_worker.py tasks/ > .queue.log 2>&1 &
# worker 空闲 40 分钟自动退出；运行期间禁止修改 adet/ 源码
```

### A.3 结果汇总 / 快检 / 大小与映射验收

```bash
python tools/summarize_all.py [--ttest]      # 汇总（--ttest 需扩展 scipy 配对检验）
python tools/quick_check.py "MODEL.VIG.VERSION m" ...   # CPU 前向/反向快检
python tests/count_params.py                  # 大小约束验收（≤53.06M）
python tests/verify_cspvigv2.py               # 预训练映射 + C3K2 恒等验收
```

### A.4 数据集切换

`train_bl+.py --dataset <名>` 自动扫描注册；或配置 `DATASETS.NAME`。
NUM_CLASSES 口径：wheat=12、Plantv2=16、Strawberry=7。

---

## 附录 B：历史里程碑归档（2026-09-03 前的 M1–M5）

| 里程碑 | 状态 | 要点（详见 ABLATION_RESULTS.md 与 git 历史） |
| --- | --- | --- |
| M1 强平台基线 | ✅ 完成 | R50 锚点：protonet 15.36 / v2 16.68（wheat, 8000 iter, seed42）；效应可测 |
| M2 cspvig 预训练 | 🔀 被取代 | cspvigv2（本蓝图）直接解决预训练映射，原 MobileViG v1 转换方案废弃 |
| M3 数据扩充 | ✅ 数据就位 | Plantv2/Strawberry 已清洗注册；其 r50-protonet 基线训练并入 M6.3 验收矩阵 |
| M4 强平台正式消融 | 🔀 并入 | 其 3 seed 配对 t 检验规范与 r50-v2 补 seed 需求并入 M6.2 L2 / M6.3 |
| M5 分叉方向 | 🔀 并入 | 结论由 M6.4 消融总表承载 |

**方法论红线（全程有效）**：单 seed ±1~2 AP 是噪声，任何宣称必须 3 seed 配对
检验；禁止用中途评估下结论；队列运行期间不改 `adet/` 源码。
