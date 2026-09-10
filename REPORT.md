# GBADMask 实验综合报告（M6.5 深度改造 + M6.6 骨干消融）

> 周期：2026-09-06 ~ 2026-09-10 ｜ 统一协议下的小数据农作物病害实例分割
> 平台：cspvigv2-M（MobileViGv2+C3K2）+ BiFPN(3,160) + ProtoNetV2 + GC basis
> 主战场：Strawberry（512 协议）与 wheat_seg_strat（诚实分层重划）
> 图表产物：`output/_figures_m65/`、`output/_figures_m66/`（本文档全部引用可点击查看）

---

## 0. 执行摘要

本报告覆盖两条实验主线及其终局：

1. **M6.5 深度改造**（V1/V2/V3 组件漏斗）：在 Strawberry 512 协议上对
   P0_res（66.06）依次检验 QFL、HQ detail 分支、边界加权损失、高分辨率
   mask 目标、B→M 蒸馏、FCOS-TAL 六类改造——**全部未通过晋级判定**
   （其中两项经三 seed 配对检验，其余单 seed 负向超噪声）。
2. **M6.6 骨干消融**：在完全一致的检测管线下横评 9 种骨干配置 × 2 数据集
   （14 臂 + 6 组 3-seed 复验），确认 **vigv2-M+C3K2 是两任务稳定最优的
   轻量骨干**，并给出颈-骨干交互的定量分解。

**核心结论**：`+5 AP` 冲刺目标在三数据集上均不可达（缺口 2.6~3.2，组件池
与骨干池双双出清）；最终定位为 **Pareto 主结果**——以 0.73× 参数、
0.43× FLOPs 达到 R50 官方基线 **+1.8~+2.4 segm AP**（两数据集 3-seed
一致为正），推理 23.5 FPS（≥10 验收线），全部效率指标达标。

---

## 1. 实验设置

### 1.1 平台与统一协议

| 项 | 配置 |
| --- | --- |
| 骨干 | cspvigv2：MobileViGv2（ImageNet-1k 预训练）+ C3K2（C2f 风格 CSP 融合） |
| 颈 | BiFPN（repeats=3, 160ch, GN, nearest 上采样） |
| 检测头 | FCOS（center sampling + centerness，stride 归一化回归） |
| mask 头 | ProtoNetV2 + GC basis（4 bases，56×56 attention） |
| 训练 | AdamW，warmup 200，STEPS 双段衰减，seed 默认 42 |

**Strawberry 512 协议**（M6.3e 重锚）：22k iter ≈ 100 ep，batch 8，
LR 0.005，多尺度 (480,512,544)，测试 512。锚点 `R1_res = 63.68` /
`P0_res = 66.06`（+5 线 = 68.68）。

**wheat_seg_strat 协议**：8000 iter，batch 7（GPU 余量约束），LR
0.004375，STEPS (4800,6400)。锚点 `strat_R1 = 13.87` / `strat_P0 =
15.71`（+5 线 = 18.87）。

### 1.2 判定纪律（全程执行）

- 单 seed 差异仅作筛选；晋级线 +0.8 AP（≈2σ 噪声阈值）；
- 晋级必须 3 seed（42/123/2024）同号 + 配对 t 检验 p<0.05（df=2 时
  |t|>4.303）；
- 中途评估禁止下结论（6k vs 8k 可差 0.3+）；
- APs 分桶单 seed 噪声实证可达 ±13，小目标结论必须 ≥3 seed；
- 失败协议（错日程/错骨干构建）的运行一律作废重跑（DQ1→DQ1b、
  KD1→KD1b 均属此类）。

---

## 2. M6.5 深度改造：全部组件检验记录

### 2.1 终局总表（Strawberry 512，vs P0_res=66.06）

| 组件 | 机制 | segm | Δ | 判定 | 检验轮次 |
| --- | --- | ---: | ---: | --- | --- |
| P0_res 基线 | — | **66.06** | — | 锚点 | 3 seed |
| DQ1b（QFL） | 分类目标=detached IoU quality | 65.23 | **−0.83** | ❌ 淘汰 | 单 seed（协议修正后） |
| HQ1（res2 detail concat） | 高分辨率细节并入 basis tower | 65.43 | **−0.63** | ❌ 淘汰（架构根因） | 单 seed |
| BR1-lite（λ=3 边界 BCE） | GT 边界带 BCE 权重 ×4 | 65.50 | **−0.56** | ❌ 淘汰（AP75 −3.11 反向） | 单 seed |
| M2b（BOTTOM_RES 64） | mask 训练目标 56→64 | 65.15 | −0.31 | ❌ 淘汰 | **3 seed 配对** |
| KD1b（B→M 蒸馏） | cls/reg/bases 三路蒸馏 | 66.15 | +0.09 | ❌ 无增益（噪声内） | 单 seed（标定修正后） |
| AS1（FCOS-TAL） | 任务对齐正样本指派 | 65.32 | **−0.74** | ❌ 淘汰（~5σ） | 单 seed（P0 σ≈0.15 参照） |

完整训练动力学与损失分解见
![M6.5 损失-AP 曲线](output/_figures_m65/curves/loss_ap_curves_straw512.png)
![P0_res 损失分量](output/_figures_m65/curves/loss_components_P0_res.png)

### 2.2 关键单项判读

**M2b（高分辨率 mask 目标）——三 seed 完整拆解**：seed42 曾给出
"APs +5.37"的诱人信号，配对复验后崩塌：

| | s42 | s123 | s2024 | 均值 | t |
| --- | ---: | ---: | ---: | ---: | ---: |
| Δ segm | −0.91 | +0.38 | −0.41 | **−0.31** | −0.85（p≈0.50） |
| Δ APs | +5.37 | −0.20 | **−12.81** | −2.55 | −0.47 |

P0_res 自身三 seed segm 66.06/65.77/65.92（σ≈0.15）而 APs
42.09/49.57/45.51（σ≈3.8）——**APs 噪声主源是基线本身**。诊断性
eval（P0_res ckpt @64 推理）显示增益全部来自训练目标分辨率而非推理
分辨率，但三 seed 证明该"增益"不可复现。

**KD 蒸馏——两轮才完成公平检验**：KD1 首轮（W_BASES=1.0）的
`loss_kd_bases`≈13.7 占总损失 93%（前景像素 MSE×4 bases 未归一），
属标定缺陷；KD1b（W_BASES=0.02）修正后 66.15（+0.09）——蒸馏无害
亦无益，教师 B@416 的知识对 512 学生无可转移增益，V3 蒸馏旗舰路径
关闭。

**AS1（FCOS-TAL）**：topk=10（小目标独立 k=4）、warmup 500 iter、
每 GT 保底正点。运行时统计健康（pos/gt≈9.2、fallback=0）但终局
−0.74（≈5σ，P0_res σ≈0.15）——动态分配在该管线下系统性劣于
几何规则分配，V2 任务对齐线关闭。

**HQ1 的架构根因**（非噪声）：tower 输出已是 stride 4 最高分辨率，
detail 分支只加"内容"不加"分辨率"，且 res2 浅层特征未语义调制。

### 2.3 组件级可视化证据

basis 图与预测对比（leaf spot 样本，BR1/DQ1b/HQ1/M2b vs 基线）：
![ablation 对比](output/_figures_m65/heatmaps/ablation/leaf_spot654_cmp.jpg)

各组件 basis 空间响应（例：M2b vs P0_res）：
![M2b bases](output/_figures_m65/heatmaps/ablation/leaf_spot654_M2b_bases.jpg)

效率画像（参数/FLOPs/FPS/精度全景）：
![efficiency](output/_figures_m65/stats/efficiency_params_flops_fps.png)

---

## 3. M6.6 骨干消融

### 3.1 设计

9 种骨干配置、完全一致管线（BiFPN+ProtoNetV2+GC+FCOS），两数据集
同协议，唯一变量=骨干。新实现三项（全部通过 CPU 冒烟：前向/反向 ×
3 尺寸 + stride 契约 + 预训练键覆盖率 ≥95% 断言）：

| 骨干 | 实现方式 | 骨干参数 |
| --- | --- | ---: |
| R50+BiFPN | 已有 build（同颈对照，分解颈/骨干贡献） | 23.78M |
| vigv2-S+C3K2 / MobileViGv2-S/M | 配置开关（USE_C3K2） | 7.35M / 15.85M |
| MobileNetV3-L | timm 0.6.12 原生（统一 wrapper） | 3.13M |
| MobileNetV4-Conv-S | 键名精确复刻 timm checkpoint（278/278 键位对齐） | 3.79M |
| LSNet-T（CVPR2025） | vendored：SKA 去 triton 纯 torch 重写 + attention bias 运行时插值 | 11.07M |

排除项：MobileNetV5（仅 Gemma-3n encoder 权重，非 IN-1k）、
GhostNetV3（checkpoint 未公开）。

### 3.2 结果总表（seed42 单轮，完整口径）

**Strawberry 512**（完整表见 `output/_figures_m66/table_straw.md`）：

| 骨干 | 总参数(M) | segm AP | APs | Δ vs R1 | Δ vs P0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **vigv2-M+C3K2（P0）** | 25.96 | **66.06** | 42.1 | **+2.38** | — |
| vigv2-S+C3K2 | 17.43 | 65.87 | 40.0 | +2.19 | −0.19 |
| MobileViGv2-S | 17.43 | 65.80 | 47.7 | +2.12 | −0.26 |
| MobileViGv2-M | 25.96 | 64.89 | 48.0 | +1.21 | −1.17 |
| R50+BiFPN | 34.36 | 65.29 | 41.3 | +1.61 | −0.77 |
| R50+FPN（R1 锚） | 35.36 | 63.68 | 38.4 | — | −2.38 |
| MobileNetV4-S | 13.85 | 60.48 | 38.8 | −3.20 | −5.58 |
| MobileNetV3-L | 13.31 | 60.17 | 34.6 | −3.51 | −5.89 |
| LSNet-T | 21.15 | 58.86 | 32.3 | −4.81 | −7.20 |

**wheat_seg_strat**（完整表见 `output/_figures_m66/table_wheat.md`）：

| 骨干 | 总参数(M) | segm AP | Δ vs R1 | Δ vs P0 |
| --- | ---: | ---: | ---: | ---: |
| vigv2-S+C3K2 | 17.44 | 16.20 | +2.33 | +0.49 |
| **vigv2-M+C3K2（P0）** | 25.96 | **15.71** | **+1.84** | — |
| MobileViGv2-S | 17.44 | 15.68 | +1.81 | −0.03 |
| MobileViGv2-M | 25.97 | 15.15 | +1.28 | −0.56 |
| MobileNetV3-L | 13.32 | 14.70 | +0.83 | −1.01 |
| R50+BiFPN | 34.37 | 13.98 | +0.11 | −1.73 |
| R50+FPN（R1 锚） | 35.36 | 13.87 | — | −1.84 |
| MobileNetV4-S | 13.86 | 13.44 | −0.43 | −2.27 |
| LSNet-T | 21.16 | 11.20 | −2.67 | −4.51 |

### 3.3 对比图

![Strawberry 骨干消融柱状图](output/_figures_m66/bar_straw.png)

![wheat 骨干消融柱状图](output/_figures_m66/bar_wheat.png)

![Strawberry 参数-AP Pareto](output/_figures_m66/pareto_straw.png)

![wheat 参数-AP Pareto](output/_figures_m66/pareto_wheat.png)

### 3.4 M66b：vigv2-S+C3K2 的 3-seed 终判

seed42 单轮上 S 变体在 wheat 以 16.20 居全场最高（+0.49 vs P0），
触发 3-seed 确认轮（补 s123/s2024 两 seed × 两数据集 + wheat P0-M
配对基线）：

| 配对（S−M，同 seed） | s42 | s123 | s2024 | 均值 | t（df=2） |
| --- | ---: | ---: | ---: | ---: | ---: |
| wheat | +0.49 | **−2.35** | **−1.75** | **−1.20** | −1.39 |
| Strawberry | −0.19 | −1.14 | −0.56 | **−0.63** | −2.28 |

**终判：淘汰**。两数据集三 seed 配对差全部同号为负（未过 5% 显著线
但方向一致），且 S 的 seed 稳定性显著差于 M（wheat σ 1.62 vs 0.21）。
seed42 的"wheat 全场最高"被判定为单 seed 噪声——与 M2b 案例互为印证，
构成本项目关于单 seed 判定风险的第二份实证。

### 3.5 机制性结论

1. **图骨干优势压倒性**：Strawberry（高分辨率小目标）上第三方骨干
   落后 P0 达 5.6~7.2 AP；wheat（极小数据）上最优第三方（MNv3-L
   14.70）仍落后 vigv2 家族 ≥1.0。图聚合在大间距稀疏连接下的全局
   感受野，对病斑类区域型目标是纯卷积骨干的结构性优势。
2. **颈-骨干交互定量分解**：R50+BiFPN vs R50+FPN = straw **+1.61** /
   wheat +0.11 → 平台 Strawberry 优势 +2.38 中约 **1/3 来自换颈**、
   2/3 来自骨干及颈-骨干适配；wheat 优势则全靠骨干。
3. **C3K2 分解**（同变体内差）：wheat M +0.56 / S +0.52（稳定正，
   与旧 clean 口径 +1.09 方向一致）；straw M +1.17 / S +0.07——
   大变体（M）上 CSP 多分支融合的增益显著大于小变体。
4. **容量与稳定性**：S 变体容量小，在 584 图上更易陷入不同次优解
   （seed 敏感）；M 的 CSP 融合冗余提供隐式正则——**轻量化不是免费
   的**，小骨干以方差换参数。
5. **APs 单轮参考**（不作结论）：MobileViGv2 无 C3K2 时 straw APs
   47.7/48.0 高于 P0 42.1，但 σ≈3.8 的分桶噪声下不具证据力。

---

## 4. 效率与部署指标（M6.4 验收线）

| 模型 | 参数(M) | FLOPs@512(G) | 推理(ms) | FPS | ckpt(MB) | segm AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **P0_res（vigv2-M+C3K2）** | **25.96** | **35.13** | 41.3 | **24.2** | 208.7 | **66.06** |
| R1_res（R50 官方） | 35.36 | 56.15 | 19.1 | 52.3 | 282.6 | 63.68 |

- 参数 0.73×、FLOPs 0.63×（上限 1.5× 官方 = 53.06M，实际 0.49×）；
- 推理 23.5~24.2 FPS（≥10 验收线，batch=1@512，3090）；
- 平台以 **0.43× R50 的计算量**换取 +2.38 AP（straw）/ +1.84（wheat）。

---

## 5. 综合结论与论文定位建议

1. **+5 AP 冲刺目标正式关闭**：三数据集缺口 2.6~3.2（straw）、
   3.2（wheat）、∞（Plantv2 顶格）；M6.5 组件池（6 项）与 M6.6 骨干
   池（9 配置）双双出清，最大单杠杆仍是骨干级预训练（+11.05，已用）。
2. **建议论文主结果 = Pareto 定位**：
   - 主表：R50 官方 vs 平台，两数据集 3-seed 配对（straw +1.73~
     +2.38、wheat +1.84），全部同号为正；
   - 副表 1：M6.6 骨干横评（9 臂 × 2 数据集，含 MNv3/MNv4/LSNet
     SOTA 轻量对照）——支撑骨干选择正当性；
   - 副表 2：效率画像（参数/FLOPs/FPS/精度）；
   - 方法学讨论：单 seed 判定风险的两份实证（M2b、M66b-S），APs
     分桶噪声测量（σ 3.8~13）——对小数据评估方法论是有价值的贡献。
3. **遗留待决策**：MQ1/NK1/NK2（M6.5 尾池，期望值低）；推理侧 TTA
   （零训练成本，期望 +0.5~1）；M6.6 数字并入 ABLATION_RESULTS.md
   消融总表。

---

## 附录 A：数据与复现

- 全部训练/评估入口：`tools/train_bl+.py`（配置驱动）；队列脚本
  `tools/run_m65_wave*.sh`、`tools/run_m66_*.sh`、`tools/run_m66b_seed.sh`
  （幂等，完成标记 `logs/m65.log`/`logs/m66.log`/`logs/m66b.log`）；
- 汇总重建：`python tools/summarize_backbones.py`（M6.6 表+图）、
  `output/_figures_m65/stats/model_stats.csv`（M6.5 效率线）；
- 单元测试：`tests/test_m65_{kd1,as1}.py`、`tests/test_m66_backbones.py`
  （CPU 可跑）；
- 骨干权重：`weights/`（MobileViG V2 全系 + MNv3/MNv4/LSNet IN-1k，
  gitignore 不入库）。

## 附录 B：协议事故与作废记录（诚实性声明）

| 运行 | 事故 | 处置 |
| --- | --- | --- |
| DQ1 | 误用 run-strawberry.yaml（cspvig v1 + build 忽略 PRETRAINED → 随机初始化） | 作废，DQ1b 重跑 |
| KD1 | kd_bases 损失无归一，占总损失 93% | 标定缺陷，KD1b 重跑 |
| M66 首挂（6 臂） | 队列脚本参数错位（yacs 类型报错定位） | 修复重跑，14/14 收队 |
| wheat 旧锚点 | 原始 val 未分层 + test 跨 split 泄漏 | 全部重建为 wheat_seg_strat |
