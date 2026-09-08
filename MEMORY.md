# GBADMask 工作记忆（MEMORY.md）

> 最后更新：2026-09-08（M6.6 骨干消融挂队）
> 用途：会话交接。新会话请先读本文件，再读 ROADMAP.md 的 M6.2/M6.3/M6.5 章节。

---

## 1. 项目目标与当前验收口径

- **目标**：小数据农作物病害实例分割，模型 ≤ 1.5× 官方 BlendMask(R50)（≤53.06M），
  segm AP **绝对提升 ≥ +5.0**（3 seed 均值、全部为正、配对 t 检验 p<0.05）。
- **平台**（M6.1 决出）：cspvigv2-M（MobileViGv2+C3K2，ImageNet 预训练）+ BiFPN(3,160)
  + ProtoNetV2 + GC basis，25.97M 参数。
- **✅ 验收锚点（2026-09-05 定稿，均同协议、seed42，1-seed）**：
  - **wheat_seg_strat**（诚实分层重划，8000iter/batch7）：`R1 = 13.87` → **+5 线 18.87**；
    平台 P0 = **15.71**（差 **3.16**，近平台上限）
  - **Strawberry**（全日程 22k≈100ep/batch8）：`R1_full = 63.69` → **+5 线 68.69**；
    平台 P0_full = **65.42**（差 **3.27**）
  - **Plantv2**：R1 顶格 98.88，+5 数学上不可能 → 弃作主张数据集
  - 旧 clean 口径（R1 15.00 → 20.00，best 17.33/缺口 2.67）已作废，仅存史（§2）。
  - **硬事实**：组件池出清，平台 vs R50 收敛到 ~+1.8（wheat +1.84 / Strawberry +1.73），
    +5 需跨任务/跨数据层级新杠杆（详见 §6 台账）。

### 统一实验协议（务必沿用）

```
数据集 wheat_seg_clean（584 train / 82 val，12 类）
8000 iter / IMS_PER_BATCH 7 / BASE_LR 0.004375 / SEED 42
STEPS (4800,6400) / EVAL_PERIOD 2000 / 只取最终 iter 评估
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**batch 7 的原因**：GPU1 有 root 常驻进程 `server.py` 占 6.9GB（动不了），
可用 ~16.3GB；batch 8 的 ProtoNetV2 平台峰值 16.1GB，实测两次 OOM。
batch 7 峰值 14.15GB。**M6.3 验收时基线（R1）已按同协议重跑，口径自洽。**

---

## 2. 全部已出分结果（wheat_seg_clean, 8000 iter, batch 7, seed 42）

| 组 | segm AP | bbox | Δ vs P0 | 结论 |
| --- | --- | --- | --- | --- |
| **R1** r50-protonet（官方基准，新锚点） | **15.00** | 19.15 | — | **+5 线 = 20.00** |
| **S1** vigv2m 从零（无预训练） | **5.88** | 8.09 | −11.05 | 预训练贡献 **+11.05 AP**（F1 定量，M6.4 拆解表） |
| **P0** vigv2m 预训练+ProtoNetV2+gc | **16.93** | 17.25 | 0（锚点） | L1 参照 |
| **B2** +EMA 注意力 | **17.33** | — | **+0.40** | 未过线（17.73），L2 候选 |
| **D3** 12000 iter + WarmupCosineLR | **17.44** | — | **+0.51** | **新全批最高**；收敛 12.59/15.96/17.07/17.44 单调上升无回落；未过线（差 0.29，噪声内）；L2 候选 |
| **C2** DROP_PATH 0（vs 0.1） | **17.18** | — | +0.25 | 噪声内，保持 0.1 |
| **C3** 关闭 C3K2 | **15.84** | — | **−1.09** | **C3K2 贡献 +1.09 实锤**（保留） |
| **B1''** PSA 仅 tower（ATTN_LOW=none） | **15.73** | — | **−1.20** | ❌ **PSA 淘汰**（>2σ 噪声；GC 保持 basis 注意力最优） |
| **D1** STEPS 提前 (4200,5600) | **15.90** | — | **−1.03** | ❌ 提前衰减更糟：6k=15.92 → 8k=15.90，峰值没兑现且整体更低 → STEPS 不动 |
| **D2** 12000 iter + STEPS (7200,9600) | **17.23** | — | **+0.30** | 噪声内（9k 峰 17.54 回落到 17.23）；延长训练不解决回落，1.5× 算力不值 |
| A1 CP+LSJ 全强度 | 11.19 | 11.15 | −5.74 | ❌ 淘汰 |
| A1' CP+LSJ 半强度 | 11.04 | — | −5.89 | ❌ 二连败 → 机制不匹配（短日程+小数据） |
| C1 NUM_BASES 8 | OOM | — | — | ❌ 放弃（显存超限，收益弱） |

**关键效应分解**：预训练 +11.05 ≫ C3K2 +1.09 > 骨干+basis vs R50 +1.93 > EMA +0.40 > PSA −1.20

**B1'' 收敛**：2k 7.19 / 4k 12.54 / 6k 15.83 / 8k 15.73（同样 6k 后回落）。
attention 线全部出清：**GC 锁定**，EMA（+0.40 未过线）为 L2 唯一候选，PSA/CBAM 淘汰。

**D 系列分项判读**：
- D1（提前衰减）**负向**：4k 才 12.47，全程低于 P0，"6k 峰值早收割"假设被证伪；
- D2（延长阶梯）12.52/14.38/17.54@9k/17.23@12k——峰值更高但 9k 后回落 −0.31；
- D3（cosine）12.59/15.96/17.07/17.44——**唯一无回落的日程，终局见第 3 节**。

**噪声基线**：同配置跨批次 ±0.6 AP（17.35/17.94/17.99 三次同配置实测）。
晋级线 +0.8 与噪声同量级 → 单 seed 判定从严，L2 必须 3 seed + t 检验。

---

## 2.5 M6.3 锚点队列终局（2026-09-05 01:18 UTC 收队，七组全部 exit=0）

**wheat_seg_strat 诚实分层重划三组（8000 iter 同 wheat 协议，口径正确）**：

| 组 | segm | bbox | Δ vs strat_R1 | 判读 |
| --- | --- | --- | --- | --- |
| **strat_R1**（r50-protonet 新锚点） | **13.87** | 15.80 | — | **新 +5 线 = 18.87** |
| **strat_P0**（vigv2m 平台） | **15.71** | 16.91 | **+1.84** | 平台优势在诚实切分下依然成立（与 clean 的 +1.93 自洽） |
| **strat_B2**（EMA） | 15.16 | 17.22 | +1.29 | **EMA 不敌 P0** → 旧 clean 上 +0.40 系噪声/泄漏伪影；**L2 候选降级，可剔除** |

- 数字如预期"变诚实"（13-15 区间）；分层后每类 13-39 实例全部可测，WCN 不再 0 分。
- **B2 > P0 在 clean 上成立、在 strat 上反转** → 判定 EMA 无真实增益，attention 线只剩 GC。

**Plantv2 / Strawberry 四组 ⚠️ 跑错日程（非预期口径，不得作验收锚点）**：
- `run_m63_anchors.sh` 头注释写明 Plantv2 100k≈101ep / Strawberry 22k≈100ep（沿专用
  `configs/run-plantv2.yaml` MAX_ITER=100000、`run-strawberry.yaml` MAX_ITER=22000），
  但 `run()` 复用 run-wheat-r50.yaml / run-vigv2.yaml（MAX_ITER=8000、STEPS(4800,6400)、
  512 输入、LR 0.004375）且**未传 `SOLVER.MAX_ITER` 覆盖** → 四组实际只跑 **8000 iter**
  （plantv2≈7ep、strawberry≈32ep），256/419 输入图也被按 wheat 放大到 ~512。
  根因：误以为改 DATASETS.NAME 会带出各数据集日程（MAX_ITER 在 YAML 不随数据集变）。
- 出分（仅方向参考，非验收口径）：

| 组 | segm | bbox | Δ(P0−R1) | 说明 |
| --- | --- | --- | --- | --- |
| plantv2_R1 | **98.88** | 97.41 | — | **Plantv2 饱和 ~99**（7ep 即顶格，疑似 val 近重复） |
| plantv2_P0 | 98.45 | 96.07 | −0.43 | 与 R50 顶格无差 → **Plantv2 无 +5 头寸，非可用验收数据集** |
| strawberry_R1 | **63.43** | 65.07 | — | 8000iter 下 R50 领先 |
| strawberry_P0 | 61.10 | 62.36 | **−2.33** | 平台落后（>噪声 2σ），但日程/分辨率不匹配 → **须全日程复跑才能定论** |

**判读（战略级）**：平台（vigv2m+C3K2+BiFPN+ProtoNetV2+GC）只在 **wheat（小数据长尾，
584图）显著占优**（+1.84~+2.44）；Plantv2 顶格无法体现差异；Strawberry 8000iter 下平台
反而落后 R50（+yolo26s 100ep 也仅 60.6 mAP50-95，大家同档 ~61-63）。
→ **三数据集当前全部够不到 +5 AP 冲刺线**：strat 需 18.87（P0 15.71，差 3.16 已接近该
平台上限）；plantv2 顶格（数学上不可能 +5）；strawberry 落后。见 §6 校准决策。

**2026-09-05 04:10 更新：Strawberry 全日程复跑收队（M6.3b，22k≈100ep，batch8/LR0.005，
输入 384-448，输入分辨率与日程均按 run-strawberry.yaml 专属配置）**：

| 组 | segm | bbox | Δ(P0−R1) |
| --- | --- | --- | --- |
| strawberry_R1_full（r50-protonet） | 63.69 | 66.06 | — |
| **strawberry_P0_full**（vigv2m 平台） | **65.42** | 66.74 | **+1.73** |

→ **8000iter 的 −2.33 确系日程伪影**（wheat 协议错配）；全日程下平台在 Strawberry 也占优。
平台 vs R50 现为：**wheat strat +1.84 / Strawberry +1.73 / Plantv2 顶格无差**——非饱和数据
集方向一致且稳健 ~+1.8（2/3 显著为正），但距 +5 冲刺线仍远，验收口径待定（见 §6）。
strat_Bcap（MobileViGv2-B@strat）batch7 首步 **OOM**（峰值 16.2GB / 可用 ~16.4GB；B 太重，
server.py 6.9GB 拖累）→ 已降 **batch6 / LR 0.00375**（线性缩放）重跑，仅作容量诊断
（口径略异于 P0 的 batch7，注明即可）。

---

## 3. 运行中 / 历史记录

**（当前）GPU1 空闲。M6.5 DQ1/QFL 实验作废（2026-09-06 14:35 UTC 复盘更正）：

- **双重协议错误**：①脚本误用 `configs/run-strawberry.yaml`——其
  `BACKBONE.NAME=build_fcos_cspvig_bifpn_backbone` 是 **cspvig v1**（模型 dump
  显示 `(backbone): MobileViG(`），而 P0_res 用 `run-vigv2.yaml` 的 cspvigv2；
  ②**cspvig v1 的 build 完全忽略 MODEL.VIG.PRETRAINED**（cspvig.py:436 无加载
  逻辑）→ DQ1 实际 = cspvig v1 随机初始化 + QFL，56.03@416 vs P0_full 65.42
  的 −9.4 与"无预训练损失 ~11AP"（wheat S1 实测）量级吻合；
- **判定更正：QFL 未被检验，DQ1 结果作废**（既不淘汰也不晋级 QFL）；
  教训入运维纪律：**写 run 脚本必须核对 config 的 BACKBONE.NAME 与加载逻辑**，
  `run-strawberry.yaml` 是 v1 时代的旧配置（勿再用于 vigv2 平台实验，
  应像 wave1/2 一样用 run-vigv2.yaml + DATASETS.NAME Strawberry 覆盖）；
- **HQ1（res2 detail branch）已出终局：淘汰**（16:29 UTC exit=0，
  `output/m65_hq1_detail_straw512`）：segm **65.43**（bbox 65.87）vs P0_res
  66.06（67.49）→ **Δ −0.63 / bbox −1.62**；APs 40.63 vs 42.09（−1.46）、
  APm −0.26、APl −1.17；逐类 Anthracnose −1.9 / PMF −2.3 / ALS +1.1。
  轨迹 54.51/62.23/63.66/65.44/65.37/65.43（16k 后平台）。
  **架构层根因：tower 输出 fre 本就是 stride 4 = 骨干最高分辨率**（Blender
  POOLER_SCALES=0.25 也按 stride 4 采样），detail 分支只加了"内容"没加
  "分辨率"，且 res2 是未语义调制的浅层噪声 → **HQ 线整线关闭（D2 门控不做，
  上限被封死）**。代码保留（默认关闭、旧 checkpoint 兼容、有单测）。
- **（当前）M6.5 wave1b 全队收队（09-07 05:19 UTC，两组 exit=0）**：
  - **DQ1b（QFL 正确协议）：淘汰**。segm **65.23** / bbox 67.07 vs P0_res
    66.06/67.49 → **−0.83/−0.42**；AP75 −1.49、APs −1.63（轨迹
    54.47/60.86/64.05/65.38/65.20/65.23）。QFL 在正确协议下方向仍负 →
    **V1 质量线（QFL）关闭**，DFL 同线且手术更大，降级不做。
  - **M2b（BOTTOM_RESOLUTION 56→64）：方向性信号，单 seed 未确认**。
    segm 65.15 / bbox 66.43 → 总分 −0.91/−1.06，但 **APs 47.46 vs 42.09
    = +5.37（M6.5 首个真实瓶颈级信号）**；APm −1.61 / APl −1.89（轨迹
    55.23/62.77/64.37/64.90/65.25/65.15）。
  - **关键诊断（05:31）**：P0_res checkpoint 以 BOTTOM_RES=64 **eval-only**
    = 66.02（APs 42.09 不变）→ **推理端分辨率零效果，M2b 的 APs 增益全部
    来自训练目标分辨率**（小实例 64×64 目标更细 → BCE 学到更细边界）；
    bbox −1.06 说明部分跌幅是共享特征训练扰动/seed 噪声，非 mask 头本身。
    判读：APs +5.4 远超分桶噪声 → 信号真实；总分跌幅需 seed 复验，
    BOTTOM_RES=64 暂不改默认。
- **下一步（wave1c 队列 `tools/run_m65_wave1c.sh`）**：① **BR1-lite**
    （边界加权 mask BCE：GT 形态学边界带内 BCE 权重 ×(1+λ)，零参数、
    直击 AP75/边界——正是 M2b 中 APm/APl 的痛点；λ=3 单变量 @56）→
    ② **M2b-s123**（BOTTOM_RES 64 seed123，复验 APs 信号量级）。
- **wave1c 全队收队（09-07 09:32 UTC，两组 exit=0）**：
  - **BR1-lite（λ=3）：淘汰**。segm 65.50 / bbox 66.60 → −0.56/−0.89；
    **AP75 70.70 vs 73.81 = −3.11 反向**，APs −6.32（35.77）。逐类 ALS −1.9。
    判读：边界带过加权挤占主体 BCE 信号（loss_mask 终值 0.166 vs 基线
    ~0.078，λ=3 相当于边界像素 4× 权重，56×56 下边界带占比过高）→
    **损失重加权路径整线关闭**（BR2/MaskTransfiner 亦依赖同类信号，
    降级不做）；代码保留（`BOUNDARY_LOSS_WEIGHT=0.0` 默认关，有单测）。
  - **M2b-s123：信号复现且总分转正**。segm **66.14**（bbox 67.05）vs
    P0_res(s42) 66.06 → +0.08；**APs 49.37 / AP75 74.04**（s42 版为
    47.46/72.23）。APs 两 seed 轨迹高度一致（末段 47.5/48.9/49.4）→
    **BOTTOM_RES=64 的 APs 增益坚实（+5~7），且 seed123 无总分损失**
    （s42 的 −0.91 疑为该 seed 特有扰动）。
- **（当前）wave1d 运行中**（10:18 UTC 启动，`tools/run_m65_wave1d.sh`）：
  ① **P0_res-s123 已完成**（12:15 UTC）：segm **65.77**（bbox 66.76，
    APs 49.57/AP75 72.86）→ **seed123 配对判定（反转）**：M2b-s123 66.14
    vs P0_res-s123 65.77 → **Δ +0.38/AP75 +1.18，但 APs −0.20**——
    **seed123 的高 APs（49.57）是基线特性而非 M2b 贡献**。
    两个 seed 的真实图景：Δ APs = s42 **+5.37** / s123 **−0.20**（seed
    强依赖，P0_res APs 基线本身在 42.09~49.57 间波动 ~7.5）；
    Δ segm = s42 −0.91 / s123 +0.38 → 均值 **−0.27**。
    ⚠️ APs 分桶在小数据 val 上的桶内实例数少，单 seed 波动大，"APs +5.37"
    的 seed42 信号大概率被 s123 证伪一半。
  ② **M2b-s2024 已完成**（14:12 UTC）：segm **65.51**（bbox 66.98，
    APs 32.70/AP75 72.19；轨迹 53.48/62.88/64.95/65.82/65.72/65.51）。
    **M2b 三 seed APs = 47.46/49.37/32.70**（内部波动 16.8，s2024 甚至
    低于 P0_res-s42 的 42.09）→ **APs 优势进一步崩塌**；三 seed segm =
    65.15/66.14/65.51（均值 65.60，vs P0_res 已知 66.06/65.77 均值 65.92
    → 至少 −0.3，无正向总分趋势）。
- **（当前）M6.6 骨干消融运行中（2026-09-08 08:55 UTC 重启）+ AS1 已终局**：
  - **AS1（FCOS-TAL）终局：淘汰**（08:49 UTC exit=0）：segm **65.32**
    （bbox 66.15，APs 48.79/AP75 72.58）vs P0_res(s42) 66.06 → **Δ −0.74**。
    P0_res 三 seed σ≈0.15 → −0.74 是真实负效应（~5σ）。TAL 轨迹全程低于
    P0_res（20k 时 Blossom 82.74 vs 基线 ~83）→ **V2 任务对齐线关闭**。
    至此 M6.5 组件池：DQ1/HQ1/BR1/M2b/KD1b/AS1 全负，仅剩 MQ1/NK1/NK2
    未试（期望值低、且 +5 缺口 2.6 已不可达）→ **M6.5 实质收官，建议
    Pareto 回退当论文主结果（+1.7~+2.4 / 25.96M / 23.5 FPS），待用户
    最终拍板**。
  - **M6.6 首挂 08:50 秒崩（已修复重启）**：两个脚本 bug——①r50 臂
    `MODEL.RESNETS.OUT_FEATURES "(res3,res4,res5)"` 字符串 vs yaml 列表
    类型冲突（yaml 本就正确，删除覆盖即可）；②`runvig` 内 `shift 4` 后
    引用 `$2/$3/$4` + `set -u` → 未绑定变量崩溃。**教训：shell 函数
    用 shift 后绝不能再引用移走的参数；列表型 cfg 键不要用括号字符串
    覆盖**。毒 START 标记已清（logs/m66_failed_0908.log.bak 留档），
    08:55 重挂 watcher（`tools/watch_m66_after_as1.sh`）正常起跑：
    首臂 r50bifpn 0.26 s/iter / 4.1GB（比 vigv2 平台更快更轻）。
  - **M6.6 骨干消融（用户 09-08 下达）**：两数据集 × 7 新臂，唯一变量=
    骨干，BiFPN(3,160)+ProtoNetV2+GC+FCOS 全同口径：
    `r50bifpn / vigv2-S+C3K2 / MobileViGv2-S / MobileViGv2-M（无 C3K2）/
    MobileNetV3-L / MobileNetV4-Conv-S / LSNet-T` + 已有锚点
    （straw：P0_res 66.06 / R1_res 63.68；wheat strat：P0 15.71 / R1 13.87）。
    统一 1-seed（42）筛选，入围者后补 3 seed。
  - **骨干可得性核查结论**：MobileNetV5 在 timm main 存在（mobilenetv5.py）
    但仅 Gemma-3n encoder 权重（非 IN-1k、含 MQA/MSFA、需 vendor 现代化
    timm 层）→ **放弃**；GhostNetV3 用户选择跳过（权重未公开）；
    MNv3-L/MNv4-S/LSNet-T 权重齐（github release / hf-mirror / HF）。
  - **实现**（全部过 CPU 冒烟 `tests/test_m66_backbones.py`，7 臂
    forward/backward × 3 尺寸 + stride 契约 + 预训练覆盖率 ≥95% 断言）：
    - `lsnet_vendor.py`：LSNet-T vendored（detection 移植版语义），**SKA
      去 triton 化**（纯 torch ks² 循环等价重写）+ FFN 改分类版命名
      （pw1/pw2）+ attention bias 运行时 bicubic 插值（多尺度兼容）。
    - `mnv4.py`：MNv4-Conv-S **按 checkpoint 键名精确复刻**（timm
      'uir' 解码：a/k/p=start/mid/end dw 核，MNv4-S 全块 ReLU），
      键位校验 278/278。
    - `mobile_bb.py`：三骨干统一 d2 Backbone 包装（res3/4/5@8/16/32），
      `MODEL.MOBILE_BB.{MODEL_NAME,WEIGHTS,INPUT_SIZE}` 一套配置接入。
    - 参数量实测：mnv3 3.13M / mnv4 3.79M / lsnet 11.07M / vigv2-S
      7.35M / mv2v2-M 15.85M（+C3K2 零参数差，与 C3 消融设计一致）/
      r50-BiFPN 23.78M。
  - **队列接力**：`tools/watch_m66_after_as1.sh`（PID 47848）等 AS1 真实
    成功标记 → `run_m66_backbone_straw.sh`（7 臂 × ~3.2h ≈ 22h，明晨
    ~07:00 UTC 完）→ `run_m66_backbone_wheat.sh`（7 臂 × ~1.3h ≈ 9h，
    明下午 ~16:00 UTC 完）。失败臂标记后不阻塞后续臂。
  - **汇总出表**：`tools/summarize_backbones.py`（解析最终 eval + 重建
    模型计参数 + markdown 表 + AP 柱状图 + 参数-AP Pareto 散点图，
    产物 `output/_figures_m66/`）。收队后跑一次即可。
  - **M6.5 状态**：AS1（TAL）08:49 UTC 收尾中；组件池仅剩 MQ1/NK1/NK2
    未试，全负则 M6.5 收官走 Pareto 回退。**M66 队列运行期间禁改
    `adet/`**（新进程逐臂 import）。
- **（前史）KD1b 收队 + AS1 运行中（2026-09-08 07:20 UTC 更新）**：
  - **KD1b 终局：KD 假设关闭（中性）**（09-08 06:40 UTC exit=0）：
    segm **66.15**（bbox 67.47，APs 44.69/AP75 72.77）vs P0_res(s42)
    66.06/67.49 → **Δ segm +0.09 / bbox −0.02（噪声内）**。标定修正后
    蒸馏不再有害（KD1 −0.39 → KD1b +0.09），但**无真实增益信号**，
    单 seed 0.09 远不及晋级线，3-seed 确认不值得 → **KD 线整线关闭**
    （V3 蒸馏旗舰路径结束；KD2 mask 蒸馏降级不做）。教师 B=66.35
    @416 的定位/掩码知识在 512 学生上无可转移增益，学生已到自身上限。
  - **AS1 全量运行中**（06:42 UTC 由 watcher 自动接棒启动，
    `tools/run_m65_wave2_as1.sh`，iter ~6400/22k，eta ~1:27，预计
    ~08:45 UTC 完）。冒烟已过（150 iter + TAL_WARMUP=50，[TAL] 统计
    存在、损失有限）。TAL 运行时统计健康：pos/gt ~9.0-9.6（topk=10
    近饱和，candidate 不再限制）、**fallback=0**（每 GT 候选充足）、
    损失量级与基线一致（total 0.91 / mask 0.113）、峰值 9.6GB、
    s/iter 0.333（vs P0_res 0.308，正点数↑致 mask ROI 略多，可接受）。
    **AS1 全量期间禁改 `adet/`。**
  - **P0_res-s2024 完成**（09-07 16:52 UTC）：segm **65.92**（bbox 67.14，
    APs 45.51/AP75 73.30）→ **M2b 三 seed 终判：淘汰**。
    配对 Δ segm = s42 −0.91 / s123 +0.38 / s2024 −0.41（均值 −0.31，t=−0.85
    p≈0.50）；Δ APs = +5.37/−0.20/**−12.81**（均值 −2.55，t=−0.47）。
    **BOTTOM_RES 保持 56**；P0_res 三 seed segm 66.06/65.77/65.92（σ≈0.15
    极稳），APs 42.09/49.57/45.51（σ≈3.8——**APs 噪声主源是基线本身**，
    单 seed 小目标结论实证噪声可达 ±13，必须 ≥3 seed）。
  - **KD1 首轮（历史）**：标定缺陷未公平检验（09-07 19:57，segm 65.67，
    loss_kd_bases ~13.7 占总损失 93%）→ 已由 KD1b（W_BASES 0.02）公平
    重测并关闭，见上。
  - **FPS 基准已实测（M6.4 效率线收账）**：平台 M@512 batch1 =
    **23.5 FPS / 42.5ms / 25.96M 参数 / 0.26GB**；R50-protonet @512 =
    54.6 FPS / 18.3ms / 35.36M。验收线 ≥10 FPS 双达标；平台 0.49× 参数
    上限（53.06M）。论文表述：平台以 0.43× R50 速度换 +1.7~+2.4 AP。
  - **下一步**：KD1b 出分后判 KD 线去留；**AS1（FCOS-TAL）为 M6.5 最后
    未实现的高期望组件**（漏斗 Wave 2，预期 +0.5~1.5），实现期间 KD1b
    占 GPU 无冲突（TAL 只改 `fcos.py` assignment，需单测+冒烟后才上 GPU）。
  ④ **KD1 实现档案（已复用于 KD1b）**：`MODEL.DISTILL.*` 配置 +
     `BlendMask._distill_losses()`（teacher 冻结、每次蒸馏前强制 eval 防 BN
     漂移；三路损失 = cls sigmoid MSE / reg L1 / bases 前景加权 MSE；
     TEACHER_OPTS 用 KEY=VALUE 格式 shell 安全；单测 `tests/test_m65_kd1.py`）。
     teacher = `output/m63d_straw_B`（B 变体 66.35），student = M@512，
     teacher 吃 student 同款 512 输入（全卷积尺寸鲁棒）。
     watcher（watch_kd1_after_s2024.sh）已完成使命消亡，无残留进程。
     KD1/KD1b 期间禁改 `adet/`（KD1b 运行中，~06:35 UTC 前）。

Strawberry 分辨率重议（M6.3e）全队收队 12:17：
**P0_res = segm 66.06**（bbox 67.49，APs 42.1/APm 49.7/APl 71.6）vs R1_res 63.68 →
**Δ +2.38**（416 协议 Δ 1.73 → 512 提升 0.65）。分辨率主要帮平台：P0_res 比 P0_full 65.42
高 0.64，且 APs 42.1 反超 R1_res 38.4（小目标差转正）。但 P0_res 距新 +5 线 68.68 仍差
**2.62**。**Strawberry 全部路径盘点**：廉价杠杆全负；B 66.35(+0.93 over M)；512 分辨率
66.06(Δ+2.38)。最佳绝对值 66.35(B)/66.06(M@512)，+5 均差 ~2.6+。→ 唯一剩余 = **M6.5
深度改造**，且建议在 **512 协议（P0_res 66.06 / R1_res 63.68）** 上起跑（基线更高、小目标
已改善），M6.5 预估 V1+0.8~1.8 / V2+1.5~3.0 叠上去才有戏。待用户定是否开 M6.5（归属谁）。**
MobileViGv2-B 容量上限已收队（08:22）：**B segm 66.35**（bbox 67.63）＝ +2.66 vs R1 /
仅 +0.93 vs M，APs 35.6 比 M 39.6 更差（APl 72.1 最高）→ **容量只补大/中、不补小**，
+3.27 缺口容量路不通，瓶颈=416 小目标像素密度 → 故决议重试分辨率。
草莓廉价杠杆（wave1+2）全负：cosine −1.32 / CP −1.26 / 30k −0.50 / iou 崩。M=65.42、
R1=63.69（均 416 协议）；多尺度 eval 512 时 P0 APs 39.6→50.1 反超 R1。**
```
wave1 出结：
  p0_cos 64.10（−1.32，cosine 在莓上伤、wheat +0.5 不复现）→ 剔除
  p0_iou / p0_cos_iou exit=1（BOX_QUALITY=iou 分支需 gt_ctrs 但 dataloader 未挂，
    属 adet 未接通项）→ 弃用
wave2 出结：
  p0_cp 64.16（−1.26；轨迹 47.97/57.66/59.90/64.12/64.06/64.16，CP 在草莓也伤）→ 剔除
  p0_ext30 64.92（−0.50；30k→137ep 反降，延长训练无益）→ 剔除
多尺度 eval 诊断（m63c_evalsweep，09-05 15:19）：416 是 R1/P0 各自最优；512 使 APs
  暴增（P0 39.62→50.08、R1 43.60→48.39，P0 反超）但整体略降 → 小目标差是像素密度
  问题；640 大跌。不加分辨率前提下需训练侧补 small。
```

```
✅ wheat 全线收官（D 系列终局见第 2 节；wheat 转为"小数据极限案例"，
   全部消融数据保留进 M6.4 论文拆解表）
（当前）yolo26 数据诊断 · Plantv2   11:51 启动，~2.6 it/s × 1131 iter/ep
        × 100 ep ≈ 12h → 预计 ~明早 00:00 完成
   ↓ run_yolo26_diag.sh 串行
        yolo26 数据诊断 · Strawberry  ~1.5-2h → ~02:00 完成
   ↓ orchestrate_m63.sh（PID 25876，判据 YOLO26_strawberry_DONE exit=0
     或 yolo26 进程死亡兜底放行）
（接棒）M6.3 锚点队列 run_m63_anchors.sh（七组，配置全部冒烟通过）：
        plantv2_R1（r50 锚点）→ plantv2_P0 → strawberry_R1 → strawberry_P0
        → **wheat_seg_strat 三组重划对照**（13:05 用户确认，追加在队尾）：
          strat_R1 / strat_P0 / strat_B2（8000 iter 同 wheat 协议，仅数据变）
        统一 batch 7 / 0.004375 / SEED 42；Plantv2 四组约 2-3 天
```

**wheat test2017 评估（12:53 完成，决定性证据）**：
| 模型 | val segm | **test segm** | Δ |
|---|---:|---:|---:|
| R1 | 15.00 | **12.60** | −2.40 |
| P0 | 16.93 | **13.15** | −3.78 |
| B2 | 17.33 | **13.90**（最抗跌） | −3.43 |
| D3 | 17.44 | **12.55**（垫底！） | −4.89 |

- **val 是高估不是低估**（全部模型 test 掉 2.4~4.9）；D3 的 17.44 是模型选择偏差；
- 稀有类在 test（每类 15-76 实例）上：WCN 3.7 / WSE 5.7 / WTA 4.4——**学到一点
  但没学透**，细粒度相似 + 数据量不足是真瓶颈，LeafRust 独占 46-50；
- B2（EMA）在诚实评估下最优：13.90 > P0 13.15 > R1 12.60，平台优势依然成立。

**wheat_seg_strat 新数据集（13:05 建立并注册，原 wheat_seg_clean 完整保留）**：
- 动机：旧 val 未分层（WCN 1图测出 0 分）+ test 已被比较污染；用户确认重划；
- 生成：`tools/make_wheat_strat.py`，818 图全合并 → 按主类分层 85/15（seed42）：
  **train 695 图/1827 实例，val 123 图/294 实例，每类 13-39 实例全部可测**；
- 图像软链（不复制），原数据集零改动；
- **预期数字"变诚实"（~13-14）而非变高**——+5 线按新锚点 strat_R1 同步重建；
- 队列追加在 strawberry_P0 之后（wheat 已是次要数据集，不抢占主战场算力）。

**yolo26 wheat 诊断终局（11:31，数据集是共同瓶颈实锤）**：
- yolo26s-seg（11.5M，x 权重 COCO 迁移）100 ep：seg mAP50=20.1，**mAP50-95=9.92**
  ——比 GBADMask（17.44）低 7.5 AP；
- 训练曲线 100 ep 未收敛仍在上升（ep98 才到 best 0.094）；长尾类全零
  （WheatCystNematode val 仅 3 实例；8/12 类 mAP<0.06）；
- **结论：584 图 + 12 类长尾是该数据集的天花板特征，wheat 上 +5 AP 目标不可行**
  （GBADMask 17.44 vs R50 15.00 的 +2.44 已经接近该平台上限）。
  按用户指示（11:47 确认）主战场迁 Plantv2/Strawberry。

**D 系列终局判读（11:10）**：
- D3 cosine 成为**新全批最高 17.44**：12k 单调上升（12.59→15.96→17.07→17.44），
  与 D2 阶梯的 9k 回落形成对照 → **cosine 退火抑制了过拟合回落**，日程线未失效；
- 但 +0.51 vs P0 仍在噪声内（±0.6），未过 17.73 晋级线 → D3 与 B2 并列进入
  **L2 3-seed 候选**（若 3-seed 过 t 检验，可作为旗舰日程）；
- 更长日程（12k）与 EMA（B2）是正交改动，L2 可测试组合。

**yolo26 GPU 事故与修复（10:54 失败 → 11:08 重启成功）**：
- **根因**：ultralytics `select_device()`（torch_utils.py:220-221）会**覆写**
  `os.environ["CUDA_VISIBLE_DEVICES"] = device`。脚本 export CVD=1 + train(device=0)
  → CVD 被改写为 "0" → cuda:0 指向物理 GPU0（sglang 等占 21GB）→ 首个 batch 即 OOM。
- **修复**：`train(device=1)`（覆写值="1" → cuda:0=物理 GPU1），已验证 CUDA:1 落卡正确。
- **次生 bug（毒标记再现）**：旧脚本无条件写 `YOLO26_WHEAT_DONE exit=0`——失败也写！
  已改为仅在 exit=0 时写，并清除日志中的假标记。**教训：完成标记必须以真实退出码为条件。**

**yolo26 数据诊断（用户 09:10 指示，wheat 已出终局见第 3 节）**：
数据集转换工具 `tools/coco2yolo.py`（COCO→YOLO 分割格式，可直接复用）。
- ✅ wheat：9.92 → 数据集是共同瓶颈（详见第 3 节终局判读）
- 🔄 Plantv2（7916/2024，16 类均衡，单实例/图）：跑 yolo26s 100ep 诊断中
- ⏭ Strawberry（1750/750，7 类，2.25 实例/图）：排在 Plantv2 后
注意：这是**数据诊断**不是论文基准（YOLO seg mAP50-95 与 COCO segm AP 定义
接近但实现有差异，只作量级判断）。

**D 系列动机**：所有组普遍出现 **6k 峰值回落**（P0 17.22@6k→16.93@8k；
vigv2m-pre 17.55@6k→17.35@8k），且组件池已见底，训练日程是零风险杠杆。

进度查看：
```bash
for f in logs/m62_D_*.log logs/m62_L1c_B1.log; do
  echo "$f: $(grep -oE 'iter: [0-9]+' $f | tail -1)"; done
python tools/summarize_all.py output/m62_ --base L1b_P0
```

---

## 4. 已修复的关键缺陷（都曾造成实验作废，勿回退）

| 文件 | 缺陷 | 修复 |
| --- | --- | --- |
| `configs/run-vigv2.yaml` | 漏 `BASIS_MODULE.NAME: ProtoNetV2` → defaults 用官方 ProtoNet，**ATTN 静默失效**（B1/B2 首批全废） | 显式写 ProtoNetV2 |
| `blendmask/psa.py` | ① 手写 N² 注意力（低层分支需 84GB）② 改 SDPA 后 torch 2.0.1 训练模式 **kernel 回退**（4.9 s/iter，8.4×） | 改**窗口注意力**（window=14，RT-DETR 原生）+ 梯度检查点；tower 22ms |
| `data/augmentation.py` | LSJ 把非方形图**拉伸成正方形**（wheat 64% 非方形） | 等比缩放 + 固定尺寸 crop/pad |
| `blendmask/ema.py` | conv 带 BN+ReLU → sigmoid 门只能放大（∈[0.5,1]） | 复原裸卷积（参考实现） |
| `data/copypaste.py` | read_image 返回只读数组，粘贴即崩 | 写入前复制 |
| `data/dataset_mapper.py` | 缺 `import random`；cp_rng fork 同源；`_load_basis_sem` 的 `transforms` 未定义；在线 basis_sem 画布用原图尺寸 | 全部修复 |
| `basis_module2.py` + defaults | 新增 `BASIS_MODULE.ATTN_LOW`（`auto`/`none`/具体名）——重组件超显存时只保留 tower 侧 | — |

验证入口：`python tests/test_m62_components.py`（单元 + 窗口语义 + LSJ + 只读输入）

---

## 5. 三条已发现的事实性错误（文档已更正，勿再引用旧数）

1. **M6.1 的 17.35/17.21 不是 "ProtoNetV2+GC"**，而是官方 ProtoNet（ATTN 失效）。
   骨干决策方向仍成立（r50 锚点同为 ProtoNet）。
2. **旧 r50 锚点 15.36/16.68 跑在 `wheat_seg`**（631/90，含 30 组跨 split 同图
   泄漏，val 虚高），而 vigv2 全部在 `wheat_seg_clean`。→ 已用 R1=15.00 重建锚点
   （泄漏仅虚高 0.36）。
3. **所有 M6.1/L1 首批运行 SEED=-1**（协议声称 seed42）。L1b 起已强制 SEED 42。

---

## 6. 待决策 / 下一步（2026-09-05 04:10 修订：用户坚持 +5，先暂停实验整理）

### +5 缺口台账（segm AP，同协议、同 seed42）

| 数据集 | R1 锚点 | +5 线 | P0 | **缺口(P0→+5线)** | 可行性 |
| --- | ---: | ---: | ---: | ---: | --- |
| wheat_seg_strat | 13.87 | 18.87 | 15.71 | **+3.16** | 平台 ceiling 估 ~17-18（clean 的 D3 17.44 为史上最高），18.87 已近/超 ceiling |
| Strawberry（22k 全日程） | 63.69 | 68.69 | 65.42 | **+3.27** | 有空间但需 +3 量级新增益 |
| Plantv2 | 98.88 | ~103.9 | 98.45 | — | **数学上不可能**（R50 已顶格），弃 |

**硬事实**：组件池（A1/B1/B2/C1/C2/D1/D2/D3/EMA/PSA）全部出清，最大真实单杠杆只有
预训练（+11.05，已用）。平台 vs R50 已收敛到 ~+1.8，**要达 +5 需把优势翻 ~3 倍**，
靠堆单点组件不可行；必须换「跨任务/跨数据」或「容量+蒸馏」层级的杠杆。

### 剩余可试杠杆（2026-09-08 用户拍板后修订）

1. ~~**任务级迁移预训练**~~ **❌ 用户否决（2026-09-08）**——不做 COCO/近域全模型
   预训练再微调。剩余杠杆全部集中在 Strawberry 512 协议上的 M6.5 深度改造 + 推理侧。
2. **MobileViGv2-B 容量上限**：已收队（66.35，仅 +0.93 over M，容量路不通）。
3. **蒸馏**：KD1（W_BASES 标定缺陷）→ **KD1b 修正版运行中**（09-08 ~06:35 UTC 出）。
4. **AS1（FCOS-TAL）**：已实现全测通过，watcher 接 KD1b 自动跑（~10:00 UTC 出）。
5. **推理侧 TTA**：期望 +0.5-1，零训练成本，GPU 空闲即可跑（FPS 基准已收）。
6. **MQ1/NK1/NK2**（M6.5 尾池）：仅在 KD1b/AS1 出正向信号后按漏斗继续；
   全负则 M6.5 提前收官。

**决策记录**：a) 任务级迁移预训练 ❌ 不做（2026-09-08）；b) c) 仍未定——
若 KD1b/AS1/MQ1 仍凑不满 Strawberry ~2.6 缺口，回退方案 = ROADMAP 0.3
第二判定（Pareto 回退：+1.7~+2.4 稳定优势 + 0.73× 参数 + 23.5 FPS 当论文主结果）。

M6.5 已开始：DQ1/QFL-only 已实现并通过 `tests/test_m62_components.py`、配置构建和
`py_compile` 验证；实验脚本为 `tools/run_m65_dq1.sh`，以 Strawberry 512 协议启动。
QFL 只改 FCOS 分类目标为 detached box IoU quality，保留 scalar regression、centerness
和其余 P0_res 组件，结果待出。M6.4 尚缺：`summarize_all.py --ttest` 已实现；FPS 实测未跑（`tools/benchmark_fps.py`
已就绪）；basis CAM 可视化未做。已出的 1-seed 结果（strat/strawberry/plantv2 锚点 +
平台）建议先固化成 M6.4 消融总表骨架。

---

## 7. 运维纪律（血泪教训）

1. **队列运行期间禁改 `adet/`**：每个新任务会 fork 新进程重新 import，
   中途改源码会让后续任务与前面不同口径。改 `tools/` / 文档安全。
2. **手动补写完成标记前，先确认所有 watcher 已死**：2026-09-03 因旧 watcher
   只查 `ALL_DONE` 存在性，导致 R1/S1 撞车 OOM 秒挂并写下"毒标记"，
   后续编排器直接跳过。幂等判据要查**内容**（如 `END X exit=0`）。
3. **后台启动必须 `setsid bash -c 'nohup ... &'`**：直接 `nohup ... &` 会被
   工具超时连带杀掉整个进程组。
4. **单 seed 差异 <1 AP 不作结论**；中途评估禁止下结论（6k vs 8k 可差 0.3+）。
5. GPU 只有 1 号可用（0/2/3 被他人 vLLM/sglang 占满）。

---

## 8. 常用命令

```bash
# 汇总（--ttest 需多 seed）
python tools/summarize_all.py output/m62_ --base L1b_P0
python tools/summarize_all.py --ttest              # 旧 23 组消融
# 参数量 / 预训练映射验收
python tests/count_params.py && python tests/verify_cspvigv2.py
# 组件单元+集成测试
python tests/test_m62_components.py
# FPS（M6.3 效率线，GPU 空闲时）
python tools/benchmark_fps.py --config-file configs/run-vigv2.yaml \
    MODEL.WEIGHTS output/m62_L1b_P0/model_final.pth
# 单组训练模板
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
    SOLVER.MAX_ITER 8000 SEED 42 SOLVER.IMS_PER_BATCH 7 \
    SOLVER.BASE_LR 0.004375 OUTPUT_DIR output/<tag>
```

产物位置：`output/`（已清理至 11G，作废批次与临时冒烟已删）、日志 `logs/m62_*.log`
（`.log` 在 .gitignore 中，不入库）。
