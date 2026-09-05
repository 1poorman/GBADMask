# GBADMask 工作记忆（MEMORY.md）

> 最后更新：2026-09-05
> 用途：会话交接。新会话请先读本文件，再读 ROADMAP.md 的 M6.2/M6.3 章节。

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

**（当前）无 GPU 任务。strat_Bcap 降档 batch6 仍 OOM（15.7GB/16.4GB，B 在 wheat
512-640 输入的激活占用几乎不随 batch 降）→ MobileViGv2-B 在 wheat 协议上跑不动，
暂停等用户定夺（见 §6 校准决策，B 是否转 strawberry 原生低分辨率或放弃）**：
```
背景：M6.3b 主队列已于 04:09 收队——
  strawberry_R1_full ✅ 63.69 segm（r50, 22k≈100ep）
  strawberry_P0_full ✅ 65.42 segm（vigv2m, +1.73）
  strat_Bcap batch7 ❌ OOM（16.2GB）→ batch6 ❌ OOM（15.7GB）
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

### 剩余可试杠杆（按对 +3 缺口的期望贡献排序，均为待决策项）

1. **任务级迁移预训练**（期望大，成本最高，最可能兑现缺口）：先在某大规模通用/近域
   分割数据集把完整 BlendMask（或仅 decoder+detect 头）预训练到收敛，再在小数据
   wheat/Strawberry 微调——把 "+11 的预训练红利"从骨干级抬到任务级。⚠️ 需先明确
   用哪个源数据（COCO-instance？/ 近域农业？），类目/分辨率迁移风险高，GPU 2-5 天。
2. **MobileViGv2-B 容量上限**（~+1-2 期望，仅 Strawberry 跑得动）：B 在 wheat 协议
   OOM（b7 16.2G/b6 15.7G > 16.4G 可用）→ 只在 Strawberry 原生低分辨率跑（~3h），
   或等 GPU0/2/3 空闲再回 wheat。单独 B 补不满缺口。
3. **蒸馏**：用 R50/大模型当 teacher 训 M 平台（架构接近更易迁移），期望 +1-2。
4. **推理侧**：多尺度 TTA / 更大 test 分辨率，期望 +0.5-1（零训练成本，先跑 FPS）。
5. **训练策略残余**：wheat strat 试 cosine+12k（clean 上 D3 +0.5 记录）；mask loss 加
   小权重 Dice（M1）；FCOS quality=iou（E1）。单点都在 ±0.5 噪声级，凑不出 +3。

**需用户拍板**：a) 是否接受"任务级迁移预训练"（大成本高风险，最可能兑现）；
b) 或把 +5 目标绑定到某单一数据集（Strawberry 更现实）并集中资源；
c) 或回归 ROADMAP 0.3 第二判定（Pareto 回退）当论文主结果。GPU 暂停中。

M6.4 尚缺：`summarize_all.py --ttest` 已实现；FPS 实测未跑（`tools/benchmark_fps.py`
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
