# GBADMask 工作记忆（MEMORY.md）

> 最后更新：2026-09-04 04:00 UTC
> 用途：会话交接。新会话请先读本文件，再读 ROADMAP.md 的 M6.2 章节。

---

## 1. 项目目标与当前验收口径

- **目标**：小数据农作物病害实例分割，模型 ≤ 1.5× 官方 BlendMask(R50)（≤53.06M），
  segm AP **绝对提升 ≥ +5.0**（3 seed 均值、全部为正、配对 t 检验 p<0.05）。
- **平台**（M6.1 决出）：cspvigv2-M（MobileViGv2+C3K2，ImageNet 预训练）+ BiFPN(3,160)
  + ProtoNetV2 + GC basis，25.97M 参数。
- **✅ 新验收线（2026-09-04 确立）**：`R1 = 15.00` → **达标线 20.00 segm**。
  当前最佳 17.33 → **缺口 2.67 AP**。

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
| **B2** +EMA 注意力 | **17.33** | — | **+0.40** | 未过线（17.73），全批最高，L2 候选 |
| **C2** DROP_PATH 0（vs 0.1） | **17.18** | — | +0.25 | 噪声内，保持 0.1 |
| **C3** 关闭 C3K2 | **15.84** | — | **−1.09** | **C3K2 贡献 +1.09 实锤**（保留） |
| A1 CP+LSJ 全强度 | 11.19 | 11.15 | −5.74 | ❌ 淘汰 |
| A1' CP+LSJ 半强度 | 11.04 | — | −5.89 | ❌ 二连败 → 机制不匹配（短日程+小数据） |
| C1 NUM_BASES 8 | OOM | — | — | ❌ 放弃（显存超限，收益弱） |

**关键效应分解**：预训练 +11.05 ≫ C3K2 +1.09 > 骨干+basis vs R50 +1.93 > EMA +0.40

**噪声基线**：同配置跨批次 ±0.6 AP（17.35/17.94/17.99 三次同配置实测）。
晋级线 +0.8 与噪声同量级 → 单 seed 判定从严，L2 必须 3 seed + t 检验。

---

## 3. 正在运行 / 已编排（后台）

```
（当前）B1''  仅 tower 的窗口 PSA          output/m62_L1c_B1     ~05:00 完成
   ↓ tools/orchestrate_schedule.sh（PID 57966，每 5min 轮询）
D1  STEPS 提前 (4200,5600) @8000 iter      output/m62_D_D1       ~78 min
D2  MAX_ITER 12000 + STEPS (7200,9600)     output/m62_D_D2       ~2h
D3  MAX_ITER 12000 + WarmupCosineLR        output/m62_D_D3       ~2h
```

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

## 6. 待决策 / 下一步（缺口 2.67 AP）

组件池已见底，需新杠杆。按预期收益排序：

1. **训练日程**（D1/D2/D3 正在跑）—— 零风险，看 6k 峰值能否兑现
2. **vigv2-b 变体**：需下载 `weights/MobileViG_V2_B_Class.pth`（代码已支持 VARIANTS["b"]，
   参数 ~40M 仍在 53.06M 预算内）。**预训练是已验证最强杠杆（+11.05 AP）**，
   更大预训练骨干最可能兑现缺口
3. **推理侧**：多尺度测试 / TTA（零训练成本，M6.3 效率线 ≥10 FPS 需先验证）
4. **basis 分辨率**：bases 现 1/4，试 1/2
5. **L2 确认**：若 D 系列或 b 变体过线，走 3 seed {42,123,2024} + 配对 t 检验

M6.4 尚缺：`summarize_all.py --ttest` 已实现；FPS 实测未跑（`tools/benchmark_fps.py`
已就绪，GPU 空闲时跑）；basis CAM 可视化未做。

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
