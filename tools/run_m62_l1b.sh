#!/usr/bin/env bash
# M6.2 L1b 组件筛选队列（修订版，2026-09-03）
#
# 与作废的旧批次（output/m62_L1_*, run_m62_l1.sh）的区别：
#   1. run-vigv2.yaml 已补 BASIS_MODULE.NAME=ProtoNetV2——旧批次漏了该键，
#      defaults 默认官方 ProtoNet 不读 ATTN，导致 B1/B2 的 psa/ema 静默失效
#      （实测仅为基线重复运行：17.94 / 17.99 vs 17.35，即批次噪声 ±0.6）。
#   2. SEED 42 显式传入（旧批次 SEED=-1，违反 L1 协议）。
#   3. 组件修复：PSA 手写 N² 注意力 → SDPA（+head_dim 8 对齐零填充，
#      显存 84GB → 0.4GB）；LSJ 改等比缩放（原实现拉伸非方形图，wheat 64%
#      非方形）；EMA 复原裸卷积（原 BN+ReLU 使门控只能放大）；
#      Copy-Paste 只读图像修复 + per-worker 播种。
#   4. 新增 P0 = 平台基线（vigv2-m + ProtoNetV2 + gc）作为本批锚点。
#      R50 平台上 ProtoNetV2 效应为 +1.32 segm（16.68 vs 15.36），
#      P0 大概率显著高于旧 17.35（那是 vigv2-m + 官方 ProtoNet 的分）。
#   5. C1（NUM_BASES 8）殿后：旧批次 batch 8 OOM；batch 7 预计 ~14.5GB 可容纳。
#
# 晋级线：Δsegm ≥ +0.8 vs P0（本批锚点，排除批次噪声）
#
# ⚠️ 协议变更（2026-09-03 12:15）：IMS_PER_BATCH 8 → 7，BASE_LR 0.005 →
# 0.004375（线性缩放 7/8）。原因：GPU1 有 6.9GB root 常驻进程（server.py），
# ProtoNetV2 平台 batch 8 真实峰值 16.1GB，超出可用 ~16.3GB（含碎片必炸，
# 两次 OOM 实测）。batch 7 峰值 ~14.1GB（B1 PSA ~14.5GB、C1 bases8 ~14.5GB
# 均可容纳）。MAX_ITER/STEPS 保持 8000/4800,6400（按 iter 口径与锚点可比）。
# M6.3 验收矩阵的 r50-protonet 基线需以同协议（batch 7）重跑，口径才自洽。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
# max_split_size_mb:128 抑制分配器碎片（GPU1 可用 ~16.3GB，余量贴边）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
CFG=configs/run-vigv2.yaml
BATCH_OPTS="SOLVER.IMS_PER_BATCH 7 SOLVER.BASE_LR 0.004375"
run() {  # $1=name  $2..=额外 --opts
  local name="$1"; shift
  echo "===== START $name $(date) =====" >> "logs/m62_L1b_$name.log" 2>&1
  $PY tools/train_bl+.py --config-file $CFG --num-gpus 1 \
      SOLVER.MAX_ITER 8000 SEED 42 $BATCH_OPTS \
      OUTPUT_DIR "output/m62_L1b_$name" "$@" \
      >> "logs/m62_L1b_$name.log" 2>&1
  echo "===== END $name exit=$? $(date) =====" >> "logs/m62_L1b_$name.log" 2>&1
}

run P0                                        # 平台基线：vigv2-m+ProtoNetV2+gc（锚点）
run A1 INPUT.COPYPASTE.ENABLED True INPUT.LSJ.ENABLED True
run B1 MODEL.BASIS_MODULE.ATTN psa
run B2 MODEL.BASIS_MODULE.ATTN ema
run C2 MODEL.VIG.DROP_PATH 0.0
run C3 MODEL.VIG.USE_C3K2 False
run C1 MODEL.BASIS_MODULE.NUM_BASES 8         # OOM 风险最高，殿后

echo "M62_L1b_ALL_DONE $(date)" >> logs/m62_L1b.log 2>&1
