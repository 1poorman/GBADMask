#!/usr/bin/env bash
# M6.2 收尾队列（2026-09-04 01:20）：R1 → S1 → B1''
#
# 背景（L1c 终局）：
#   - B2=17.33 / C2=17.18 / C3=15.84（C3K2 贡献 +1.09 实锤）/ A1'=11.04（淘汰）
#   - B1 窗口版 batch7 双插入点仍 OOM（15487M > 预算）→ B1'' 只留 tower
#     插入点（ATTN_LOW=none，省 attn_low 的 PSA ~0.8GB）
#   - C1（NUM_BASES 8）放弃：显存放不下且预期收益弱
#   - R1/S1 曾被"毒标记"（旧 watcher 提前触发 + ALL_DONE 幂等标记）坑死，
#     已清理重跑
#
# 协议：wheat_seg_clean / 8000 iter / batch 7 + 0.004375 / SEED 42（与 L1b/c 同）
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python

# --- R1：+5 AP 验收锚点重建（r50-protonet @ wheat_seg_clean, batch 7）---
echo "===== START R1 $(date) =====" >> logs/m62_fu_R1.log 2>&1
$PY tools/train_bl+.py --config-file configs/run-wheat-r50.yaml --num-gpus 1 \
    --dataset wheat_seg_clean \
    SOLVER.MAX_ITER 8000 SOLVER.IMS_PER_BATCH 7 SOLVER.BASE_LR 0.004375 \
    SEED 42 OUTPUT_DIR output/m62_fu_R1_r50p7 \
    >> logs/m62_fu_R1.log 2>&1
echo "===== END R1 exit=$? $(date) =====" >> logs/m62_fu_R1.log 2>&1

# --- S1：vigv2m 从零对照（M6.4 骨干拆解表的"从零"行）---
echo "===== START S1 $(date) =====" >> logs/m62_fu_S1.log 2>&1
$PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
    SOLVER.MAX_ITER 8000 SOLVER.IMS_PER_BATCH 7 SOLVER.BASE_LR 0.004375 \
    SEED 42 MODEL.VIG.PRETRAINED "" OUTPUT_DIR output/m62_fu_S1_scratch \
    >> logs/m62_fu_S1.log 2>&1
echo "===== END S1 exit=$? $(date) =====" >> logs/m62_fu_S1.log 2>&1

# --- B1''：仅 tower 的窗口 PSA ---
echo "===== START B1 $(date) =====" >> logs/m62_L1c_B1.log 2>&1
$PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
    SOLVER.MAX_ITER 8000 SOLVER.IMS_PER_BATCH 7 SOLVER.BASE_LR 0.004375 \
    SEED 42 OUTPUT_DIR output/m62_L1c_B1 \
    MODEL.BASIS_MODULE.ATTN psa MODEL.BASIS_MODULE.ATTN_LOW none \
    >> logs/m62_L1c_B1.log 2>&1
echo "===== END B1 exit=$? $(date) =====" >> logs/m62_L1c_B1.log 2>&1

echo "M62_FINAL_QUEUE_DONE $(date)" >> logs/m62_fu.log 2>&1
