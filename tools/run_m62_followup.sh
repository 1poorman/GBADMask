#!/usr/bin/env bash
# M6.2 follow-up 队列 —— 在 L1b 完成后自动接续（tools/watch_l1b_then_followup.sh）
#
# 内容（与 L1b 组件筛选无依赖，可安全自动接续）：
#   R1  r50-protonet @ wheat_seg_clean, batch 7, seed 42, 8000 iter
#       —— 重建 +5 AP 验收锚点。原因（2026-09-03 排查发现）：
#       旧 r50 锚点 15.36 实际跑在 wheat_seg（631/90，含 30 组跨 split 同图
#       泄漏，val 虚高），而 vigv2/L1b 全部在 wheat_seg_clean（584/82，无泄漏）
#       —— M6.1 对比跨了数据集，15.36 不可用作 M6.3 验收锚点。
#       R1 与 L1b 同协议（batch 7 / 0.004375 / seed 42 / 8000 iter），出分后
#       +5 AP 线 = R1 + 5.0，且 vigv2-m vs R1 构成真正的同数据集骨干对决。
#   S1  vigv2m-scratch（MODEL.VIG.PRETRAINED ""）
#       —— M6.4 骨干拆解表缺"从零"行（F1 的预训练结论量化）。
#
# L2（3 seed 确认）依赖 L1b 出分，不能预排：晋级名单出来后按
# "tools/run_m62_l2.sh <组件名...>" 生成（暂未创建）。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python

echo "===== START R1 $(date) =====" >> logs/m62_fu_R1.log 2>&1
$PY tools/train_bl+.py --config-file configs/run-wheat-r50.yaml --num-gpus 1 \
    --dataset wheat_seg_clean \
    SOLVER.MAX_ITER 8000 SOLVER.IMS_PER_BATCH 7 SOLVER.BASE_LR 0.004375 \
    SEED 42 OUTPUT_DIR output/m62_fu_R1_r50p7 \
    >> logs/m62_fu_R1.log 2>&1
echo "===== END R1 exit=$? $(date) =====" >> logs/m62_fu_R1.log 2>&1

echo "===== START S1 $(date) =====" >> logs/m62_fu_S1.log 2>&1
$PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
    SOLVER.MAX_ITER 8000 SOLVER.IMS_PER_BATCH 7 SOLVER.BASE_LR 0.004375 \
    SEED 42 MODEL.VIG.PRETRAINED "" OUTPUT_DIR output/m62_fu_S1_scratch \
    >> logs/m62_fu_S1.log 2>&1
echo "===== END S1 exit=$? $(date) =====" >> logs/m62_fu_S1.log 2>&1

echo "M62_FOLLOWUP_ALL_DONE $(date)" >> logs/m62_fu.log 2>&1
