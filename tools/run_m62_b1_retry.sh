#!/usr/bin/env bash
# L1c 补跑 B1（window-PSA + 梯度检查点版）：L1c 首跑 B1 因 attn 矩阵反向
# 暂存 OOM（batch7 下两插入点 ~2GB×2 份），checkpoint 用重算换显存。
# 等 L1c 主体（B2/A1'/C2/C3/C1）结束后跑，然后才放行 followup。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
CFG=configs/run-vigv2.yaml

echo "===== START B1 $(date) =====" >> "logs/m62_L1c_B1.log" 2>&1
$PY tools/train_bl+.py --config-file $CFG --num-gpus 1 \
    SOLVER.MAX_ITER 8000 SEED 42 \
    SOLVER.IMS_PER_BATCH 7 SOLVER.BASE_LR 0.004375 \
    OUTPUT_DIR output/m62_L1c_B1 MODEL.BASIS_MODULE.ATTN psa \
    >> "logs/m62_L1c_B1.log" 2>&1
echo "===== END B1 exit=$? $(date) =====" >> "logs/m62_L1c_B1.log" 2>&1
