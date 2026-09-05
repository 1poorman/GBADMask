#!/usr/bin/env bash
# D3 断点续训（2026-09-04 09:31 OOM 后恢复）
#
# OOM 原因：iter 6019（6000 eval+checkpoint 之后），GPU1 上 root server.py 占
# 6.92GB + 训练峰值 14.16GB + eval 后碎片（reserved 16.28GB vs allocated 13.54GB），
# 差 224MiB。model_0005999.pth 完整保存（含 optimizer/scheduler 状态）。
#
# 恢复：OUTPUT_DIR 不变 → train_bl+.py 的 resume_or_load 自动从 iter 6000 续训。
# 附加 garbage_collection_threshold:0.85 —— 仅改变 allocator 回收策略（保留
# max_split_size_mb:128），无任何数值影响；缓解 eval 后 reserved 内存滞留。
# D2 同配置（12000 iter + 4 eval）成功跑完，证明内存余量本身足够，OOM 是边缘事件。
#
# ⚠️ M62_D_ALL_DONE 标记已在 09:31 被 run_m62_schedule.sh 写下（D3 exit=1 时），
# 该标记不反映真实状态。真实判据 = 本日志的 END D3 exit=0。
# orchestrate_yolo26.sh（PID 33757）等的是内容 "END D3 exit=0"，不受影响。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.85
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python

echo "===== RESUME D3 (from ckpt) $(date) =====" >> logs/m62_D_D3.log 2>&1
$PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
    --resume \
    SEED 42 SOLVER.IMS_PER_BATCH 7 SOLVER.BASE_LR 0.004375 \
    OUTPUT_DIR output/m62_D_D3 \
    SOLVER.MAX_ITER 12000 SOLVER.LR_SCHEDULER_NAME WarmupCosineLR \
    SOLVER.CHECKPOINT_PERIOD 3000 TEST.EVAL_PERIOD 3000 \
    >> logs/m62_D_D3.log 2>&1
echo "===== END D3 exit=$? $(date) =====" >> logs/m62_D_D3.log 2>&1
