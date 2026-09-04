#!/usr/bin/env bash
# M6.2 L1c 队列 —— L1b 的续批（2026-09-03 15:10 重组）
#
# L1b 中途结论（详见 ROADMAP M6.2 修订记录）：
#   - P0 = 16.93（锚点）
#   - A1 失败：segm 11.19（−5.74），曲线欠收敛（4k 才 7.4），强增广与
#     8000 iter 短日程错配 → 本批 A1' 降强度重试（ROADMAP 风险预案）：
#     PROB 0.5→0.25 / MAX_DONORS 8→4 / LSJ (0.3,1.5)→(0.5,1.2)
#   - B1 作废：SDPA 训练模式 kernel 回退（4.9 s/iter，8.4×）→ PSA 改窗口式
#     注意力（window=14，RT-DETR 原生设计；tower 实测 22ms/次，~200× 提速），
#     本批 B1' 重跑
#   - B2/C2/C3/C1 未受影响，继续跑
#
# 协议与 L1b 相同：wheat_seg_clean / 8000 iter / batch 7 + 0.004375 / SEED 42
# 晋级线：segm ≥ 17.73（P0 + 0.8）
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
CFG=configs/run-vigv2.yaml
BATCH_OPTS="SOLVER.IMS_PER_BATCH 7 SOLVER.BASE_LR 0.004375"
run() {  # $1=name  $2..=额外 --opts
  local name="$1"; shift
  echo "===== START $name $(date) =====" >> "logs/m62_L1c_$name.log" 2>&1
  $PY tools/train_bl+.py --config-file $CFG --num-gpus 1 \
      SOLVER.MAX_ITER 8000 SEED 42 $BATCH_OPTS \
      OUTPUT_DIR "output/m62_L1c_$name" "$@" \
      >> "logs/m62_L1c_$name.log" 2>&1
  echo "===== END $name exit=$? $(date) =====" >> "logs/m62_L1c_$name.log" 2>&1
}

run B1 MODEL.BASIS_MODULE.ATTN psa                 # 窗口版 PSA（新实现）
run B2 MODEL.BASIS_MODULE.ATTN ema
run A1 INPUT.COPYPASTE.ENABLED True INPUT.COPYPASTE.PROB 0.25 \
      INPUT.COPYPASTE.MAX_DONORS 4 INPUT.LSJ.ENABLED True \
      INPUT.LSJ.SCALE_RANGE "(0.5,1.2)"            # 降强度重试
run C2 MODEL.VIG.DROP_PATH 0.0
run C3 MODEL.VIG.USE_C3K2 False
run C1 MODEL.BASIS_MODULE.NUM_BASES 8              # OOM 风险最高，殿后

echo "M62_L1c_ALL_DONE $(date)" >> logs/m62_L1c.log 2>&1
