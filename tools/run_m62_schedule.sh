#!/usr/bin/env bash
# M6.2 D 系列：训练日程消融（2026-09-04）
#
# 动机：P0/M6.1/L1c 各组普遍出现 **6k 峰值回落**（P0 17.22@6k → 16.93@8k；
# vigv2m-pre 17.55@6k → 17.35@8k），提示 8000 iter + STEPS(4800,6400) 的
# LR 日程与本数据规模不匹配。且组件池已见底（A1 淘汰/C1 放弃/B2 仅 +0.40），
# 距 +5 AP 线（= R1 15.00 + 5 = 20.00）缺口 2.67 AP，训练日程是零风险杠杆。
#
# 三组（平台与 P0 完全一致，唯一变量为 SOLVER 日程）：
#   D1  STEPS 提前：(4200, 5600) / MAX_ITER 8000  —— 峰值处早衰减，等算力
#   D2  延长训练：MAX_ITER 12000 / STEPS (7200, 9600) —— 1.5× 算力
#   D3  Cosine 退火：MAX_ITER 12000，无阶梯（WarmupCosineLR）
#
# 协议：wheat_seg_clean / batch 7 + 0.004375 / SEED 42（与 P0 锚点可比）
# 参照：P0 = 16.93；晋级线 ≥ 17.73（+0.8）
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
  echo "===== START $name $(date) =====" >> "logs/m62_D_$name.log" 2>&1
  $PY tools/train_bl+.py --config-file $CFG --num-gpus 1 \
      SEED 42 $BATCH_OPTS OUTPUT_DIR "output/m62_D_$name" "$@" \
      >> "logs/m62_D_$name.log" 2>&1
  echo "===== END $name exit=$? $(date) =====" >> "logs/m62_D_$name.log" 2>&1
}

# D1：等算力，STEPS 提前到 0.525/0.70（原 0.60/0.80）
run D1 SOLVER.MAX_ITER 8000 SOLVER.STEPS "(4200,5600)"
# D2：1.5× 算力，STEPS 维持 0.60/0.80 比例
run D2 SOLVER.MAX_ITER 12000 SOLVER.STEPS "(7200,9600)" \
      SOLVER.CHECKPOINT_PERIOD 3000 TEST.EVAL_PERIOD 3000
# D3：1.5× 算力 + cosine 退火（无阶梯）
run D3 SOLVER.MAX_ITER 12000 SOLVER.LR_SCHEDULER_NAME WarmupCosineLR \
      SOLVER.CHECKPOINT_PERIOD 3000 TEST.EVAL_PERIOD 3000

echo "M62_D_ALL_DONE $(date)" >> logs/m62_D.log 2>&1
