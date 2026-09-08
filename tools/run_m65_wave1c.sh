#!/usr/bin/env bash
# M6.5 wave1c：BR1-lite（边界加权 mask BCE）+ M2b seed123 复验。
# 均严格 P0_res 协议（run-vigv2.yaml / 480-544/768 / 512 / 22k / batch8）：
#   br1_lite : BOUNDARY_LOSS_WEIGHT=3（单变量 @56，直击 AP75/边界）
#   m2b_s123 : BOTTOM_RESOLUTION=64 + SEED 123（复验 APs +5.4 信号量级；
#              无 seed123 配对基线，只看 APs 是否仍在 ~47）
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
MARK=logs/m65.log

run() {  # $1=tag ${@:2}=opts（含 OUTPUT_DIR）
  local tag="$1"; shift
  if grep -q "M65_${tag}_DONE exit=0" "$MARK" 2>/dev/null; then
    echo "$tag 已完成，跳过"; return 0
  fi
  echo "===== START M65_${tag} $(date -u) =====" >> "$MARK"
  $PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
      DATASETS.NAME Strawberry \
      MODEL.VIG.VERSION m MODEL.VIG.PRETRAINED weights/MobileViG_V2_M_Class.pth \
      MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
      MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
      MODEL.WEIGHTS "" \
      SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 SOLVER.WARMUP_ITERS 200 \
      SOLVER.STEPS "(13100,17500)" SOLVER.MAX_ITER 22000 \
      SOLVER.CHECKPOINT_PERIOD 4000 \
      INPUT.MIN_SIZE_TRAIN "(480,512,544)" INPUT.MIN_SIZE_TEST 512 \
      INPUT.MAX_SIZE_TRAIN 768 INPUT.MAX_SIZE_TEST 768 \
      TEST.EVAL_PERIOD 4000 "$@" >> "logs/${tag}.log" 2>&1
  local rc=$?
  echo "===== END M65_${tag} exit=$rc $(date -u) =====" >> "$MARK"
  [ "$rc" -eq 0 ] && echo "M65_${tag}_DONE exit=0 $(date -u)" >> "$MARK"
  return "$rc"
}

run m65_br1_lite_bw3_straw512 OUTPUT_DIR output/m65_br1_lite_bw3_straw512 \
    SEED 42 MODEL.BLENDMASK.BOUNDARY_LOSS_WEIGHT 3.0
run m65_m2b_s123_straw512 OUTPUT_DIR output/m65_m2b_s123_straw512 \
    SEED 123 MODEL.BLENDMASK.BOTTOM_RESOLUTION 64

echo "===== ALL M6.5 wave1c DONE $(date -u) =====" >> "$MARK"
