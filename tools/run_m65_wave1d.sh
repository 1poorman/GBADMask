#!/usr/bin/env bash
# M6.5 wave1d：M2b 三 seed 收口。
#   p0res_s123 : P0_res 协议 + SEED 123（补配对基线，与 M2b-s123 同种子对照）
#   m2b_s2024  : BOTTOM_RESOLUTION=64 + SEED 2024（凑第三 seed，供 t 检验）
# 已有：P0_res(s42)=66.06、M2b(s42)=65.15(APs 47.46)、M2b(s123)=66.14(APs 49.37)
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

run m65_p0res_s123 OUTPUT_DIR output/m65_p0res_s123 SEED 123
run m65_m2b_s2024 OUTPUT_DIR output/m65_m2b_s2024 \
    SEED 2024 MODEL.BLENDMASK.BOTTOM_RESOLUTION 64

echo "===== ALL M6.5 wave1d DONE $(date -u) =====" >> "$MARK"
