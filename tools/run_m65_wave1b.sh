#!/usr/bin/env bash
# M6.5 wave1b：DQ1b（QFL 正确协议重跑）+ M2b（BOTTOM_RESOLUTION 64）。
# DQ1 作废教训：必须 run-vigv2.yaml（cspvigv2）；run-strawberry.yaml 是 cspvig v1
# 且其 build 忽略 PRETRAINED。两组均严格 P0_res 协议，单变量 vs P0_res=66.06：
#   dq1b : MODEL.FCOS.CLS_LOSS=qfl（QFL_BETA=2.0），保留 centerness，其余不动
#   m2b  : MODEL.BLENDMASK.BOTTOM_RESOLUTION 56→64（旧 M6.2 池 M2 项，从未实测）
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
      TEST.EVAL_PERIOD 4000 SEED 42 "$@" >> "logs/${tag}.log" 2>&1
  local rc=$?
  echo "===== END M65_${tag} exit=$rc $(date -u) =====" >> "$MARK"
  [ "$rc" -eq 0 ] && echo "M65_${tag}_DONE exit=0 $(date -u)" >> "$MARK"
  return "$rc"
}

run m65_dq1b_qfl_straw512 OUTPUT_DIR output/m65_dq1b_qfl_straw512 \
    MODEL.FCOS.CLS_LOSS qfl MODEL.FCOS.QFL_BETA 2.0
run m65_m2b_botres64_straw512 OUTPUT_DIR output/m65_m2b_botres64_straw512 \
    MODEL.BLENDMASK.BOTTOM_RESOLUTION 64

echo "===== ALL M6.5 wave1b DONE $(date -u) =====" >> "$MARK"
