#!/usr/bin/env bash
# M6.3d：Strawberry MobileViGv2-B 容量上限（2026-09-06 用户口径：先走 B、M6.5 延后）
# 与 P0_full 同协议（22k/batch8/LR0.005/STEPS(13100,17500)/416 输入），仅骨干
#   M(25.97M) → B(~40M，≤53.06M 预算)。对照：P0_full 65.42 / R1_full 63.69 / +5 线 68.69。
# 若 B≥68.69 → 容量可及 +5；若 ~66-68 → M 已近平台、容量边际有限 → 回 M6.5。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.85
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
MARK=logs/m63d.log

run() {  # $1=tag ${@:2}=opts
  local tag="$1"; shift
  if grep -q "M63D_${tag}_DONE exit=0" "$MARK" 2>/dev/null; then
    echo "$tag 已完成，跳过"; return 0
  fi
  echo "===== START $tag $(date) =====" >> "$MARK"
  $PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
      DATASETS.NAME Strawberry \
      MODEL.VIG.VERSION b MODEL.VIG.PRETRAINED weights/MobileViG_V2_B_Class.pth \
      MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
      MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
      MODEL.WEIGHTS "" \
      SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 SOLVER.WARMUP_ITERS 200 \
      SOLVER.STEPS "(13100,17500)" SOLVER.MAX_ITER 22000 SOLVER.CHECKPOINT_PERIOD 4000 \
      INPUT.MIN_SIZE_TRAIN "(384,416,448)" INPUT.MIN_SIZE_TEST 416 \
      INPUT.MAX_SIZE_TRAIN 512 INPUT.MAX_SIZE_TEST 512 \
      TEST.EVAL_PERIOD 4000 SEED 42 "$@" >> "logs/m63d_${tag}.log" 2>&1
  local rc=$?
  echo "===== END $tag exit=$rc $(date) =====" >> "$MARK"
  [ "$rc" -eq 0 ] && echo "M63D_${tag}_DONE exit=0 $(date)" >> "$MARK"
  return "$rc"
}

run B OUTPUT_DIR output/m63d_straw_B

echo "===== ALL M6.3d DONE $(date) =====" >> "$MARK"
