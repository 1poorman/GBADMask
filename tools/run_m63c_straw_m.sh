#!/usr/bin/env bash
# M6.3c：Strawberry 攻坚第一波 —— M 平台非分辨率杠杆筛选（2026-09-05 用户口径：
#   +5 绑定 Strawberry、先推 M 到底、不改输入分辨率）
# 基线：P0_full = 65.42（run-vigv2 架构 + Strawberry 全日程 22k/batch8/LR0.005/
#       STEPS(13100,17500)，eval 416）。R1_full = 63.69，+5 线 = 68.69。
# 本波纯配置、正交可叠加，各与 P0_full 比（单变量）：
#   1. p0_cos      : WarmupCosineLR @22k（P0 22k 时仍在缓升；wheat 上 D3 cosine +0.5）
#   2. p0_iou      : FCOS BOX_QUALITY=iou（排序质量，可能利好小斑）
#   3. p0_cos_iou  : 两者组合（提前验证可加性，≥单项和 50% 视为可保留）
# 队列后视结果决定：再加 CP/LSJ、extend、Dice(需改码) 等第二波；R1 控制组随后补。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.85
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
MARK=logs/m63c.log

run() {  # $1=tag ${@:2}=extra opts (已含 OUTPUT_DIR)
  local tag="$1"; shift
  if grep -q "M63C_${tag}_DONE exit=0" "$MARK" 2>/dev/null; then
    echo "$tag 已完成，跳过"; return 0
  fi
  echo "===== START $tag $(date) =====" >> "$MARK"
  $PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
      DATASETS.NAME Strawberry \
      MODEL.VIG.VERSION m MODEL.VIG.PRETRAINED weights/MobileViG_V2_M_Class.pth \
      MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
      MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
      MODEL.WEIGHTS "" \
      SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 SOLVER.WARMUP_ITERS 200 \
      SOLVER.STEPS "(13100,17500)" SOLVER.MAX_ITER 22000 SOLVER.CHECKPOINT_PERIOD 4000 \
      INPUT.MIN_SIZE_TRAIN "(384,416,448)" INPUT.MIN_SIZE_TEST 416 \
      INPUT.MAX_SIZE_TRAIN 512 INPUT.MAX_SIZE_TEST 512 \
      TEST.EVAL_PERIOD 4000 SEED 42 "$@" >> "logs/m63c_${tag}.log" 2>&1
  local rc=$?
  echo "===== END $tag exit=$rc $(date) =====" >> "$MARK"
  [ "$rc" -eq 0 ] && echo "M63C_${tag}_DONE exit=0 $(date)" >> "$MARK"
  return "$rc"
}

run p0_cos     OUTPUT_DIR output/m63c_p0_cos     SOLVER.LR_SCHEDULER_NAME WarmupCosineLR
run p0_iou     OUTPUT_DIR output/m63c_p0_iou     MODEL.FCOS.BOX_QUALITY iou
run p0_cos_iou OUTPUT_DIR output/m63c_p0_cos_iou SOLVER.LR_SCHEDULER_NAME WarmupCosineLR MODEL.FCOS.BOX_QUALITY iou

echo "===== ALL M6.3c wave1 DONE $(date) =====" >> "$MARK"
