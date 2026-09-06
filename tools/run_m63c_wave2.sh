#!/usr/bin/env bash
# M6.3c wave2：Strawberry M 平台剩余非分辨率候选（2026-09-06）
# wave1 结论（m63c.log）：cosine −1.32 剔除；BOX_QUALITY=iou 崩（gt_ctrs 未接通，弃）。
# 本波：仍 416 协议 / batch8 / LR0.005 / SEED42，基线 P0_full=65.42：
#   1. p0_cp     : Copy-Paste 单独启用（不叠 LSJ）。草莓 1750 图/100ep 机制与 wheat
#                  不同（wheat 584 图 A1 系全灭），值得单测。defaults PROB.5/MAX8。
#   2. p0_ext30  : 延训 22k→30k（STEPS 仍 60/80ep=(13100,17500)，吃 16-22k 仍在爬的尾段）。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.85
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
MARK=logs/m63c.log

run() {  # $1=tag ${@:2}=opts (含 OUTPUT_DIR)
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

run p0_cp    OUTPUT_DIR output/m63c_p0_cp    INPUT.COPYPASTE.ENABLED True
run p0_ext30 OUTPUT_DIR output/m63c_p0_ext30 SOLVER.MAX_ITER 30000

echo "===== ALL M6.3c wave2 DONE $(date) =====" >> "$MARK"
