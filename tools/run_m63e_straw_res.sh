#!/usr/bin/env bash
# M6.3e：Strawberry 分辨率重议（2026-09-06 用户选 2 = 松绑"不加分辨率"约束）
# 依据：多尺度 eval（m63c_evalsweep）512 时 P0 APs 39.62→50.08 反超 R1 48.39，
#   小目标差是像素密度问题 → 提到 512 档训练+测试，让小目标真正吃满像素。
# 新协议（R1/P0 同口径，改分辨率后锚点随 R1 重锚）：
#   22k / batch8 / LR0.005 / STEPS(13100,17500) / INPUT 训练多尺度 (480,512,544)
#   / 测试 512 / MAX 768。对比：旧协议 P0_full 65.42 / R1_full 63.69（416）。
# 1. R1_res : r50-protonet（新锚点）  2. P0_res : vigv2m（候选）
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.85
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
MARK=logs/m63e.log
SCHED='SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 SOLVER.WARMUP_ITERS 200 SOLVER.STEPS (13100,17500) SOLVER.MAX_ITER 22000 SOLVER.CHECKPOINT_PERIOD 4000'
INP='INPUT.MIN_SIZE_TRAIN (480,512,544) INPUT.MIN_SIZE_TEST 512 INPUT.MAX_SIZE_TRAIN 768 INPUT.MAX_SIZE_TEST 768'

run() {  # $1=tag $2=config ${@:3}=opts
  local tag="$1" cfg="$2"; shift 2
  if grep -q "M63E_${tag}_DONE exit=0" "$MARK" 2>/dev/null; then
    echo "$tag 已完成，跳过"; return 0
  fi
  echo "===== START $tag $(date) =====" >> "$MARK"
  $PY tools/train_bl+.py --config-file "$cfg" --num-gpus 1 \
      "$@" TEST.EVAL_PERIOD 4000 SEED 42 >> "logs/m63e_${tag}.log" 2>&1
  local rc=$?
  echo "===== END $tag exit=$rc $(date) =====" >> "$MARK"
  [ "$rc" -eq 0 ] && echo "M63E_${tag}_DONE exit=0 $(date)" >> "$MARK"
  return "$rc"
}

run R1_res configs/run-wheat-r50.yaml \
    DATASETS.NAME Strawberry \
    MODEL.BASIS_MODULE.NAME ProtoNet \
    MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
    MODEL.WEIGHTS "detectron2://ImageNetPretrained/MSRA/R-50.pkl" \
    $SCHED $INP OUTPUT_DIR output/m63e_straw_R1_res

run P0_res configs/run-vigv2.yaml \
    DATASETS.NAME Strawberry \
    MODEL.VIG.VERSION m MODEL.VIG.PRETRAINED weights/MobileViG_V2_M_Class.pth \
    MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
    MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
    MODEL.WEIGHTS "" \
    $SCHED $INP OUTPUT_DIR output/m63e_straw_P0_res

echo "===== ALL M6.3e DONE $(date) =====" >> "$MARK"
