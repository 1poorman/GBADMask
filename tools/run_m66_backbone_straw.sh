#!/usr/bin/env bash
# M6.6 骨干消融（Strawberry 512 协议，与 P0_res/R1_res 完全同口径）：
#   r50bifpn / vigv2s(+C3K2) / mvigv2s / mvigv2m（无 C3K2）/
#   mnv3l / mnv4s / lsnett
# 每臂唯一变量 = 骨干，BiFPN(3,160)+ProtoNetV2+GC+FCOS 全一致。
# 参照锚点：P0_res(vigv2m+C3K2)=66.06 / R1_res(r50+FPN 官方)=63.68。
# 单 seed 筛选（42）；失败臂标记后继续（不阻塞后续臂）。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.85
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
MARK=logs/m66.log

SCHED='SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 SOLVER.WARMUP_ITERS 200 SOLVER.STEPS (13100,17500) SOLVER.MAX_ITER 22000 SOLVER.CHECKPOINT_PERIOD 4000'
INP='INPUT.MIN_SIZE_TRAIN (480,512,544) INPUT.MIN_SIZE_TEST 512 INPUT.MAX_SIZE_TRAIN 768 INPUT.MAX_SIZE_TEST 768'

run() {  # $1=tag；其余为 opts
  local tag="m66_straw_$1"; shift
  if grep -q "M66_${tag}_DONE exit=0" "$MARK" 2>/dev/null; then
    echo "$tag 已完成，跳过"; return 0
  fi
  echo "===== START M66_${tag} $(date -u) =====" >> "$MARK"
  $PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
      DATASETS.NAME Strawberry \
      MODEL.BACKBONE.NAME build_fcos_mobile_bb_bifpn_backbone \
      MODEL.MOBILE_BB.MODEL_NAME "$1" \
      MODEL.MOBILE_BB.WEIGHTS "$2" \
      MODEL.MOBILE_BB.INPUT_SIZE 512 \
      MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
      MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
      MODEL.WEIGHTS "" \
      $SCHED $INP TEST.EVAL_PERIOD 4000 SEED 42 \
      OUTPUT_DIR output/${tag} >> logs/${tag}.log 2>&1
  local rc=$?
  echo "===== END M66_${tag} exit=$rc $(date -u) =====" >> "$MARK"
  [ "$rc" -eq 0 ] && echo "M66_${tag}_DONE exit=0 $(date -u)" >> "$MARK"
  return 0
}

runvig() {  # $1=tag $2=version $3=use_c3k2 $4=pretrained
  local tag="m66_straw_$1"
  if grep -q "M66_${tag}_DONE exit=0" "$MARK" 2>/dev/null; then
    echo "$tag 已完成，跳过"; return 0
  fi
  echo "===== START M66_${tag} $(date -u) =====" >> "$MARK"
  $PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
      DATASETS.NAME Strawberry \
      MODEL.VIG.VERSION "$2" MODEL.VIG.PRETRAINED "$4" \
      MODEL.VIG.USE_C3K2 "$3" \
      MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
      MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
      MODEL.WEIGHTS "" \
      $SCHED $INP TEST.EVAL_PERIOD 4000 SEED 42 \
      OUTPUT_DIR output/${tag} >> logs/${tag}.log 2>&1
  local rc=$?
  echo "===== END M66_${tag} exit=$rc $(date -u) =====" >> "$MARK"
  [ "$rc" -eq 0 ] && echo "M66_${tag}_DONE exit=0 $(date -u)" >> "$MARK"
  return 0
}

# 1) R50 + BiFPN（同颈对照；RESNETS.OUT_FEATURES 沿用 yaml 的 ["res3","res4","res5"]）
tag=m66_straw_r50bifpn
if ! grep -q "M66_${tag}_DONE exit=0" "$MARK" 2>/dev/null; then
  echo "===== START M66_${tag} $(date -u) =====" >> "$MARK"
  $PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
      DATASETS.NAME Strawberry \
      MODEL.BACKBONE.NAME build_fcos_resnet_bifpn_backbone \
      MODEL.WEIGHTS "detectron2://ImageNetPretrained/MSRA/R-50.pkl" \
      MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
      MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
      $SCHED $INP TEST.EVAL_PERIOD 4000 SEED 42 \
      OUTPUT_DIR output/${tag} >> logs/${tag}.log 2>&1
  rc=$?
  echo "===== END M66_${tag} exit=$rc $(date -u) =====" >> "$MARK"
  [ "$rc" -eq 0 ] && echo "M66_${tag}_DONE exit=0 $(date -u)" >> "$MARK"
fi

# 2) vigv2 家族
runvig vigv2s    s m True  weights/MobileViG_V2_S_Class.pth
runvig mvigv2s   s m False weights/MobileViG_V2_S_Class.pth
runvig mvigv2m   m m False weights/MobileViG_V2_M_Class.pth

# 3) 第三方轻量骨干
run mnv3l  mnv3_l  weights/mobilenetv3_large_100_ra_in1k.pth
run mnv4s  mnv4_s  weights/mobilenetv4_conv_small_in1k.pth
run lsnett lsnet_t weights/lsnet_t.pth

echo "===== ALL M66 STRAW DONE $(date -u) =====" >> "$MARK"
