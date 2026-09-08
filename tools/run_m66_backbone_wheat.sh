#!/usr/bin/env bash
# M6.6 骨干消融（wheat_seg_strat 协议，与 strat_R1/strat_P0 同口径）：
#   8000 iter / batch 7 / LR 0.004375 / STEPS(4800,6400) / 12 类 / seed42。
# 参照锚点：strat_P0(vigv2m+C3K2)=15.71 / strat_R1(r50+FPN)=13.87。
# 失败臂标记后继续。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.85
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
MARK=logs/m66.log

SCHED='SOLVER.IMS_PER_BATCH 7 SOLVER.BASE_LR 0.004375 SOLVER.WARMUP_ITERS 200 SOLVER.STEPS (4800,6400) SOLVER.MAX_ITER 8000 SOLVER.CHECKPOINT_PERIOD 2000'

run() {  # $1=tag；$2/$3 = MOBILE_BB 名称/权重
  local tag="m66_wheat_$1"; shift
  if grep -q "M66_${tag}_DONE exit=0" "$MARK" 2>/dev/null; then
    echo "$tag 已完成，跳过"; return 0
  fi
  echo "===== START M66_${tag} $(date -u) =====" >> "$MARK"
  $PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
      DATASETS.NAME wheat_seg_strat \
      MODEL.BACKBONE.NAME build_fcos_mobile_bb_bifpn_backbone \
      MODEL.MOBILE_BB.MODEL_NAME "$1" \
      MODEL.MOBILE_BB.WEIGHTS "$2" \
      MODEL.MOBILE_BB.INPUT_SIZE 512 \
      MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
      MODEL.BASIS_MODULE.NUM_CLASSES 12 MODEL.FCOS.NUM_CLASSES 12 \
      MODEL.WEIGHTS "" \
      $SCHED TEST.EVAL_PERIOD 2000 SEED 42 \
      OUTPUT_DIR output/${tag} >> logs/${tag}.log 2>&1
  local rc=$?
  echo "===== END M66_${tag} exit=$rc $(date -u) =====" >> "$MARK"
  [ "$rc" -eq 0 ] && echo "M66_${tag}_DONE exit=0 $(date -u)" >> "$MARK"
  return 0
}

runvig() {  # $1=tag $2=version $3=use_c3k2 $4=pretrained
  local tag="m66_wheat_$1"
  if grep -q "M66_${tag}_DONE exit=0" "$MARK" 2>/dev/null; then
    echo "$tag 已完成，跳过"; return 0
  fi
  echo "===== START M66_${tag} $(date -u) =====" >> "$MARK"
  $PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
      DATASETS.NAME wheat_seg_strat \
      MODEL.VIG.VERSION "$2" MODEL.VIG.PRETRAINED "$4" \
      MODEL.VIG.USE_C3K2 "$3" \
      MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
      MODEL.BASIS_MODULE.NUM_CLASSES 12 MODEL.FCOS.NUM_CLASSES 12 \
      MODEL.WEIGHTS "" \
      $SCHED TEST.EVAL_PERIOD 2000 SEED 42 \
      OUTPUT_DIR output/${tag} >> logs/${tag}.log 2>&1
  local rc=$?
  echo "===== END M66_${tag} exit=$rc $(date -u) =====" >> "$MARK"
  [ "$rc" -eq 0 ] && echo "M66_${tag}_DONE exit=0 $(date -u)" >> "$MARK"
  return 0
}

# 1) R50 + BiFPN（同颈对照；RESNETS.OUT_FEATURES 沿用 yaml）
tag=m66_wheat_r50bifpn
if ! grep -q "M66_${tag}_DONE exit=0" "$MARK" 2>/dev/null; then
  echo "===== START M66_${tag} $(date -u) =====" >> "$MARK"
  $PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
      DATASETS.NAME wheat_seg_strat \
      MODEL.BACKBONE.NAME build_fcos_resnet_bifpn_backbone \
      MODEL.WEIGHTS "detectron2://ImageNetPretrained/MSRA/R-50.pkl" \
      MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
      MODEL.BASIS_MODULE.NUM_CLASSES 12 MODEL.FCOS.NUM_CLASSES 12 \
      $SCHED TEST.EVAL_PERIOD 2000 SEED 42 \
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

echo "===== ALL M66 WHEAT DONE $(date -u) =====" >> "$MARK"
