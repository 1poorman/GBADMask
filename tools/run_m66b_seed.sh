#!/usr/bin/env bash
# M6.6b：vigv2-S+C3K2 3-seed 确认轮（用户 09-09 拍板）。
# 已有 seed42（m66_straw_vigv2s=65.87 / m66_wheat_vigv2s=16.20），
# 本轮补 s123/s2024 两 seed × 两数据集，并补 wheat P0-M 配对基线
# s123/s2024（straw 侧 P0_res 三 seed 66.06/65.77/65.92 已存在，无需补）。
# 判定口径：3-seed 均值 ± 配对 t 检验（vs P0 同 seed 配对）。
# 顺序：wheat 3 组（~1h each，先出轻量侧结论）→ straw 3 组（~1.9h each）。
# 失败组标记后不阻塞。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.85
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
MARK=logs/m66b.log

WSCHED='SOLVER.IMS_PER_BATCH 7 SOLVER.BASE_LR 0.004375 SOLVER.WARMUP_ITERS 200 SOLVER.STEPS (4800,6400) SOLVER.MAX_ITER 8000 SOLVER.CHECKPOINT_PERIOD 4000'
SSCHED='SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 SOLVER.WARMUP_ITERS 200 SOLVER.STEPS (13100,17500) SOLVER.MAX_ITER 22000 SOLVER.CHECKPOINT_PERIOD 4000'
SINP='INPUT.MIN_SIZE_TRAIN (480,512,544) INPUT.MIN_SIZE_TEST 512 INPUT.MAX_SIZE_TRAIN 768 INPUT.MAX_SIZE_TEST 768'

runwheat() {  # $1=tag $2=version $3=c3k2 $4=pretrained $5=seed
  local tag="m66b_wheat_$1"
  if grep -q "M66B_${tag}_DONE exit=0" "$MARK" 2>/dev/null; then
    echo "$tag 已完成，跳过"; return 0
  fi
  echo "===== START M66B_${tag} $(date -u) =====" >> "$MARK"
  $PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
      DATASETS.NAME wheat_seg_strat \
      MODEL.VIG.VERSION "$2" MODEL.VIG.PRETRAINED "$4" \
      MODEL.VIG.USE_C3K2 "$3" \
      MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
      MODEL.BASIS_MODULE.NUM_CLASSES 12 MODEL.FCOS.NUM_CLASSES 12 \
      MODEL.WEIGHTS "" \
      $WSCHED TEST.EVAL_PERIOD 2000 SEED "$5" \
      OUTPUT_DIR output/${tag} >> logs/${tag}.log 2>&1
  local rc=$?
  echo "===== END M66B_${tag} exit=$rc $(date -u) =====" >> "$MARK"
  [ "$rc" -eq 0 ] && echo "M66B_${tag}_DONE exit=0 $(date -u)" >> "$MARK"
  return 0
}

runstraw() {  # $1=tag $2=version $3=c3k2 $4=pretrained $5=seed
  local tag="m66b_straw_$1"
  if grep -q "M66B_${tag}_DONE exit=0" "$MARK" 2>/dev/null; then
    echo "$tag 已完成，跳过"; return 0
  fi
  echo "===== START M66B_${tag} $(date -u) =====" >> "$MARK"
  $PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
      DATASETS.NAME Strawberry \
      MODEL.VIG.VERSION "$2" MODEL.VIG.PRETRAINED "$4" \
      MODEL.VIG.USE_C3K2 "$3" \
      MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
      MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
      MODEL.WEIGHTS "" \
      $SSCHED $SINP TEST.EVAL_PERIOD 4000 SEED "$5" \
      OUTPUT_DIR output/${tag} >> logs/${tag}.log 2>&1
  local rc=$?
  echo "===== END M66B_${tag} exit=$rc $(date -u) =====" >> "$MARK"
  [ "$rc" -eq 0 ] && echo "M66B_${tag}_DONE exit=0 $(date -u)" >> "$MARK"
  return 0
}

# ---- wheat（含 P0-M 配对基线）----
runwheat vigv2s_s123 m True  weights/MobileViG_V2_S_Class.pth 123
runwheat p0m_s123    m True  weights/MobileViG_V2_M_Class.pth 123
runwheat vigv2s_s2024 m True weights/MobileViG_V2_S_Class.pth 2024
runwheat p0m_s2024    m True weights/MobileViG_V2_M_Class.pth 2024

# ---- strawberry ----
runstraw vigv2s_s123  s True weights/MobileViG_V2_S_Class.pth 123
runstraw vigv2s_s2024 s True weights/MobileViG_V2_S_Class.pth 2024

echo "===== ALL M66B DONE $(date -u) =====" >> "$MARK"
