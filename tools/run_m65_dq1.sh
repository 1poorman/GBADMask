#!/usr/bin/env bash
# M6.5 DQ1：Strawberry 512 协议上的 QFL-only 首轮验证。
# 单变量：仅把 FCOS sigmoid focal 分类目标改为 detached box IoU quality；
# scalar box regression、centerness、ProtoNetV2、GC 和 BiFPN 全部保持 P0_res。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.85
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
TAG=m65_dq1_qfl_straw512
OUT=output/${TAG}
LOG=logs/${TAG}.log
MARK=logs/m65.log

if grep -q "M65_${TAG}_DONE exit=0" "$MARK" 2>/dev/null; then
  echo "$TAG 已完成，跳过"
  exit 0
fi

echo "===== START M65_${TAG} $(date -u) =====" >> "$MARK"
$PY tools/train_bl+.py --config-file configs/run-strawberry.yaml --num-gpus 1 \
    MODEL.VIG.VERSION m MODEL.VIG.PRETRAINED weights/MobileViG_V2_M_Class.pth \
    MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
    MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
    MODEL.FCOS.CLS_LOSS qfl MODEL.FCOS.QFL_BETA 2.0 \
    MODEL.WEIGHTS "" \
    SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 \
    SOLVER.WARMUP_ITERS 200 SOLVER.STEPS "(13100,17500)" SOLVER.MAX_ITER 22000 \
    SOLVER.CHECKPOINT_PERIOD 4000 \
    INPUT.MIN_SIZE_TRAIN "(384,416,448)" INPUT.MIN_SIZE_TEST 512 \
    INPUT.MAX_SIZE_TRAIN 512 INPUT.MAX_SIZE_TEST 800 \
    TEST.EVAL_PERIOD 4000 SEED 42 OUTPUT_DIR "$OUT" \
    > "$LOG" 2>&1
rc=$?
echo "===== END M65_${TAG} exit=$rc $(date -u) =====" >> "$MARK"
[ "$rc" -eq 0 ] && echo "M65_${TAG}_DONE exit=0 $(date -u)" >> "$MARK"
exit "$rc"
