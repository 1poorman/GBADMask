#!/usr/bin/env bash
# M6.5 HQ1：Strawberry 512 协议上的 res2 detail branch 首轮验证。
# 单变量（相对 P0_res=66.06）：BiFPN.PASSTHROUGH=['res2'] + ProtoNetV2 DETAIL_ON，
# 其余严格对齐 P0_res（run-vigv2.yaml / 480-544 训练 / 512 测试 / 22k / batch8 / seed42）。
# 教训（DQ1）：必须用 run-vigv2.yaml（cspvigv2），勿用 run-strawberry.yaml（cspvig v1）。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
TAG=m65_hq1_detail_straw512
OUT=output/${TAG}
LOG=logs/${TAG}.log
MARK=logs/m65.log

if grep -q "M65_${TAG}_DONE exit=0" "$MARK" 2>/dev/null; then
  echo "$TAG 已完成，跳过"
  exit 0
fi

echo "===== START M65_${TAG} $(date -u) =====" >> "$MARK"
$PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
    DATASETS.NAME Strawberry \
    MODEL.VIG.VERSION m MODEL.VIG.PRETRAINED weights/MobileViG_V2_M_Class.pth \
    MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
    MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
    MODEL.BiFPN.PASSTHROUGH "['res2']" \
    MODEL.BASIS_MODULE.DETAIL_ON True \
    MODEL.WEIGHTS "" \
    SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 \
    SOLVER.WARMUP_ITERS 200 SOLVER.STEPS "(13100,17500)" SOLVER.MAX_ITER 22000 \
    SOLVER.CHECKPOINT_PERIOD 4000 \
    INPUT.MIN_SIZE_TRAIN "(480,512,544)" INPUT.MIN_SIZE_TEST 512 \
    INPUT.MAX_SIZE_TRAIN 768 INPUT.MAX_SIZE_TEST 768 \
    TEST.EVAL_PERIOD 4000 SEED 42 OUTPUT_DIR "$OUT" \
    > "$LOG" 2>&1
rc=$?
echo "===== END M65_${TAG} exit=$rc $(date -u) =====" >> "$MARK"
[ "$rc" -eq 0 ] && echo "M65_${TAG}_DONE exit=0 $(date -u)" >> "$MARK"
exit "$rc"
