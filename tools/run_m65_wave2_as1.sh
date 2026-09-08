#!/usr/bin/env bash
# M6.5 wave2：AS1（FCOS-TAL，TOOD 风格任务对齐正样本指派）。
# 单变量 vs P0_res=66.06（s42，Strawberry 512 协议）：MODEL.FCOS.ASSIGN
# default→tal，其余完全一致。TAL 参数取默认：topk=10 / topk_small=4
# （面积 <32²）/ alpha=1 / beta=6 / warmup=500（前 500 iter 保持默认
# 指派）/ 每 GT 保底 1 正点（全局最大 IoU 兜底）。推理路径零改动。
# 实现与单测：adet/modeling/fcos/fcos_outputs.py::_tal_reassign、
# tests/test_m65_as1.py（全过 2026-09-08）。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
TAG=m65_as1_tal_straw512
OUT=output/${TAG}
LOG=logs/${TAG}.log
MARK=logs/m65.log

if grep -q "M65_${TAG}_DONE exit=0" "$MARK" 2>/dev/null; then
  echo "$TAG 已完成，跳过"; exit 0
fi

echo "===== START M65_${TAG} $(date -u) =====" >> "$MARK"
$PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
    DATASETS.NAME Strawberry \
    MODEL.VIG.VERSION m MODEL.VIG.PRETRAINED weights/MobileViG_V2_M_Class.pth \
    MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
    MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
    MODEL.WEIGHTS "" \
    MODEL.FCOS.ASSIGN tal \
    SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 SOLVER.WARMUP_ITERS 200 \
    SOLVER.STEPS "(13100,17500)" SOLVER.MAX_ITER 22000 \
    SOLVER.CHECKPOINT_PERIOD 4000 \
    INPUT.MIN_SIZE_TRAIN "(480,512,544)" INPUT.MIN_SIZE_TEST 512 \
    INPUT.MAX_SIZE_TRAIN 768 INPUT.MAX_SIZE_TEST 768 \
    TEST.EVAL_PERIOD 4000 SEED 42 OUTPUT_DIR "$OUT" \
    > "$LOG" 2>&1
rc=$?
echo "===== END M65_${TAG} exit=$rc $(date -u)" >> "$MARK"
[ "$rc" -eq 0 ] && echo "M65_${TAG}_DONE exit=0 $(date -u)" >> "$MARK"
exit "$rc"
