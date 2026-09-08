#!/usr/bin/env bash
# M6.5 wave1f：KD1 检测+mask 蒸馏首轮。
# teacher = output/m63d_straw_B（B 变体，416 协议训练，segm 66.35），
# student = M 平台 @512 协议（与 P0_res 完全一致的架构与日程）。
# 蒸馏：cls sigmoid MSE + reg L1 + bases 前景加权 MSE（B 冻结）。
# 单变量 vs P0_res=66.06（s42）；teacher 前向吃 student 同款 512 输入
# （backbone 全卷积，尺寸鲁棒；避免双份 ImageList/分辨率分裂）。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
TAG=m65_kd1_straw512
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
    MODEL.DISTILL.WEIGHTS output/m63d_straw_B/model_final.pth \
    "MODEL.DISTILL.TEACHER_OPTS" "MODEL.VIG.VERSION=b MODEL.VIG.PRETRAINED=weights/MobileViG_V2_B_Class.pth MODEL.FCOS.NUM_CLASSES=7 MODEL.BASIS_MODULE.NUM_CLASSES=7" \
    MODEL.DISTILL.W_CLS 1.0 MODEL.DISTILL.W_REG 0.25 \
    MODEL.DISTILL.W_BASES 1.0 MODEL.DISTILL.FG_THRESH 0.1 \
    SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 SOLVER.WARMUP_ITERS 200 \
    SOLVER.STEPS "(13100,17500)" SOLVER.MAX_ITER 22000 \
    SOLVER.CHECKPOINT_PERIOD 4000 \
    INPUT.MIN_SIZE_TRAIN "(480,512,544)" INPUT.MIN_SIZE_TEST 512 \
    INPUT.MAX_SIZE_TRAIN 768 INPUT.MAX_SIZE_TEST 768 \
    TEST.EVAL_PERIOD 4000 SEED 42 OUTPUT_DIR "$OUT" \
    > "$LOG" 2>&1
rc=$?
echo "===== END M65_${TAG} exit=$rc $(date -u) =====" >> "$MARK"
[ "$rc" -eq 0 ] && echo "M65_${TAG}_DONE exit=0 $(date -u)" >> "$MARK"
exit "$rc"
