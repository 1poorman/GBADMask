#!/usr/bin/env bash
# M6.5 wave1f-b：KD1b 蒸馏权重标定修正重跑。
# 背景：KD1（W_BASES=1.0）的 loss_kd_bases ~13.7 占总损失 93%，
# 蒸馏项梯度主导训练，属标定缺陷而非 KD 假设的公平检验。
# 唯一改动：W_BASES 1.0 -> 0.02（13.7×0.02≈0.27，与任务损失
# 0.07~0.6 同量级）。其余与 wave1f 完全一致（teacher=B@416 ckpt，
# student=M@512 协议，cls sigmoid MSE / reg L1 / bases 前景加权 MSE）。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
TAG=m65_kd1b_straw512
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
    MODEL.DISTILL.W_BASES 0.02 MODEL.DISTILL.FG_THRESH 0.1 \
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
