#!/usr/bin/env bash
# AS1 接力 watcher：等 KD1b 真实成功标记（内容判据，非存在性）后，
# 先跑 AS1 GPU 冒烟（150 iter、TAL_WARMUP=50——确保跨过 warmup 真正
# 激活 TAL 路径），校验 [TAL] 统计与损失有限后才启动 wave2 全量。
# KD1b 进程死亡且无成功标记时放弃接棒（不写任何标记，人工介入）。
set -u
cd /home/huachenghao/codes/GBADMask
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
MARK=logs/m65.log
echo "[watcher] AS1 watcher 启动 $(date -u)"

while true; do
  if grep -q "M65_m65_kd1b_straw512_DONE exit=0" "$MARK" 2>/dev/null; then
    break
  fi
  if ! pgrep -f "output/m65_kd1b_straw512" > /dev/null; then
    if ! grep -q "M65_m65_kd1b_straw512_DONE exit=0" "$MARK" 2>/dev/null; then
      echo "[watcher] KD1b 进程消失且无成功标记，放弃接棒 $(date -u)"
      exit 1
    fi
  fi
  sleep 60
done
echo "[watcher] KD1b 完成，开始 AS1 冒烟 $(date -u)"

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
SMOKE_OUT=output/_smoke_as1
rm -rf "$SMOKE_OUT"
$PY tools/train_bl+.py --config-file configs/run-vigv2.yaml --num-gpus 1 \
    DATASETS.NAME Strawberry \
    MODEL.VIG.VERSION m MODEL.VIG.PRETRAINED weights/MobileViG_V2_M_Class.pth \
    MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
    MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
    MODEL.WEIGHTS "" \
    MODEL.FCOS.ASSIGN tal MODEL.FCOS.TAL_WARMUP 50 MODEL.FCOS.TAL_LOG_PERIOD 20 \
    SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 SOLVER.WARMUP_ITERS 200 \
    SOLVER.STEPS "(13100,17500)" SOLVER.MAX_ITER 150 \
    INPUT.MIN_SIZE_TRAIN "(480,512,544)" INPUT.MIN_SIZE_TEST 512 \
    INPUT.MAX_SIZE_TRAIN 768 INPUT.MAX_SIZE_TEST 768 \
    TEST.EVAL_PERIOD 0 SOLVER.CHECKPOINT_PERIOD 100000 \
    SEED 42 OUTPUT_DIR "$SMOKE_OUT" \
    > logs/m65_as1_smoke.log 2>&1
rc=$?
echo "[watcher] AS1 冒烟 exit=$rc $(date -u)"
if [ "$rc" -ne 0 ]; then
  echo "[watcher] 冒烟失败，不启动全量"; rm -rf "$SMOKE_OUT"; exit 1
fi
if ! grep -q "\[TAL\]" logs/m65_as1_smoke.log; then
  echo "[watcher] 冒烟日志无 [TAL] 统计（TAL 未激活？），不启动全量"
  rm -rf "$SMOKE_OUT"; exit 1
fi
if grep -qE "total_loss: (nan|inf)" logs/m65_as1_smoke.log; then
  echo "[watcher] 冒烟损失非有限，不启动全量"; rm -rf "$SMOKE_OUT"; exit 1
fi
rm -rf "$SMOKE_OUT"
echo "[watcher] 冒烟通过，启动 AS1 全量（wave2）$(date -u)"
bash tools/run_m65_wave2_as1.sh
