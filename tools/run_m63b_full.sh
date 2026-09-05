#!/usr/bin/env bash
# M6.3b：Strawberry 全日程复跑 + MobileViGv2-B 容量上限（2026-09-05 用户确认）
#
# 背景：M6.3 锚点队列的 Plantv2/Strawberry 四组误用 wheat 协议（MAX_ITER=8000、
# 512 输入），非验收口径。本队列：
#   1. strawberry_R1_full —— r50-protonet 官方基线，run-strawberry.yaml 全日程
#      （MAX_ITER 22000 ≈ 100ep，batch 8 / LR 0.005 / STEPS(13100,17500)，
#       输入 384-448，EVAL 4000）
#   2. strawberry_P0_full —— vigv2m 平台同协议全日程（判定 8000iter 的 −2.33
#      是日程伪影还是平台真落后；决定论文主张能否扩到更大数据集）
#   3. strat_Bcap —— MobileViGv2-B 容量上限（wheat_seg_strat，batch 7 同 P0
#      协议，探能否逼近 +5 线 18.87）
# Plantv2 已判顶格饱和（7ep 即 ~99），不再复跑。
#
# 显存：GPU1 有 server.py 6.9GB 常驻，可用 ~16.3GB。
#   strawberry batch8 vigv2 ≈13GB（batch7 实测 11.3GB）✓；
#   B batch7 512 输入接近 vigv2m 的 14.15GB 上限，风险中等——若 OOM 见日志后
#   改 batch 6 + LR 0.00375 重跑（记录一次即可，不得静默降）。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.85
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
MARK=logs/m63b.log

run() {  # $1=tag $2=config $3=OUTPUT_DIR ${@:4}=opts
  local tag="$1" cfg="$2" out="$3"; shift 3
  if grep -q "M63B_${tag}_DONE exit=0" "$MARK" 2>/dev/null; then
    echo "$tag 已完成，跳过"; return 0
  fi
  echo "===== START $tag $(date) =====" >> "$MARK"
  $PY tools/train_bl+.py --config-file "$cfg" --num-gpus 1 \
      OUTPUT_DIR "$out" "$@" >> "logs/m63b_${tag}.log" 2>&1
  local rc=$?
  echo "===== END $tag exit=$rc $(date) =====" >> "$MARK"
  [ "$rc" -eq 0 ] && echo "M63B_${tag}_DONE exit=0 $(date)" >> "$MARK"
  return "$rc"
}

# ---------- 1/3 Strawberry R50 基线 全日程 ----------
run strawberry_R1_full configs/run-wheat-r50.yaml output/m63b_strawberry_R1 \
    DATASETS.NAME Strawberry \
    MODEL.BASIS_MODULE.NAME ProtoNet \
    MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
    MODEL.WEIGHTS "detectron2://ImageNetPretrained/MSRA/R-50.pkl" \
    SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 SOLVER.WARMUP_ITERS 200 \
    SOLVER.STEPS "(13100,17500)" SOLVER.MAX_ITER 22000 SOLVER.CHECKPOINT_PERIOD 4000 \
    INPUT.MIN_SIZE_TRAIN "(384,416,448)" INPUT.MIN_SIZE_TEST 416 \
    INPUT.MAX_SIZE_TRAIN 512 INPUT.MAX_SIZE_TEST 512 \
    TEST.EVAL_PERIOD 4000 SEED 42

# ---------- 2/3 Strawberry vigv2m 平台 全日程 ----------
run strawberry_P0_full configs/run-vigv2.yaml output/m63b_strawberry_P0 \
    DATASETS.NAME Strawberry \
    MODEL.VIG.VERSION m MODEL.VIG.PRETRAINED weights/MobileViG_V2_M_Class.pth \
    MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
    MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
    MODEL.WEIGHTS "" \
    SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 SOLVER.WARMUP_ITERS 200 \
    SOLVER.STEPS "(13100,17500)" SOLVER.MAX_ITER 22000 SOLVER.CHECKPOINT_PERIOD 4000 \
    INPUT.MIN_SIZE_TRAIN "(384,416,448)" INPUT.MIN_SIZE_TEST 416 \
    INPUT.MAX_SIZE_TRAIN 512 INPUT.MAX_SIZE_TEST 512 \
    TEST.EVAL_PERIOD 4000 SEED 42

# ---------- 3/3 wheat_seg_strat MobileViGv2-B 容量上限（batch7 同 strat_P0）----------
run strat_Bcap configs/run-vigv2.yaml output/m63b_strat_Bcap \
    DATASETS.NAME wheat_seg_strat \
    MODEL.VIG.VERSION b MODEL.VIG.PRETRAINED weights/MobileViG_V2_B_Class.pth \
    MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
    MODEL.BASIS_MODULE.NUM_CLASSES 12 MODEL.FCOS.NUM_CLASSES 12 \
    MODEL.WEIGHTS "" \
    SOLVER.IMS_PER_BATCH 7 SOLVER.BASE_LR 0.004375 \
    SEED 42

echo "===== ALL M6.3b DONE $(date) =====" >> "$MARK"
