#!/usr/bin/env bash
# M6.3 前置：Plantv2 / Strawberry 锚点重建（2026-09-04 用户确认）
#
# 背景：wheat 数据诊断结论 = 数据集是共同瓶颈（yolo26s 100ep seg mAP50-95 仅 9.92
# 未收敛、长尾类全零；GBADMask 17.44 已是该平台 584 图规模的天花板区间）。
# 主战场迁移到 Plantv2（7916 train / 16 类 / 类均衡）与 Strawberry（1750 train /
# 7 类 / 2.25 实例每图）。wheat 全部消融数据保留，作为论文"小数据极限案例"。
#
# 队列（每数据集先 R1 基线后 P0 平台组，两者共用协议）：
#   Plantv2-R1  r50-protonet（官方基线，验收锚点）
#   Plantv2-P0  vigv2-m + ProtoNetV2 + gc（GBADMask 轻量平台）
#   Strawberry-R1 / Strawberry-P0 同上
#
# 协议统一（沿用 wheat 批次口径，保证跨数据集可比）：
#   batch 7 + BASE_LR 0.004375（线性缩放，GPU1 余量所限）
#   SEED 42 / EVAL 只取最终 iter / MAX_ITER 沿用各数据集既有配置
#   （Plantv2 100k iter ≈ 101 ep / Strawberry 22k iter ≈ 100 ep，都是
#   100 epoch 量级的完整日程，与 ROADMAP M6.3 验收矩阵一致）
#
# 显存预算：Plantv2 256×256 图 + batch 7 远低于 wheat 512 的占用；
# Strawberry 419×419 + batch 7 与 wheat 同档（wheat batch7 峰值 14.15GB）。
# ⚠️ GPU 现状：yolo26 诊断占 GPU1 约 2.9GB（还有 server.py 6.9GB）。
# 本队列与 yolo26 共卡：GBADMask 峰值 14GB + yolo26 2.9GB + server 6.9GB ≈ 23.8GB
# 超出 24GB 预算 → 本队列等待 yolo26 诊断完成后启动（watcher 编排）。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.85
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
BATCH_OPTS="SOLVER.IMS_PER_BATCH 7 SOLVER.BASE_LR 0.004375 SEED 42"

run() {  # $1=log标记 $2=config $3=OUTPUT_DIR $4..=额外 --opts（含 DATASETS.NUM_CLASSES）
  local tag="$1" cfg="$2" out="$3"; shift 3
  if grep -q "M63_${tag}_DONE exit=0" logs/m63_anchors.log 2>/dev/null; then
    echo "$tag 已完成，跳过"
    return 0
  fi
  echo "===== START $tag $(date) =====" >> logs/m63_anchors.log 2>&1
  $PY tools/train_bl+.py --config-file "$cfg" --num-gpus 1 \
      $BATCH_OPTS OUTPUT_DIR "$out" "$@" >> "logs/m63_${tag}.log" 2>&1
  local rc=$?
  echo "===== END $tag exit=$rc $(date) =====" >> logs/m63_anchors.log 2>&1
  if [ "$rc" -eq 0 ]; then
    echo "M63_${tag}_DONE exit=0 $(date)" >> logs/m63_anchors.log 2>&1
  fi
  return "$rc"
}

# --- Plantv2（R50 基线锚点 → vigv2-M 平台）---
run plantv2_R1 configs/run-wheat-r50.yaml output/m63_plantv2_R1 \
    DATASETS.NAME Plantv2 \
    MODEL.BASIS_MODULE.NAME ProtoNet \
    MODEL.BASIS_MODULE.NUM_CLASSES 16 MODEL.FCOS.NUM_CLASSES 16 \
    MODEL.WEIGHTS "detectron2://ImageNetPretrained/MSRA/R-50.pkl"

run plantv2_P0 configs/run-vigv2.yaml output/m63_plantv2_P0 \
    DATASETS.NAME Plantv2 \
    MODEL.VIG.VERSION m MODEL.VIG.PRETRAINED weights/MobileViG_V2_M_Class.pth \
    MODEL.BASIS_MODULE.NAME ProtoNetV2 \
    MODEL.BASIS_MODULE.NUM_CLASSES 16 MODEL.FCOS.NUM_CLASSES 16 \
    MODEL.WEIGHTS ""

# --- Strawberry ---
run strawberry_R1 configs/run-wheat-r50.yaml output/m63_strawberry_R1 \
    DATASETS.NAME Strawberry \
    MODEL.BASIS_MODULE.NAME ProtoNet \
    MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
    MODEL.WEIGHTS "detectron2://ImageNetPretrained/MSRA/R-50.pkl"

run strawberry_P0 configs/run-vigv2.yaml output/m63_strawberry_P0 \
    DATASETS.NAME Strawberry \
    MODEL.VIG.VERSION m MODEL.VIG.PRETRAINED weights/MobileViG_V2_M_Class.pth \
    MODEL.BASIS_MODULE.NAME ProtoNetV2 \
    MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7 \
    MODEL.WEIGHTS ""

# --- wheat_seg_strat 重划对照（2026-09-04 用户确认，原 wheat_seg_clean 全部保留）---
# 动机：旧 val 未分层（WCN 1图3实例测出 0 分）且 test 已被比较污染；
# 新划分 818 图全合并按类分层 85/15（train 695/1827，val 123/294，每类 13-39 实例）。
# 预期：数字"变诚实"（~13-14）而非变高；+5 线按新锚点 R1' 同步重建。
# 日程沿用 8000 iter（与 wheat_seg_clean 14 组历史同口径，协议不变仅数据变）。
run strat_R1 configs/run-wheat-r50.yaml output/m63_strat_R1 \
    DATASETS.NAME wheat_seg_strat \
    MODEL.BASIS_MODULE.NAME ProtoNet \
    MODEL.BASIS_MODULE.NUM_CLASSES 12 MODEL.FCOS.NUM_CLASSES 12 \
    MODEL.WEIGHTS "detectron2://ImageNetPretrained/MSRA/R-50.pkl"

run strat_P0 configs/run-vigv2.yaml output/m63_strat_P0 \
    DATASETS.NAME wheat_seg_strat \
    MODEL.VIG.VERSION m MODEL.VIG.PRETRAINED weights/MobileViG_V2_M_Class.pth \
    MODEL.BASIS_MODULE.NAME ProtoNetV2 \
    MODEL.BASIS_MODULE.NUM_CLASSES 12 MODEL.FCOS.NUM_CLASSES 12 \
    MODEL.WEIGHTS ""

run strat_B2 configs/run-vigv2.yaml output/m63_strat_B2 \
    DATASETS.NAME wheat_seg_strat \
    MODEL.VIG.VERSION m MODEL.VIG.PRETRAINED weights/MobileViG_V2_M_Class.pth \
    MODEL.BASIS_MODULE.NAME ProtoNetV2 \
    MODEL.BASIS_MODULE.ATTN ema \
    MODEL.BASIS_MODULE.NUM_CLASSES 12 MODEL.FCOS.NUM_CLASSES 12 \
    MODEL.WEIGHTS ""
