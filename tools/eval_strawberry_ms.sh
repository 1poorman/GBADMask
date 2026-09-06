#!/usr/bin/env bash
# Strawberry 全日程两组 checkpoint 的多尺度 eval 试探（2026-09-05）
# 目的：判断 P0 在 small 目标上输 R1 ~4 AP 是否是【输入分辨率】造成。
#   P0_full 65.42 vs R1_full 63.69（+1.73，全靠 medium，small −3.98）。
#   若提高 eval 分辨率 P0 的 small 追平/反超 → 训练分辨率是主要杠杆。
# 复用 eval_wheat_test.sh 的 eval-only 模式（train_bl+.py --eval-only + 训练架构配置）。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
MARK=logs/m63c_evalsweep.log

run_eval() {  # $1=tag $2=config $3=weights $4=res ${@:5}=arch opts
  local tag="$1" cfg="$2" weights="$3" res="$4"; shift 4
  if grep -q "M63C_${tag}_${res}_DONE exit=0" "$MARK" 2>/dev/null; then
    echo "$tag@$res 已完成，跳过"; return 0
  fi
  echo "===== START $tag@$res $(date) =====" >> "$MARK"
  $PY tools/train_bl+.py --config-file "$cfg" --num-gpus 1 --eval-only \
      MODEL.WEIGHTS "$weights" "$@" \
      INPUT.MIN_SIZE_TEST "$res" INPUT.MAX_SIZE_TEST 800 \
      OUTPUT_DIR "output/m63c_evalsweep_${tag}_${res}" >> "$MARK" 2>&1
  local rc=$?
  echo "===== END $tag@$res exit=$rc $(date) =====" >> "$MARK"
  [ "$rc" -eq 0 ] && echo "M63C_${tag}_${res}_DONE exit=0 $(date)" >> "$MARK"
}

BASE_OPTS='DATASETS.NAME Strawberry'
R1_OPTS="$BASE_OPTS MODEL.BASIS_MODULE.NAME ProtoNet MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7"
P0_OPTS="$BASE_OPTS MODEL.VIG.VERSION m MODEL.VIG.PRETRAINED weights/MobileViG_V2_M_Class.pth \
  MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
  MODEL.BASIS_MODULE.NUM_CLASSES 7 MODEL.FCOS.NUM_CLASSES 7"

for res in 416 512 640; do
  run_eval R1 configs/run-wheat-r50.yaml output/m63b_strawberry_R1/model_final.pth "$res" $R1_OPTS
  run_eval P0 configs/run-vigv2.yaml      output/m63b_strawberry_P0/model_final.pth "$res" $P0_OPTS
done
echo "===== ALL M63C evalsweep DONE $(date) =====" >> "$MARK"
