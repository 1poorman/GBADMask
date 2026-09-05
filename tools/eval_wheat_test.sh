#!/usr/bin/env bash
# wheat test2017 评估 v2（2026-09-04，修正两个 bug）
#
# bug1：--dataset wheat_seg_clean 会把 DATASETS.TEST 覆写回 val
#       （train_bl+.py setup() 中 apply_dataset 在 merge_from_list 之后执行）
#       → 去掉 --dataset，直接用 opts 写 DATASETS.TRAIN/TEST（注册器按名反推注册）
# bug2：每组必须用**训练时的架构配置**评估——
#       B2 是 ATTN=ema 训练的（用默认 gc 配置评估 = 架构错配，数字无效）
#       R1 是 r50+FPN+ProtoNet（用 vigv2 配置 = 权重全部错配 → 0 AP）
# 验证：D3/P0 用正确配置在 val 上复现 17.443/16.928（与训练时一致），管线可信。
set -u
cd /home/huachenghao/codes/GBADMask
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/huachenghao/.conda/envs/gbadmask/bin/python

# bug3：run-vigv2.yaml 的 DATASETS.NAME=wheat_seg_clean 优先级高于 opts 的
#       TRAIN/TEST（apply_dataset 顺序：--dataset > DATASETS.NAME > TRAIN/TEST），
#       会把 TEST 覆盖回 val → 必须 DATASETS.NAME "" 一并传入
TEST_OPTS='DATASETS.NAME "" DATASETS.TRAIN ("wheat_seg_clean_train",) DATASETS.TEST ("wheat_seg_clean_test",)'

run_eval() {  # $1=tag $2=config $3=weights $4=arch opts
  local tag="$1" cfg="$2" weights="$3"; shift 3
  if grep -q "TEST2_${tag}_DONE exit=0" logs/m63_wheat_test.log 2>/dev/null; then
    return 0
  fi
  echo "===== START-v2 $tag $(date) =====" >> logs/m63_wheat_test.log 2>&1
  $PY tools/train_bl+.py --config-file "$cfg" --num-gpus 1 \
      --eval-only \
      MODEL.WEIGHTS "$weights" $TEST_OPTS "$@" \
      OUTPUT_DIR "output/m63_test_$tag" >> logs/m63_wheat_test.log 2>&1
  local rc=$?
  echo "===== END-v2 $tag exit=$rc $(date) =====" >> logs/m63_wheat_test.log 2>&1
  if [ "$rc" -eq 0 ]; then
    echo "TEST2_${tag}_DONE exit=0 $(date)" >> logs/m63_wheat_test.log 2>&1
  fi
}

# D3/P0：run-vigv2.yaml 默认架构即训练架构（gc attention）
run_eval D3 configs/run-vigv2.yaml output/m62_D_D3/model_final.pth
run_eval P0 configs/run-vigv2.yaml output/m62_L1b_P0/model_final.pth
# B2：ema attention
run_eval B2 configs/run-vigv2.yaml output/m62_L1c_B2/model_final.pth \
    MODEL.BASIS_MODULE.ATTN ema
# R1：r50 + FPN + 官方 ProtoNet
run_eval R1 configs/run-wheat-r50.yaml output/m62_fu_R1_r50p7/model_final.pth \
    MODEL.BASIS_MODULE.NAME ProtoNet
