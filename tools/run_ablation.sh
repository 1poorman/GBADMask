# 小麦病害实例分割：消融实验脚本
#
# 数据集：datasets/HBueHxOW/wheat_seg_clean（584 train / 82 val / 152 test，12 类）
# 固定 SEED=42，所有组共用 configs/run-wheat.yaml 的超参。
#
# 实验分两阶段：
#   阶段 1（A 组）：当前 ProtoNetV2 各组件的贡献
#   阶段 2（B 组）：新引入的三项改进的贡献（FDC / 坐标 / 空间注意力）
#
# 每组 4000 iter（约 55 epoch），单卡 3090 约 20 分钟。
# 结果用 tests/tmp/summarize_ablation.py 汇总。
#
# 用法:
#   bash tools/run_ablation.sh            # 跑全部
#   bash tools/run_ablation.sh a          # 只跑阶段 1
#   bash tools/run_ablation.sh b          # 只跑阶段 2
set -u

cd /home/huachenghao/codes/GBADMask
export GBADMASK_DATA_ROOT=/home/huachenghao/codes/GBADMask/datasets/HBueHxOW/wheat_seg_clean
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1

PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
CFG=configs/run-wheat.yaml
LOG=${LOG:-.ablation.log}

run () {
  local tag=$1; shift
  echo "=== [$(date +%H:%M:%S)] START $tag : $* ==="
  $PY tools/train_bl+.py --config-file $CFG --num-gpus 1 \
      SEED 42 DATALOADER.NUM_WORKERS 8 \
      SOLVER.MAX_ITER 4000 SOLVER.STEPS 2400,3200 \
      SOLVER.CHECKPOINT_PERIOD 4000 TEST.EVAL_PERIOD 2000 \
      OUTPUT_DIR output/ab-$tag "$@"
  echo "=== [$(date +%H:%M:%S)] DONE $tag ==="
}

stage=${1:-all}

if [ "$stage" = "a" ] || [ "$stage" = "all" ]; then
  # ---- 阶段 1：ProtoNetV2 各组件 ----
  # base  : 官方 ProtoNet
  # nogc  : ProtoNetV2 去掉注意力       -> 测「低层分支 + conv2」的贡献
  # gc    : ProtoNetV2 + ATTN=gc        -> 与 nogc 对比得 GC 净贡献
  # cbam  : ProtoNetV2 + ATTN=cbam      -> 通道+空间  vs  纯通道
  run base  MODEL.BASIS_MODULE.NAME ProtoNet
  run nogc  MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN none
  run gc    MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc
  run cbam  MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN cbam
fi

if [ "$stage" = "b" ] || [ "$stage" = "all" ]; then
  # ---- 阶段 2：新引入的三项改进 ----
  # 均以 ProtoNetV2 + ATTN=gc 为起点（即上面的 gc 组），逐项叠加：
  # fdc     : + 开启语义损失(FDC)        -> 测改动 D
  # spatial : 注意力换成纯空间           -> 测「空间 vs 通道」这一理论判断
  # coord   : + 坐标编码                 -> 测位置编码
  # full    : 三项全开                   -> 看是否有协同增益
  run fdc     MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
              MODEL.BASIS_MODULE.LOSS_ON True
  run spatial MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN spatial
  run coord   MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc \
              MODEL.BASIS_MODULE.COORD_ON True
  run full    MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN spatial \
              MODEL.BASIS_MODULE.COORD_ON True MODEL.BASIS_MODULE.LOSS_ON True
fi

echo "ALLDONE [$(date +%H:%M:%S)]"
