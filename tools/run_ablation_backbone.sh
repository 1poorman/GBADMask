# 小麦病害实例分割：骨干维度消融（阶段 C）
#
# 数据集：datasets/HBueHxOW/wheat_seg_clean（584 train / 82 val，12 类）
# 固定 SEED=42，与阶段 A/B 完全一致的超参，唯一变量为骨干。
#
# basis 模块统一用阶段 A 的最优配置：ProtoNetV2 + ATTN=gc（segm AP 6.797），
# 因此本阶段结果可直接与 6.797 对比。
#
# 三组：
#   vig     : MobileViG（cspvig）              -> 即阶段 A 的 gc 组，作为基线参照
#   lskoff  : Lcspvig + USE_LSK=False          -> 骨干架构差异（此时 Lcspvig ≡ cspvig）
#   ls Kon  : Lcspvig + USE_LSK=True           -> LSK 大核选择注意力的净贡献
#
# 对比关系：
#   lskon - lskoff  = LSK 本身的贡献
#   lskon - vig     = 换成 Lcspvig 骨干的总收益
#
# 每组 4000 iter，单卡 3090 约 30 分钟，共约 1.5 小时。
#
# 用法:
#   bash tools/run_ablation_backbone.sh
set -u

cd /home/huachenghao/codes/GBADMask
export GBADMASK_DATA_ROOT=/home/huachenghao/codes/GBADMask/datasets/HBueHxOW/wheat_seg_clean
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1

PY=/home/huachenghao/.conda/envs/gbadmask/bin/python
CFG=configs/run-wheat.yaml
# basis 模块固定为阶段 A 最优配置
BASIS="MODEL.BASIS_MODULE.NAME ProtoNetV2 MODEL.BASIS_MODULE.ATTN gc"

run () {
  local tag=$1; shift
  echo "=== [$(date +%H:%M:%S)] START $tag : $* ==="
  $PY tools/train_bl+.py --config-file $CFG --num-gpus 1 \
      SEED 42 DATALOADER.NUM_WORKERS 8 \
      SOLVER.MAX_ITER 4000 SOLVER.STEPS 2400,3200 \
      SOLVER.CHECKPOINT_PERIOD 4000 TEST.EVAL_PERIOD 2000 \
      OUTPUT_DIR output/ab-$tag $BASIS "$@"
  echo "=== [$(date +%H:%M:%S)] DONE $tag ==="
}

# vig 组即阶段 A 的 gc 组，若已存在权重可跳过（此处重跑以确保同批次可比）
run vig    MODEL.BACKBONE.NAME build_fcos_cspvig_bifpn_backbone
run lskoff MODEL.BACKBONE.NAME build_fcos_Lcspvig_bifpn_backbone MODEL.VIG.USE_LSK False
run lskon  MODEL.BACKBONE.NAME build_fcos_Lcspvig_bifpn_backbone MODEL.VIG.USE_LSK True

echo "ALLDONE [$(date +%H:%M:%S)]"
