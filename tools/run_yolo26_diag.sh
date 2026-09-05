#!/usr/bin/env bash
# YOLO26-seg 数据诊断：Plantv2 + Strawberry（2026-09-04）
#
# 背景：wheat 诊断结论 = 数据集是共同瓶颈（yolo26s seg mAP50-95 仅 9.92，
# 100 ep 未收敛，长尾类全零）。转向两个更大更均衡的数据集验证数据健康度。
#
# ⚠️ GPU：ultralytics select_device() 会覆写 CUDA_VISIBLE_DEVICES，
#   必须用 train(device=1)（覆写值="1" → cuda:0 = 物理 GPU1）。
#
# 训练量：按 ultralytics 惯例 100 ep（Plantv2 7916 图是主战场；Strawberry
#   1750 图参照）。imgsz 512 对齐 GBADMask 协议。幂等标记仅成功时写。
set -u
cd /home/huachenghao/codes/GBADMask
PY=/home/huachenghao/.conda/envs/yolo26/bin/python

run_one() {  # $1=dataset_name $2=data.yaml $3=epochs
  local name="$1" data="$2" epochs="$3"
  if grep -q "YOLO26_${name}_DONE exit=0" logs/yolo26_diag.log 2>/dev/null; then
    echo "$name 已完成，跳过"
    return 0
  fi
  echo "===== START yolo26s-seg $name $(date) =====" >> logs/yolo26_diag.log 2>&1
  $PY - "$name" "$data" "$epochs" <<'EOF' >> logs/yolo26_diag.log 2>&1
import sys
sys.path.insert(0, '/home/huachenghao/codes/ultralytics')
from ultralytics import YOLO

name, data, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
model = YOLO('yolo26s-seg.yaml')
model.load('/home/huachenghao/codes/ultralytics/yolo26x-seg.pt')

model.train(
    data=data, epochs=epochs, imgsz=512, batch=7, device=1, seed=42,
    workers=4, project='/home/huachenghao/codes/GBADMask/output/yolo26_diag',
    name=f's_{name}', patience=0, val=True, verbose=True,
)
metrics = model.val(data=data, imgsz=512, batch=7, device=1)
print(f'FINAL-{name} box mAP50=%.4f mAP50-95=%.4f' % (metrics.box.map50, metrics.box.map))
print(f'FINAL-{name} seg mAP50=%.4f mAP50-95=%.4f' % (metrics.seg.map50, metrics.seg.map))
EOF
  local rc=$?
  echo "===== END yolo26-$name exit=$rc $(date) =====" >> logs/yolo26_diag.log 2>&1
  if [ "$rc" -eq 0 ]; then
    echo "YOLO26_${name}_DONE exit=0 $(date)" >> logs/yolo26_diag.log 2>&1
  fi
  return "$rc"
}

# Plantv2 优先（主战场，7900×100ep 在 3090 上约 6~8 h）
run_one plantv2 datasets/yolo_plantv2/data.yaml 100
# Strawberry 次之（1750×100ep 约 1.5~2 h）
run_one strawberry datasets/yolo_strawberry/data.yaml 100
