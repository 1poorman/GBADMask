#!/usr/bin/env bash
# YOLO26-seg 外部基准：验证 wheat_seg_clean 数据集本身的可学习性（2026-09-04）
#
# ⚠️ GPU 选择修复（10:54 OOM 事故根因）：ultralytics select_device() 会**覆写**
# CUDA_VISIBLE_DEVICES（torch_utils.py:220-221：os.environ["CUDA_VISIBLE_DEVICES"]=device）。
# 传 device=0 → CVD 被改写成 "0" → cuda:0 指向物理 GPU0（被 sglang/start.py 占满 21GB）→ OOM。
# 正确做法：train(device=1) → CVD 覆写为 "1" → cuda:0 = 物理 GPU1。
# 本脚本不再 export CUDA_VISIBLE_DEVICES（反正会被覆写；覆写值即目标卡）。
#
# 判读（数据诊断口径，非论文基准）：
#   - yolo26 AP 明显高（>20）→ 数据没问题，GBADMask 平台是瓶颈
#   - yolo26 AP 相近（15~18）→ 数据集规模/标注是共同上限，wheat 上的 +5 目标
#     可能不现实，按用户指示转向 Plantv2 / Strawberry
#   - yolo26 AP 更低 → YOLO 短训练或不适配，不能下结论（只作参考）
set -u
cd /home/huachenghao/codes/GBADMask
PY=/home/huachenghao/.conda/envs/yolo26/bin/python

echo "===== START yolo26s-seg wheat $(date) =====" >> logs/yolo26_wheat.log 2>&1
$PY - <<'EOF' >> logs/yolo26_wheat.log 2>&1
import sys
sys.path.insert(0, '/home/huachenghao/codes/ultralytics')
from ultralytics import YOLO

# s 规格模型 + x 的 COCO 预训练 backbone 迁移
model = YOLO('yolo26s-seg.yaml')
model.load('/home/huachenghao/codes/ultralytics/yolo26x-seg.pt')

results = model.train(
    data='datasets/yolo_wheat_seg_clean/data.yaml',
    epochs=100,          # ≈ 8000 iter × 7 / 584 ≈ 96 ep，取整
    imgsz=512,
    batch=7,             # 与 GBADMask 协议一致
    device=1,            # 物理 GPU1（select_device 会把 CVD 覆写为 "1"）
    seed=42,
    workers=4,
    project='/home/huachenghao/codes/GBADMask/output/yolo26_wheat',
    name='s_pre_100ep',
    patience=0,          # 不早停（对齐 GBADMask 只取最终 iter 的口径）
    val=True,
    verbose=True,
)
# 最终评估：seg mAP50 / mAP50-95（COCO 口径近似对照 BlendMask 的 AP50/AP）
metrics = model.val(data='datasets/yolo_wheat_seg_clean/data.yaml', imgsz=512, batch=7, device=1)
print('FINAL box mAP50=%.4f mAP50-95=%.4f' % (metrics.box.map50, metrics.box.map))
print('FINAL seg mAP50=%.4f mAP50-95=%.4f' % (metrics.seg.map50, metrics.seg.map))
EOF
RC=$?
echo "===== END yolo26 exit=$RC $(date) =====" >> logs/yolo26_wheat.log 2>&1
# 幂等标记只在成功时写（失败写 exit=0 = 毒标记，会骗过编排器跳过重跑）
if [ "$RC" -eq 0 ]; then
  echo "YOLO26_WHEAT_DONE exit=0 $(date)" >> logs/yolo26_wheat.log 2>&1
fi

