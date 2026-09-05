#!/usr/bin/env bash
# 等 yolo26 诊断（plantv2 + strawberry 两连跑）真正结束 → 启动 M6.3 锚点队列。
# 判据：strawberry 的成功完成标记（串行队列的最后一组），内容校验非存在性。
set -u
cd /home/huachenghao/codes/GBADMask
while true; do
  if grep -q "YOLO26_strawberry_DONE exit=0" logs/yolo26_diag.log 2>/dev/null; then
    break
  fi
  # 若 yolo26 诊断进程死掉且无成功标记，也放行（GPU 空了，别互相死等）
  if ! pgrep -f "run_yolo26_diag" > /dev/null; then
    echo "$(date) yolo26 诊断进程已退出（无成功标记），直接启动锚点队列" >> logs/m63_anchors.log
    break
  fi
  sleep 300
done
if grep -q "M63_strawberry_P0_DONE exit=0" logs/m63_anchors.log 2>/dev/null; then
  echo "锚点队列已完成，跳过"
  exit 0
fi
echo "$(date) yolo26 诊断结束，启动 M6.3 锚点队列" >> logs/m63_anchors.log
bash tools/run_m63_anchors.sh
