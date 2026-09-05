#!/usr/bin/env bash
# 等待 D3（cosine 日程组）真正结束 → 启动 yolo26-seg wheat 外部基准。
# 幂等判据用内容：END D3 exit=0（吸取毒标记教训，不查标记存在性）。
set -u
cd /home/huachenghao/codes/GBADMask
while true; do
  if grep -q "END D3 exit=0" logs/m62_D_D3.log 2>/dev/null; then
    break
  fi
  sleep 300
done
if grep -q "YOLO26_WHEAT_DONE exit=0" logs/yolo26_wheat.log 2>/dev/null; then
  echo "yolo26 wheat 已完成，跳过"
  exit 0
fi
bash tools/run_yolo26_wheat.sh
