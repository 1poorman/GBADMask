#!/usr/bin/env bash
# M6.6 接力 watcher：等 AS1（wave2）真实成功标记后，串行跑
# Strawberry 骨干消融队列 → wheat 骨干消融队列。
# AS1 进程死亡且无成功标记 → 放弃接棒（人工介入，不写任何标记）。
set -u
cd /home/huachenghao/codes/GBADMask
MARK=logs/m65.log
M66=logs/m66.log
echo "[watcher] M66 backbone watcher 启动 $(date -u)"

while true; do
  if grep -q "M65_m65_as1_tal_straw512_DONE exit=0" "$MARK" 2>/dev/null; then
    break
  fi
  if ! pgrep -f "output/m65_as1_tal_straw512" > /dev/null; then
    if ! grep -q "M65_m65_as1_tal_straw512_DONE exit=0" "$MARK" 2>/dev/null; then
      echo "[watcher] AS1 进程消失且无成功标记，放弃接棒 $(date -u)"
      exit 1
    fi
  fi
  sleep 60
done
echo "[watcher] AS1 完成，启动 M66 Strawberry 骨干队列 $(date -u)"
bash tools/run_m66_backbone_straw.sh >> logs/m66_straw_launcher.log 2>&1
echo "[watcher] M66 straw 队列结束，启动 wheat 骨干队列 $(date -u)"
bash tools/run_m66_backbone_wheat.sh >> logs/m66_wheat_launcher.log 2>&1
echo "[watcher] M66 全部结束 $(date -u)"
