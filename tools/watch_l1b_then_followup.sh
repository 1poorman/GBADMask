#!/usr/bin/env bash
# 等 L1b 全部完成后自动启动 followup 队列（R1 锚点重建 + S1 从零对照）。
# 幂等：已完成（logs/m62_fu.log 有 ALL_DONE）则不重复跑。
set -u
cd /home/huachenghao/codes/GBADMask
while true; do
  if grep -q "M62_L1b_ALL_DONE" logs/m62_L1b.log 2>/dev/null; then
    break
  fi
  sleep 300
done
if grep -q "M62_FOLLOWUP_ALL_DONE" logs/m62_fu.log 2>/dev/null; then
  echo "followup 已完成，跳过"
  exit 0
fi
echo "$(date) L1b 完成，启动 followup" >> logs/m62_fu.log
bash tools/run_m62_followup.sh
