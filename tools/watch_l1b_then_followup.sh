#!/usr/bin/env bash
# 等 L1b+L1c 全部完成后自动启动 followup 队列（R1 锚点重建 + S1 从零对照）。
# 幂等：已完成（logs/m62_fu.log 有 ALL_DONE）则不重复跑。
set -u
cd /home/huachenghao/codes/GBADMask
while true; do
  done_b=$(grep -c "M62_L1b_ALL_DONE" logs/m62_L1b.log 2>/dev/null || echo 0)
  done_c=$(grep -c "M62_L1c_ALL_DONE" logs/m62_L1c.log 2>/dev/null || echo 0)
  if [ "${done_b:-0}" -ge 1 ] && [ "${done_c:-0}" -ge 1 ]; then
    break
  fi
  sleep 300
done
if grep -q "M62_FOLLOWUP_ALL_DONE" logs/m62_fu.log 2>/dev/null; then
  echo "followup 已完成，跳过"
  exit 0
fi
echo "$(date) L1b+L1c 完成，启动 followup" >> logs/m62_fu.log
bash tools/run_m62_followup.sh
