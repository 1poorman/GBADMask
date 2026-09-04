#!/usr/bin/env bash
# 编排：等 B1''（收尾队列最后一组）真正结束 → 启动 D 系列日程消融。
#
# ⚠️ 幂等判据用**内容**而非标记存在性（吸取 2026-09-03 "毒标记"事故教训：
# 旧 watcher 只查 ALL_DONE 是否存在，导致失败运行也被当成完成）。
set -u
cd /home/huachenghao/codes/GBADMask
while true; do
  # 收尾队列写 M62_FINAL_QUEUE_DONE，且 B1 必须有 END 记录才算真结束
  if grep -q "M62_FINAL_QUEUE_DONE" logs/m62_fu.log 2>/dev/null \
     && grep -q "END B1 exit" logs/m62_L1c_B1.log 2>/dev/null; then
    break
  fi
  sleep 300
done
if grep -q "M62_D_ALL_DONE" logs/m62_D.log 2>/dev/null; then
  echo "D 系列已完成，跳过"
  exit 0
fi
echo "$(date) 收尾队列结束，启动 D 系列日程消融" >> logs/m62_D.log
bash tools/run_m62_schedule.sh
