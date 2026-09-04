#!/usr/bin/env bash
# 编排器：等 L1c 主体完成 → 跑 B1 补跑（window-PSA checkpoint 版）→ 放行 followup。
# 取代原 watch_l1b_then_followup.sh 的职责（原脚本已随本轮重启被杀）。
set -u
cd /home/huachenghao/codes/GBADMask
# 1) 等 L1c ALL_DONE（B1 首跑失败也会写 END，主体由 run 函数顺序推进）
while true; do
  if grep -q "M62_L1c_ALL_DONE" logs/m62_L1c.log 2>/dev/null; then
    break
  fi
  sleep 300
done
# 2) B1 重跑（若上次 exit=0 则跳过）
if ! grep -q "END B1 exit=0" logs/m62_L1c_B1.log 2>/dev/null; then
  bash tools/run_m62_b1_retry.sh
else
  echo "$(date) B1 已成功，跳过重跑" >> logs/m62_fu.log
fi
# 3) followup（R1 锚点重建 + S1 从零对照）
if grep -q "M62_FOLLOWUP_ALL_DONE" logs/m62_fu.log 2>/dev/null; then
  echo "followup 已完成，跳过"
  exit 0
fi
echo "$(date) L1c+B1 完成，启动 followup" >> logs/m62_fu.log
bash tools/run_m62_followup.sh
