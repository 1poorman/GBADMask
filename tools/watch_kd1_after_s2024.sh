#!/usr/bin/env bash
# 等待 P0_res-s2024 真实完成标记（内容校验，防毒标记），然后启动 KD1。
set -u
cd /home/huachenghao/codes/GBADMask
MARK=logs/m65.log

while true; do
  if grep -q "M65_m65_p0res_s2024_DONE exit=0" "$MARK" 2>/dev/null; then
    echo "[watcher] P0_res-s2024 完成，启动 KD1 $(date -u)"
    bash tools/run_m65_wave1f_kd1.sh
    exit $?
  fi
  # 兜底：进程死亡且无成功标记（含失败情形）也不挂死
  if ! pgrep -f 'm65_p0res_s2024' > /dev/null 2>&1; then
    if grep -q "END M65_m65_p0res_s2024 exit=" "$MARK" 2>/dev/null; then
      echo "[watcher] P0_res-s2024 已结束但无成功标记，不启动 KD1 $(date -u)"
      exit 1
    fi
  fi
  sleep 60
done
