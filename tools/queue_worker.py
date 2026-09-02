# -*- coding: utf-8 -*-
"""消融实验队列 worker。

从 tasks 目录按文件名顺序逐个取任务执行，支持运行过程中动态追加新任务
（便于在实验进行的同时继续开发新的改造方案并加入队列）。

任务文件格式（每行一个任务，字段以制表符分隔）::

    <tag>\\t<额外参数...>

例如::

    base_s42\\tMODEL.BASIS_MODULE.NAME ProtoNet\\tSEED 42
    c2f_s42\\tMODEL.VIG.CSP_STYLE c2f\\tSEED 42

公共参数（SEED 除外）取自 configs/run-wheat.yaml 与脚本内的 COMMON。

用法::
    conda activate gbadmask
    cd /home/huachenghao/codes/GBADMask
    export GBADMASK_DATA_ROOT=$PWD/datasets/HBueHxOW/wheat_seg_clean
    nohup python tools/queue_worker.py tasks/ > .queue.log 2>&1 &
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

COMMON = [
    "--config-file", "configs/run-wheat.yaml",
    "--num-gpus", "1",
    "DATALOADER.NUM_WORKERS", "8",
    "SOLVER.MAX_ITER", "4000",
    "SOLVER.STEPS", "2400,3200",
    "SOLVER.CHECKPOINT_PERIOD", "4000",
    "TEST.EVAL_PERIOD", "2000",
    # basis 模块默认用 ProtoNetV2+gc（阶段 A 的最优配置）
    "MODEL.BASIS_MODULE.NAME", "ProtoNetV2",
    "MODEL.BASIS_MODULE.ATTN", "gc",
]


def run_task(task_file, done_dir):
    with open(task_file, encoding="utf-8") as f:
        line = f.read().strip()
    if not line or line.startswith("#"):
        return None
    parts = [p for p in line.split("\t") if p.strip()]
    tag = parts[0]
    extra = []
    for p in parts[1:]:
        extra.extend(p.split())

    out = os.path.join("output", "ab-" + tag)
    cmd = [PY, "tools/train_bl+.py"] + COMMON + extra + ["OUTPUT_DIR", out]
    print("[{}] START {} : {}".format(
        time.strftime("%H:%M:%S"), tag, " ".join(extra)), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE)
    dt = (time.time() - t0) / 60.0
    print("[{}] DONE  {}  exit={}  {:.1f}min".format(
        time.strftime("%H:%M:%S"), tag, r.returncode, dt), flush=True)
    if r.returncode != 0:
        print("[{}] FAILED {}".format(time.strftime("%H:%M:%S"), tag), flush=True)
    os.rename(task_file, os.path.join(done_dir, os.path.basename(task_file)))
    return tag, r.returncode


def main():
    task_dir = sys.argv[1] if len(sys.argv) > 1 else "tasks"
    task_dir = os.path.abspath(task_dir)
    done_dir = os.path.join(task_dir, "done")
    os.makedirs(done_dir, exist_ok=True)

    print("queue dir:", task_dir, flush=True)
    idle = 0
    while True:
        files = sorted(
            f for f in os.listdir(task_dir)
            if f.endswith(".task") and os.path.isfile(os.path.join(task_dir, f))
        )
        if not files:
            idle += 1
            # 连续 40 分钟（每 30s 一轮，共 80 轮）无新任务则退出
            if idle > 80:
                print("[{}] 无待执行任务，worker 退出".format(
                    time.strftime("%H:%M:%S")), flush=True)
                break
            time.sleep(30)
            continue
        idle = 0
        run_task(os.path.join(task_dir, files[0]), done_dir)


if __name__ == "__main__":
    main()
