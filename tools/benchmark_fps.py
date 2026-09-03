# -*- coding: utf-8 -*-
"""推理速度基准 —— 对应 ROADMAP M6.3 效率验收。

验收线：≥ 10 FPS（RTX 3090，batch=1，512 输入）。

用法（GPU 空闲时运行）:
    python tools/benchmark_fps.py --config-file configs/run-vigv2.yaml \
        MODEL.WEIGHTS output/m62_L1b_P0/model_final.pth

    # 不加载权重（只测结构速度）：
    python tools/benchmark_fps.py --config-file configs/run-vigv2.yaml

说明:
    - 输入构造方式与 quick_check.py 一致（模型自带归一化，喂 0~255 的
      (C,H,W) float 张量即可）。
    - warmup N 次后计时 M 次（torch.cuda.synchronize 包裹），报告
      ms/iter 与 FPS、参数量、峰值显存。
"""
import argparse
import sys
import time

sys.path.insert(0, ".")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--size", type=int, default=512, help="输入边长（默认 512）")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("opts", nargs=argparse.REMAINDER, default=[],
                    help="配置覆盖 KEY VALUE ...")
    args = ap.parse_args()

    import torch
    from adet.config import get_cfg
    from detectron2.modeling import build_model

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    # 仅当命令行显式给 MODEL.WEIGHTS 时才加载权重（权重不影响速度，
    # 还可避免配置里 detectron2:// 分类权重触发联网下载）
    cfg.MODEL.WEIGHTS = ""
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    if cfg.MODEL.WEIGHTS:
        from detectron2.checkpoint import DetectionCheckpointer
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        print("已加载权重: {}".format(cfg.MODEL.WEIGHTS))

    device = torch.device(args.device)
    H = W = args.size
    inputs = [{
        "image": (torch.rand(3, H, W, device=device) * 255),
        "height": H, "width": W,
    }]

    with torch.no_grad():
        for _ in range(args.warmup):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    ms = dt / args.iters * 1000.0
    peak = (torch.cuda.max_memory_allocated() / 2**30
            if device.type == "cuda" else 0.0)
    print("输入: 1×3×{}×{} | iters={} (warmup={})".format(
        H, W, args.iters, args.warmup))
    print("参数量: {:.2f} M".format(n_params / 1e6))
    print("速度: {:.1f} ms/iter | FPS: {:.1f}".format(ms, 1000.0 / ms))
    if peak:
        print("峰值显存: {:.2f} GiB".format(peak))
    verdict = "✅ 达标" if 1000.0 / ms >= 10 else "❌ 未达标（<10 FPS）"
    print("验收（≥10 FPS）: {}".format(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
