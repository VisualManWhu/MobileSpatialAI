"""Offline evaluation of a trained scene: per-view PSNR and a floater score.

Usage (Ubuntu, from server/):
  uv run python scripts/eval_scene.py <job_dir> [--profile laptop]
      evaluates <job_dir>/scene.ply (or <job_dir>/retrain-<tag>/scene.ply with --tag)
  uv run python scripts/eval_scene.py <job_dir> --train [--tag NAME] [--pose-lr X] [--depth-lambda Y]
      retrains capture.zip into <job_dir>/retrain-<tag>/ (default tag "new") with the job
      pipeline, then evaluates the new scene.ply

Metrics (also written to eval.json next to the evaluated ply):
  - PSNR of every training view rendered at its capture pose: median, min, worst 5 indices
    (0-based position in frames.jsonl)
  - floater score: among Gaussians with sigmoid(opacity) > 0.5, the fraction whose nearest
    confidence-filtered initial point (init_points, conf >= 0.5) is farther than 0.10 m
  - Gaussian count; training wall time when --train
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recon3d.config import PROFILES, Profile  # noqa: E402

FLOATER_OPACITY = 0.5
FLOATER_DIST_M = 0.10
WORST_K = 5


def _train(job_dir: Path, out_dir: Path, profile: Profile) -> float:
    """Run the job pipeline on job_dir/capture.zip inside out_dir; returns wall seconds."""
    from recon3d.jobs.pipeline import run_pipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_dst = out_dir / "capture.zip"
    if not zip_dst.exists():
        try:
            os.link(job_dir / "capture.zip", zip_dst)
        except OSError:
            shutil.copyfile(job_dir / "capture.zip", zip_dst)

    last = [-1]

    def on_progress(p: float) -> None:
        pct = int(p * 100)
        if pct // 10 != last[0] // 10:
            last[0] = pct
            print(f"训练进度 {pct}%（{time.monotonic() - t0:.0f} s）", flush=True)

    t0 = time.monotonic()
    run_pipeline(out_dir, profile, on_progress)
    return time.monotonic() - t0


def evaluate(job_dir: Path, ply: Path, profile: Profile) -> dict:
    import torch
    from scipy.spatial import cKDTree

    from recon3d.capture.format import load_capture
    from recon3d.recon.dataset import TrainingSet, build_training_set
    from recon3d.recon.export_splat import load_ply
    from recon3d.recon.init_points import init_points
    from recon3d.recon.trainer import render

    with tempfile.TemporaryDirectory(prefix="recon3d-eval-") as tmp:
        cap = load_capture(job_dir / "capture.zip", Path(tmp))
        ts = build_training_set(cap, profile)
        del cap

    splats = load_ply(ply, device="cuda")
    n_gauss = len(splats.means)

    # One view at a time: stacking 193 renders would not fit next to the model on 8 GB.
    psnr = np.empty(len(ts.c2w), np.float64)
    for i in range(len(ts.c2w)):
        view = TrainingSet(images=ts.images[i : i + 1], K=ts.K[i : i + 1], c2w=ts.c2w[i : i + 1], depths=[None])
        img = render(splats, view)[0]
        mse = ((img - ts.images[i].to(img.device)) ** 2).mean().item()
        psnr[i] = 10.0 * np.log10(1.0 / max(mse, 1e-12))
        del img

    xyz, _ = init_points(ts, min_points=0)
    alpha = torch.sigmoid(splats.opacities).cpu().numpy()
    means = splats.means.cpu().numpy()[alpha > FLOATER_OPACITY]
    del splats
    torch.cuda.empty_cache()
    if len(means):
        d, _ = cKDTree(xyz).query(means, k=1, workers=-1)
        floater = float((d > FLOATER_DIST_M).mean())
    else:
        floater = 0.0

    worst = np.argsort(psnr, kind="stable")[:WORST_K]
    return {
        "ply": str(ply),
        "profile": profile.name,
        "depth_lambda": profile.depth_lambda,
        "pose_lr": profile.pose_lr,
        "views": int(len(psnr)),
        "psnr_median": float(np.median(psnr)),
        "psnr_min": float(psnr.min()),
        "psnr_worst5": [{"index": int(i), "psnr": float(psnr[i])} for i in worst],
        "gaussians": int(n_gauss),
        "opaque_gaussians": int(len(means)),
        "init_points": int(len(xyz)),
        "floater_score": floater,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="离线评估重建结果（逐视角 PSNR、floater 分数）")
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--profile", default="laptop", choices=sorted(PROFILES))
    ap.add_argument("--train", action="store_true", help="先用当前代码重训到 retrain-<tag>/ 再评估")
    ap.add_argument("--tag", default=None, help="重训目录名后缀（默认 new）")
    ap.add_argument("--pose-lr", type=float, default=None)
    ap.add_argument("--depth-lambda", type=float, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    job_dir = args.job_dir.resolve()
    profile = PROFILES[args.profile]
    overrides = {
        k: v for k, v in (("pose_lr", args.pose_lr), ("depth_lambda", args.depth_lambda)) if v is not None
    }
    if overrides and not args.train:
        ap.error("--pose-lr / --depth-lambda 只能和 --train 一起用")
    profile = dataclasses.replace(profile, **overrides)

    train_s = None
    if args.train:
        out_dir = job_dir / f"retrain-{args.tag or 'new'}"
        print(f"重训到 {out_dir}（depth_lambda={profile.depth_lambda}, pose_lr={profile.pose_lr}）", flush=True)
        train_s = _train(job_dir, out_dir, profile)
    else:
        out_dir = job_dir / f"retrain-{args.tag}" if args.tag else job_dir
    ply = out_dir / "scene.ply"

    res = evaluate(job_dir, ply, profile)
    if not args.train:
        # Training settings of an existing ply are unknown; don't report the profile's.
        res.pop("depth_lambda")
        res.pop("pose_lr")
    res["train_seconds"] = train_s

    print(f"模型：{ply}")
    print(f"视角数：{res['views']}")
    print(f"PSNR 中位数 {res['psnr_median']:.2f} dB，最小 {res['psnr_min']:.2f} dB")
    print("最差 5 帧：" + "，".join(f"#{w['index']} {w['psnr']:.2f}" for w in res["psnr_worst5"]))
    print(
        f"floater 分数：{res['floater_score']:.4f}"
        f"（不透明度>{FLOATER_OPACITY} 的 {res['opaque_gaussians']} 个高斯中，"
        f"离初始点云（{res['init_points']} 点）>{FLOATER_DIST_M:.2f} m 的比例）"
    )
    print(f"高斯总数：{res['gaussians']}")
    if train_s is not None:
        print(f"训练用时：{train_s / 60:.1f} 分钟（{train_s:.0f} s）")
    (out_dir / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out_dir / 'eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
