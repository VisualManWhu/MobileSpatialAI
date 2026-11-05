"""Offline evaluation of a trained scene: per-view PSNR and a floater score.

Usage (Ubuntu, from server/):
  uv run python scripts/eval_scene.py <job_dir> [--profile laptop]
      evaluates <job_dir>/scene.ply (or <job_dir>/retrain-<tag>/scene.ply with --tag)
  uv run python scripts/eval_scene.py <job_dir> --train [--tag NAME] [--pose-lr X] [--depth-lambda Y]
                                                         [--far-m R] [--bbox-margin-m M] [--support-far-m S] [--support-dist-m D]
      retrains capture.zip into <job_dir>/retrain-<tag>/ (default tag "new") with the job
      pipeline, then evaluates the new scene.ply

Metrics (also written to eval.json next to the evaluated ply):
  - PSNR of every training view rendered at its capture pose: median, min, worst 5 indices; plus the median
    after a per-view least-squares affine colour fit (exposure-independent)
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
# Profile fields recorded in eval.json; only known (and reported) when this run trained the ply.
TRAIN_SETTINGS = (
    "depth_lambda", "pose_lr", "strategy", "far_m", "bbox_margin_m", "support_far_m", "support_dist_m", "contrib_min_views",
    "contrib_min_px", "contrib_view_px", "sor_sigma", "opacity_reg", "scale_reg", "appearance_lr", "appearance_reg",
    "mono_lambda", "mono_lambda_end", "mono_patch", "mcmc_noise_lr", "max_scale_m",
)


def _matched_mse(img, gt) -> float:
    """MSE after the best per-view affine colour fit rgb' = A rgb + b (least squares on a pixel subsample).

    Separates exposure / white-balance mismatch between the exported colours and this frame from
    geometry and blur: a scene trained with per-frame colour correction renders the *mean* frame's
    colours, so its plain PSNR drops on every frame whose exposure differs from the mean.
    """
    import torch

    x = img.reshape(-1, 3).double()
    y = gt.reshape(-1, 3).double()
    step = max(1, len(x) // 200_000)
    xs = torch.cat([x[::step], torch.ones(len(x[::step]), 1, dtype=x.dtype, device=x.device)], dim=1)
    m = torch.linalg.lstsq(xs, y[::step]).solution  # [4,3]
    pred = x @ m[:3] + m[3]
    return ((pred - y) ** 2).mean().item()


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


def evaluate(job_dir: Path, ply: Path, profile: Profile, out_dir: Path, offtrack_m: float = 0.5, offtrack_views: int = 12) -> dict:
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
    psnr_matched = np.empty(len(ts.c2w), np.float64)
    for i in range(len(ts.c2w)):
        view = TrainingSet(images=ts.images[i : i + 1], K=ts.K[i : i + 1], c2w=ts.c2w[i : i + 1], depths=[None])
        img = render(splats, view)[0]
        gt = ts.images[i].to(img.device)
        mse = ((img - gt) ** 2).mean().item()
        psnr[i] = 10.0 * np.log10(1.0 / max(mse, 1e-12))
        psnr_matched[i] = 10.0 * np.log10(1.0 / max(_matched_mse(img, gt), 1e-12))
        del img, gt

    offtrack_png = None
    if offtrack_m > 0 and offtrack_views > 0:
        import cv2  # lazy: only available via `uv run --with opencv-python-headless`

        from recon3d.recon.offtrack import contact_sheet, pick_views, raise_cameras

        c2w_up = raise_cameras(ts.c2w, offtrack_m)
        tiles = []
        for i in pick_views(len(ts.c2w), offtrack_views):
            view = TrainingSet(images=ts.images[i : i + 1], K=ts.K[i : i + 1], c2w=c2w_up[i : i + 1], depths=[None])
            img = render(splats, view)[0].clamp(0, 1).mul(255).round().byte().cpu().numpy()
            tiles.append(np.ascontiguousarray(img[..., ::-1]))  # RGB -> BGR for cv2
            del img
        if tiles:
            offtrack_png = out_dir / "offtrack.png"
            cv2.imwrite(str(offtrack_png), contact_sheet(tiles, cols=4))

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
        "strategy": profile.strategy,
        "far_m": profile.far_m,
        "bbox_margin_m": profile.bbox_margin_m,
        "support_far_m": profile.support_far_m,
        "support_dist_m": profile.support_dist_m,
        "contrib_min_views": profile.contrib_min_views,
        "contrib_min_px": profile.contrib_min_px,
        "contrib_view_px": profile.contrib_view_px,
        "sor_sigma": profile.sor_sigma,
        "opacity_reg": profile.opacity_reg,
        "scale_reg": profile.scale_reg,
        "appearance_lr": profile.appearance_lr,
        "appearance_reg": profile.appearance_reg,
        "mono_lambda": profile.mono_lambda,
        "mono_lambda_end": profile.mono_lambda_end,
        "mono_patch": profile.mono_patch,
        "mcmc_noise_lr": profile.mcmc_noise_lr,
        "max_scale_m": profile.max_scale_m,
        "views": int(len(psnr)),
        "psnr_median": float(np.median(psnr)),
        "psnr_matched_median": float(np.median(psnr_matched)),
        "psnr_min": float(psnr.min()),
        "psnr_worst5": [{"index": int(i), "psnr": float(psnr[i])} for i in worst],
        "gaussians": int(n_gauss),
        "opaque_gaussians": int(len(means)),
        "init_points": int(len(xyz)),
        "floater_score": floater,
        "offtrack_png": str(offtrack_png) if offtrack_png else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="离线评估重建结果（逐视角 PSNR、floater 分数）")
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--profile", default="laptop", choices=sorted(PROFILES))
    ap.add_argument("--train", action="store_true", help="先用当前代码重训到 retrain-<tag>/ 再评估")
    ap.add_argument("--tag", default=None, help="重训目录名后缀（默认 new）")
    ap.add_argument("--pose-lr", type=float, default=None)
    ap.add_argument("--depth-lambda", type=float, default=None)
    ap.add_argument("--offtrack-m", type=float, default=0.5, help="偏离轨迹渲染：相机抬高的米数（0 关闭）")
    ap.add_argument("--offtrack-views", type=int, default=12, help="偏离轨迹渲染：拼图的视角数")
    ap.add_argument("--far-m", type=float, default=None, help="场景范围：距最近相机的最大距离（0 关闭）")
    ap.add_argument("--bbox-margin-m", type=float, default=None, help="场景范围：相机包围盒外扩距离（0 关闭）")
    ap.add_argument("--support-far-m", type=float, default=None, help="点云支持裁剪：距相机超过此距离的高斯需贴近深度点云（0 关闭）")
    ap.add_argument("--support-dist-m", type=float, default=None, help="点云支持裁剪：到深度点云的最大距离（0 关闭）")
    ap.add_argument("--contrib-min-views", type=int, default=None, help="贡献度剪枝：最少可见视角数（与 --contrib-min-px 都为 0 时关闭）")
    ap.add_argument("--contrib-min-px", type=float, default=None, help="贡献度剪枝：最小贡献像素数（与 --contrib-min-views 都为 0 时关闭）")
    ap.add_argument("--contrib-view-px", type=float, default=None, help="贡献度剪枝：单个视角算作可见的最小混合像素数（半分辨率）")
    ap.add_argument("--sor-sigma", type=float, default=None, help="离群点滤波：均值外的标准差倍数（0 关闭）")
    ap.add_argument("--opacity-reg", type=float, default=None, help="不透明度正则权重（0 关闭）")
    ap.add_argument("--scale-reg", type=float, default=None, help="尺度正则权重（0 关闭）")
    ap.add_argument("--appearance-lr", type=float, default=None, help="逐帧仿射颜色校正的学习率（0 关闭）")
    ap.add_argument("--appearance-reg", type=float, default=None, help="颜色校正向单位阵的正则权重")
    ap.add_argument("--mono-lambda", type=float, default=None, help="单目深度（Depth Anything V2）patch Pearson 约束的起始权重（0 关闭）")
    ap.add_argument("--mono-lambda-end", type=float, default=None, help="单目深度约束的结束权重（线性衰减）")
    ap.add_argument("--mono-patch", type=int, default=None, help="单目深度约束的图块边长（像素）")
    ap.add_argument("--mcmc-noise-lr", type=float, default=None, help="MCMC 位置噪声尺度（gsplat noise_lr，默认 5e5）")
    ap.add_argument("--max-scale-m", type=float, default=None, help="裁掉最长轴超过此米数的高斯（0 关闭）")
    ap.add_argument("--strategy", default=None, choices=["default", "mcmc"], help="致密化策略（default 或 mcmc）")
    ap.add_argument("--dump-pruned", action="store_true", help="把贡献度剪枝和离群点滤波移除的高斯写入 pruned.ply（需 --train）")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    job_dir = args.job_dir.resolve()
    profile = PROFILES[args.profile]
    overrides = {
        k: v
        for k, v in (
            ("pose_lr", args.pose_lr),
            ("depth_lambda", args.depth_lambda),
            ("far_m", args.far_m),
            ("bbox_margin_m", args.bbox_margin_m),
            ("support_far_m", args.support_far_m),
            ("support_dist_m", args.support_dist_m),
            ("contrib_min_views", args.contrib_min_views),
            ("contrib_min_px", args.contrib_min_px),
            ("contrib_view_px", args.contrib_view_px),
            ("sor_sigma", args.sor_sigma),
            ("opacity_reg", args.opacity_reg),
            ("scale_reg", args.scale_reg),
            ("appearance_lr", args.appearance_lr),
            ("appearance_reg", args.appearance_reg),
            ("mono_lambda", args.mono_lambda),
            ("mono_lambda_end", args.mono_lambda_end),
            ("mono_patch", args.mono_patch),
            ("mcmc_noise_lr", args.mcmc_noise_lr),
            ("max_scale_m", args.max_scale_m),
            ("strategy", args.strategy),
        )
        if v is not None
    }
    if (overrides or args.dump_pruned) and not args.train:
        ap.error(
            "--pose-lr / --depth-lambda / --far-m / --bbox-margin-m / --support-far-m / --support-dist-m / --contrib-min-views / --contrib-min-px"
            " / --contrib-view-px / --sor-sigma / --opacity-reg / --scale-reg / --appearance-lr / --appearance-reg / --mono-lambda / --mono-lambda-end / --mono-patch / --mcmc-noise-lr / --max-scale-m / --strategy / --dump-pruned 只能和 --train 一起用"
        )
    profile = dataclasses.replace(profile, **overrides)

    train_s = None
    if args.train:
        out_dir = job_dir / f"retrain-{args.tag or 'new'}"
        print(f"重训到 {out_dir}（depth_lambda={profile.depth_lambda}, pose_lr={profile.pose_lr}, strategy={profile.strategy}）", flush=True)
        if args.dump_pruned:
            os.environ["RECON3D_DUMP_PRUNED"] = "1"
        try:
            train_s = _train(job_dir, out_dir, profile)
        finally:
            os.environ.pop("RECON3D_DUMP_PRUNED", None)
        if args.dump_pruned and (out_dir / "pruned.ply").exists():
            print(f"被剪高斯已写入 {out_dir / 'pruned.ply'}", flush=True)
    else:
        out_dir = job_dir / f"retrain-{args.tag}" if args.tag else job_dir
    ply = out_dir / "scene.ply"

    res = evaluate(job_dir, ply, profile, out_dir, args.offtrack_m, args.offtrack_views)
    if not args.train:
        # Training settings of an existing ply are unknown; don't report the profile's.
        for k in TRAIN_SETTINGS:
            res.pop(k)
    res["train_seconds"] = train_s

    print(f"模型：{ply}")
    print(f"视角数：{res['views']}")
    print(f"PSNR 中位数 {res['psnr_median']:.2f} dB，最小 {res['psnr_min']:.2f} dB；逐帧颜色匹配后中位数 {res['psnr_matched_median']:.2f} dB")
    print("最差 5 帧：" + "，".join(f"#{w['index']} {w['psnr']:.2f}" for w in res["psnr_worst5"]))
    print(
        f"floater 分数：{res['floater_score']:.4f}"
        f"（不透明度>{FLOATER_OPACITY} 的 {res['opaque_gaussians']} 个高斯中，"
        f"离初始点云（{res['init_points']} 点）>{FLOATER_DIST_M:.2f} m 的比例）"
    )
    print(f"高斯总数：{res['gaussians']}")
    if res["offtrack_png"]:
        print(f"偏离轨迹（抬高 {args.offtrack_m:g} m）渲染：{res['offtrack_png']}")
    if train_s is not None:
        print(f"训练用时：{train_s / 60:.1f} 分钟（{train_s:.0f} s）")
    (out_dir / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out_dir / 'eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
