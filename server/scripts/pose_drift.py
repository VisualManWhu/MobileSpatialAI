"""How far did COLMAP move the ARCore poses of one capture? CSV + top-view trajectory plot.

Usage (Ubuntu, from server/):
  uv run python scripts/pose_drift.py <job_dir> [--out DIR]

Unpacks <job_dir>/capture.zip to a temp dir, runs recon/precise_pose.py (CPU, minutes for
hundreds of frames) and writes to --out (default <job_dir>):
  pose_drift.csv     index,name,registered,shift_m,rot_deg   (per frame, refined vs ARCore)
  trajectory.png     top view x/z of both trajectories, start marker, every 50th index annotated
and prints the stats dict (registration ratio, shift mean/median/max, scale, timings).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recon3d.capture.format import load_capture  # noqa: E402
from recon3d.capture.poses import gl_to_cv_c2w  # noqa: E402

ANNOTATE_EVERY = 50


def _rot_deg(a: np.ndarray, b: np.ndarray) -> float:
    r = a[:3, :3] @ b[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))))


def write_report(arcore: np.ndarray, refined: np.ndarray, registered: np.ndarray, names: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shift = np.linalg.norm(refined[:, :3, 3] - arcore[:, :3, 3], axis=1)
    with open(out / "pose_drift.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "name", "registered", "shift_m", "rot_deg"])
        for i, nm in enumerate(names):
            w.writerow([i, nm, int(registered[i]), f"{shift[i]:.4f}", f"{_rot_deg(refined[i], arcore[i]):.3f}"])

    p1, p2 = arcore[:, :3, 3], refined[:, :3, 3]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(p1[:, 0], p1[:, 2], "-", lw=1, label="ARCore")
    ax.plot(p2[:, 0], p2[:, 2], "-", lw=1, label="COLMAP BA")
    ax.scatter(p1[0, 0], p1[0, 2], c="k", marker="o", label="start")
    ax.scatter(p1[-1, 0], p1[-1, 2], c="r", marker="x", label="end")
    for i in range(0, len(p1), ANNOTATE_EVERY):
        ax.annotate(str(i), p2[i, [0, 2]], fontsize=7)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"top view x/z — shift mean {shift.mean():.3f} m, median {np.median(shift):.3f} m, max {shift.max():.3f} m")
    fig.savefig(out / "trajectory.png", dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="COLMAP 位姿优化相对 ARCore 的漂移报告（CSV + 轨迹图）")
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="输出目录（默认 job_dir）")
    args = ap.parse_args(argv)

    from recon3d.recon.precise_pose import precise_pose

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    job_dir = args.job_dir.resolve()
    out = (args.out or job_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="recon3d-drift-") as tmp:
        cap = load_capture(job_dir / "capture.zip", Path(tmp))
        arcore = np.stack([gl_to_cv_c2w(f.pose) for f in cap.frames])
        names = [f.image.rsplit("/", 1)[-1] for f in cap.frames]
        res = precise_pose(cap, Path(tmp) / "colmap", lambda q: print(f"位姿优化 {int(q * 100)}%", flush=True))
    write_report(arcore, res.c2w_cv, res.registered, names, out)
    print(json.dumps(res.stats, ensure_ascii=False, indent=2))
    print(f"已写入 {out / 'pose_drift.csv'} 和 {out / 'trajectory.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
