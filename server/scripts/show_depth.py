"""Contact sheet of RGB | depth (colour-mapped) | depth overlaid on RGB for a capture dir.

Usage (Ubuntu):
  uv run python server/scripts/show_depth.py <dir with frames.jsonl> [out.png] [--frames 1,40,80] [--every N]
Depth colour scale: 0.3 m (blue) .. 6 m (red); black = 0 / invalid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

NEAR, FAR = 0.3, 6.0
TILE_W = 480  # width of each panel


def colormap(d: np.ndarray) -> np.ndarray:
    """Turbo-ish: near=blue, mid=green/yellow, far=red; invalid=black. d in metres."""
    t = np.clip((d - NEAR) / (FAR - NEAR), 0, 1)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    rgb = np.stack([r, g, b], -1)
    rgb[d <= 0] = 0
    return (rgb * 255).astype(np.uint8)


def panel(root: Path, fr: dict) -> Image.Image:
    im = Image.open(root / fr["image"]).convert("RGB")
    s = TILE_W / im.width
    size = (TILE_W, round(im.height * s))
    im = im.resize(size, Image.BILINEAR)
    d = np.asarray(Image.open(root / fr["depth"]), dtype=np.float32) / 1000.0
    dep = Image.fromarray(colormap(d)).resize(size, Image.NEAREST)
    over = Image.blend(im, dep, 0.55)
    valid = (d > 0).mean() * 100
    far = (d > FAR).mean() * 100
    sheet = Image.new("RGB", (TILE_W * 3 + 8, size[1] + 22), (20, 20, 20))
    for k, p in enumerate((im, dep, over)):
        sheet.paste(p, (k * (TILE_W + 4), 22))
    ImageDraw.Draw(sheet).text(
        (6, 4),
        f"frame {fr['index']:03d}   depth {d.shape[1]}x{d.shape[0]}   valid {valid:.0f}%   >{FAR:.0f}m {far:.0f}%   "
        f"median {np.median(d[d > 0]) if (d > 0).any() else 0:.2f} m",
        fill=(255, 255, 255),
    )
    return sheet


def main(argv: list[str]) -> None:
    root = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 and not argv[1].startswith("--") else root / "depth_preview.png"
    frames = [json.loads(l) for l in (root / "frames.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    frames = [f for f in frames if f.get("depth")]
    pick = frames[:: max(1, len(frames) // 8)][:8]
    if "--frames" in argv:
        want = {int(x) for x in argv[argv.index("--frames") + 1].split(",")}
        pick = [f for f in frames if f["index"] in want]
    if "--every" in argv:
        n = int(argv[argv.index("--every") + 1])
        pick = frames[::n]
    panels = [panel(root, f) for f in pick]
    sheet = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)), (0, 0, 0))
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height
    sheet.save(out)
    print(f"wrote {out} ({len(panels)} frames: {[f['index'] for f in pick]})")


if __name__ == "__main__":
    main(sys.argv[1:])
