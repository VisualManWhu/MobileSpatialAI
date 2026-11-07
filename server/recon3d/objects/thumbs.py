"""Representative thumbnail of an object: its best detection box, padded and shrunk."""
from __future__ import annotations

from PIL import Image


def crop_thumbnail(image: Image.Image, box: tuple[float, float, float, float], pad: float = 0.15, long_side: int = 256) -> Image.Image:
    x0, y0, x1, y1 = box
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    x0, x1 = x0 - pad * bw, x1 + pad * bw
    y0, y1 = y0 - pad * bh, y1 + pad * bh
    W, H = image.size
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(W, int(round(x1))), min(H, int(round(y1)))
    if x1 <= x0:
        x1 = min(W, x0 + 1)
    if y1 <= y0:
        y1 = min(H, y0 + 1)
    crop = image.crop((x0, y0, x1, y1))
    scale = long_side / max(crop.size)
    if scale < 1:
        crop = crop.resize((max(1, round(crop.size[0] * scale)), max(1, round(crop.size[1] * scale))), Image.LANCZOS)
    return crop
