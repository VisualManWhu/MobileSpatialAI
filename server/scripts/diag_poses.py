"""Pose-accuracy diagnostic for an unpacked capture (CPU only).

Two independent checks over frame pairs (i, i+k):

A. Depth consistency. Back-project frame i's ARCore depth, move it into frame i+k with the
   ARCore poses, compare with frame i+k's own depth along the viewing ray, then solve a
   point-to-plane least-squares 6-DoF correction (small-angle). Rotation/translation of that
   correction = pose error the depth can see; residual after it = depth-sensor noise floor.

B. Image-only epipolar check (needs opencv). ORB matches between the two images, RANSAC
   fundamental matrix; then (1) the Sampson error of the inlier matches under the
   fundamental matrix implied by ARCore's relative pose, in pixels, and (2) the angle
   between ARCore's relative rotation / translation direction and the ones recovered from
   the essential matrix. (2) is independent of depth entirely.

Usage (Ubuntu):  uv run python server/scripts/diag_poses.py <dir with frames.jsonl>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recon3d.capture.poses import gl_to_cv_c2w  # noqa: E402

MIN_Z, MAX_Z = 0.3, 6.0  # ARCore depth is unreliable beyond a few metres
GROSS = 0.5  # |residual| above this is occlusion / dynamic, not pose error
IMG_SIDE = 960  # ORB on downscaled images (same scale the trainer uses)


def load(root: Path):
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frames = [json.loads(l) for l in (root / "frames.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for fr in frames:
        if not fr.get("depth"):
            continue
        d = np.asarray(Image.open(root / fr["depth"]), dtype=np.float64) / 1000.0
        dh, dw = d.shape
        sx, sy = dw / man["image_width"], dh / man["image_height"]
        K = np.array([[fr["fx"] * sx, 0, fr["cx"] * sx], [0, fr["fy"] * sy, fr["cy"] * sy], [0, 0, 1]])
        Kimg = np.array([[fr["fx"], 0, fr["cx"]], [0, fr["fy"], fr["cy"]], [0, 0, 1]])
        out.append({"idx": fr["index"], "t": fr["t_ns"] * 1e-9, "depth": d, "K": K, "Kimg": Kimg,
                    "image": root / fr["image"], "c2w": gl_to_cv_c2w(np.array(fr["pose"]).reshape(4, 4).T)})
    return out


def backproject(d, K):
    valid = (d > MIN_Z) & (d < MAX_Z)
    v, u = np.nonzero(valid)
    z = d[v, u]
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)


def normals_from_depth(d, K):
    """Per-pixel normals in camera frame from the depth map (central differences)."""
    dh, dw = d.shape
    v, u = np.mgrid[0:dh, 0:dw]
    z = d
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], -1)
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(dx, dy)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(ln, 1e-9)
    ok = (ln[..., 0] > 1e-9) & (z > MIN_Z) & (z < MAX_Z)
    # steep normals (grazing / depth discontinuities) are unreliable
    ok &= np.abs(n[..., 2]) > 0.2
    return n, ok


def point_to_plane(p, q, n, iters=3):
    """Small-angle 6-DoF (omega, t) minimising sum (n·(p + omega x p + t - q))^2, with reweighting."""
    w = np.ones(len(p))
    omega = np.zeros(3)
    t = np.zeros(3)
    for _ in range(iters):
        pc = p + np.cross(omega, p) + t
        r = np.einsum("ij,ij->i", n, pc - q)
        thr = max(0.03, 1.5 * np.median(np.abs(r)))
        w = (np.abs(r) < thr).astype(float)
        if w.sum() < 20:
            break
        A = np.concatenate([np.cross(pc, n), n], 1)  # d r / d omega = n·(dω x p) = (p x n)·dω
        Aw, rw = A * w[:, None], r * w
        try:
            d = np.linalg.lstsq(Aw, -rw, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        omega, t = omega + d[:3], t + d[3:]
    pc = p + np.cross(omega, p) + t
    after = np.abs(np.einsum("ij,ij->i", n, pc - q))
    return omega, t, after[w > 0]


def depth_pair(a, b):
    pa = backproject(a["depth"], a["K"])
    w2b = np.linalg.inv(b["c2w"])
    T = w2b @ a["c2w"]
    pb = (T[:3, :3] @ pa.T).T + T[:3, 3]
    z = pb[:, 2]
    ok = z > MIN_Z
    pb, z = pb[ok], z[ok]
    Kb = b["K"]
    u = np.round(pb[:, 0] / z * Kb[0, 0] + Kb[0, 2]).astype(int)
    v = np.round(pb[:, 1] / z * Kb[1, 1] + Kb[1, 2]).astype(int)
    dh, dw = b["depth"].shape
    inb = (u >= 0) & (u < dw) & (v >= 0) & (v < dh)
    if inb.sum() < 50:
        return None
    u, v, z, pb = u[inb], v[inb], z[inb], pb[inb]
    db = b["depth"][v, u]
    nb, nok = normals_from_depth(b["depth"], Kb)
    good = (db > MIN_Z) & (db < MAX_Z)
    r = z[good] - db[good]
    if len(r) < 50:
        return None
    fine = good.copy()
    fine[good] = np.abs(r) < GROSS
    fine &= nok[v, u]
    if fine.sum() < 50:
        return None
    q = np.stack([(u[fine] - Kb[0, 2]) / Kb[0, 0] * db[fine], (v[fine] - Kb[1, 2]) / Kb[1, 1] * db[fine], db[fine]], 1)
    omega, t, after = point_to_plane(pb[fine], q, nb[v[fine], u[fine]])
    return {
        "overlap": float(good.mean()), "med_abs": float(np.median(np.abs(r))),
        "within5cm": float((np.abs(r) < 0.05).mean()), "bias": float(np.median(r)),
        "rot_deg": float(np.degrees(np.linalg.norm(omega))), "trans_cm": float(np.linalg.norm(t) * 100),
        "after_cm": float(np.median(after) * 100) if len(after) else float("nan"),
    }


# ---------------------------------------------------------------- image-only check
_orb_cache: dict = {}


def orb(fr):
    import cv2

    if fr["idx"] in _orb_cache:
        return _orb_cache[fr["idx"]]
    im = cv2.imread(str(fr["image"]), cv2.IMREAD_GRAYSCALE)
    s = IMG_SIDE / max(im.shape)
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    det = cv2.ORB_create(nfeatures=3000)
    kp, desc = det.detectAndCompute(im, None)
    K = fr["Kimg"].copy()
    K[:2] *= s
    _orb_cache[fr["idx"]] = (kp, desc, K)
    return _orb_cache[fr["idx"]]


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def image_pair(a, b):
    import cv2

    kpa, da, Ka = orb(a)
    kpb, db_, Kb = orb(b)
    if da is None or db_ is None:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db_, k=2)
    good = [x for x, y in (p for p in m if len(p) == 2) if x.distance < 0.75 * y.distance]
    if len(good) < 30:
        return None
    pa = np.float32([kpa[g.queryIdx].pt for g in good])
    pb = np.float32([kpb[g.trainIdx].pt for g in good])
    E, mask = cv2.findEssentialMat(pa, pb, Ka, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return None
    inl = mask.ravel().astype(bool)
    if inl.sum() < 20:
        return None
    # ARCore relative pose a -> b (OpenCV convention): x_b = R x_a + t
    T = np.linalg.inv(b["c2w"]) @ a["c2w"]
    R, t = T[:3, :3], T[:3, 3]
    baseline = np.linalg.norm(t)
    F = np.linalg.inv(Kb).T @ skew(t / max(baseline, 1e-9)) @ R @ np.linalg.inv(Ka)
    xa = np.concatenate([pa[inl], np.ones((inl.sum(), 1), np.float32)], 1)
    xb = np.concatenate([pb[inl], np.ones((inl.sum(), 1), np.float32)], 1)
    Fx = xa @ F.T
    Ftx = xb @ F
    num = np.einsum("ij,ij->i", xb, Fx) ** 2
    den = Fx[:, 0] ** 2 + Fx[:, 1] ** 2 + Ftx[:, 0] ** 2 + Ftx[:, 1] ** 2
    sampson = np.sqrt(num / np.maximum(den, 1e-12))
    _, Re, te, _ = cv2.recoverPose(E, pa[inl], pb[inl], Ka)
    rot_err = np.degrees(np.arccos(np.clip((np.trace(Re.T @ R) - 1) / 2, -1, 1)))
    t_dir_err = np.degrees(np.arccos(np.clip(abs(np.dot(te.ravel(), t / max(baseline, 1e-9))), -1, 1)))
    return {"matches": int(inl.sum()), "sampson_px": float(np.median(sampson)),
            "rot_err_deg": float(rot_err), "tdir_err_deg": float(t_dir_err), "baseline_cm": float(baseline * 100)}


def main(root: Path):
    fr = load(root)
    print(f"{len(fr)} frames with depth; depth {fr[0]['depth'].shape[1]}x{fr[0]['depth'].shape[0]}")
    cams = np.array([f["c2w"][:3, 3] for f in fr])
    step = np.linalg.norm(np.diff(cams, axis=0), axis=1)
    dt = np.diff([f["t"] for f in fr])
    print(f"camera step between keyframes: median {np.median(step)*100:.1f} cm, max {step.max()*100:.1f} cm; "
          f"median dt {np.median(dt):.2f} s; path length {step.sum():.1f} m")
    try:
        import cv2  # noqa: F401
        have_cv = True
    except ImportError:
        have_cv = False
        print("(opencv not installed: image-only check skipped; uv pip install opencv-python-headless)")

    for k in (1, 3, 10, 30):
        rows = [(fr[i]["idx"], fr[i + k]["idx"], s) for i in range(len(fr) - k)
                if (s := depth_pair(fr[i], fr[i + k])) and s["overlap"] > 0.2]
        print(f"\n=== A. depth consistency, pairs (i, i+{k}): {len(rows)} pairs with >20% overlap ===")
        if rows:
            g = lambda key: np.array([r[2][key] for r in rows])
            print(f"  |dz| along ray: median {np.median(g('med_abs'))*100:.1f} cm, p90 {np.percentile(g('med_abs'),90)*100:.1f} cm; "
                  f"pixels within 5 cm: {np.median(g('within5cm'))*100:.0f}%; signed bias {np.median(g('bias'))*100:+.1f} cm")
            print(f"  point-to-plane correction: rotation median {np.median(g('rot_deg')):.2f} deg, p90 {np.percentile(g('rot_deg'),90):.2f}, "
                  f"max {g('rot_deg').max():.2f}; translation median {np.median(g('trans_cm')):.1f} cm, "
                  f"p90 {np.percentile(g('trans_cm'),90):.1f}, max {g('trans_cm').max():.1f}")
            print(f"  residual after correction (noise floor): median {np.nanmedian(g('after_cm')):.1f} cm")
            worst = sorted(rows, key=lambda r: -r[2]["trans_cm"])[:5]
            print("  worst 5: " + ", ".join(f"{a}->{b} {s['trans_cm']:.0f}cm/{s['rot_deg']:.1f}deg" for a, b, s in worst))
        if not have_cv:
            continue
        rows = [(fr[i]["idx"], fr[i + k]["idx"], s) for i in range(len(fr) - k) if (s := image_pair(fr[i], fr[i + k]))]
        print(f"=== B. image-only epipolar check, pairs (i, i+{k}): {len(rows)} pairs with >=20 inlier matches ===")
        if rows:
            g = lambda key: np.array([r[2][key] for r in rows])
            print(f"  inlier matches median {np.median(g('matches')):.0f}; baseline median {np.median(g('baseline_cm')):.0f} cm")
            print(f"  Sampson error under ARCore pose: median {np.median(g('sampson_px')):.2f} px, p90 {np.percentile(g('sampson_px'),90):.2f} px "
                  f"(at {IMG_SIDE}px width; <1 px = pose consistent with the images)")
            print(f"  ARCore vs essential-matrix pose: rotation diff median {np.median(g('rot_err_deg')):.2f} deg, p90 {np.percentile(g('rot_err_deg'),90):.2f}; "
                  f"translation-direction diff median {np.median(g('tdir_err_deg')):.1f} deg, p90 {np.percentile(g('tdir_err_deg'),90):.1f} "
                  f"(direction is ill-defined when the baseline is a few cm)")
            worst = sorted(rows, key=lambda r: -r[2]["sampson_px"])[:5]
            print("  worst 5 by Sampson: " + ", ".join(f"{a}->{b} {s['sampson_px']:.1f}px/{s['rot_err_deg']:.1f}deg" for a, b, s in worst))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
