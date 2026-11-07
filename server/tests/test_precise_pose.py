"""precise_pose on a synthetic textured room: loop closure, unregistered frames, two cameras."""
import numpy as np
import pytest

from recon3d.capture.format import load_capture
from tests.fixtures.make_capture import make_capture
from tests.fixtures.room_render import loop_trajectory, render_room

W, H, N = 320, 240, 40
K = np.array([[280.0, 0, 160.0], [0, 280.0, 120.0], [0, 0, 1]])
GL = np.diag([1.0, -1.0, -1.0, 1.0])


def _gl_colmajor(c2w_cv):
    """make_capture wants 16 floats, column-major, OpenGL c2w."""
    return (c2w_cv @ GL).T.flatten().tolist()


def _drifted(truth):
    """Accumulated VIO-like drift: end frame off by 0.3 m / 3 deg."""
    out = []
    for i, T in enumerate(truth):
        s = i / (len(truth) - 1)
        a = np.deg2rad(3.0 * s)
        Rd = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        D = np.eye(4)
        D[:3, :3] = Rd
        D[:3, 3] = [0.3 * s, 0, 0]
        out.append(D @ T)
    return np.stack(out)


def _loop_capture(tmp_path, blank=()):
    truth = loop_trajectory(N)
    noisy = _drifted(truth)
    images = [render_room(truth[i], K, W, H) for i in range(N)]
    for b in blank:
        images[b] = np.full((H, W, 3), 128, np.uint8)
    z = make_capture(tmp_path, n_frames=N, with_depth=False, image_width=W, image_height=H,
                     fx=280.0, fy=280.0, cx=160.0, cy=120.0,
                     poses=[_gl_colmajor(p) for p in noisy], images=images)
    return load_capture(z, tmp_path / "w"), truth, noisy


def _aligned_center_err(a, truth):
    """Per-frame centre error after the best similarity fit of a onto truth.

    The priors keep the refined poses in the (drifted) ARCore frame, so absolute
    error against truth is dominated by that frame; the shape error is what
    precise mode improves.
    """
    src, dst = a[:, :3, 3], truth[:, :3, 3]
    ms, md = src.mean(0), dst.mean(0)
    sc, dc = src - ms, dst - md
    U, S, Vt = np.linalg.svd(dc.T @ sc / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / ((sc ** 2).sum() / len(src))
    fit = s * (R @ src.T).T + (md - s * R @ ms)
    return np.linalg.norm(fit - dst, axis=1)


@pytest.mark.slow
def test_loop_closes_and_poses_improve(tmp_path):
    from recon3d.recon.precise_pose import precise_pose
    cap, truth, noisy = _loop_capture(tmp_path)
    seen = []
    res = precise_pose(cap, tmp_path / "colmap", seen.append)
    assert res.c2w_cv.shape == (N, 4, 4) and res.registered.shape == (N,)
    assert res.registered.mean() >= 0.9
    rel_in = np.linalg.inv(noisy[0]) @ noisy[-1]
    rel_out = np.linalg.inv(res.c2w_cv[0]) @ res.c2w_cv[-1]
    rel_gt = np.linalg.inv(truth[0]) @ truth[-1]
    err_in = np.linalg.norm(rel_in[:3, 3] - rel_gt[:3, 3])
    err_out = np.linalg.norm(rel_out[:3, 3] - rel_gt[:3, 3])
    assert err_out * 5 < err_in
    assert np.median(_aligned_center_err(res.c2w_cv, truth)) < np.median(_aligned_center_err(noisy, truth)) / 3
    assert 0.98 <= res.stats["scale"] <= 1.02
    assert seen == sorted(seen) and seen[-1] == 1.0 and seen[0] <= 0.35
    for k in ("registered_ratio", "shift_mean_m", "rot_mean_deg", "points3d", "reproj_px", "t_extract_s"):
        assert k in res.stats


@pytest.mark.slow
def test_blank_frames_unregistered_follow_neighbour_correction(tmp_path):
    from recon3d.recon.precise_pose import precise_pose
    cap, truth, noisy = _loop_capture(tmp_path, blank=(10, 11))
    res = precise_pose(cap, tmp_path / "colmap", lambda p: None)
    assert not res.registered[10] and not res.registered[11] and res.registered[9] and res.registered[12]
    # frame 10 inherits frame 9's rigid correction; frame 11 inherits frame 12's (nearest registered, ties -> earlier)
    exp10 = res.c2w_cv[9] @ np.linalg.inv(noisy[9]) @ noisy[10]
    exp11 = res.c2w_cv[12] @ np.linalg.inv(noisy[12]) @ noisy[11]
    assert np.allclose(res.c2w_cv[10], exp10, atol=1e-6)
    assert np.allclose(res.c2w_cv[11], exp11, atol=1e-6)


@pytest.mark.slow
def test_precise_pose_two_intrinsic_groups(tmp_path):
    """Frames 0..19 fx=280, frames 20..39 fx=300: one COLMAP camera per group, all frames still register."""
    from recon3d.recon.precise_pose import precise_pose
    truth = loop_trajectory(N)
    Ks = [K if i < 20 else np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1]]) for i in range(N)]
    images = [render_room(truth[i], Ks[i], W, H) for i in range(N)]
    z = make_capture(tmp_path, n_frames=N, with_depth=False, image_width=W, image_height=H,
                     poses=[_gl_colmajor(p) for p in _drifted(truth)], images=images,
                     fx=280.0, fy=280.0, cx=160.0, cy=120.0)
    cap = load_capture(z, tmp_path / "w")
    for i in range(20, N):  # make_capture has one global fx; patch the loaded frames
        cap.frames[i].fx = cap.frames[i].fy = 300.0
    res = precise_pose(cap, tmp_path / "colmap", lambda p: None)
    assert res.stats["cameras"] == 2
    assert res.registered.mean() >= 0.9
