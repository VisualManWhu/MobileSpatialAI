import numpy as np

from recon3d.capture.poses import gl_to_cv_c2w


def test_gl_identity_looks_down_minus_z():
    c2w = gl_to_cv_c2w(np.eye(4))
    assert np.allclose(c2w[:3, 2], [0, 0, -1])  # CV +Z (forward) == GL -Z
    assert np.allclose(c2w[:3, 1], [0, -1, 0])  # CV +Y (down)


def test_translation_preserved():
    p = np.eye(4)
    p[:3, 3] = [1, 2, 3]
    assert np.allclose(gl_to_cv_c2w(p)[:3, 3], [1, 2, 3])
