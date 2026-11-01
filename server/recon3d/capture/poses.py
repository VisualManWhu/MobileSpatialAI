"""Pose convention conversion."""
from __future__ import annotations

import numpy as np

_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def gl_to_cv_c2w(pose_gl: np.ndarray) -> np.ndarray:
    """OpenGL camera-to-world (X right, Y up, -Z forward) -> OpenCV (X right, Y down, +Z forward)."""
    return np.asarray(pose_gl, dtype=np.float64) @ _GL_TO_CV
