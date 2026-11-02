import numpy as np
from recon3d.recon.confidence import depth_confidence, MAX_DEPTH_M, EDGE_M

def test_conf_png_scales_and_masks_invalid():
    d = np.array([[1.0, 0.0], [5.0, 2.0]], np.float32)
    c = np.array([[255, 255], [255, 51]], np.uint8)
    out = depth_confidence(d, c)
    assert out[0, 0] == 1.0 and out[0, 1] == 0.0 and out[1, 0] == 0.0
    assert abs(out[1, 1] - 0.2) < 1e-6

def test_heuristic_penalises_depth_edges():
    d = np.full((5, 5), 2.0, np.float32)
    d[:, 3:] = 2.5  # 50 cm step between columns 2 and 3
    out = depth_confidence(d, None)
    assert out[2, 0] == 1.0          # flat region
    assert out[2, 2] == 0.0 and out[2, 3] == 0.0  # both sides of the step
    d2 = np.full((5, 5), 2.0, np.float32); d2[:, 3:] = 2.05
    assert abs(depth_confidence(d2, None)[2, 2] - 0.5) < 1e-6

def test_heuristic_far_is_zero():
    d = np.full((3, 3), MAX_DEPTH_M + 0.5, np.float32)
    assert depth_confidence(d, None).max() == 0.0
