import math, torch
from recon3d.recon.pose_opt import PoseRefine, se3_exp

def test_zero_delta_is_identity():
    T = se3_exp(torch.zeros(2, 3), torch.zeros(2, 3))
    assert torch.allclose(T, torch.eye(4).expand(2, 4, 4))

def test_rotation_about_z_90deg():
    T = se3_exp(torch.tensor([[0.0, 0.0, math.pi / 2]]), torch.zeros(1, 3))
    p = T[0, :3, :3] @ torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(p, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)

def test_module_applies_per_camera_and_reg():
    m = PoseRefine(3, "cpu")
    with torch.no_grad():
        m.delta[1, 3] = 0.1  # 10 cm x-translation on camera 1 only
    v = torch.eye(4).expand(3, 4, 4).clone()
    out = m(v, torch.arange(3))
    assert torch.allclose(out[0], torch.eye(4)) and torch.allclose(out[2], torch.eye(4))
    assert abs(float(out[1, 0, 3]) - 0.1) < 1e-6
    assert abs(float(m.reg()) - (0.1 ** 2) / 3) < 1e-9
