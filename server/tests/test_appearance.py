import torch

from recon3d.recon.appearance import AffineColor


def test_identity_at_init():
    m = AffineColor(4, "cpu")
    rgb = torch.rand(2, 5, 6, 3)
    idx = torch.tensor([1, 3])
    assert torch.equal(m(rgb, idx), rgb)
    assert float(m.reg()) == 0.0
    s = m.stats()
    assert s["gain_max"] == 0.0 and s["off_max"] == 0.0


def test_applies_per_frame_affine():
    m = AffineColor(3, "cpu")
    with torch.no_grad():
        m.dA[2] = torch.diag(torch.tensor([1.0, 0.0, -0.5]))  # A = diag(2, 1, 0.5)
        m.db[2] = torch.tensor([0.1, 0.0, 0.0])
    rgb = torch.full((1, 2, 2, 3), 0.4)
    # Deltas are mean-centred: frame 2 gets 2/3 of its delta, frames 0 and 1 get -1/3 of it.
    A2 = torch.eye(3) + torch.diag(torch.tensor([1.0, 0.0, -0.5])) * 2 / 3
    b2 = torch.tensor([0.1, 0.0, 0.0]) * 2 / 3
    out = m(rgb, torch.tensor([2]))
    assert torch.allclose(out[0, 0, 0], A2 @ rgb[0, 0, 0] + b2)
    A0 = torch.eye(3) - torch.diag(torch.tensor([1.0, 0.0, -0.5])) / 3
    assert torch.allclose(m(rgb, torch.tensor([0]))[0, 1, 1], A0 @ rgb[0, 1, 1] - b2 / 2)
    # The mean correction over all frames is exactly the identity.
    A, b = m.matrices(torch.arange(3))
    assert torch.allclose(A.mean(0), torch.eye(3), atol=1e-7) and torch.allclose(b.mean(0), torch.zeros(3), atol=1e-7)
    # reg is on the raw deltas: (1 + 0.25 + 0.01) spread over 3 frames.
    assert abs(float(m.reg()) - (1.0 + 0.25 + 0.01) / 3) < 1e-6
    assert m.stats()["gain_max"] > 0


def test_gradients_flow_through_centring():
    m = AffineColor(3, "cpu")
    rgb = torch.rand(1, 4, 4, 3, requires_grad=True)
    m(rgb, torch.tensor([1])).sum().backward()
    assert m.dA.grad[1].abs().sum() > 0 and m.db.grad[1].abs().sum() > 0
    # Centring couples the frames: the others get the opposite (mean-removal) gradient, half each.
    assert torch.allclose(m.dA.grad[0], m.dA.grad[2]) and torch.allclose(m.dA.grad[0], -m.dA.grad[1] / 2)
    assert rgb.grad is not None and rgb.grad.abs().sum() > 0
