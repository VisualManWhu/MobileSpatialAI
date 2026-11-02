import numpy as np
import pytest
import torch


from recon3d.recon.export_splat import export_ply, export_splat, load_ply  # noqa: E402
from recon3d.recon.trainer import Splats  # noqa: E402


def make_splats(n, opacity_logit=None, k=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    if opacity_logit is None:
        opacity_logit = [2.0] * n
    return Splats(
        means=torch.randn(n, 3, generator=g),
        scales=torch.randn(n, 3, generator=g) * 0.1 - 3.0,
        quats=torch.randn(n, 4, generator=g) * 3.0,
        opacities=torch.tensor(opacity_logit, dtype=torch.float32),
        sh0=torch.randn(n, 1, 3, generator=g) * 0.5,
        shN=torch.randn(n, k, 3, generator=g) * 0.1,
    )


def test_splat_record_layout(tmp_path):
    s = make_splats(n=3, opacity_logit=[5.0, 5.0, -10.0])
    n = export_splat(s, tmp_path / "a.splat")
    data = (tmp_path / "a.splat").read_bytes()
    assert n == 2 and len(data) == 2 * 32
    # ordering: equal opacity -> larger volume first; find which record is which
    vol = torch.exp(s.scales[:2].sum(1))
    first = int(torch.argmax(vol))
    rec = data[:32] if first == 0 else data[32:]
    pos = np.frombuffer(rec[:12], "<f4")
    assert np.allclose(pos, s.means[first].numpy())
    assert rec[12 + 12 + 3] == round(torch.sigmoid(torch.tensor(5.0)).item() * 255)
    scale = np.frombuffer(rec[12:24], "<f4")
    assert np.allclose(scale, torch.exp(s.scales[first]).numpy())
    rgb = np.clip(0.5 + 0.28209479177387814 * s.sh0[first, 0].numpy(), 0, 1) * 255
    assert np.all(np.abs(np.frombuffer(rec[24:27], np.uint8).astype(float) - rgb) <= 1)
    q = s.quats[first] / s.quats[first].norm()
    exp_q = np.clip(np.trunc(q.numpy() * 128 + 128), 0, 255)
    assert np.array_equal(np.frombuffer(rec[28:32], np.uint8), exp_q.astype(np.uint8))


def test_splat_sorted_by_importance_desc(tmp_path):
    s = make_splats(n=50, opacity_logit=list(np.linspace(-3, 3, 50)))
    n = export_splat(s, tmp_path / "b.splat", min_opacity=0.0)
    assert n == 50
    arr = np.frombuffer((tmp_path / "b.splat").read_bytes(), dtype=np.dtype(
        [("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)]))
    imp = arr["s"].prod(1) * (arr["c"][:, 3] / 255.0)
    # alpha is u8-quantized, so allow a small relative slack
    assert np.all(imp[1:] <= imp[:-1] * 1.02)


def test_splat_all_filtered(tmp_path):
    s = make_splats(n=2, opacity_logit=[-20.0, -20.0])
    assert export_splat(s, tmp_path / "c.splat") == 0
    assert (tmp_path / "c.splat").read_bytes() == b""


@pytest.mark.parametrize("k", [0, 3, 8])
def test_ply_header_and_size(tmp_path, k):
    n = 5
    s = make_splats(n=n, opacity_logit=[-20.0] * n, k=k)  # no filtering for ply
    assert export_ply(s, tmp_path / "s.ply") == n
    raw = (tmp_path / "s.ply").read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    lines = raw[:end].decode("ascii").splitlines()
    assert lines[0] == "ply" and lines[1] == "format binary_little_endian 1.0"
    assert f"element vertex {n}" in lines
    props = [l.split()[-1] for l in lines if l.startswith("property float ")]
    expected = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
                + [f"f_rest_{i}" for i in range(3 * k)]
                + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    assert props == expected
    assert len(raw) - end == n * len(expected) * 4
    body = np.frombuffer(raw[end:], "<f4").reshape(n, len(expected))
    assert np.allclose(body[:, :3], s.means.numpy())
    assert np.all(body[:, 3:6] == 0)
    assert np.allclose(body[:, 6:9], s.sh0[:, 0].numpy())
    if k:
        # channel-major: f_rest_{c*K+j} = shN[:, j, c]
        rest = body[:, 9:9 + 3 * k].reshape(n, 3, k)
        assert np.allclose(rest, s.shN.permute(0, 2, 1).numpy())
    o = 9 + 3 * k
    assert np.allclose(body[:, o], s.opacities.numpy())
    assert np.allclose(body[:, o + 1:o + 4], s.scales.numpy())
    assert np.allclose(body[:, o + 4:o + 8], s.quats.numpy())


@pytest.mark.parametrize("k", [0, 3])
def test_ply_roundtrip(tmp_path, k):
    s = make_splats(n=7, opacity_logit=list(np.linspace(-4, 4, 7)), k=k)
    export_ply(s, tmp_path / "r.ply")
    t = load_ply(tmp_path / "r.ply")
    for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        a, b = getattr(s, name), getattr(t, name)
        assert b.shape == a.shape and b.dtype == torch.float32, name
        assert torch.allclose(a, b), name
    assert t.sh_degree == s.sh_degree
