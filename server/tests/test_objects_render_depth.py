import numpy as np
import pytest
import torch

from recon3d.objects.render_depth import read_splat_positions, render_depths
from recon3d.recon.export_splat import export_ply, export_splat
from recon3d.recon.trainer import Splats


def wall_splats(n=400, z=-2.0):
    g = torch.linspace(-1, 1, 20)
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    means = torch.stack([xx.flatten(), yy.flatten(), torch.full((n,), z)], 1)
    return Splats(
        means=means,
        scales=torch.full((n, 3), float(np.log(0.08))),
        quats=torch.tensor([[1.0, 0, 0, 0]] * n),
        opacities=torch.full((n,), 4.0),
        sh0=torch.zeros(n, 1, 3),
        shN=torch.zeros(n, 0, 3),
    )


def test_read_splat_positions_matches_export_order(tmp_path):
    s = wall_splats()
    export_splat(s, tmp_path / "scene.splat")
    pos = read_splat_positions(tmp_path / "scene.splat")
    assert pos.shape == (400, 3) and pos.dtype == np.float32
    assert set(map(tuple, np.round(pos, 4))) == set(map(tuple, np.round(s.means.numpy(), 4)))


@pytest.mark.gpu
def test_rendered_depth_of_a_wall_is_its_distance(tmp_path):
    export_ply(wall_splats(), tmp_path / "scene.ply")
    # OpenCV camera (+Z forward) at the origin, turned to look down world -Z at the wall z=-2.
    c2w = np.eye(4)[None]
    c2w[0, :3, :3] = np.diag([1, -1, -1])
    K = np.array([[[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]]])
    depths = list(render_depths(tmp_path / "scene.ply", c2w, K, 320, 240, scale=0.5))
    assert len(depths) == 1 and depths[0].shape == (120, 160)
    centre = depths[0][50:70, 70:90]
    assert np.all(centre > 0) and np.allclose(centre, 2.0, atol=0.1)
