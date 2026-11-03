import numpy as np
import torch

from recon3d.recon.offtrack import contact_sheet, pick_views, raise_cameras


def test_raise_cameras_moves_world_y_only():
    c2w = torch.eye(4).repeat(3, 1, 1); c2w[:, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    out = raise_cameras(c2w, 0.5)
    assert torch.allclose(out[:, :3, 3], torch.tensor([1.0, 2.5, 3.0]).expand(3, 3))
    assert torch.equal(out[:, :3, :3], c2w[:, :3, :3]) and torch.equal(c2w[:, 1, 3], torch.full((3,), 2.0))


def test_pick_views_even_and_distinct():
    assert pick_views(100, 4) == [0, 33, 66, 99]
    assert pick_views(3, 12) == [0, 1, 2]
    assert pick_views(0, 12) == []


def test_contact_sheet_pads_last_row():
    tiles = [np.full((2, 3, 3), v, np.uint8) for v in (10, 20, 30, 40, 50)]
    sheet = contact_sheet(tiles, cols=4)
    assert sheet.shape == (4, 12, 3)
    assert sheet[0, 0, 0] == 10 and sheet[0, 11, 0] == 40
    assert sheet[2, 0, 0] == 50 and sheet[2, 11, 0] == 0
