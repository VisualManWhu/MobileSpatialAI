import pytest


def pytest_collection_modifyitems(config, items):
    gpu_items = [it for it in items if "gpu" in it.keywords]
    if not gpu_items:
        return
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if has_cuda:
        return
    skip = pytest.mark.skip(reason="needs CUDA (torch.cuda.is_available() is False)")
    for it in gpu_items:
        it.add_marker(skip)


@pytest.fixture(autouse=True)
def _cheap_mono_depth(monkeypatch):
    """The profiles run Depth Anything V2 before training by default; tests swap it for a smooth map.

    Loading the real model takes ~16 s and needs the Hugging Face cache; test_mono_depth.py exercises it
    on its own. Only the trainer's reference is patched, so recon3d.recon.mono_depth itself is untouched.
    """
    try:
        import torch

        import recon3d.recon.trainer as tr
    except ImportError:
        return

    def fake(images, device):
        n, H, W = images.shape[:3]
        return torch.linspace(1, 2, H)[None, :, None].expand(n, H, W).to(torch.float16).contiguous()

    monkeypatch.setattr(tr, "predict_disparity", fake)
