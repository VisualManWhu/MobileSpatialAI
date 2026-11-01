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
