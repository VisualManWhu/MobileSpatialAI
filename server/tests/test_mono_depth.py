import pytest
import torch

from recon3d.recon.mono_depth import _input_size, pearson_patch_loss, predict_disparity


def test_input_size_keeps_aspect_and_patch_multiple():
    h, w = _input_size(540, 960)
    assert h == 518 and w % 14 == 0 and abs(w / h - 960 / 540) < 0.03
    assert _input_size(960, 540) == (w, h)


def test_predict_disparity_resizes_and_normalises():
    seen = {}

    def fake(x):
        seen["shape"] = tuple(x.shape)
        seen["mean"] = float(x.mean())
        return x[:, 0, ::2, ::2]  # any [B,h',w'] map

    imgs = torch.full((5, 54, 96, 3), 0.5)
    d = predict_disparity(imgs, "cpu", batch=2, model=fake)
    assert d.shape == (5, 54, 96) and d.dtype == torch.float16
    h, w = _input_size(54, 96)
    assert seen["shape"] == (1, 3, h, w)  # last batch of 5 with batch=2
    # 0.5 normalised with ImageNet mean/std is near zero on average.
    assert abs(seen["mean"]) < 0.5
    assert torch.isfinite(d.float()).all()


def test_pearson_loss_zero_for_affine_match_and_positive_otherwise():
    g = torch.Generator().manual_seed(0)
    depth = torch.rand(128, 192, generator=g) * 4 + 0.5
    mask = torch.ones(128, 192, dtype=torch.bool)
    # disparity = 3 / depth + 2: perfectly (affinely) correlated with 1/depth.
    good = pearson_patch_loss(depth, 3.0 / depth + 2.0, mask)
    assert good is not None and float(good) < 1e-4
    # Anti-correlated (depth itself) scores near 2 per patch is not required, but must be clearly worse.
    bad = pearson_patch_loss(depth, torch.rand(128, 192, generator=g), mask)
    assert float(bad) > 0.5


def test_pearson_loss_respects_mask_and_fill():
    depth = torch.rand(64, 128) + 0.5
    disp = 1.0 / depth
    mask = torch.zeros(64, 128, dtype=torch.bool)
    assert pearson_patch_loss(depth, disp, mask) is None  # nothing masked
    mask[:, :64] = True  # left tile full, right tile empty
    disp[:, 64:] = torch.rand(64, 64)  # garbage only where unmasked
    loss = pearson_patch_loss(depth, disp, mask)
    assert loss is not None and float(loss) < 1e-4
    # Partial border (not a multiple of the patch) is dropped, not an error.
    assert pearson_patch_loss(depth[:70, :100], disp[:70, :100], mask[:70, :100]) is not None
    # Too small for one patch.
    assert pearson_patch_loss(depth[:10, :10], disp[:10, :10], mask[:10, :10]) is None


def test_pearson_loss_backprops_to_depth():
    depth = (torch.rand(64, 64) + 0.5).requires_grad_(True)
    target = torch.rand(64, 64)
    loss = pearson_patch_loss(depth, target, torch.ones(64, 64, dtype=torch.bool))
    loss.backward()
    assert depth.grad is not None and torch.isfinite(depth.grad).all() and depth.grad.abs().sum() > 0


@pytest.mark.gpu
def test_real_model_runs_if_cached():
    from huggingface_hub import try_to_load_from_cache

    from recon3d.recon.mono_depth import MODEL_ID

    if not try_to_load_from_cache(MODEL_ID, "config.json"):
        pytest.skip("Depth Anything V2 Small not in the Hugging Face cache")
    # A synthetic gradient has no semantics the net can read, so only check shape, finiteness and that the
    # map is not constant (the model ran and produced a relief).
    g = torch.Generator().manual_seed(0)
    img = torch.rand(1, 270, 480, 3, generator=g)
    d = predict_disparity(img, "cuda").float()
    assert d.shape == (1, 270, 480) and torch.isfinite(d).all()
    assert float(d.std()) > 1e-3
