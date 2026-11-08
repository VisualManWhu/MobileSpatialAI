import pytest
from PIL import Image, ImageDraw

from recon3d.objects.vlm import grid


def test_grid_layout_pastes_crops_in_one_row():
    imgs = [Image.new("RGB", (100, 50), "red"), Image.new("RGB", (30, 90), "green"), Image.new("RGB", (64, 64), "blue")]
    g = grid(imgs, cell=384)
    assert g.size == (3 * 384, 384)
    assert grid(imgs[:1], cell=128).size == (128, 128)


@pytest.mark.gpu
@pytest.mark.slow
def test_qwen_3b_answers_a_shape_question():
    from recon3d.objects.vlm import QwenVlm

    img = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(img).ellipse([48, 48, 208, 208], fill="red")
    vlm = QwenVlm("Qwen/Qwen2.5-VL-3B-Instruct", four_bit=True)
    try:
        answer = vlm.ask([img], "图中的红色图形是圆形还是方形？只回答一个词。")
    finally:
        vlm.close()
    assert "圆" in answer, answer
