"""Qwen2.5-VL client for label confirmation (spec 2026-11-08 §4.3): a few crops of one object pasted
into a single row plus a Chinese question; greedy decoding, short answer."""
from __future__ import annotations

import logging

from PIL import Image

from recon3d.objects.detect import DownloadError

log = logging.getLogger(__name__)

MAX_NEW_TOKENS = 16
GRID_CELL = 384


def grid(images: list[Image.Image], cell: int = GRID_CELL) -> Image.Image:
    """Pastes up to five crops side by side, each letterboxed into a cell x cell square on white."""
    images = list(images)[:5] or [Image.new("RGB", (cell, cell), "white")]
    out = Image.new("RGB", (cell * len(images), cell), "white")
    for i, im in enumerate(images):
        im = im.convert("RGB")
        scale = min(cell / im.width, cell / im.height)
        w, h = max(1, round(im.width * scale)), max(1, round(im.height * scale))
        thumb = im.resize((w, h), Image.BILINEAR)
        out.paste(thumb, (i * cell + (cell - w) // 2, (cell - h) // 2))
    return out


class QwenVlm:
    """verify.Vlm implementation. four_bit uses bitsandbytes NF4 (laptop profile)."""

    def __init__(self, model_id: str, device: str = "cuda", four_bit: bool = False):
        self.model_id = model_id
        self.device = device
        self.four_bit = four_bit
        self._model = None
        self._proc = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        try:
            kwargs: dict = {"device_map": self.device}
            if self.four_bit:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                kwargs["dtype"] = torch.bfloat16
            self._proc = AutoProcessor.from_pretrained(self.model_id, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs).eval()
        except (OSError, ValueError) as e:
            self.close()
            raise DownloadError(str(e)) from e
        self._torch = torch

    def close(self) -> None:
        self._model = self._proc = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def ask(self, images: list[Image.Image], prompt: str) -> str:
        self._load()
        torch = self._torch
        image = grid(images)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self._proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._proc(text=[text], images=[image], return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1] :]
        answer = self._proc.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        log.debug("vlm: %r -> %r", prompt[-60:], answer)
        return answer
