"""Open-vocabulary detection (Grounding DINO tiny) + box-prompted segmentation (SAM 2.1 small).

Both models load lazily on the first detect() and are released by close(). Weights come from the
Hugging Face hub on first use (~1 GB); a failed download raises DownloadError."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from recon3d.objects.merge import Detection, nms_frame
from recon3d.objects.vocab import label_of, prompt_batches

log = logging.getLogger(__name__)

GDINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_ID = "facebook/sam2.1-hiera-small"


class DownloadError(Exception):
    """Model weights could not be fetched (network / hub problem)."""


@dataclass
class FrameDetection:
    label: str  # vocab entry (English)
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 pixels
    mask: np.ndarray  # [H, W] bool


class Detector:
    def __init__(self, device: str = "cuda", box_threshold: float = 0.35, text_threshold: float = 0.25, nms_iou: float = 0.7):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou = nms_iou
        self._gd = None
        self._gd_proc = None
        self._sam = None
        self._sam_proc = None

    # --- loading ---

    def _load(self) -> None:
        if self._gd is not None:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

        try:
            self._gd_proc = AutoProcessor.from_pretrained(GDINO_ID)
            self._gd = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(self.device).eval()
            self._sam_proc = Sam2Processor.from_pretrained(SAM2_ID)
            self._sam = Sam2Model.from_pretrained(SAM2_ID).to(self.device).eval()
        except (OSError, ValueError) as e:  # hub errors surface as OSError (connection, 404, timeout)
            self.close()
            raise DownloadError(str(e)) from e
        self._torch = torch

    def close(self) -> None:
        self._gd = self._gd_proc = self._sam = self._sam_proc = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    # --- inference ---

    def detect(self, image: Image.Image) -> list[FrameDetection]:
        self._load()
        torch = self._torch
        W, H = image.size
        cands: list[Detection] = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device != "cpu"):
            for text in prompt_batches():
                inp = self._gd_proc(images=image, text=text, return_tensors="pt").to(self.device)
                out = self._gd(**inp)
                res = self._gd_proc.post_process_grounded_object_detection(
                    out, inp.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold, target_sizes=[(H, W)]
                )[0]
                phrases = res["text_labels"] if "text_labels" in res else res["labels"]
                for phrase, score, box in zip(phrases, res["scores"].tolist(), res["boxes"].tolist()):
                    label = label_of(str(phrase))
                    if label is None:
                        continue
                    cands.append(Detection(0, label, float(score), tuple(float(v) for v in box), np.zeros((0, 3))))
        kept = nms_frame(cands, self.nms_iou)
        if not kept:
            return []
        boxes = [list(d.box) for d in kept]
        with torch.inference_mode():
            inp = self._sam_proc(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)
            out = self._sam(**inp, multimask_output=False)
            masks = self._sam_proc.post_process_masks(out.pred_masks, inp["original_sizes"])[0]
        masks = masks.reshape(len(kept), -1, H, W)[:, 0] > 0
        masks_np = masks.cpu().numpy().astype(bool)
        return [FrameDetection(d.label, d.score, d.box, masks_np[i]) for i, d in enumerate(kept)]
