"""Open-vocabulary detection (Grounding DINO tiny) + box-prompted segmentation (SAM 2.1 small).

Both models load lazily on the first detect() and are released by close(). Weights come from the
Hugging Face hub on first use (~1 GB); a failed download raises DownloadError."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from recon3d.objects.vocab import label_of, prompt_batches

log = logging.getLogger(__name__)

GDINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_ID = "facebook/sam2.1-hiera-small"


class DownloadError(Exception):
    """Model weights could not be fetched (network / hub problem)."""


BoxDet = tuple[str, float, tuple[float, float, float, float]]  # (vocab label, score, x0 y0 x1 y1 pixels)


def box_iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(dets: list[BoxDet], iou: float = 0.7) -> list[BoxDet]:
    """Cross-class NMS inside one frame (prompt batches return duplicate boxes): keep the higher score."""
    kept: list[BoxDet] = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if all(box_iou(d[2], k[2]) < iou for k in kept):
            kept.append(d)
    return kept


@dataclass
class FrameDetection:
    label: str  # vocab entry (English)
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 pixels
    mask: np.ndarray  # [H, W] bool


class Detector:
    def __init__(
        self,
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        nms_iou: float = 0.7,
        model_id: str = GDINO_ID,
        sam_id: str = SAM2_ID,
    ):
        self.device = device
        self.model_id = model_id
        self.sam_id = sam_id
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
            self._gd_proc = AutoProcessor.from_pretrained(self.model_id)
            self._gd = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()
            if self.sam_id:
                self._sam_proc = Sam2Processor.from_pretrained(self.sam_id)
                self._sam = Sam2Model.from_pretrained(self.sam_id).to(self.device).eval()
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

    def detect_boxes(self, image: Image.Image) -> list[BoxDet]:
        """Grounding DINO boxes mapped to vocab labels, NMS'd, no segmentation (the tracker's proposer)."""
        self._load()
        torch = self._torch
        W, H = image.size
        cands: list[BoxDet] = []
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
                    cands.append((label, float(score), tuple(float(v) for v in box)))
        return nms_boxes(cands, self.nms_iou)

    def detect(self, image: Image.Image) -> list[FrameDetection]:
        kept = self.detect_boxes(image)
        if not kept:
            return []
        torch = self._torch
        W, H = image.size
        boxes = [list(d[2]) for d in kept]
        with torch.inference_mode():
            inp = self._sam_proc(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)
            out = self._sam(**inp, multimask_output=False)
            masks = self._sam_proc.post_process_masks(out.pred_masks, inp["original_sizes"])[0]
        masks = masks.reshape(len(kept), -1, H, W)[:, 0] > 0
        masks_np = masks.cpu().numpy().astype(bool)
        return [FrameDetection(label, score, box, masks_np[i]) for i, (label, score, box) in enumerate(kept)]
