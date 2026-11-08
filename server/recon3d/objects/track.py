"""Object identity from SAM 2 video tracking seeded by Grounding DINO proposals (spec 2026-11-08 §4.2).

Frames stream through one `Sam2VideoModel` session; every PROPOSE_EVERY-th frame the proposer suggests
boxes, and a box that does not overlap a live track becomes a new tracked object. A track is closed
after LOST_AFTER consecutive empty masks. The transformers session cannot drop an object, so the
session is rebuilt every RESET_EVERY frames with only the live tracks re-seeded from their last masks."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from recon3d.objects.detect import Detector, DownloadError, box_iou
from recon3d.objects.segments import Track, pack_mask

log = logging.getLogger(__name__)

PROPOSE_EVERY = 10
PROPOSE_THRESHOLD = 0.25
NEW_TRACK_MAX_IOU = 0.5
VOTE_MIN_IOU = 0.5
LOST_AFTER = 15
RESET_EVERY = 100  # never on a proposal frame: the session takes one batch of new inputs per frame
MIN_MASK_PX = 16  # at mask resolution; smaller is "lost" this frame


def iou_xyxy(a, b) -> float:
    return box_iou(a, b)


def _mask_box(mask: np.ndarray, scale_x: float, scale_y: float) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min() * scale_x), float(ys.min() * scale_y), float((xs.max() + 1) * scale_x), float((ys.max() + 1) * scale_y))


class _Live:
    __slots__ = ("track", "obj_id", "lost", "last_mask", "last_box")

    def __init__(self, track: Track, obj_id: int):
        self.track = track
        self.obj_id = obj_id
        self.lost = 0
        self.last_mask: np.ndarray | None = None
        self.last_box: tuple | None = None


class Tracker:
    def __init__(self, device: str, detector_id: str, sam_id: str):
        self.device = device
        self.sam_id = sam_id
        self.detector = Detector(device, box_threshold=PROPOSE_THRESHOLD, model_id=detector_id, sam_id="")
        self._model = None
        self._proc = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        try:
            self._proc = Sam2VideoProcessor.from_pretrained(self.sam_id)
            self._model = Sam2VideoModel.from_pretrained(self.sam_id, dtype=torch.bfloat16).to(self.device).eval()
        except (OSError, ValueError) as e:
            self.close()
            raise DownloadError(str(e)) from e
        self._torch = torch

    def close(self) -> None:
        self.detector.close()
        self._model = self._proc = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    # --- session helpers ---

    def _new_session(self):
        return self._proc.init_video_session(inference_device=self.device, dtype=self._torch.bfloat16)

    # The session keeps ONE set of "objects with new inputs" per call (the processor overwrites it),
    # so every object seeded on a frame must go in through a single call.

    def _seed_boxes(self, session, frame_idx: int, obj_ids: list[int], boxes: list, full_hw) -> None:
        self._proc.add_inputs_to_inference_session(
            session, frame_idx=frame_idx, obj_ids=obj_ids, input_boxes=[[list(b) for b in boxes]], original_size=full_hw
        )

    def _seed_masks(self, session, frame_idx: int, obj_ids: list[int], masks: list[np.ndarray], full_hw) -> None:
        session.video_height, session.video_width = full_hw
        self._proc.add_inputs_to_inference_session(
            session, frame_idx=frame_idx, obj_ids=obj_ids, input_masks=[m.astype(np.float32) for m in masks]
        )

    def _step(self, session, pixel_values, mask_hw) -> dict[int, np.ndarray]:
        """Propagate one streamed frame; returns obj_id -> bool mask at mask resolution."""
        out = self._model(inference_session=session, frame=pixel_values)
        masks = self._proc.post_process_masks([out.pred_masks], original_sizes=[mask_hw], binarize=True)[0]
        masks = masks.reshape(-1, mask_hw[0], mask_hw[1]).cpu().numpy().astype(bool)
        return {session.obj_idx_to_id(i): masks[i] for i in range(masks.shape[0])}

    # --- main loop ---

    def run(self, image_paths: list[Path], mask_hw: tuple[int, int], on_frame: Callable[[int], None] | None = None) -> list[Track]:
        self._load()
        torch = self._torch
        done: list[Track] = []
        live: list[_Live] = []
        next_track_id = 1
        session = None
        fed = 0  # frames fed to the current session: the index the next forward() will get
        full_hw = None
        with torch.inference_mode():
            for fi, path in enumerate(image_paths):
                img = Image.open(path).convert("RGB")
                W, H = img.size
                full_hw = (H, W)
                sx, sy = W / mask_hw[1], H / mask_hw[0]
                inputs = self._proc(images=img, device=self.device, return_tensors="pt")
                pixel = inputs.pixel_values[0]
                propose = fi % PROPOSE_EVERY == 0
                # periodic rebuild (never on a proposal frame): drop dead objects; re-seed only the tracks
                # seen in the previous frame (a mask lost for several frames would stick to whatever is
                # there now), the rest end here.
                if session is None or (fed >= RESET_EVERY and not propose):
                    session = self._new_session()
                    fed = 0
                    seeded = [lv for lv in live if lv.lost == 0 and lv.last_mask is not None]
                    for lv in live:
                        if lv not in seeded and lv.track.frames:
                            done.append(lv.track)
                    live = seeded
                    if seeded:
                        self._seed_masks(session, 0, [lv.obj_id for lv in seeded], [lv.last_mask for lv in seeded], full_hw)
                # proposals
                proposals = self.detector.detect_boxes(img) if propose else []
                new_ids: list[int] = []
                new_boxes: list = []
                new_live: list[_Live] = []
                for label, score, box in proposals:
                    matched = None
                    for lv in live:
                        if lv.last_box is not None and box_iou(box, lv.last_box) >= VOTE_MIN_IOU:
                            matched = lv
                            break
                    if matched is not None:
                        matched.track.votes[label] = matched.track.votes.get(label, 0.0) + score
                        continue
                    if any(lv.last_box is not None and box_iou(box, lv.last_box) >= NEW_TRACK_MAX_IOU for lv in live):
                        continue
                    # a second box of this frame on the same thing (cross-class duplicate the NMS let through)
                    dup = next((lv for lv, nb in zip(new_live, new_boxes) if box_iou(box, nb) >= NEW_TRACK_MAX_IOU), None)
                    if dup is not None:
                        dup.track.votes[label] = dup.track.votes.get(label, 0.0) + score
                        continue
                    t = Track(next_track_id, [], [], [], {label: score})
                    lv = _Live(t, next_track_id)
                    next_track_id += 1
                    new_ids.append(lv.obj_id)
                    new_boxes.append(box)
                    new_live.append(lv)
                    live.append(lv)
                if new_ids:
                    # seeds refer to the frame the next forward() will add: index `fed`, not the capture position
                    self._seed_boxes(session, fed, new_ids, new_boxes, full_hw)
                if not live:
                    if on_frame:
                        on_frame(fi)
                    continue
                masks = self._step(session, pixel, mask_hw)
                fed += 1
                still: list[_Live] = []
                for lv in live:
                    m = masks.get(lv.obj_id)
                    if m is None or m.sum() < MIN_MASK_PX:
                        lv.lost += 1
                        if lv.lost >= LOST_AFTER:
                            if lv.track.frames:
                                done.append(lv.track)
                            continue
                        still.append(lv)
                        continue
                    lv.lost = 0
                    lv.last_mask = m
                    lv.last_box = _mask_box(m, sx, sy)
                    lv.track.frames.append(fi)
                    lv.track.masks.append(pack_mask(m))
                    lv.track.boxes.append(lv.last_box)
                    still.append(lv)
                live = still
                if on_frame:
                    on_frame(fi)
        done.extend(lv.track for lv in live if lv.track.frames)
        done.sort(key=lambda t: t.id)
        log.info("tracker: %d frames -> %d tracks", len(image_paths), len(done))
        return done
