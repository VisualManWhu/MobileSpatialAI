# Objects v2 (structure → tracking → VLM verification) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v1 object pipeline (per-frame detection + 3D overlap merging) with a precision-first pipeline: room structure from plane fitting, object identity from SAM 2 video tracking, labels confirmed by size/support constraints and a VLM, Gaussians assigned by voxel hits; deployed on the lab A6000.

**Architecture:** Server `recon3d/objects/` gets `structure.py` (planes, pure numpy), `track.py` (GDINO proposer + SAM 2 video), `segments.py` (track → voxel segment, merge, support; pure numpy), `verify.py` (vocab constraints + VLM two-ask confirmation + harmony filter; VLM injectable), `vlm.py` (Qwen2.5-VL client), rewritten `vocab.py`, `assign.py`, `table.py` (version 2), `pipeline.py`. `merge.py` is deleted. The job model, API, `objects.zip` shape and the phone flow are unchanged; the phone parses the new fields and lists objects in three level sections.

**Tech Stack:** Python 3.10, torch 2.4.1 (pinned by gsplat), transformers 4.57 (`Sam2VideoModel`, `Qwen2_5_VLForConditionalGeneration`), bitsandbytes (laptop 4-bit), numpy/scipy; Kotlin/Compose on Android.

**Spec:** `docs/superpowers/specs/2026-11-08-recon3d-objects-v2-design.md` (supersedes §4.3–4.4 of `2026-11-07-recon3d-objects-design.md`; the rest of that spec still applies).

## Global Constraints

- User-visible strings Chinese; code, comments, commit messages English. Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- torch stays `2.4.1`, transformers `>=4.45,<5` (whatever `uv.lock` resolves; currently 4.57). New Linux-only deps go behind `sys_platform == 'linux'` like torchvision.
- Progress stage strings exactly: `识别物体 (结构)` 0→0.15, `识别物体 (跟踪)` 0.15→0.55, `识别物体 (确认)` 0.55→0.9, `识别物体 (归属)` 0.9→1.0.
- All keyframes participate (no 1/5 sampling). Depth rendered from `scene.ply` at `DEPTH_SCALE = 0.5`, kept in RAM as float16.
- Coordinates: ARCore world, +Y up, poses gravity-aligned. Precise jobs use `poses_cv.npy`, fast jobs `gl_to_cv_c2w(frame.pose)` (unchanged from v1).
- Models: `server` profile → GDINO `IDEA-Research/grounding-dino-base`, SAM `facebook/sam2.1-hiera-large`, VLM `Qwen/Qwen2.5-VL-7B-Instruct` bf16, all resident if ≥ 30 GB free, else serial; `laptop` → `grounding-dino-tiny`, `sam2.1-hiera-small`, `Qwen/Qwen2.5-VL-3B-Instruct` 4-bit, always serial.
- Everything released from the GPU when the run ends (reconstruction needs the card).
- Server tests: `wsl -d Ubuntu -- bash -lc 'cd /mnt/d/projects/Recon3D/server && export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/recon3d UV_LINK_MODE=copy && uv run --no-sync pytest -q -m "not gpu and not slow"'` must stay green (currently 211). Android: `cd D:\projects\Recon3D\android; $env:JAVA_HOME="C:\Program Files\Android\Android Studio\jbr"; .\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --offline`.
- Work on `main` directly (user's standing choice for this repo).

## Review Focus

1. A capture with no detectable floor (camera never looked down, or an outdoor capture): `fit_room` must not crash and the pipeline must still produce objects, with `support` falling back to `none` and the harmony "outside the walls" rule disabled when there are < 2 walls. (Test in Task 2 and Task 4.)
2. A VLM answer that is a candidate name wrapped in extra text (`「门」`, `这是门。`): accept the exact-match-after-stripping-punctuation case, reject anything that is not exactly one candidate. (Test in Task 4.)
3. Two tracks of the same object that share exactly one frame because the proposer re-fired on a still-tracked object with IoU just under 0.5: they must stay separate (spec rule) and the harmony furniture-IoU rule must then collapse them. (Test in Task 3 and Task 4.)
4. A track whose every mask has no rendered depth (object seen only through glass / a mirror): zero voxels → dropped, never a NaN bbox. (Test in Task 3.)
5. An old `objects.json` version 1 on the phone (table downloaded before this change): still opens, all objects listed under 「物品」. (Test in Task 11.)

---

### Task 1: Vocabulary v2 with levels, size ranges and supports

**Files:**
- Rewrite: `server/recon3d/objects/vocab.py`
- Test: `server/tests/test_objects_vocab_table.py` (extend the vocab part; the table tests move to Task 8)

**Interfaces:**
- Produces:
  ```python
  Level = Literal["structure", "furniture", "item"]
  @dataclass(frozen=True)
  class VocabEntry:
      en: str; zh: str; level: Level
      h_min: float; h_max: float      # height range, metres
      l_min: float; l_max: float      # longest horizontal side range, metres
      supports: frozenset[str]        # subset of {"floor", "wall", "ceiling", "on"}
  VOCAB: list[VocabEntry]            # ~60 entries, every v1 entry kept (same en/zh)
  def zh_label(en: str) -> str
  def entry(en: str) -> VocabEntry
  def prompt_batches(max_per_batch: int = 20) -> list[str]   # unchanged semantics
  def label_of(phrase: str) -> str | None                     # unchanged semantics
  def by_level(level: Level) -> list[VocabEntry]
  def candidates(size_lwh: tuple[float, float, float], support: str, level: Level) -> list[VocabEntry]
  ```
  `candidates` returns the entries of `level` whose `h_min ≤ h ≤ h_max`, `l_min ≤ max(l, w) ≤ l_max` and whose `supports` contains `support`'s kind (`on:<id>` → `"on"`, `wall:<id>` → `"wall"`, `none` matches every entry).

- [ ] **Step 1: Write the failing tests**

```python
def test_v1_vocab_is_preserved():
    v1 = {"cup", "bottle", "book", "remote control", "chair", "table", "monitor", "laptop", "keyboard",
          "mouse", "phone", "box", "bag", "lamp", "plant", "whiteboard", "door", "window", "trash can",
          "pen", "sofa", "bed", "pillow", "shelf", "cabinet", "tv", "speaker", "headphones", "clock",
          "picture frame", "curtain", "mirror", "fan", "printer", "backpack", "shoes", "towel", "sink",
          "refrigerator", "microwave"}
    assert v1 <= {e.en for e in VOCAB}
    assert 55 <= len(VOCAB) <= 70
    assert zh_label("door") == "门" and entry("door").level == "structure"

def test_candidates_use_size_and_support():
    # a 2.0 m tall, 0.9 m wide thing flush with a wall: door and window pass the size/support gate
    # (the floor-distance split between them is Task 4's door_window_ok), phone does not
    c = {e.en for e in candidates((0.9, 0.1, 2.0), "wall:1", "structure")}
    assert {"door", "window"} <= c
    assert "phone" not in {e.en for e in candidates((0.9, 0.1, 2.0), "floor", "item")}
    # a 15 cm flat thing on a table: phone yes, remote yes, monitor no
    c = {e.en for e in candidates((0.15, 0.07, 0.01), "on:3", "item")}
    assert {"phone", "remote control"} <= c and "monitor" not in c
    # "none" support matches everything that fits the size
    assert "chair" in {e.en for e in candidates((0.5, 0.5, 0.9), "none", "furniture")}

def test_every_entry_has_sane_ranges():
    for e in VOCAB:
        assert 0 < e.h_min < e.h_max and 0 < e.l_min < e.l_max and e.supports
```
- [ ] **Step 2: Run tests → FAIL** (`ImportError: cannot import name 'VocabEntry'`).
- [ ] **Step 3: Rewrite `vocab.py`.** Keep the v1 40, add ~20 common indoor classes (desk, stool, bench, nightstand, wardrobe, bookshelf, coffee table, armchair, radiator, air conditioner, ceiling light, floor lamp, router, tablet, water dispenser, kettle, bowl, vase, basket, tissue box). Levels: door/window `structure`; furniture = table, desk, chair, stool, bench, sofa, armchair, bed, nightstand, cabinet, wardrobe, shelf, bookshelf, coffee table, refrigerator, whiteboard, tv, curtain, mirror, printer, trash can, floor lamp, radiator, air conditioner, water dispenser, sink; everything else `item`. Size/support values per the spec examples (door 1.8–2.4 h, 0.6–1.2 l, floor+wall; chair 0.7–1.3 h, 0.4–0.8 l, floor; phone 0.005–0.2 h, 0.12–0.2 l, on+floor; plant 0.1–2.0 h, floor+on; window 0.4–2.2 h, 0.3–3.0 l, wall), the rest by common sense.
- [ ] **Step 4: Run the vocab tests and the existing `test_objects_detect.py` label tests → PASS.**
- [ ] **Step 5: Commit** `feat(objects): vocabulary v2 with levels, size ranges and supports`.

---

### Task 2: Room structure from plane fitting

**Files:**
- Create: `server/recon3d/objects/structure.py`
- Test: `server/tests/test_objects_structure.py`

**Interfaces:**
- Consumes: depth maps `[h, w]` float (0 = none), `Ks_small [n,3,3]`, `c2w_cv [n,4,4]` (same conventions as v1 `_backproject` in `pipeline.py`).
- Produces:
  ```python
  @dataclass(frozen=True)
  class Wall:
      id: int; normal: np.ndarray  # unit, horizontal, pointing into the room
      d: float                     # plane: normal · p + d = 0
      u: np.ndarray                # horizontal in-plane unit axis
      extent: tuple[float, float, float, float]  # u_min, u_max, y_min, y_max
  @dataclass(frozen=True)
  class Room:
      floor_y: float | None; ceiling_y: float | None; walls: list[Wall]
      def wall_distance(self, p: np.ndarray, wall: Wall) -> np.ndarray      # signed, metres
      def inside_xz(self, p: np.ndarray, margin_m: float) -> np.ndarray     # bool; all True when < 2 walls
      def to_json(self) -> dict  # {"floor_y", "ceiling_y", "walls": [{"id","normal","d","extent"}]}
  def depth_point_cloud(depths, Ks_small, c2w_cv, voxel_m=0.03, stride=4) -> np.ndarray  # [m,3] world
  def fit_room(points: np.ndarray, cam_positions: np.ndarray, *, min_floor_area_m2=2.0,
               min_wall_area_m2=1.5, inlier_m=0.03, horizontal_deg=5.0, vertical_deg=85.0,
               merge_deg=10.0, merge_dist_m=0.3, seed=0) -> Room
  ```
  Plane area = inlier count × voxel_m². Floor = lowest horizontal plane with enough area; ceiling = highest horizontal plane above every camera; walls sorted by area desc, merged when normals within `merge_deg` and offsets within `merge_dist_m`.

- [ ] **Step 1: Write failing tests** on a synthetic box room: sample points on floor y=0 (4×5 m), ceiling y=2.6, four walls, plus 500 random interior noise points; cameras at y=1.4.
  - `test_fit_room_finds_floor_ceiling_and_four_walls`: `floor_y ≈ 0 (±0.02)`, `ceiling_y ≈ 2.6`, `len(walls) == 4`, every wall normal points inward (dot with vector from wall to room centre > 0).
  - `test_no_floor_in_view_yields_none`: drop the floor points → `floor_y is None`, walls still 4.
  - `test_inside_xz_margin`: centre → True; 0.5 m outside a wall → False with margin 0.3; `Room(None, None, [])` → all True.
  - `test_depth_point_cloud_backprojects`: a flat depth map of 2.0 m, identity pose → all points z≈2.0 in camera-OpenCV coordinates converted through c2w; voxel dedup shrinks count.
- [ ] **Step 2: Run → FAIL** (module not found).
- [ ] **Step 3: Implement.** RANSAC: sample 3 points, fit plane, count inliers within `inlier_m`, 300 iterations per plane, peel inliers and repeat until the best plane's area is below the threshold. Horizontal/vertical selection by the normal's angle to +Y.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `feat(objects): room structure (floor, ceiling, walls) from rendered-depth point cloud`.

---

### Task 3: Tracks → voxel segments, merging, support

**Files:**
- Create: `server/recon3d/objects/segments.py`
- Test: `server/tests/test_objects_segments.py`

**Interfaces:**
- Consumes: `Room` (Task 2); masks are bit-packed depth-resolution bool arrays (`np.packbits`, as v1 did).
- Produces:
  ```python
  VOXEL_M = 0.02; MIN_TRACK_FRAMES = 5; MIN_VOXELS = 20; MIN_HITS = 3; MERGE_OVERLAP = 0.5
  @dataclass
  class Track:                       # produced by track.py (Task 5); defined here so tests need no GPU
      id: int
      frames: list[int]              # frame indices, ascending
      masks: list[np.ndarray]        # packed bits, depth resolution, parallel to frames
      boxes: list[tuple[float, float, float, float]]  # full-res pixels, parallel to frames
      votes: dict[str, float]        # proposer label -> summed score over the track
  @dataclass
  class Segment:
      track_ids: list[int]; frames: list[int]
      voxels: np.ndarray             # [k,3] int64 voxel coords
      bbox_min: np.ndarray; bbox_max: np.ndarray   # 2–98 percentile of voxel centres
      size_lwh: tuple[float, float, float]         # horizontal min-area rectangle (l ≥ w), height
      votes: dict[str, float]
      best_views: list[tuple[int, tuple]]          # (frame, box) of up to 5 best views
      def volume(self) -> float
  def track_to_segment(track: Track, depths, Ks_small, c2w_cv, h: int, w: int) -> Segment | None
  def merge_segments(segments: list[Segment]) -> list[Segment]
  def support_of(seg: Segment, room: Room, others: list[Segment]) -> str   # "floor"|"ceiling"|"on:<i>"|"wall:<id>"|"none"
  def voxel_key(points: np.ndarray) -> np.ndarray   # floor(p / VOXEL_M) int64 [k,3]
  ```
  `others` are segments with larger volume; `on:<i>` uses the index in `others`. Best views: top 5 by mask area, greedily skipping a frame whose camera direction is within 20° of an already chosen one.

- [ ] **Step 1: Write failing tests** with synthetic depth maps (a 0.4 m box at 2 m in front of a camera, 3 cameras) :
  - `test_track_to_segment_keeps_voxels_hit_in_three_frames`: voxels on the box faces visible in all 3 frames survive; an extra frame whose mask covers empty background adds nothing; `size_lwh ≈ (0.4, 0.4, 0.4)` ± 0.06.
  - `test_track_without_depth_is_dropped`: depth all zero → `None` (Review Focus 4).
  - `test_short_track_is_dropped`: 4 frames → `None`.
  - `test_merge_requires_no_shared_frame_and_half_overlap`: same voxels, disjoint frames → merged (frames union, votes summed); same voxels but one shared frame → not merged (Review Focus 3); disjoint frames but 30 % overlap → not merged.
  - `test_support_floor_on_wall_none`: box on floor_y → `floor`; box with bottom 5 cm above a bigger segment's top and inside its XZ → `on:0`; thin segment with 70 % voxels within 15 cm of a wall → `wall:<id>`; floating box → `none`; `Room(None, None, [])` → `none`.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** (back-projection as v1 `_backproject`, counting hits per voxel with `np.unique(..., return_counts=True)`; min-area rectangle via `scipy.spatial.ConvexHull` + rotating calipers or a 1° sweep).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `feat(objects): voxel segments from tracks, merging and support relations`.

---

### Task 4: Verification layer (candidates, VLM two-ask, harmony)

**Files:**
- Create: `server/recon3d/objects/verify.py`
- Test: `server/tests/test_objects_verify.py`

**Interfaces:**
- Consumes: `VocabEntry`, `candidates` (Task 1); `Segment`, `support_of` (Task 3); `Room` (Task 2).
- Produces:
  ```python
  class Vlm(Protocol):
      def ask(self, images: list[Image.Image], prompt: str) -> str: ...
  PROMPT = ("这是同一个室内物体的几张照片。它高约 {h:.2f} 米、最长边约 {l:.2f} 米，{where}。"
            "它是下列哪一个：{options}？只回答其中一个名称；都不是或看不清就回答「不确定」。")
  WHERE = {"floor": "放在地面上", "on": "放在家具上", "wall": "贴墙", "ceiling": "装在天花板上", "none": "位置不明"}
  def candidate_entries(seg: Segment, support: str, level: Level, top_k: int = 5) -> list[VocabEntry]
      # vocab.candidates(...) ordered by seg.votes (desc), unvoted ones after, truncated to top_k
  def parse_answer(text: str, options: list[VocabEntry]) -> VocabEntry | None
      # strip whitespace and 「」""''。.! ; exact match on zh; else None
  def confirm(seg: Segment, support: str, options: list[VocabEntry], crops: list[Image.Image],
              vlm: Vlm, rng: random.Random) -> VocabEntry | None
      # ask twice, second with shuffled options; both parse to the same entry → it, else None
  @dataclass
  class Verified:
      seg: Segment; entry: VocabEntry; support: str; confidence: float
  def confidence_of(seg: Segment, entry: VocabEntry) -> float   # votes[entry.en] / sum(votes), 0 if no votes
  def harmony_filter(verified: list[Verified], room: Room) -> list[Verified]
  def door_window_ok(seg: Segment, entry: VocabEntry, room: Room) -> bool
  ```
  `harmony_filter` rules (spec §4.3): furniture pairs with bbox IoU > 0.3 → keep larger; item with IoU > 0.5 against a furniture bbox → drop; centre outside `room.inside_xz(margin 0.3)` → drop (no-op when < 2 walls, Review Focus 1); non-door/window with ≥ 80 % voxels within 15 cm of a wall plane and height > 1.5 m → drop. `door_window_ok`: ≥ 80 % voxels within 15 cm of some wall; door: bottom − floor_y < 0.15 (needs floor), h 1.8–2.4, l 0.6–1.2; window: bottom − floor_y ≥ 0.3 (or no floor), h 0.4–2.2.

- [ ] **Step 1: Write failing tests** with a `FakeVlm(answers: list[str])` and hand-built `Segment`s (helper `seg(size, bottom_y, votes, frames)`):
  - `test_candidates_ordered_by_votes_then_truncated`.
  - `test_parse_answer_accepts_wrapped_name_rejects_extra_text`: `「门」` → door; `这是门。` → None; `不确定` → None (Review Focus 2).
  - `test_confirm_needs_two_matching_answers`: `["门", "门"]` → door; `["门", "柜子"]` → None; second call's options differ in order (assert via a recording fake).
  - `test_white_door_never_becomes_cabinet`: a 2.0 m tall, 0.9 m wide, wall-flush segment labelled cabinet by the VLM → removed by `harmony_filter`; same segment as door with bottom on the floor → `door_window_ok` True.
  - `test_furniture_overlap_keeps_larger` (Review Focus 3), `test_item_inside_furniture_dropped`, `test_outside_walls_dropped_only_with_two_walls`.
  - `test_confidence_is_vote_share`.
- [ ] **Step 2: Run → FAIL.**  **Step 3: Implement.**  **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `feat(objects): verification layer — candidates, VLM two-ask confirmation, harmony filter`.

---

### Task 5: Proposer + SAM 2 video tracker

**Files:**
- Create: `server/recon3d/objects/track.py`
- Modify: `server/recon3d/objects/detect.py` (make `GDINO_ID` a constructor arg `model_id`; add `detect_boxes(image) -> list[(label, score, box)]` that skips SAM)
- Test: `server/tests/test_objects_track.py` (gpu)

**Interfaces:**
- Consumes: `Track` (Task 3), `Detector` (existing).
- Produces:
  ```python
  PROPOSE_EVERY = 10; PROPOSE_THRESHOLD = 0.25; NEW_TRACK_MAX_IOU = 0.5; LOST_AFTER = 15
  class Tracker:
      def __init__(self, device: str, detector_id: str, sam_id: str): ...
      def run(self, image_paths: list[Path], mask_hw: tuple[int, int],
              on_frame: Callable[[int], None] | None = None) -> list[Track]
      def close(self) -> None
  ```
  Behaviour: frames in order; every `PROPOSE_EVERY`-th frame run `detector.detect_boxes` (threshold 0.25) and open a new SAM 2 video object for every box whose IoU with every live track's current mask box is < 0.5; propagate all live objects one frame at a time (`Sam2VideoModel` + `Sam2VideoProcessor`: `init_video_session`, `add_inputs_to_inference_session`, `propagate_in_frame` / streaming; if the installed transformers lacks these, fall back to the `sam2` PyPI package — implementer's call, spec §8); a track is closed after `LOST_AFTER` consecutive empty masks; masks stored packed at `mask_hw` (depth resolution); `votes[label] += score` whenever a proposal box with IoU ≥ 0.5 against the track's current mask box fires.

- [ ] **Step 1: Write the gpu test** against the real office capture (skip when `data/jobs/40a80bd6c17f/capture.zip` is absent, like `test_objects_detect.py`): unpack, take the first 60 frames, `Tracker("cuda", "IDEA-Research/grounding-dino-tiny", "facebook/sam2.1-hiera-small").run(paths, (540, 960))`; assert ≥ 3 tracks of ≥ 5 frames, every track's frames strictly increasing, masks unpack to `(540, 960)`, and at least one track has a vote for `monitor` or `laptop` or `keyboard`.
- [ ] **Step 2: Run → FAIL** (no module).  **Step 3: Implement.**
- [ ] **Step 4: Run the gpu test on the laptop → PASS**; note frames/s in the ledger.
- [ ] **Step 5: Commit** `feat(objects): GDINO proposer + SAM 2 video tracker`.

---

### Task 6: Qwen2.5-VL client

**Files:**
- Create: `server/recon3d/objects/vlm.py`
- Modify: `server/pyproject.toml` (add `accelerate>=0.33`, `bitsandbytes>=0.43; sys_platform == 'linux'`)
- Test: `server/tests/test_objects_vlm.py` (gpu, slow)

**Interfaces:**
- Produces:
  ```python
  class QwenVlm:                      # implements verify.Vlm
      def __init__(self, model_id: str, device: str = "cuda", four_bit: bool = False): ...
      def ask(self, images: list[Image.Image], prompt: str) -> str   # max_new_tokens=16, greedy
      def close(self) -> None
  def grid(images: list[Image.Image], cell: int = 384) -> Image.Image   # pastes ≤5 crops into one row
  ```
  Raises `detect.DownloadError` on download failure (the worker already maps it to 模型下载失败).

- [ ] **Step 1: gpu+slow test**: `QwenVlm("Qwen/Qwen2.5-VL-3B-Instruct", four_bit=True)` on the laptop; a synthetic image of a red circle; prompt asks 「这是圆形还是方形？只回答一个词」 → answer contains 圆. `test_grid_layout`: 3 images → width 3×384.
- [ ] **Step 2 → 5** as usual; commit `feat(objects): Qwen2.5-VL client`.

---

### Task 7: Gaussian assignment by voxels and planes

**Files:**
- Rewrite: `server/recon3d/objects/assign.py`
- Test: `server/tests/test_objects_assign.py` (rewrite)

**Interfaces:**
- Produces:
  ```python
  def assign_by_voxels(positions: np.ndarray, segments: list[Segment]) -> np.ndarray   # uint16, id = index+1
  def assign_structure(positions: np.ndarray, room: Room, ids: np.ndarray, first_id: int,
                       openings: list[tuple[int, Segment]]) -> tuple[np.ndarray, list[int]]
  ```
  Voxels: a Gaussian whose `voxel_key` (or any of its 26 neighbours) is in a segment's voxel set belongs to it; several → smallest volume wins; only Gaussians still 0 are touched by `assign_structure`, which assigns floor / ceiling / each wall (|distance| < 5 cm, within extent) and door/window openings (within 15 cm of the wall plane and inside the opening's bbox, assigned *before* the wall). Returns ids and the list of structure ids created in order [floor?, ceiling?, walls…, openings…].

- [ ] **Step 1: Failing tests**: two segments (one inside the other's bbox but smaller) → the smaller wins on shared voxels; a Gaussian 3 cm from a voxel → assigned; 6 cm → not; floor plane Gaussians get the floor id; a door opening Gaussian gets the door id, not the wall id.
- [ ] **Step 2 → 5**; commit `feat(objects): Gaussian assignment by voxel hits and structure planes`.

---

### Task 8: Table version 2

**Files:**
- Modify: `server/recon3d/objects/table.py`, `docs/api.md`
- Test: `server/tests/test_objects_vocab_table.py` (table part)

**Interfaces:**
- `VERSION = 2`; `ObjectRecord` gains `level: str`, `size: tuple[float,float,float]`, `support: str`, `confidence: float`, `thumb` unchanged; `write_objects(out_dir, job_id, frames_used, ids, objects, relations, structure: dict)` writes `structure` at top level.

- [ ] **Step 1: Failing test**: written json has `version == 2`, each object has the four new keys, `structure.floor_y`.
- [ ] **Step 2 → 5**; update `docs/api.md` objects.zip section; commit `feat(objects): objects.json version 2 (level, size, support, confidence, structure)`.

---

### Task 9: Pipeline v2, profile fields, memory policy

**Files:**
- Rewrite: `server/recon3d/objects/pipeline.py`
- Delete: `server/recon3d/objects/merge.py`, `server/tests/test_objects_merge.py`
- Modify: `server/recon3d/config.py` (Profile fields `objects_detector`, `objects_sam`, `objects_vlm`, `objects_vlm_4bit`, `objects_resident_min_free_gb: float = 30.0`; laptop/server values per Global Constraints), `server/recon3d/objects/relations.py` (accept the new `ObjectRecord`), `server/tests/test_objects_pipeline.py` (rewrite)

**Interfaces:**
- Keeps `run_objects(job_dir, profile, on_progress) -> int` and `ModelDownloadFailed`.
- Injection points: `tracker_factory(device, profile) -> Tracker-like`, `vlm_factory(device, profile) -> Vlm`, `render_depths_fn`.
- Stages: render all depths (float16 list) → `depth_point_cloud` + `fit_room` (progress 0.15) → `Tracker.run` over all frames (0.15→0.55, `on_frame` drives progress) → `track_to_segment` each, `merge_segments`, sort by volume desc → structure openings: segments whose top proposer vote is door/window and `door_window_ok` → confirm with VLM → furniture → items, each with `candidate_entries` + `confirm` + `confidence_of`; `harmony_filter` (0.55→0.9) → `assign_by_voxels` + `assign_structure`, drop zero-Gaussian objects and renumber (0.9→0.97) → thumbnails from `best_views[0]`, relations, `write_objects` (→1.0). Structure records: label `floor`/`ceiling`/`wall` with zh 地面/天花板/墙面, `views = len(frames)` (all), `confidence 1.0`, bbox from the plane extent. Memory policy: `torch.cuda.mem_get_info()` free ≥ `objects_resident_min_free_gb` and profile is not laptop → tracker and VLM resident together; else close the tracker before loading the VLM. `_log` per stage with counts (tracks, segments, candidates per level, confirmed, dropped by each harmony rule).

- [ ] **Step 1: Failing end-to-end test** with fakes: synthetic room (Task 2 geometry rendered to 20 fake depth maps by a fake `render_depths_fn`), a `FakeTracker` returning two tracks for a 0.5 m box on the floor (disjoint frames, same voxels) and one 0.15 m flat track on top of it, `FakeVlm` answering 椅子 twice then 手机 twice. Assert: `run_objects` returns 2 objects + structure records (floor, ceiling, 4 walls) ; the box is `chair`, level furniture, support floor; the small one is `phone`, support `on:<chair id>`; `objects.bin` has ids for both; progress messages visited the four stage strings in order; `merge.py` import fails.
  Second test: `FakeVlm` answering 柜子 for a 2.0 m wall-flush box → not in the table (white door rule).
- [ ] **Step 2 → 4**; run the whole non-gpu suite → PASS (count ≥ 211 minus merge tests plus new ones).
- [ ] **Step 5: Commit** `feat(objects): pipeline v2 — structure, tracking, verification, voxel assignment`.

---

### Task 10: Deployment plumbing and docs

**Files:**
- Modify: `server/scripts/run.sh` (pass `CUDA_VISIBLE_DEVICES` through; print it), `docs/setup/02-server-lab.md` (Tailscale install/login, `bash server/scripts/predownload_models.sh server`, shared-GPU note, first-run time), `docs/setup/01-server-laptop.md` (laptop runs v2 degraded), `docs/acceptance/phase2.md` (new §4 for v2 acceptance per spec §7)
- Create: `server/scripts/predownload_models.sh [laptop|server]` (uses `huggingface-cli download` for the three model ids of that profile)

- [ ] **Step 1:** write the script and docs; **Step 2:** `bash -n` the script; run `predownload_models.sh laptop` on the laptop (expect the three laptop models cached; ~8 GB).
- [ ] **Step 3: Commit** `docs: objects v2 deployment (A6000, Tailscale, model predownload)`.

---

### Task 11: Phone — level sections and detail line

**Files:**
- Modify: `android/app/src/main/java/dev/recon3d/viewer/ObjectsTable.kt`, `ViewerScreen.kt` (list only)
- Test: `android/app/src/test/java/dev/recon3d/viewer/ObjectsTableTest.kt`

**Interfaces:**
- `ObjectEntry` gains `level: String = "item"`, `size: List<Float>? = null`, `support: String? = null`, `confidence: Float? = null`.
- `data class LevelSection(val level: String, val title: String, val groups: List<ObjectGroup>)`; `fun ObjectsTable.sections(): List<LevelSection>` in order structure 「结构」, furniture 「家具」, item 「物品」, empty sections omitted; `itemDetail(entry)` → `出现 N 帧 · 高 0.90 米` when `size != null` and level != structure, else `出现 N 帧`.

- [ ] **Step 1: Failing tests**: version-1 json (no level) → one section 「物品」 (Review Focus 5); v2 json with a door, a chair, a cup → three sections in order with correct titles; `itemDetail` formats height with two decimals.
- [ ] **Step 2 → 4**: implement; `LazyColumn` gets a section header (`titleMedium`) above the existing sticky group headers.
- [ ] **Step 5: Commit** `feat(app): objects list in structure / furniture / item sections with size`.

---

### Task 12: A6000 deployment and GPU acceptance (needs the lab machine)

**Files:**
- Modify: `docs/acceptance/phase2.md` §4 (results)

- [ ] **Step 1:** over Tailscale ssh: clone/pull `main`, `bash server/scripts/install.sh`, `bash server/scripts/predownload_models.sh server`, `nvidia-smi` to pick a free GPU, start with `CUDA_VISIBLE_DEVICES=<n> bash server/scripts/run.sh server` in tmux.
- [ ] **Step 2:** copy `server/data/jobs/40a80bd6c17f/` from the laptop (scp), `POST /api/v1/jobs/40a80bd6c17f/objects`, record per-stage times from `log.txt` and total; expect ≤ 10 min.
- [ ] **Step 3:** run the gpu tests there: `uv run --no-sync pytest -q -m gpu tests/test_objects_track.py tests/test_objects_vlm.py` → PASS.
- [ ] **Step 4:** phone pointed at the A6000 (settings → server address): acceptance items spec §7 1–6 on the office scene, then reconstruct one of the user's recent door/window scenes and repeat 1–4. Record results in `phase2.md` §4; if a rule misfires, tune the constant in `verify.py` / `vocab.py` and note the change.
- [ ] **Step 5: Commit** `docs: objects v2 acceptance on A6000`.
