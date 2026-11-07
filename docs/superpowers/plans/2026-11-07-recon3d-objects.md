# Recon3D 第二期前半段（识别物体 / 物体表 / 高亮）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On demand, the server turns a finished reconstruction into an object table (`objects.zip`); the phone lists the objects and, on tap, flies to one and highlights its Gaussians.

**Architecture:** A second work type ("objects") hangs off an existing done job and runs on the same worker queue. `recon3d/objects/` does: sample frames → Grounding DINO + SAM 2 per frame → depth rendered from the trained Gaussians → 3D merge → per-Gaussian object id (uint16, `.splat` order) → thumbnails + relations → `objects.zip`. The phone polls the job's `objects` block, downloads/unzips the result next to `scene.splat`, and the WebView viewer dims non-selected splats through Spark's per-splat `splatRgba` override.

**Tech Stack:** Python 3.10, torch 2.4.1 + torchvision 0.19.1 (cu124), transformers 4.57 (`grounding-dino-tiny`, `sam2.1-hiera-small`), gsplat 1.5.3, FastAPI; Kotlin/Compose, OkHttp, kotlinx.serialization; three.js 0.187 + Spark 2.4.0 (esbuild bundle).

**Spec:** `docs/superpowers/specs/2026-11-07-recon3d-objects-design.md` (builds on `2026-11-01-recon3d-design.md` §5.4).

## Global Constraints

- User-visible strings Chinese; code, comments, commit messages English. Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Server tests: `wsl -d Ubuntu -- bash -lc 'cd /mnt/d/projects/Recon3D/server && export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/recon3d UV_LINK_MODE=copy && uv run pytest -q -m "not gpu and not slow"'` (CPU suite must stay green; GPU tests `-m gpu` run on the laptop when the server is idle — check `GET /api/v1/jobs` first). Android: `cd D:\projects\Recon3D\android; $env:JAVA_HOME="C:\Program Files\Android\Android Studio\jbr"; .\gradlew.bat :app:testDebugUnitTest --offline`. viewer-web: `cd D:\projects\Recon3D\android\viewer-web && npm test && npm run build` (the bundle in `android/app/src/main/assets/viewer/` is committed).
- Models: `IDEA-Research/grounding-dino-tiny` (box 0.35, text 0.25), `facebook/sam2.1-hiera-small`; new dependency `torchvision==0.19.1` via the `pytorch-cu124` index (Linux). Weights load lazily and are released after each run.
- Frame sampling: every k-th frame, k = ceil(n / 240). Poses: `poses_cv.npy` when present (precise job), else ARCore poses via `gl_to_cv_c2w`. Gaussian indices refer to `scene.splat` order.
- Thresholds (spec §4.4): per-frame NMS IoU ≥ 0.7; same-frame detections never merge; same-class merge: ≥ 30 % of points within 5 cm OR bbox IoU ≥ 0.3; cross-class merge: ≥ 60 % within 5 cm AND bbox IoU ≥ 0.5; objects seen in < 2 frames dropped; centre = median, bbox = 5–95 percentile; Gaussian vote: inside mask AND |z_gauss − depth| < 0.1 m + 5 % depth; assigned if votes ≥ 2 AND centre inside bbox + 10 cm; thumbnail: max(score × mask area), box padded 15 %, long side 256 px JPEG. Relations: pairs with centre distance < 1.5 m; `on`/`in`/`next_to`/`left_of`/`right_of` as spec §4.4-5.
- Progress: 检测 0→0.6, 合并 0.6→0.75, 归属 0.75→0.95, 截图 0.95→1.0; messages `识别物体 (检测)` / `(合并)` / `(归属)` / `(截图)`; failure messages `显存不足，请降低分辨率或改用服务器` (existing `OOM_MESSAGE`), `模型下载失败，请检查网络`, generic `识别失败，请重试；如仍失败，把服务端 data/jobs/{id}/log.txt 发给开发者`.
- Product: `objects/objects.json` (version 1), `objects/objects.bin` (uint16 LE per Gaussian, 0 = none), `objects/thumbs/{id}.jpg`, zipped as `objects.zip` in the job dir. Phone unzips to `<scenesDir>/<roomId>/<captureId>/objects/`.
- Highlight: non-selected splats → alpha × 0.15, colour desaturated (grey = luma); selected unchanged; id 0 clears. Fly target: distance d = max(1.5 × bbox diagonal, 0.8 m), horizontal direction from object centre towards the camera (fallback: opposite of home look), height = centre.y + 0.3 d, lookAt centre, 1000 ms.

## Review Focus

1. A scene where detection finds nothing (empty room / synthetic capture): `objects.zip` is still produced with zero objects, the phone shows 「物体 (0)」 and an empty list, not a failure. (Task 7, Task 11)
2. Re-running recognition over an existing result while the phone is viewing the old one: old `objects/` stays usable until the new zip is fully unzipped; a failed re-run keeps the old table. (Task 11)
3. The server restarts mid-recognition: the `objects` block goes back to `queued` and the run resumes; the job's `scene.splat` stays viewable throughout. (Task 8)
4. DELETE of a job while recognition is running must be refused (409), like a training job; while queued it must succeed and the worker must skip the stale queue entry. (Task 8)
5. `objects.bin` whose length does not match the loaded `.splat` count (stale result after a rerun / different scene): the viewer must refuse to apply it (console warning, no highlight) instead of mis-colouring or throwing. (Task 12)

---

## File structure

Server (`server/`):
- `recon3d/objects/__init__.py`
- `recon3d/objects/vocab.py` — fixed vocabulary (en prompt, zh label), prompt batching.
- `recon3d/objects/table.py` — `ObjectRecord` dataclass, `objects.json` write/read, `objects.bin` encode/decode, `write_objects_zip`.
- `recon3d/objects/merge.py` — pure numpy: `Detection`, `nms_frame`, `merge_detections`, `MergedObject`.
- `recon3d/objects/relations.py` — pure numpy relation rules.
- `recon3d/objects/render_depth.py` — depth from `scene.ply` via gsplat; `read_splat_positions`.
- `recon3d/objects/detect.py` — `Detector` (GDINO + SAM 2, lazy, `close()`).
- `recon3d/objects/assign.py` — torch (device-agnostic) Gaussian → object voting.
- `recon3d/objects/thumbs.py` — thumbnail crop.
- `recon3d/objects/pipeline.py` — `run_objects(job_dir, profile, on_progress) -> int`.
- `recon3d/jobs/store.py` — `ObjectsStatus`, `JobStatus.objects`, `request_objects`, queue of `(kind, job_id)`.
- `recon3d/jobs/worker.py` — dispatch by kind; `ObjectsPipeline`.
- `recon3d/api/app.py` — `POST /jobs/{id}/objects`, `GET /jobs/{id}/objects.zip`, `objects` in GET bodies; `recon3d/api/status_page.py` — 「物体」 column.
- `recon3d/main.py` — wires `run_objects`.
- Tests: `tests/test_objects_vocab_table.py`, `tests/test_objects_merge.py`, `tests/test_objects_relations.py`, `tests/test_objects_render_depth.py` (gpu), `tests/test_objects_detect.py` (gpu), `tests/test_objects_assign.py`, `tests/test_objects_pipeline.py`, additions to `tests/test_jobs.py`, `tests/test_api.py`.
- Docs: `docs/api.md`, `docs/setup/01-server-laptop.md`, `docs/acceptance/phase2.md`.

viewer-web (`android/viewer-web/`):
- `src/objects-math.js` — `dimRgba`, `flyPoseFor`, `parseObjectIds`.
- `src/viewer.js` — `loadObjects(url)`, `highlightObject(id)`; roam button moved.
- `src/index.html` — roam button position.
- `test/objects-math.test.js`, `test/index-html.test.js`.

Android (`android/app/src/main/java/dev/recon3d/`):
- `net/ApiClient.kt` — `ObjectsStatus`, `JobStatus.objects`, `requestObjects`, `downloadObjects`.
- `rooms/RoomRepository.kt` — `CaptureRecord.objects*` fields.
- `viewer/ScenePathPolicy.kt` — `objects.bin`.
- `viewer/ObjectsTable.kt` — kotlinx-serializable model of `objects.json`, grouping helper.
- `viewer/ObjectsViewModel.kt` — request / poll / download / unzip / labels.
- `viewer/ViewerScreen.kt` — chip, bottom sheet, highlight bar, JS bridge calls.
- `upload/UploadScreen.kt` — rerun clears objects fields.
- `AppGraph.kt`, `nav/AppNav.kt` — wiring.
- Tests: `test/.../viewer/ObjectsTableTest.kt`, `ObjectsViewModelTest.kt`, `ScenePathPolicyTest.kt`, `net/ApiClientTest.kt`, `upload/UploadViewModelTest.kt`, `upload/Fakes.kt`.

---

### Task 1: Vocabulary, object table and `objects.bin` codec

**Files:**
- Create: `server/recon3d/objects/__init__.py`, `server/recon3d/objects/vocab.py`, `server/recon3d/objects/table.py`
- Modify: `server/pyproject.toml` (add `"torchvision==0.19.1; sys_platform == 'linux'"` with a `[tool.uv.sources]` entry on the `pytorch-cu124` index like torch; run `uv lock`)
- Test: `server/tests/test_objects_vocab_table.py`

**Interfaces:**
- Produces: `vocab.VOCAB: list[tuple[str, str]]` (en, zh) of 40 entries from spec §4.3; `vocab.zh_label(en: str) -> str`; `vocab.prompt_batches(max_per_batch: int = 20) -> list[str]` (each `"cup. bottle. ... ."`); `vocab.label_of(text: str) -> str | None` (maps GDINO's returned phrase, e.g. `"remote control phone"`, to the single vocab entry that is a substring, longest match wins, else None).
- Produces: `table.ObjectRecord(id: int, label: str, label_zh: str, score: float, views: int, center: tuple[float,float,float], bbox_min: tuple, bbox_max: tuple, gaussian_count: int)`; `table.encode_ids(ids: np.ndarray[uint16]) -> bytes`; `table.decode_ids(data: bytes) -> np.ndarray[uint16]` (raises ValueError on odd length); `table.write_objects(out_dir: Path, job_id: str, frames_used: int, ids: np.ndarray, objects: list[ObjectRecord], relations: list[dict]) -> Path` (writes `objects.json`, `objects.bin`, expects `thumbs/` already filled, then zips `out_dir` into `out_dir.parent / "objects.zip"` and returns that path; `thumb` field is `thumbs/{id}.jpg`); `table.read_objects_json(path) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_objects_vocab_table.py
import json, zipfile
import numpy as np
import pytest
from recon3d.objects import vocab, table

def test_vocab_has_40_entries_with_chinese_labels():
    assert len(vocab.VOCAB) == 40
    assert vocab.zh_label("cup") == "杯子" and vocab.zh_label("remote control") == "遥控器"
    assert all(zh and not zh.isascii() for _, zh in vocab.VOCAB)

def test_prompt_batches_cover_vocab_in_groups_of_20():
    batches = vocab.prompt_batches()
    assert len(batches) == 2 and all(b.endswith(".") for b in batches)
    assert sum(b.count(". ") + 1 for b in batches) == 40

def test_label_of_maps_detector_phrases_back_to_vocab():
    assert vocab.label_of("remote control phone") == "remote control"
    assert vocab.label_of("trash") is None and vocab.label_of("trash can") == "trash can"
    assert vocab.label_of("cup") == "cup"

def test_ids_roundtrip_little_endian_uint16():
    ids = np.array([0, 1, 65535, 7], dtype=np.uint16)
    data = table.encode_ids(ids)
    assert data == b"\x00\x00\x01\x00\xff\xff\x07\x00"
    assert np.array_equal(table.decode_ids(data), ids)
    with pytest.raises(ValueError):
        table.decode_ids(b"\x01")

def test_write_objects_produces_json_bin_and_zip(tmp_path):
    out = tmp_path / "job" / "objects"
    (out / "thumbs").mkdir(parents=True)
    (out / "thumbs" / "1.jpg").write_bytes(b"jpg")
    rec = table.ObjectRecord(1, "chair", "椅子", 0.64, 17, (1.0, 0.5, -2.0), (0.7, 0.0, -2.3), (1.3, 1.0, -1.7), 3)
    ids = np.array([1, 0, 1, 1], dtype=np.uint16)
    zpath = table.write_objects(out, "abc123", frames_used=12, ids=ids, objects=[rec], relations=[{"a": 1, "rel": "on", "b": 2}])
    assert zpath == tmp_path / "job" / "objects.zip"
    j = json.loads((out / "objects.json").read_text(encoding="utf-8"))
    assert j["version"] == 1 and j["job_id"] == "abc123" and j["frames_used"] == 12 and j["gaussians"] == 4
    o = j["objects"][0]
    assert o["id"] == 1 and o["label_zh"] == "椅子" and o["thumb"] == "thumbs/1.jpg" and o["gaussian_count"] == 3
    assert o["bbox"] == {"min": [0.7, 0.0, -2.3], "max": [1.3, 1.0, -1.7]}
    assert j["relations"] == [{"a": 1, "rel": "on", "b": 2}]
    assert (out / "objects.bin").read_bytes() == table.encode_ids(ids)
    assert sorted(zipfile.ZipFile(zpath).namelist()) == ["objects.bin", "objects.json", "thumbs/1.jpg"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_objects_vocab_table.py`
Expected: FAIL — `ModuleNotFoundError: recon3d.objects`

- [ ] **Step 3: Implement `vocab.py` and `table.py`**

`VOCAB` = the 40 pairs listed in spec §4.3 in that order. `label_of`: among vocab entries that are a whitespace-bounded substring of the phrase, return the longest; `prompt_batches` joins `en` with `". "` and ends with `"."`. `write_objects` uses `json.dumps(..., ensure_ascii=False, indent=1)` and `zipfile.ZIP_DEFLATED` for json, `ZIP_STORED` for bin/jpg.

- [ ] **Step 4: Add torchvision to `pyproject.toml`, run `uv lock` + `uv sync --group dev`, verify import**

Run: `uv run python -c "import torchvision; print(torchvision.__version__)"`
Expected: `0.19.1+cu124`

- [ ] **Step 5: Run tests**

Run: `uv run pytest -q tests/test_objects_vocab_table.py`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add server/recon3d/objects server/tests/test_objects_vocab_table.py server/pyproject.toml server/uv.lock
git commit -m "feat(objects): vocabulary, object table and objects.bin codec"
```

---

### Task 2: 3D merge of per-frame detections

**Files:**
- Create: `server/recon3d/objects/merge.py`
- Test: `server/tests/test_objects_merge.py`

**Interfaces:**
- Produces: `Detection(frame: int, label: str, score: float, box: tuple[float,float,float,float], points: np.ndarray[M,3])` (dataclass; `mask` is NOT kept here — see Task 6 for how masks flow); `nms_frame(dets: list[Detection], iou: float = 0.7) -> list[Detection]` (cross-class, keep higher score; `box` is x0,y0,x1,y1 pixels); `MergedObject(label: str, score: float, views: int, center: np.ndarray[3], bbox_min: np.ndarray[3], bbox_max: np.ndarray[3], members: list[int])` (`members` = indices into the input `dets` list); `merge_detections(dets: list[Detection], same_overlap=0.3, same_iou=0.3, cross_overlap=0.6, cross_iou=0.5, radius_m=0.05, min_views=2) -> list[MergedObject]`; `point_overlap(a: np.ndarray, b: np.ndarray, radius_m: float) -> float` (fraction of `a` within radius of some point of `b`, via `scipy.spatial.cKDTree`); `bbox_iou(a_min, a_max, b_min, b_max) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_objects_merge.py
import numpy as np
from recon3d.objects.merge import Detection, MergedObject, bbox_iou, merge_detections, nms_frame, point_overlap

def cloud(center, size=0.2, n=200, seed=0):
    r = np.random.default_rng(seed)
    return np.asarray(center) + r.uniform(-size / 2, size / 2, (n, 3))

def det(frame, label, center, score=0.5, seed=0, box=(0, 0, 10, 10)):
    return Detection(frame, label, score, box, cloud(center, seed=seed))

def test_nms_drops_overlapping_lower_score_box_across_classes():
    a = det(0, "monitor", (0, 0, 0), 0.6, box=(0, 0, 100, 100))
    b = det(0, "tv", (0, 0, 0), 0.5, box=(5, 5, 100, 100))
    c = det(0, "cup", (1, 0, 0), 0.9, box=(200, 200, 220, 220))
    kept = nms_frame([a, b, c])
    assert [d.label for d in kept] == [c.label, a.label] or [d.label for d in kept] == [a.label, c.label]
    assert all(d is not b for d in kept)

def test_point_overlap_and_bbox_iou():
    a = cloud((0, 0, 0)); b = cloud((0, 0, 0), seed=1); far = cloud((2, 0, 0))
    assert point_overlap(a, b, 0.05) > 0.9 and point_overlap(a, far, 0.05) == 0.0
    assert bbox_iou(np.zeros(3), np.ones(3), np.zeros(3), np.ones(3)) == 1.0
    assert bbox_iou(np.zeros(3), np.ones(3), np.array([0.5, 0, 0]), np.array([1.5, 1, 1])) == pytest.approx(1 / 3)

def test_same_object_seen_in_three_frames_merges_to_one():
    dets = [det(f, "chair", (1, 0.5, -2), seed=f) for f in range(3)]
    objs = merge_detections(dets)
    assert len(objs) == 1 and objs[0].views == 3 and sorted(objs[0].members) == [0, 1, 2]
    assert np.allclose(objs[0].center, (1, 0.5, -2), atol=0.03)

def test_two_touching_cups_in_the_same_frame_stay_two_objects():
    # same frame -> never merged, even though the clouds overlap heavily
    dets = [det(0, "cup", (0, 0, 0), seed=0), det(0, "cup", (0.05, 0, 0), seed=1),
            det(1, "cup", (0, 0, 0), seed=2), det(1, "cup", (0.05, 0, 0), seed=3)]
    objs = merge_detections(dets)
    assert len(objs) == 2 and all(o.views == 2 for o in objs)

def test_single_view_object_is_dropped():
    dets = [det(0, "cup", (0, 0, 0)), det(1, "chair", (3, 0, 0)), det(2, "chair", (3, 0, 0), seed=1)]
    objs = merge_detections(dets)
    assert [o.label for o in objs] == ["chair"]

def test_cross_class_merge_needs_strong_overlap_and_votes_label_by_score():
    # monitor/tv flicker on the same object: 2 monitor views at 0.5, 1 tv view at 0.9 -> label by score sum
    dets = [det(0, "monitor", (0, 1, -1), 0.5), det(1, "tv", (0, 1, -1), 0.9, seed=1), det(2, "monitor", (0, 1, -1), 0.5, seed=2)]
    objs = merge_detections(dets)
    assert len(objs) == 1 and objs[0].label == "monitor" and objs[0].views == 3
    # weak overlap across classes does not merge (clouds offset 15 cm, size 20 cm)
    dets2 = [det(0, "bag", (0, 0, 0)), det(1, "backpack", (0.15, 0, 0), seed=1), det(2, "bag", (0, 0, 0), seed=2), det(3, "backpack", (0.15, 0, 0), seed=3)]
    assert len(merge_detections(dets2)) == 2

def test_bbox_is_5_95_percentile_and_score_is_max():
    pts = np.array([[0, 0, 0]] * 98 + [[10, 10, 10], [-10, -10, -10]], dtype=float)
    dets = [Detection(0, "box", 0.4, (0, 0, 1, 1), pts), Detection(1, "box", 0.7, (0, 0, 1, 1), pts)]
    o = merge_detections(dets)[0]
    assert np.allclose(o.bbox_min, 0) and np.allclose(o.bbox_max, 0) and o.score == 0.7
```

(add `import pytest` at the top.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_objects_merge.py`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `merge.py`**

Greedy union-find over detections in input order: for each pair (i < j) with `frame_i != frame_j`, compute `ov = max(point_overlap(i,j), point_overlap(j,i))` and `iou = bbox_iou` of their per-detection 5–95 % boxes; same label → merge if `ov ≥ same_overlap or iou ≥ same_iou`; different label → merge if `ov ≥ cross_overlap and iou ≥ cross_iou`. Skip pairs whose centres are more than 2 m apart before the KD-tree (speed). After union: drop groups with fewer than `min_views` distinct frames; label = argmax of score sum per label; score = max; centre = median of concatenated points; bbox = 5/95 percentiles. Pairs of detections that are both already in the same group are skipped.

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q tests/test_objects_merge.py`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add server/recon3d/objects/merge.py server/tests/test_objects_merge.py
git commit -m "feat(objects): per-frame NMS and 3D merge of detections"
```

---

### Task 3: Spatial relations

**Files:**
- Create: `server/recon3d/objects/relations.py`
- Test: `server/tests/test_objects_relations.py`

**Interfaces:**
- Consumes: `table.ObjectRecord` (Task 1).
- Produces: `compute_relations(objects: list[ObjectRecord], home_look: np.ndarray[3] | None, max_dist_m: float = 1.5) -> list[dict]` with dicts `{"a": id, "rel": "on"|"in"|"next_to"|"left_of"|"right_of", "b": id}`; rules per spec §4.4-5 (`on`: a.bbox_min.y within [b.bbox_max.y − 0.02, b.bbox_max.y + 0.10] and a's horizontal centre inside b's horizontal bbox; `in`: ≥ 90 % of a's bbox volume inside b's; `next_to`: horizontal gap between bboxes < 0.3 m and vertical ranges overlap, emitted once per unordered pair with a < b; `left_of`/`right_of`: sign of `cross(up, look) · (a.center − b.center)` with `look` = horizontal part of `home_look`, emitted for ordered pairs within `max_dist_m`, skipped when `home_look` is None or horizontal separation < 0.1 m). Pairs with centre distance ≥ `max_dist_m` produce nothing.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_objects_relations.py
import numpy as np
from recon3d.objects.relations import compute_relations
from recon3d.objects.table import ObjectRecord

def obj(i, label, mn, mx):
    c = tuple((np.asarray(mn) + np.asarray(mx)) / 2)
    return ObjectRecord(i, label, label, 0.5, 2, c, tuple(mn), tuple(mx), 10)

table = obj(1, "table", (-0.6, 0.0, -0.4), (0.6, 0.75, 0.4))
cup = obj(2, "cup", (0.1, 0.75, 0.0), (0.2, 0.85, 0.1))
book = obj(3, "book", (-0.3, 0.76, 0.0), (-0.1, 0.78, 0.15))
chair = obj(4, "chair", (0.75, 0.0, -0.3), (1.2, 0.9, 0.2))
far = obj(5, "door", (5, 0, 0), (5.1, 2, 0.9))
box = obj(6, "box", (-2.0, 0.0, -2.0), (-1.0, 1.0, -1.0))
pen = obj(7, "pen", (-1.6, 0.4, -1.6), (-1.5, 0.42, -1.4))

def rels(*objs, look=(0, 0, -1)):
    return compute_relations(list(objs), np.array(look, dtype=float))

def test_on_and_in():
    r = rels(table, cup, box, pen)
    assert {"a": 2, "rel": "on", "b": 1} in r
    assert {"a": 7, "rel": "in", "b": 6} in r
    assert {"a": 1, "rel": "on", "b": 2} not in r

def test_next_to_is_unordered_and_needs_small_gap():
    r = rels(table, chair, far)
    assert {"a": 1, "rel": "next_to", "b": 4} in r and {"a": 4, "rel": "next_to", "b": 1} not in r
    assert not any(x["b"] == 5 or x["a"] == 5 for x in r)

def test_left_right_from_home_view():
    # looking down -Z, +X is to the right: cup (x=.15) is right of book (x=-.2)
    r = rels(cup, book)
    assert {"a": 2, "rel": "right_of", "b": 3} in r and {"a": 3, "rel": "left_of", "b": 2} in r
    assert all(x["rel"] not in ("left_of", "right_of") for x in compute_relations([cup, book], None))

def test_far_pairs_produce_nothing():
    assert compute_relations([table, far], np.array([0, 0, -1.0])) == []
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest -q tests/test_objects_relations.py` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement `relations.py`** as pure numpy over `bbox_min/bbox_max/center`.

- [ ] **Step 4: Run tests** — Expected: 4 passed.

- [ ] **Step 5: Commit** — `git commit -m "feat(objects): geometric spatial relations"`

---

### Task 4: Depth rendered from the trained Gaussians; `.splat` positions

**Files:**
- Create: `server/recon3d/objects/render_depth.py`
- Test: `server/tests/test_objects_render_depth.py`

**Interfaces:**
- Consumes: `recon3d.recon.export_splat.load_ply(path, device) -> Splats`, `export_splat.SPLAT_DTYPE`, `recon3d.recon.trainer._rasterize(params, viewmats, Ks, W, H, sh_degree, render_mode)`.
- Produces: `read_splat_positions(path: Path) -> np.ndarray[N,3] float32` (pure numpy from `SPLAT_DTYPE`); `render_depths(ply_path: Path, c2w_cv: np.ndarray[F,4,4], Ks: np.ndarray[F,3,3], width: int, height: int, device: str = "cuda", scale: float = 0.5) -> Iterator[np.ndarray[h,w] float32]` yielding expected depth (`render_mode="ED"`, metres, 0 where alpha < 0.5) per frame at `scale` resolution with `Ks` scaled accordingly (`h = round(height*scale)`, `w = round(width*scale)`); frees the Splats at the end (`generator` finally).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_objects_render_depth.py
import numpy as np, pytest, torch
from pathlib import Path
from recon3d.objects.render_depth import read_splat_positions, render_depths
from recon3d.recon.export_splat import export_ply, export_splat
from recon3d.recon.trainer import Splats

def wall_splats(n=400, z=-2.0):
    g = torch.linspace(-1, 1, 20)
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    means = torch.stack([xx.flatten(), yy.flatten(), torch.full((n,), z)], 1)
    return Splats(means=means, scales=torch.full((n, 3), np.log(0.08)), quats=torch.tensor([[1.0, 0, 0, 0]] * n),
                  opacities=torch.full((n,), 4.0), sh0=torch.zeros(n, 1, 3), shN=torch.zeros(n, 0, 3))

def test_read_splat_positions_matches_export_order(tmp_path):
    s = wall_splats()
    export_splat(s, tmp_path / "scene.splat")
    pos = read_splat_positions(tmp_path / "scene.splat")
    assert pos.shape == (400, 3) and pos.dtype == np.float32
    assert set(map(tuple, np.round(pos, 4))) == set(map(tuple, np.round(s.means.numpy(), 4)))

@pytest.mark.gpu
def test_rendered_depth_of_a_wall_is_its_distance(tmp_path):
    export_ply(wall_splats(), tmp_path / "scene.ply")
    c2w = np.eye(4)[None]  # camera at origin looking down -Z (OpenCV: +Z forward) -> flip to look at z=-2
    c2w[0, :3, :3] = np.diag([1, -1, -1])
    K = np.array([[[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]]])
    depths = list(render_depths(tmp_path / "scene.ply", c2w, K, 320, 240, scale=0.5))
    assert len(depths) == 1 and depths[0].shape == (120, 160)
    centre = depths[0][50:70, 70:90]
    assert np.all(centre > 0) and np.allclose(centre, 2.0, atol=0.1)
```

- [ ] **Step 2: Run CPU part** — `uv run pytest -q tests/test_objects_render_depth.py -m "not gpu"` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement `render_depth.py`** — `viewmats = inv(c2w)`; render in batches of 1; `depth = out[..., 3]` of `"RGB+ED"`? No: use `render_mode="ED"` (output `[1,h,w,1]`) and `alphas` from the same call; zero where alpha < 0.5.

- [ ] **Step 4: Run** CPU test → 1 passed; GPU test on the laptop when idle: `uv run pytest -q tests/test_objects_render_depth.py -m gpu` → 1 passed.

- [ ] **Step 5: Commit** — `git commit -m "feat(objects): depth rendered from trained Gaussians"`

---

### Task 5: Detector (Grounding DINO + SAM 2)

**Files:**
- Create: `server/recon3d/objects/detect.py`
- Test: `server/tests/test_objects_detect.py` (gpu)

**Interfaces:**
- Consumes: `vocab.prompt_batches`, `vocab.label_of` (Task 1), `merge.nms_frame` (Task 2; `Detection.points` left as `None`-sized placeholder is NOT allowed — the detector returns its own type below).
- Produces: `FrameDetection(label: str, score: float, box: tuple[float,float,float,float], mask: np.ndarray[H,W] bool)`; `class Detector: __init__(self, device: str = "cuda", box_threshold=0.35, text_threshold=0.25, nms_iou=0.7)`; `detect(self, image: PIL.Image) -> list[FrameDetection]` (runs every prompt batch, maps phrases with `label_of` and drops unmapped ones, applies per-frame NMS across batches via `merge.nms_frame` on a temporary `Detection` list with empty `points`, then SAM 2 masks for the survivors, in one `Sam2Model` call with all boxes); `close(self)` (drops models, `torch.cuda.empty_cache()`); `DownloadError(Exception)` raised when `from_pretrained` fails with an `OSError`/`requests` error (message in log, user text set by the pipeline). Models load lazily on the first `detect`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_objects_detect.py
import numpy as np, pytest
from PIL import Image, ImageDraw
from recon3d.objects.detect import Detector, FrameDetection

@pytest.mark.gpu
def test_detector_finds_a_drawn_cup_and_returns_masks():
    img = Image.new("RGB", (640, 480), (235, 235, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([250, 200, 390, 400], fill=(200, 40, 40))       # a red mug body
    d.arc([380, 240, 440, 360], 270, 90, fill=(200, 40, 40), width=14)  # handle
    det = Detector()
    try:
        out = det.detect(img)
    finally:
        det.close()
    assert all(isinstance(o, FrameDetection) and o.mask.shape == (480, 640) and o.mask.dtype == bool for o in out)
    labels = {o.label for o in out}
    assert labels & {"cup", "box"}  # a drawn mug reads as a cup or a box; either proves the chain
    best = max(out, key=lambda o: o.score)
    assert best.mask[300, 320]  # mask covers the body centre
```

- [ ] **Step 2: Run** (GPU, laptop idle) — `uv run pytest -q tests/test_objects_detect.py -m gpu` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement `detect.py`** using `AutoProcessor`/`AutoModelForZeroShotObjectDetection` and `Sam2Processor`/`Sam2Model` exactly as in the spike (`scripts` are not kept; fp16 autocast for GDINO, `post_process_grounded_object_detection(..., threshold, text_threshold, target_sizes)`, `text_labels`/`labels` key, `Sam2Processor.post_process_masks(out.pred_masks, inp["original_sizes"])[0]`, `multimask_output=False`, mask = `(m > 0)` of the single mask per box).

- [ ] **Step 4: Run** → 1 passed (first run downloads ~1 GB of weights).

- [ ] **Step 5: Commit** — `git commit -m "feat(objects): Grounding DINO + SAM 2 detector"`

---

### Task 6: Gaussian → object assignment and thumbnails

**Files:**
- Create: `server/recon3d/objects/assign.py`, `server/recon3d/objects/thumbs.py`
- Test: `server/tests/test_objects_assign.py`

**Interfaces:**
- Produces: `assign.ObjectMasks(frame: int, object_id: int, mask: np.ndarray[h,w] bool)` (mask at the depth resolution, i.e. the detector's mask downscaled by `scale` with `PIL.Image.resize(NEAREST)`); `assign.assign_gaussians(positions: np.ndarray[N,3], c2w_cv: np.ndarray[F,4,4], Ks_depth: np.ndarray[F,3,3], depths: list[np.ndarray[h,w]], masks: list[ObjectMasks], bbox_min: np.ndarray[O,3], bbox_max: np.ndarray[O,3], device: str = "cpu", min_votes: int = 2, depth_tol_m: float = 0.1, depth_tol_rel: float = 0.05, bbox_margin_m: float = 0.1, chunk: int = 200_000) -> np.ndarray[N] uint16` (object ids 1..O index `bbox_*` rows 0..O−1; votes counted on the device per chunk of Gaussians; ties → lower id);
- Produces: `thumbs.crop_thumbnail(image: PIL.Image, box: tuple, pad: float = 0.15, long_side: int = 256) -> PIL.Image`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_objects_assign.py
import numpy as np
from PIL import Image
from recon3d.objects.assign import ObjectMasks, assign_gaussians
from recon3d.objects.thumbs import crop_thumbnail

W, H = 64, 48
K = np.array([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]])
cam = np.eye(4)  # OpenCV camera at origin looking +Z

def frame_with_object(frame, depth_m, mask_box):
    depth = np.full((H, W), depth_m, dtype=np.float32)
    mask = np.zeros((H, W), bool)
    x0, y0, x1, y1 = mask_box
    mask[y0:y1, x0:x1] = True
    return depth, ObjectMasks(frame, 1, mask)

def test_gaussian_inside_mask_and_at_depth_in_two_frames_gets_the_object():
    pos = np.array([[0.0, 0.0, 2.0],   # projects to (32,24), depth 2 -> inside mask, right depth
                    [0.0, 0.0, 3.0],   # same pixel, 1 m behind the surface -> depth mismatch
                    [1.5, 0.0, 2.0]])  # projects to x=77 -> off image
    d0, m0 = frame_with_object(0, 2.0, (20, 12, 44, 36))
    d1, m1 = frame_with_object(1, 2.0, (20, 12, 44, 36))
    ids = assign_gaussians(pos, np.stack([cam, cam]), np.stack([K, K]), [d0, d1], [m0, m1],
                           bbox_min=np.array([[-1, -1, 1.5]]), bbox_max=np.array([[1, 1, 2.5]]))
    assert ids.dtype == np.uint16 and ids.tolist() == [1, 0, 0]

def test_one_vote_is_not_enough_and_bbox_margin_is_enforced():
    pos = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    d0, m0 = frame_with_object(0, 2.0, (0, 0, W, H))
    ids = assign_gaussians(pos, cam[None], K[None], [d0], [m0], np.array([[-1, -1, 1.5]]), np.array([[1, 1, 2.5]]))
    assert ids.tolist() == [0, 0]
    d1, m1 = frame_with_object(1, 2.0, (0, 0, W, H))
    tight = assign_gaussians(pos, np.stack([cam, cam]), np.stack([K, K]), [d0, d1], [m0, m1],
                             np.array([[0.5, 0.5, 1.5]]), np.array([[1, 1, 2.5]]))  # object box excludes x=0
    assert tight.tolist() == [0, 0]

def test_votes_pick_the_object_with_most_views():
    pos = np.array([[0.0, 0.0, 2.0]])
    frames = []
    for f in range(5):
        d = np.full((H, W), 2.0, np.float32)
        frames.append((d, ObjectMasks(f, 2 if f < 3 else 1, np.ones((H, W), bool))))
    ids = assign_gaussians(pos, np.stack([cam] * 5), np.stack([K] * 5), [d for d, _ in frames], [m for _, m in frames],
                           np.array([[-1, -1, 1.5]] * 2), np.array([[1, 1, 2.5]] * 2))
    assert ids.tolist() == [2]

def test_crop_thumbnail_pads_and_limits_long_side():
    img = Image.new("RGB", (1920, 1080))
    t = crop_thumbnail(img, (900, 500, 1100, 600))  # 200x100 box -> padded 260x130 -> long side 256
    assert max(t.size) == 256 and abs(t.size[0] / t.size[1] - 2.0) < 0.05
    edge = crop_thumbnail(img, (1800, 1000, 1920, 1080))  # clipped at the image border, still returns an image
    assert edge.size[0] > 0 and edge.size[1] > 0
```

- [ ] **Step 2: Run** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement `assign.py`** — per frame: `w2c = inv(c2w)`, transform the chunk, keep z > 0, pixel = `K @ p / z`, round to int, in-bounds; `votes[obj]` += inside mask & |z − depth[v,u]| < tol & depth > 0; after all frames `best = argmax`, apply `min_votes` and bbox-with-margin test. Use `torch` tensors on `device`; masks per frame are stacked into one `[h,w] int32` label image (0 = none) before projection so one gather suffices.

- [ ] **Step 4: Run** → 4 passed.

- [ ] **Step 5: Commit** — `git commit -m "feat(objects): Gaussian-to-object voting and thumbnails"`

---

### Task 7: Objects pipeline

**Files:**
- Create: `server/recon3d/objects/pipeline.py`
- Test: `server/tests/test_objects_pipeline.py`

**Interfaces:**
- Consumes: `load_capture`, `gl_to_cv_c2w`, Tasks 1–6.
- Produces: `run_objects(job_dir: Path, profile: Profile, on_progress: OnProgress) -> int` (returns object count); module-level injection points `detector_factory = None` (callable `(device) -> Detector`-like with `detect`/`close`) and `render_depths_fn = None` (same signature as `render_depth.render_depths`), resolved lazily like `pipeline.precise_pose`; constants `STAGE_DETECT = "识别物体 (检测)"`, `STAGE_MERGE = "识别物体 (合并)"`, `STAGE_ASSIGN = "识别物体 (归属)"`, `STAGE_THUMBS = "识别物体 (截图)"`, `MAX_FRAMES = 240`, `DEPTH_SCALE = 0.5`; `class ModelDownloadFailed(Exception)` re-raised from `detect.DownloadError`; writes `objects/` + `objects.zip` in `job_dir`, removes the extracted `capture/` in `finally`, logs per-stage timings to `log.txt` via the same `_log` helper pattern as `jobs/pipeline.py`.
- Algorithm per spec §4.4: frames `cap.frames[::k]`, `k = ceil(n / MAX_FRAMES)`; poses from `poses_cv.npy` if present (indexed by the same frame indices) else `gl_to_cv_c2w(f.pose)`; per frame: detect → for every `FrameDetection` build `merge.Detection` points by back-projecting the mask ∩ (rendered depth > 0) at depth resolution (mask downscaled NEAREST), uniformly subsampled to ≤ 2000, transformed to world; keep the full-res mask and the frame index for assignment and the best-thumbnail bookkeeping; progress inside detect stage = `0.6 * (i+1)/F`; merge → objects (ids 1..O in output order) → assignment with `ObjectMasks` built only for detections that are members of surviving objects → thumbnails (reload the chosen frame's JPEG from the capture dir) → relations with `home_look` = first frame's look direction (`c2w_cv[0][:3, 2]`) → `table.write_objects`. Empty detection set → zero objects, still writes the product.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_objects_pipeline.py
import json, shutil, zipfile
import numpy as np, pytest, torch
from PIL import Image
import recon3d.objects.pipeline as op
from recon3d.config import PROFILES
from recon3d.objects.detect import FrameDetection
from recon3d.objects.pipeline import run_objects
from recon3d.recon.export_splat import export_ply, export_splat
from recon3d.recon.trainer import Splats
from tests.fixtures.make_capture import make_capture

def _job(tmp_path, n_frames=6):
    job = tmp_path / "job"; job.mkdir()
    shutil.copy(make_capture(tmp_path, n_frames=n_frames), job / "capture.zip")
    n = 300
    means = torch.rand(n, 3) * 2 - 1
    means[:, 2] -= 2.0  # in front of the fixture cameras
    s = Splats(means=means, scales=torch.full((n, 3), np.log(0.05)), quats=torch.tensor([[1.0, 0, 0, 0]] * n),
               opacities=torch.full((n,), 3.0), sh0=torch.zeros(n, 1, 3), shN=torch.zeros(n, 0, 3))
    export_splat(s, job / "scene.splat"); export_ply(s, job / "scene.ply")
    return job

class FakeDetector:
    def __init__(self, dets_by_call): self.calls = 0; self.dets = dets_by_call; self.closed = False
    def detect(self, image):
        out = self.dets(self.calls, image.size); self.calls += 1; return out
    def close(self): self.closed = True

def fake_depths(ply, c2w, Ks, W, H, device="cuda", scale=0.5):
    for _ in range(len(c2w)):
        yield np.full((round(H * scale), round(W * scale)), 2.0, np.float32)

def _install(monkeypatch, dets):
    fake = FakeDetector(dets)
    monkeypatch.setattr(op, "detector_factory", lambda device: fake, raising=False)
    monkeypatch.setattr(op, "render_depths_fn", fake_depths, raising=False)
    return fake

def centre_box(size):
    W, H = size; m = np.zeros((H, W), bool); m[H//4:3*H//4, W//4:3*W//4] = True
    return FrameDetection("cup", 0.7, (W/4, H/4, 3*W/4, 3*H/4), m)

def test_pipeline_writes_objects_zip_with_stage_messages(tmp_path, monkeypatch):
    job = _job(tmp_path)
    fake = _install(monkeypatch, lambda i, size: [centre_box(size)])
    seen = []
    count = run_objects(job, PROFILES["laptop"], lambda p, m: seen.append((p, m)))
    assert count == 1 and fake.closed
    z = zipfile.ZipFile(job / "objects.zip")
    assert {"objects.json", "objects.bin", "thumbs/1.jpg"} <= set(z.namelist())
    j = json.loads((job / "objects" / "objects.json").read_text(encoding="utf-8"))
    assert j["objects"][0]["label_zh"] == "杯子" and j["objects"][0]["views"] == 6 and j["frames_used"] == 6
    ids = np.frombuffer((job / "objects" / "objects.bin").read_bytes(), "<u2")
    assert len(ids) == 300 and ids.max() == 1 and j["objects"][0]["gaussian_count"] == int((ids == 1).sum()) > 0
    msgs = [m for _, m in seen]
    assert msgs[0] == "识别物体 (检测)" and {"识别物体 (合并)", "识别物体 (归属)", "识别物体 (截图)"} <= set(msgs)
    ps = [p for p, _ in seen]; assert ps == sorted(ps) and ps[-1] == 1.0
    assert not (job / "capture").exists()
    Image.open(job / "objects" / "thumbs" / "1.jpg").verify()

def test_pipeline_with_no_detections_yields_an_empty_table(tmp_path, monkeypatch):
    job = _job(tmp_path)
    _install(monkeypatch, lambda i, size: [])
    assert run_objects(job, PROFILES["laptop"], lambda p, m: None) == 0
    j = json.loads((job / "objects" / "objects.json").read_text(encoding="utf-8"))
    assert j["objects"] == [] and (job / "objects.zip").is_file()
    assert np.frombuffer((job / "objects" / "objects.bin").read_bytes(), "<u2").max() == 0

def test_pipeline_uses_poses_cv_when_present(tmp_path, monkeypatch):
    job = _job(tmp_path)
    captured = {}
    def spy_depths(ply, c2w, Ks, W, H, device="cuda", scale=0.5):
        captured["c2w"] = c2w.copy(); yield from fake_depths(ply, c2w, Ks, W, H, device, scale)
    fake = _install(monkeypatch, lambda i, size: [])
    monkeypatch.setattr(op, "render_depths_fn", spy_depths, raising=False)
    poses = np.stack([np.eye(4)] * 6); poses[:, 0, 3] = 9.0
    np.save(job / "poses_cv.npy", poses)
    run_objects(job, PROFILES["laptop"], lambda p, m: None)
    assert np.allclose(captured["c2w"][:, 0, 3], 9.0)

def test_frame_sampling_caps_at_240(tmp_path, monkeypatch):
    assert op.sample_indices(1123) == list(range(0, 1123, 5)) and len(op.sample_indices(1123)) <= 240
    assert op.sample_indices(100) == list(range(100))
```

- [ ] **Step 2: Run** → `ModuleNotFoundError` / `AttributeError: sample_indices`.

- [ ] **Step 3: Implement `pipeline.py`** (add `sample_indices(n: int) -> list[int]`).

- [ ] **Step 4: Run** `uv run pytest -q tests/test_objects_pipeline.py` → 4 passed; then the whole CPU suite → all passed.

- [ ] **Step 5: Commit** — `git commit -m "feat(objects): run_objects pipeline"`

---

### Task 8: Objects work in JobStore, Worker and API

**Files:**
- Modify: `server/recon3d/jobs/store.py`, `server/recon3d/jobs/worker.py`, `server/recon3d/api/app.py`, `server/recon3d/api/status_page.py`, `server/recon3d/main.py`
- Test: `server/tests/test_jobs.py`, `server/tests/test_api.py` (and update existing fakes that read `store.queue`)

**Interfaces:**
- Produces (store): `@dataclass(frozen=True) ObjectsStatus(state: str = "none", progress: float = 0.0, message: str = "", count: int = 0)`; `JobStatus.objects: dict` (default `asdict(ObjectsStatus())`; `JobStatus.objects_status -> ObjectsStatus` property); `OBJECTS_STATES = ("none", "queued", "running", "done", "failed")`; `JobStore.queue: queue.Queue[tuple[str, str]]` with items `("recon", job_id)` / `("objects", job_id)` (every existing `queue.put(job_id)` becomes `queue.put(("recon", job_id))`); `request_objects(job_id: str) -> None` raising `KeyError` (unknown), `JobNotDone` (new exception; state != done or no capture.zip) or `ObjectsActive` (new; objects state queued/running); it sets `objects = {state: queued, progress: 0, message: "排队中", count: 0}`, logs `objects requested`, enqueues; `update_objects(job_id, **fields) -> JobStatus` (merges into the objects dict under the store lock); `recover_after_restart` re-queues objects `queued|running` as `queued`; `delete` raises `JobActive` when objects state is `running`.
- Produces (worker): `ObjectsPipeline = Callable[[Path, OnProgress], int]`; `Worker(store, pipeline, objects_pipeline: ObjectsPipeline | None = None)`; `_loop` dispatches on kind; `_run_objects(job_id)`: skips if status gone or objects state != queued; `running` + progress via `update_objects`; on success `done`, `progress 1.0`, `message "完成"`, `count`; on failure `failed` with `objects_failure_message(e, job_id)` (`OOM_MESSAGE` for OOM, `模型下载失败，请检查网络` for `ModelDownloadFailed`, else the generic text from Global Constraints) and the traceback in `log.txt`.
- Produces (API): `POST /jobs/{id}/objects` → 202 `{"job_id", "objects": {...}}`; 404 `任务不存在`; 409 `任务尚未完成` (JobNotDone) or `正在识别` (ObjectsActive). `GET /jobs/{id}` and `GET /jobs` rows include `"objects": {state, progress, message, count}`. `GET /jobs/{id}/objects.zip` → `FileResponse` or 404 `没有物体表`. Status page: new column 「物体」 rendering 无 / 排队 / 识别中 N% / N 个（link to objects.zip）/ 失败.
- `main.build_app` passes `objects_pipeline=lambda d, p: run_objects(d, profile, p)` (lazy import inside).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_jobs.py (append)
import io, time
from recon3d.jobs.store import JobNotDone, ObjectsActive

def _done_job(store, tmp_path):
    job_id = store.create(io.BytesIO(make_capture(tmp_path, n_frames=2).read_bytes()))
    store.queue.get_nowait()  # drop the recon work item
    store.update(job_id, state="done", progress=1.0, message="完成")
    return job_id

def test_request_objects_queues_and_reports(tmp_path):
    store = JobStore(tmp_path / "d")
    job_id = _done_job(store, tmp_path)
    assert store.get(job_id).objects == {"state": "none", "progress": 0.0, "message": "", "count": 0}
    store.request_objects(job_id)
    assert store.queue.get_nowait() == ("objects", job_id)
    assert store.get(job_id).objects_status.state == "queued"
    with pytest.raises(ObjectsActive):
        store.request_objects(job_id)
    with pytest.raises(KeyError):
        store.request_objects("nope")

def test_request_objects_needs_a_done_job(tmp_path):
    store = JobStore(tmp_path / "d")
    job_id = store.create(io.BytesIO(make_capture(tmp_path, n_frames=2).read_bytes()))
    with pytest.raises(JobNotDone):
        store.request_objects(job_id)

def test_restart_requeues_objects_work_and_keeps_scene_done(tmp_path):
    store = JobStore(tmp_path / "d")
    job_id = _done_job(store, tmp_path)
    store.request_objects(job_id); store.queue.get_nowait()
    store.update_objects(job_id, state="running", progress=0.4)
    store2 = JobStore(tmp_path / "d"); store2.recover_after_restart()
    s = store2.get(job_id)
    assert s.state == "done" and s.objects_status.state == "queued"
    assert store2.queue.get_nowait() == ("objects", job_id)

def test_delete_refused_while_objects_running_allowed_while_queued(tmp_path):
    store = JobStore(tmp_path / "d")
    job_id = _done_job(store, tmp_path)
    store.request_objects(job_id)
    store.update_objects(job_id, state="running")
    with pytest.raises(JobActive):
        store.delete(job_id)
    store.update_objects(job_id, state="queued")
    store.delete(job_id)
    assert store.get(job_id) is None

def test_worker_runs_objects_work_and_records_result(tmp_path):
    store = JobStore(tmp_path / "d")
    job_id = _done_job(store, tmp_path)
    seen = []
    def objects_pipeline(job_dir, on_progress):
        on_progress(0.3, "识别物体 (检测)"); on_progress(1.0, None); seen.append(job_dir); return 7
    w = Worker(store, lambda d, m, p: None, objects_pipeline=objects_pipeline); w.start()
    try:
        store.request_objects(job_id)
        end = time.monotonic() + 5
        while store.get(job_id).objects_status.state != "done" and time.monotonic() < end:
            time.sleep(0.01)
    finally:
        w.stop()
    o = store.get(job_id).objects_status
    assert o.state == "done" and o.count == 7 and o.progress == 1.0 and seen == [store.job_dir(job_id)]
    assert store.get(job_id).state == "done"

def test_worker_objects_failure_messages(tmp_path):
    from recon3d.objects.pipeline import ModelDownloadFailed
    store = JobStore(tmp_path / "d")
    job_id = _done_job(store, tmp_path)
    def boom(job_dir, on_progress): raise ModelDownloadFailed("hf down")
    w = Worker(store, lambda d, m, p: None, objects_pipeline=boom); w.start()
    try:
        store.request_objects(job_id)
        end = time.monotonic() + 5
        while store.get(job_id).objects_status.state != "failed" and time.monotonic() < end:
            time.sleep(0.01)
    finally:
        w.stop()
    assert store.get(job_id).objects_status.message == "模型下载失败，请检查网络"
    assert "hf down" in (store.job_dir(job_id) / "log.txt").read_text(encoding="utf-8")
```

```python
# tests/test_api.py (append; reuse the module's client/store fixtures and fake pipeline pattern)
def test_objects_endpoints(client, store, tmp_path):
    job_id = _done_job_via_store(store, tmp_path)  # helper like tests/test_jobs._done_job
    r = client.get(f"/api/v1/jobs/{job_id}")
    assert r.json()["objects"] == {"state": "none", "progress": 0.0, "message": "", "count": 0}
    assert client.get(f"/api/v1/jobs/{job_id}/objects.zip").status_code == 404
    r = client.post(f"/api/v1/jobs/{job_id}/objects")
    assert r.status_code == 202 and r.json()["objects"]["state"] == "queued"
    assert client.post(f"/api/v1/jobs/{job_id}/objects").status_code == 409
    assert client.post("/api/v1/jobs/000000000000/objects").status_code == 404
    assert any(j["job_id"] == job_id and j["objects"]["state"] == "queued" for j in client.get("/api/v1/jobs").json()["jobs"])
    (store.job_dir(job_id) / "objects.zip").write_bytes(b"PK\x05\x06" + b"\0" * 18)
    store.update_objects(job_id, state="done", progress=1.0, message="完成", count=3)
    r = client.get(f"/api/v1/jobs/{job_id}/objects.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"

def test_objects_request_on_unfinished_job_is_409(client, store, tmp_path):
    job_id = store.create(io.BytesIO(make_capture(tmp_path, n_frames=2).read_bytes()))
    r = client.post(f"/api/v1/jobs/{job_id}/objects")
    assert r.status_code == 409 and r.json()["detail"] == "任务尚未完成"

def test_status_page_has_objects_column():
    from recon3d.api.status_page import HTML
    assert "<th>物体</th>" in HTML and "objects.zip" in HTML
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_jobs.py tests/test_api.py` → failures on missing names; note any existing test that asserts the queue holds bare ids (fix them to the tuple form as part of Step 3).

- [ ] **Step 3: Implement** store / worker / API / status page / main wiring.

- [ ] **Step 4: Run** the CPU suite → all passed.

- [ ] **Step 5: Commit** — `git commit -m "feat(server): objects work type, POST /jobs/{id}/objects, objects.zip, status page column"`

---

### Task 9: Server docs

**Files:**
- Modify: `docs/api.md` (objects block in GET bodies; `POST /jobs/{id}/objects`; `GET /jobs/{id}/objects.zip`; product format §), `docs/setup/01-server-laptop.md` (first-run model download note, torchvision), create `docs/acceptance/phase2.md` with spec §6 items 1–9 as a checklist.

- [ ] **Step 1: Write the docs.** `docs/api.md` documents the `objects.json` schema verbatim from spec §4.5 and the `objects.bin` layout.
- [ ] **Step 2: Commit** — `git commit -m "docs: objects API, setup note, phase 2 acceptance checklist"`

---

### Task 10: viewer-web — highlight maths and Spark `splatRgba` highlight

**Files:**
- Create: `android/viewer-web/src/objects-math.js`, `android/viewer-web/test/objects-math.test.js`
- Modify: `android/viewer-web/src/viewer.js`, `android/viewer-web/src/index.html`, `android/viewer-web/test/index-html.test.js`

**Interfaces:**
- Produces (`objects-math.js`): `parseObjectIds(buffer: ArrayBuffer, splatCount: number) -> Uint16Array` (throws `Error("objects.bin has N ids, scene has M splats")` on mismatch); `extractRgba(splatBytes: ArrayBuffer) -> Uint8Array[N*4]` (bytes 24..27 of each 32-byte record); `dimRgba(rgba: Uint8Array, ids: Uint16Array, selected: number, { dimAlpha = 0.15 } = {}) -> Uint8Array` (new array; for `ids[i] !== selected`: r=g=b=round(0.299r+0.587g+0.114b), a=round(a*dimAlpha); selected unchanged); `flyPoseFor(center: {x,y,z}, bboxMin: {x,y,z}, bboxMax: {x,y,z}, cameraPos: {x,y,z}, homeLook: {x,y,z} | null) -> { position, lookAt }` per Global Constraints.
- Produces (`viewer.js` → `window`): `loadObjects(url) -> Promise<number>` (fetches `objects.bin`, validates against the loaded mesh's splat count, keeps `objectIds` and the original rgba; resolves to the number of distinct non-zero ids); `highlightObject(id: number) -> void` (id > 0: `mesh.splatRgba = new RgbaArray({ array: dimRgba(...), count: N })`, set `needsUpdate = true`; id 0: `mesh.splatRgba = null`; request a render); `clearObjects()` on `loadScene` (a new scene drops ids and highlight); `flyToObject(id: number, center: {x,y,z}, bboxMin: {x,y,z}, bboxMax: {x,y,z}) -> Promise<void>` (= `flyTo(flyPoseFor(center, bboxMin, bboxMax, camera.position, homeLook), 1000)` then `highlightObject(id)`; `homeLook` = `home.target − home.position` of the current scene). The `.splat` bytes are kept (`sceneBytes`) only while objects are loaded; `loadScene` keeps a copy of the fetched buffer since Spark may take ownership — copy before constructing `SplatMesh`.
- Spike inside Step 3: verify with the office scene (`server/data/jobs/40a80bd6c17f/scene.splat`) served from the scratchpad as in the roam work that `mesh.splatRgba` recolours; if Spark ignores it (no visual change after `needsUpdate`), fall back to: build a modified `.splat` byte buffer with `dimRgba` written back into bytes 24..27 and swap in a new `SplatMesh` (keep the camera). Record which path was taken in a code comment and in the ledger.
- `index.html`: `#roam-toggle` moves to `top: 56px` (the Compose chip takes the top-right corner).

- [ ] **Step 1: Write the failing tests**

```js
// test/objects-math.test.js
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseObjectIds, extractRgba, dimRgba, flyPoseFor } from "../src/objects-math.js";

function splatBytes(rgbas) {
  const out = new Uint8Array(rgbas.length * 32);
  rgbas.forEach((c, i) => out.set(c, i * 32 + 24));
  return out.buffer;
}

test("parseObjectIds reads little-endian uint16 and rejects a count mismatch", () => {
  const ids = parseObjectIds(new Uint8Array([1, 0, 0, 0, 255, 255]).buffer, 3);
  assert.deepEqual([...ids], [1, 0, 65535]);
  assert.throws(() => parseObjectIds(new Uint8Array([1, 0]).buffer, 3), /1 ids, scene has 3/);
});

test("extractRgba pulls bytes 24..27 of every 32-byte splat", () => {
  const rgba = extractRgba(splatBytes([[10, 20, 30, 255], [1, 2, 3, 128]]));
  assert.deepEqual([...rgba], [10, 20, 30, 255, 1, 2, 3, 128]);
});

test("dimRgba greys and fades everything but the selected object", () => {
  const rgba = new Uint8Array([200, 100, 0, 255, 50, 50, 250, 200]);
  const out = dimRgba(rgba, new Uint16Array([3, 1]), 1);
  const luma = Math.round(0.299 * 200 + 0.587 * 100 + 0.114 * 0);
  assert.deepEqual([...out], [luma, luma, luma, Math.round(255 * 0.15), 50, 50, 250, 200]);
  assert.notEqual(out, rgba); // input untouched
  assert.deepEqual([...rgba.slice(0, 4)], [200, 100, 0, 255]);
});

test("flyPoseFor stands back along the camera's horizontal direction and looks at the centre", () => {
  const c = { x: 0, y: 1, z: 0 };
  const p = flyPoseFor(c, { x: -0.1, y: 0.9, z: -0.1 }, { x: 0.1, y: 1.1, z: 0.1 }, { x: 3, y: 5, z: 0 }, null);
  const d = Math.max(1.5 * Math.hypot(0.2, 0.2, 0.2), 0.8); // -> 0.8
  assert.ok(Math.abs(p.position.x - d) < 1e-9 && Math.abs(p.position.z) < 1e-9);
  assert.ok(Math.abs(p.position.y - (1 + 0.3 * d)) < 1e-9);
  assert.deepEqual(p.lookAt, c);
});

test("flyPoseFor falls back to the reverse home look when the camera is above the object", () => {
  const c = { x: 0, y: 0, z: 0 };
  const p = flyPoseFor(c, { x: -1, y: -1, z: -1 }, { x: 1, y: 1, z: 1 }, { x: 0, y: 4, z: 0 }, { x: 0, y: -0.5, z: -1 });
  const d = 1.5 * Math.sqrt(12);
  assert.ok(Math.abs(p.position.z - d) < 1e-9 && Math.abs(p.position.x) < 1e-9); // behind = +Z when look is -Z
});
```

Append to `test/index-html.test.js`:

```js
test("the roam toggle leaves the top-right corner to the app's objects chip", () => {
  const html = readFileSync(new URL("../src/index.html", import.meta.url), "utf8");
  assert.match(html, /#roam-toggle \{[^}]*top: 56px/);
});
```

- [ ] **Step 2: Run** `npm test` → objects-math import fails; roam-toggle test fails.

- [ ] **Step 3: Implement `objects-math.js`, wire `viewer.js`, run the spike** (serve `index.html` + bundle + `scene/scene.splat` + a hand-made `scene/objects.bin` of all-zeros except ids 1 for the first 50 000 splats from `http://127.0.0.1:8777`, call `await loadObjects('/scene/objects.bin'); highlightObject(1)` in the console and screenshot). Decide dyno-free `splatRgba` vs rebuild fallback.

- [ ] **Step 4: Run** `npm test` → all passed; `npm run build`.

- [ ] **Step 5: Commit** — `git commit -m "feat(viewer): loadObjects/highlightObject with per-splat dimming"`

---

### Task 11: Android — API, record fields, objects table, ObjectsViewModel

**Files:**
- Modify: `android/app/src/main/java/dev/recon3d/net/ApiClient.kt`, `rooms/RoomRepository.kt`, `viewer/ScenePathPolicy.kt`, `upload/UploadScreen.kt`, `AppGraph.kt`
- Create: `viewer/ObjectsTable.kt`, `viewer/ObjectsViewModel.kt`
- Test: `test/.../net/ApiClientTest.kt`, `viewer/ScenePathPolicyTest.kt`, `viewer/ObjectsTableTest.kt`, `viewer/ObjectsViewModelTest.kt`, `upload/UploadViewModelTest.kt`, `upload/Fakes.kt`

**Interfaces:**
- `net`: `@Serializable data class ObjectsStatus(val state: String = "none", val progress: Float = 0f, val message: String = "", val count: Int = 0)`; `JobStatus.objects: ObjectsStatus = ObjectsStatus()`; `Api.requestObjects(jobId: String): ObjectsStatus` (POST, 404/409 → `ApiException` with server detail); `Api.downloadObjects(jobId: String, dest: File, onRetry: (Int) -> Unit = {})` (GET `objects.zip`, same resumable `.part` logic as `downloadScene`, factor the loop into a private `downloadFile(urlSegments, dest, onRetry)`).
- `rooms`: `CaptureRecord` gains `objectsState: String = "none"`, `objectsProgress: Float = 0f`, `objectsMessage: String? = null`, `objectsCount: Int = 0`, `objectsPath: String? = null` (directory holding `objects.json`, `objects.bin`, `thumbs/`). `UploadViewModel.rerun` success and `start`/`retry` reset all five to defaults.
- `viewer/ScenePathPolicy`: `const val OBJECTS_NAME = "objects.bin"`; `allows(path)` true for both names; `resolve(requestPath, scenePath, objectsPath: String? = null)` serves `File(objectsPath, "objects.bin")` for `OBJECTS_NAME`.
- `viewer/ObjectsTable.kt`: `@Serializable data class ObjectEntry(val id: Int, val label: String, @SerialName("label_zh") val labelZh: String, val score: Float, val views: Int, val center: List<Float>, val bbox: Bbox, @SerialName("gaussian_count") val gaussianCount: Int, val thumb: String)`, `@Serializable data class Bbox(val min: List<Float>, val max: List<Float>)`, `@Serializable data class ObjectsTable(val version: Int, @SerialName("job_id") val jobId: String, val objects: List<ObjectEntry>)` (ignore unknown keys); `ObjectsTable.load(dir: File): ObjectsTable`; `data class ObjectGroup(val labelZh: String, val items: List<ObjectEntry>)`; `fun ObjectsTable.grouped(): List<ObjectGroup>` (by `labelZh`, groups ordered by size desc then name, items by id); `fun itemTitle(group: ObjectGroup, index: Int) = "${group.labelZh} ${index + 1}"`; `fun groupTitle(g: ObjectGroup) = "${g.labelZh} ×${g.items.size}"`.
- `viewer/ObjectsViewModel.kt`: `class ObjectsViewModel(api: Api, repo: RoomRepository, scenesDir: File, poll: Duration = 3.seconds, scope: CoroutineScope = ...)`; `fun start(roomId, captureId)` (needs `record.jobId`, else sets `objectsState = "failed"`, `objectsMessage = MSG_OBJECTS_NO_JOB = "任务已不在服务端，请先重新上传"`); `fun resume(roomId, captureId)` (re-enter polling for `queued|running` records at app start / screen entry); polling via `api.job(jobId).objects`; on `done`: download `objects.zip` to `<scenesDir>/<roomId>/<captureId>/objects.zip`, unzip into `objects.tmp/`, verify `objects.json` + `objects.bin` exist, then delete old `objects/` and rename `objects.tmp` → `objects/`, set `objectsPath`, `objectsState = done`, `objectsCount`; on `failed`: state failed + server message, **existing `objectsPath` kept**; 404 from `api.job` → failed with `MSG_JOB_GONE` text `任务不存在`. Pure helpers: `fun objectsChipLabel(record: CaptureRecord): String` = `"识别物体"` (none), `"识别物体 ${pct}%"` with the server message's stage in parentheses when present (`"识别物体 (检测) 42%"`), `"排队中"` (queued), `"物体 (${count})"` (done), `"重试识别"` (failed); `fun objectsFailureText(record) = "识别失败：${objectsMessage}"` for failed records (shown as a toast / snackbar by the screen).
- `AppGraph`: `val objects = ObjectsViewModel(SettingsBackedApi(settings), rooms, File(app.filesDir, "rooms"))`; `SettingsBackedApi` delegates the two new calls.

- [ ] **Step 1: Write the failing tests** — `ApiClientTest`: `requestObjects` posts to `/api/v1/jobs/j1/objects` and decodes `{"job_id":"j1","objects":{"state":"queued",...}}`; 409 surfaces `正在识别`; `downloadObjects` writes `dest` from a 200 body and sends `Range` on retry (mirror the existing `downloadScene` tests); `JobStatus` decodes without an `objects` key (old server) to the default. `ScenePathPolicyTest`: `allows("objects.bin")`, `resolve("objects.bin", scene, objectsDir)` returns the file, null when `objectsPath` is null; traversal still refused. `ObjectsTableTest`: parse the spec §4.5 JSON (with an extra unknown key) → grouped `[椅子 ×2, 杯子 ×1]` for ids chair,chair,cup; `itemTitle` → `"椅子 2"`. `ObjectsViewModelTest` (fakes in `Fakes.kt`: `objectsStatuses: ArrayDeque<ObjectsStatus>`, `objectsZip: ByteArray?`, `requestObjectsCalls`): start → chip `排队中` → `识别物体 (检测) 42%` → done downloads + unzips (fake zip built with `java.util.zip.ZipOutputStream` containing `objects.json`, `objects.bin`, `thumbs/1.jpg`), `objectsPath` ends with `/objects`, `objectsCount == 1`; re-run over an existing result keeps old `objectsPath` until the new zip lands, and a server `failed` keeps it; start without `jobId` → failed with `任务已不在服务端，请先重新上传`; `UploadViewModelTest`: rerun resets `objectsState`/`objectsPath`.

- [ ] **Step 2: Run** `.\gradlew.bat :app:testDebugUnitTest --offline` → compilation errors on new names.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run** → all tests pass (expect ≈ 193 + new).

- [ ] **Step 5: Commit** — `git commit -m "feat(app): objects API, record fields, ObjectsViewModel, objects.bin serving"`

---

### Task 12: Android — viewer UI (chip, list, highlight bar, JS calls)

**Files:**
- Modify: `android/app/src/main/java/dev/recon3d/viewer/ViewerScreen.kt`, `nav/AppNav.kt`
- Test: none beyond Task 11 helpers (Compose UI verified on device per `docs/acceptance/phase2.md`); add `ViewerScreenHelpersTest.kt` only if a new pure helper appears (e.g. `flyToJs(entry, cameraState)` string builder — then test its JSON escaping).

**Interfaces:**
- `ViewerScreen(repo, roomId, captureId, objects: ObjectsViewModel)`; `SceneViewer(scenePath, home, objectsPath: String? = null, onReady: (WebView) -> Unit = {})` keeps a reference to the WebView for `evaluateJavascript`.
- Behaviour: on load (and whenever `objectsPath` changes) call `loadObjects('/scene/objects.bin')`; top-right `AssistChip` with `objectsChipLabel(record)`: tap → `objects.start(...)` when none/failed, opens `ModalBottomSheet` when done, no-op while queued/running; long-press when done → `DropdownMenu` 「重新识别」 → `objects.start(...)`. Sheet: `LazyColumn` with `stickyHeader(groupTitle)` and rows (48 dp `AsyncImage`-free `BitmapFactory.decodeFile` thumbnail, `itemTitle`); tap → `evaluateJavascript("window.flyToObject(${id}, ${centerJson}, ${minJson}, ${maxJson})")` (Task 10's export; JSON built with kotlinx.serialization so numbers are locale-safe); sheet closes; bottom bar `「椅子 1 · 取消高亮」` → `highlightObject(0)`. Failed state shows a `Snackbar` with `objectsFailureText`. `ObjectsViewModel.resume` is called in a `LaunchedEffect` on entry.
- `AppNav` passes `graph.objects`.

- [ ] **Step 1: Implement the UI** (strings: 「识别物体」「排队中」「物体 (N)」「重试识别」「重新识别」「取消高亮」「识别失败：…」).
- [ ] **Step 2: Build** `.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --offline` → BUILD SUCCESSFUL.
- [ ] **Step 3: Commit** — `git commit -m "feat(app): objects chip, list and highlight in the viewer"`

---

### Task 13: End-to-end on the laptop + phone

- [ ] **Step 1:** Restart the server on the new code (check `GET /api/v1/jobs` for active work first; stop the background task, start it again with the same command).
- [ ] **Step 2:** `curl -X POST http://127.0.0.1:8765/api/v1/jobs/40a80bd6c17f/objects`; watch `GET /jobs/40a80bd6c17f` until `objects.state == done`; record the wall time and the per-stage timings from `log.txt` in `docs/acceptance/phase2.md` §"实测".
- [ ] **Step 3:** Install the APK on the Pixel (`adb -s 56121FDCH0091J install -r …`, no active uploads), run `docs/acceptance/phase2.md` items 1–9 with the user; note results.
- [ ] **Step 4: Commit** the acceptance notes — `git commit -m "docs: phase 2 objects acceptance run"`.
