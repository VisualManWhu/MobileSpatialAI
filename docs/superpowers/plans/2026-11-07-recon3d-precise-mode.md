# Precise Reconstruction Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-job `mode` (`fast` | `precise`); in precise mode refine the ARCore poses with pycolmap (SIFT → sequential + spatial matching with ARCore position priors → known-pose triangulation → mapper global BA, intrinsics fixed) before the existing 3DGS training, with fallback to the fast path; expose it as 快速上传 / 高精上传 and 改用高精重算 / 改用快速重算 in the app via a new `POST /jobs/{id}/rerun`.

**Architecture:** Server: new `recon3d/recon/precise_pose.py` (pure function `precise_pose(cap, workdir, on_progress) -> PrecisePoseResult`), `build_training_set(..., c2w_override=)`, `run_pipeline(job_dir, profile, on_progress, mode)` where `on_progress(p, message)` carries the stage text; `JobStore` stores `mode` in `status.json` / `upload.json` and gains `rerun(job_id, mode)`; API/status page surface `mode`. Android: `CaptureRecord.mode`, `Api.upload(zip, uploadId, mode, listener)`, `Api.rerunJob(jobId, mode)`, `UploadViewModel.start(…, mode)` / `rerun(…, mode)`, two upload buttons and a rerun menu item.

**Tech Stack:** Python 3.10, FastAPI, pycolmap 4.2.1 (CPU), numpy, pytest (WSL: `cd /mnt/d/projects/Recon3D/server && export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/recon3d UV_LINK_MODE=copy && uv run pytest -q -m "not gpu"`); Kotlin/Compose, OkHttp 4.12, MockWebServer, kotlinx-coroutines-test (`cd android && $env:JAVA_HOME="C:\Program Files\Android\Android Studio\jbr"; .\gradlew.bat :app:testDebugUnitTest --offline`).

**Spec:** `docs/superpowers/specs/2026-11-05-recon3d-precise-mode-design.md` (§9 adjusts two values: `max_num_neighbors = 20`, `ba_global_max_refinements = 3`).

## Global Constraints

- User-visible strings Chinese, code/comments English. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Fast mode behaviour must not change: every new branch is behind `mode == "precise"`; default `"fast"` everywhere (API bodies, `status.json` without the field, old Android records).
- pycolmap 4.2.1 API (from the spikes): `FeatureExtractionOptions.max_image_size` (not under `.sift`), `Database.open(path)`, `frame.image_ids` yields `data_t` → `.id`, `PosePrior(position=, coordinate_system=CARTESIAN, corr_data_id=frame.data_ids[0], position_covariance=)` + `db.write_pose_prior(prior)`, `Reconstruction.add_camera` **before** `add_rig`, `Image.cam_from_world()` is a method returning `Rigid3d` (`.matrix()` → 3×4), `Reconstruction.write(dir)` needs an existing dir, mapper no-ops on an input with 0 points (triangulate first).
- Pipeline constants: `extract: max_image_size 1600, sift.max_num_features 4096, num_threads os.cpu_count()`; priors `position_covariance = diag(0.3**2)`; `match_sequential(overlap=10, loop_detection=False)`; `match_spatial(max_num_neighbors=20, max_distance=3.0, ignore_z=False)`; mapper `ba_refine_focal_length = ba_refine_principal_point = ba_refine_extra_params = False`, `ba_global_max_refinements = 3`, `use_prior_position = True`, `use_robust_loss_on_prior_position = True`, `prior_position_loss_scale = 0.3`; unregistered = `not has_pose` or < 20 3D points; fallback when `registered.mean() < 0.7`; Umeyama warn thresholds scale ±5 %, rotation 2°.
- Progress: precise stage 0 → `POSE_SHARE = 0.3` (inside it: extract 0→0.35, match 0.35→0.7, mapper 0.7→1.0, mapper estimated by wall time up to 0.95 then 1.0), training `0.3 → 0.9`, export `0.9 → 1.0`; `EXPORT_START = 0.9` unchanged.
- Messages (exact): `位姿优化 (特征提取)`, `位姿优化 (特征匹配)`, `位姿优化 (全局优化)`, `训练中`, `导出中`, `完成`, `完成（高精位姿失败，已按快速模式重建）`; API errors `任务不存在` (404), `上传未完成` (409).
- Android copy (exact): 快速上传 / 高精上传 / 快速重传 / 高精重传 / 重试 / 改用高精重算 / 改用快速重算 / status suffix ` · 高精`.
- Server tests that need pycolmap are marked `@pytest.mark.slow`; `-m "not gpu"` still runs them (they are CPU). Android: 177 tests pass today; all must keep passing.

## Review Focus

1. A capture whose frames carry two different intrinsic sets (e.g. the phone switched focal length mid-capture): precise mode must build one COLMAP camera per rounded (fx, fy, cx, cy) group and keep K per frame identical to what training uses — test in Task 1 (`test_precise_pose_two_intrinsic_groups`).
2. `POST /jobs/{id}/rerun` on a job that is itself still training: allowed (source only needs `capture.zip`), the new job queues behind it — test in Task 4 (`test_rerun_while_source_training_queues`).
3. `POST /uploads` / `POST /jobs` / `rerun` with `mode` outside `fast|precise`: 422, nothing created — test in Task 4 (`test_bad_mode_is_422`).
4. pycolmap not importable (venv without it): precise job still completes via fallback with the Chinese note, `log.txt` has the ImportError — test in Task 3 (`test_precise_import_error_falls_back`).
5. App: "改用高精重算" while the old `scene.splat` exists: "查看 3D" disappears during the rerun but `scenePath` stays until the new download overwrites it; a *failed* rerun keeps the old `scenePath` too — test in Task 7 (`rerunFailureKeepsOldScenePath`).

---

## File map

Server (`server/`):
- Create `recon3d/recon/precise_pose.py` — COLMAP pose refinement (Task 1, 3 for fallback wiring).
- Create `tests/fixtures/room_render.py` — CPU textured-box renderer for the synthetic loop test (Task 1).
- Create `tests/test_precise_pose.py` (Task 1).
- Modify `recon3d/recon/dataset.py` — `c2w_override` (Task 2); `tests/test_dataset.py`.
- Modify `recon3d/jobs/pipeline.py`, `recon3d/jobs/worker.py`, `recon3d/main.py` — mode, stage messages, fallback note (Task 3); `tests/test_pipeline.py`, `tests/test_jobs.py`, `tests/test_api.py` fakes.
- Modify `recon3d/jobs/store.py`, `recon3d/api/app.py`, `recon3d/api/resumable.py`, `recon3d/api/upload.py`, `recon3d/api/status_page.py`, `docs/api.md` — `mode` + rerun (Task 4).
- Modify `scripts/eval_scene.py`; create `scripts/pose_drift.py` (Task 5).
- Modify `pyproject.toml` (pycolmap dependency, Task 1), `docs/setup/01-server-laptop.md`, `docs/acceptance/phase1.md` (Task 9).

Android (`android/app/src/main/java/dev/recon3d/`):
- `rooms/RoomRepository.kt` (`CaptureRecord.mode`), `net/ApiClient.kt` (`Api.upload` mode, `Api.rerunJob`), `AppGraph.kt`, test `Fakes.kt` (Task 6).
- `upload/UploadScreen.kt` view model + `captureStatusText` (Task 7).
- `upload/UploadScreen.kt` `CaptureRow`, `capture/PreviewScreen.kt`, `rooms/RoomListScreen.kt`, `nav/AppNav.kt` (Task 8).

---

### Task 1: `precise_pose` core with a synthetic loop-closure test

**Files:**
- Modify: `server/pyproject.toml` (add `"pycolmap==4.2.1"` to `dependencies`)
- Create: `server/recon3d/recon/precise_pose.py`
- Create: `server/tests/fixtures/room_render.py`
- Test: `server/tests/test_precise_pose.py`

**Interfaces:**
- Consumes: `recon3d.capture.format.Capture` (`cap.root: Path`, `cap.frames[i].image` relative path like `images/000001.jpg`, `.pose` 4×4 GL c2w ndarray, `.fx .fy .cx .cy`), `recon3d.capture.poses.gl_to_cv_c2w(pose) -> np.ndarray`.
- Produces:
  ```python
  @dataclass
  class PrecisePoseResult:
      c2w_cv: np.ndarray          # [N,4,4] float64, OpenCV convention, ARCore world frame
      registered: np.ndarray      # [N] bool
      stats: dict                 # keys below

  class PrecisePoseFailed(Exception):
      def __init__(self, ratio: float, cause: BaseException | None = None): ...
      # str(e) is used in log.txt; .ratio in 0..1

  def precise_pose(cap: Capture, workdir: Path, on_progress: Callable[[float], None]) -> PrecisePoseResult
  ```
  `stats` keys: `frames, registered, registered_ratio, points3d, reproj_px, shift_mean_m, shift_median_m, shift_max_m, rot_mean_deg, scale, rot_deg, t_extract_s, t_match_s, t_mapper_s`.
  Module constants: `MAX_IMAGE_SIZE = 1600`, `MAX_FEATURES = 4096`, `SEQ_OVERLAP = 10`, `SPATIAL_NEIGHBORS = 20`, `SPATIAL_MAX_DIST_M = 3.0`, `PRIOR_SIGMA_M = 0.3`, `BA_GLOBAL_REFINEMENTS = 3`, `MIN_POINTS_REGISTERED = 20`, `MIN_REGISTERED_RATIO = 0.7`.
  Also `tests/fixtures/room_render.py`: `render_room(c2w_cv: np.ndarray, K: np.ndarray, W: int, H: int, seed: int = 0) -> np.ndarray` (uint8 H×W×3) and `loop_trajectory(n: int) -> np.ndarray` ([n,4,4] CV c2w, closed loop inside the room).

- [ ] **Step 1: Add the dependency and check it resolves**

Add `"pycolmap==4.2.1",` under `transformers` in `pyproject.toml` with the comment `# CPU SIFT + mapper for precise mode (recon/precise_pose.py)`.
Run: `uv sync` (WSL env as in Tech Stack) then `uv run python -c "import pycolmap; print(pycolmap.__version__, pycolmap.has_cuda)"`
Expected: `4.2.1 False`

- [ ] **Step 2: Write the renderer fixture (test code, so no failing test first)**

`render_room` ray-casts a pinhole camera against the inside of the axis-aligned box `[-3,3]×[-1.5,1.5]×[-3,3]` m (x, y-up, z): for every pixel, ray direction `d = R · K⁻¹ [u,v,1]`; for each axis `a` with `d[a] != 0`, the hit parameter is `t_a = (bound_a(sign d[a]) - o[a]) / d[a]`, take `t = min` of the positive ones, `p = o + t·d`, the face index is the argmin axis + sign, and the colour is a bilinear lookup of a per-face texture at `(p[other axes] + 3) / 6` (floor/ceiling use x,z; use the two non-hit axes scaled to the face size). Textures: `rng = np.random.default_rng(seed + face)`, `tex = rng.integers(0, 256, (256, 256, 3))`, `scipy.ndimage.gaussian_filter(tex, (1.5, 1.5, 0))`, contrast-stretched to 0..255 — blobby enough for SIFT. Vectorise over all pixels with numpy; no loops over pixels.

`loop_trajectory(n)`: camera centres on a circle of radius 1.2 m in the horizontal plane at y = 0 (`angle = 2π·i/n`), each looking outward at the wall along the radial direction with up = +y (build c2w with the look-at helper pattern of `tests/fixtures/synthetic.py::look_at_c2w`, but in OpenCV convention: +z forward, +y down — i.e. `R = [right, -up, forward]` columns). Frame `n-1` is 1/n of a turn before frame 0, so the loop closes.

- [ ] **Step 3: Write the failing loop-closure test**

```python
import numpy as np, pytest
from recon3d.capture.format import load_capture
from tests.fixtures.make_capture import make_capture
from tests.fixtures.room_render import render_room, loop_trajectory

W, H, N = 320, 240, 40
K = np.array([[280.0, 0, 160.0], [0, 280.0, 120.0], [0, 0, 1]])
GL = np.diag([1.0, -1.0, -1.0, 1.0])

def _gl_colmajor(c2w_cv):            # make_capture wants 16 floats, column-major, OpenGL c2w
    return (c2w_cv @ GL).T.flatten().tolist()

def _drifted(truth):                 # accumulated VIO-like drift: end frame off by 0.3 m / 3 deg
    out = []
    for i, T in enumerate(truth):
        s = i / (len(truth) - 1)
        a = np.deg2rad(3.0 * s)
        Rd = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        D = np.eye(4); D[:3, :3] = Rd; D[:3, 3] = [0.3 * s, 0, 0]
        out.append(D @ T)
    return np.stack(out)

def _loop_capture(tmp_path, blank=()):
    truth = loop_trajectory(N)
    noisy = _drifted(truth)
    images = [render_room(truth[i], K, W, H) for i in range(N)]
    for b in blank:
        images[b] = np.full((H, W, 3), 128, np.uint8)
    z = make_capture(tmp_path, n_frames=N, with_depth=False, image_width=W, image_height=H,
                     fx=280.0, fy=280.0, cx=160.0, cy=120.0,
                     poses=[_gl_colmajor(p) for p in noisy], images=images)
    return load_capture(z, tmp_path / "w"), truth, noisy

def _center_err(a, b):
    return np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1)

@pytest.mark.slow
def test_loop_closes_and_poses_improve(tmp_path):
    from recon3d.recon.precise_pose import precise_pose
    cap, truth, noisy = _loop_capture(tmp_path)
    seen = []
    res = precise_pose(cap, tmp_path / "colmap", seen.append)
    assert res.c2w_cv.shape == (N, 4, 4) and res.registered.shape == (N,)
    assert res.registered.mean() >= 0.9
    rel_in = np.linalg.inv(noisy[0]) @ noisy[-1]
    rel_out = np.linalg.inv(res.c2w_cv[0]) @ res.c2w_cv[-1]
    rel_gt = np.linalg.inv(truth[0]) @ truth[-1]
    err_in = np.linalg.norm(rel_in[:3, 3] - rel_gt[:3, 3])
    err_out = np.linalg.norm(rel_out[:3, 3] - rel_gt[:3, 3])
    assert err_out * 5 < err_in
    assert np.median(_center_err(res.c2w_cv, truth)) < np.median(_center_err(noisy, truth)) / 3
    assert 0.98 <= res.stats["scale"] <= 1.02
    assert seen == sorted(seen) and seen[-1] == 1.0 and seen[0] <= 0.35
    for k in ("registered_ratio", "shift_mean_m", "rot_mean_deg", "points3d", "reproj_px", "t_extract_s"):
        assert k in res.stats

@pytest.mark.slow
def test_blank_frames_unregistered_follow_neighbour_correction(tmp_path):
    from recon3d.recon.precise_pose import precise_pose
    cap, truth, noisy = _loop_capture(tmp_path, blank=(10, 11))
    res = precise_pose(cap, tmp_path / "colmap", lambda p: None)
    assert not res.registered[10] and not res.registered[11] and res.registered[9] and res.registered[12]
    # frame 10 inherits frame 9's rigid correction; frame 11 inherits frame 12's (nearest registered, ties -> earlier)
    exp10 = res.c2w_cv[9] @ np.linalg.inv(noisy[9]) @ noisy[10]
    exp11 = res.c2w_cv[12] @ np.linalg.inv(noisy[12]) @ noisy[11]
    assert np.allclose(res.c2w_cv[10], exp10, atol=1e-6)
    assert np.allclose(res.c2w_cv[11], exp11, atol=1e-6)

@pytest.mark.slow
def test_precise_pose_two_intrinsic_groups(tmp_path):
    """Frames 0..19 fx=280, frames 20..39 fx=300: one COLMAP camera per group, all frames still register."""
    from recon3d.recon.precise_pose import precise_pose
    truth = loop_trajectory(N)
    Ks = [K if i < 20 else np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1]]) for i in range(N)]
    images = [render_room(truth[i], Ks[i], W, H) for i in range(N)]
    z = make_capture(tmp_path, n_frames=N, with_depth=False, image_width=W, image_height=H,
                     poses=[_gl_colmajor(p) for p in _drifted(truth)], images=images, fx=280.0, fy=280.0, cx=160.0, cy=120.0)
    cap = load_capture(z, tmp_path / "w")
    for i in range(20, N):               # make_capture has one global fx; patch the loaded frames
        cap.frames[i].fx = cap.frames[i].fy = 300.0
    res = precise_pose(cap, tmp_path / "colmap", lambda p: None)
    assert res.stats["cameras"] == 2
    assert res.registered.mean() >= 0.9
```

Note for the two-group test: `tests/fixtures/make_capture.py` writes one `fx` per package; patching `cap.frames[i].fx` after `load_capture` is the cheapest way to get two groups. Add `"cameras"` to the stats keys (number of COLMAP cameras).

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_precise_pose.py -v -x`
Expected: FAIL with `ModuleNotFoundError: No module named 'recon3d.recon.precise_pose'` (the renderer import must already succeed — fix the fixture first if it is what fails).

- [ ] **Step 5: Implement `precise_pose`**

`precise_pose(cap, workdir, on_progress)`:
1. `workdir.mkdir(parents=True, exist_ok=True)`; image dir = `cap.root / "images"` (the frames' image paths all start with `images/`; assert that and use the file name part as the COLMAP image name).
2. Group frames by `tuple(round(v, 1) for v in (fx, fy, cx, cy))`; for each group call `pycolmap.extract_features(db, image_dir, image_names=[names], camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ImageReaderOptions(camera_model="PINHOLE", camera_params="fx,fy,cx,cy"), extraction_options=...)` — one call per group yields one camera per group. Report `on_progress` 0.35·(groups done / groups) during this phase.
3. Open the DB, `clear_pose_priors()`, write one prior per frame as the Global Constraints say (position = `gl_to_cv_c2w(frame.pose)[:3, 3]`), then `match_sequential` and `match_spatial` (progress 0.35 → 0.55 → 0.7).
4. Build the initial `Reconstruction` exactly as the spike did (cameras → rigs → frames with `rig_from_world = inv(c2w_cv)` → images → `register_frame`), `triangulate_points(rec, db, image_dir, workdir / "tri", refine_intrinsics=False)`, write to `workdir / "init"`, then `incremental_mapping(db, image_dir, workdir / "out", options=IncrementalPipelineOptions(...), input_path=workdir / "init")`; take the reconstruction with the most registered images from the returned dict. While the mapper runs, no callback exists: start it in a thread and from the caller loop every 2 s calling `on_progress(0.7 + 0.25 * min(1, elapsed / estimate))` with `estimate = 1.5 * (t_extract + t_match)`; call `on_progress(1.0)` at the end.
5. Registered mask per frame: `image.has_pose` and `sum(p.has_point3D() for p in image.points2D) >= MIN_POINTS_REGISTERED`. Unregistered frames: nearest registered index by `abs(j - i)` (ties → smaller j), pose = `c2w_new[j] @ inv(c2w_arcore[j]) @ c2w_arcore[i]`.
6. Umeyama with scale on the registered centres (BA → ARCore), full least squares, no RANSAC; `log.warning` if `abs(scale - 1) > 0.05` or rotation angle > 2°; record `scale`, `rot_deg`. **Do not apply** the alignment to the poses (the priors already pin the frame) — it is a sanity statistic only.
7. If `registered.mean() < MIN_REGISTERED_RATIO`: `raise PrecisePoseFailed(ratio)`.
8. Fill `stats`, `log.info` one line with them, return.

`gl_to_cv_c2w` already exists in `recon3d/capture/poses.py`; reuse it, do not re-derive the GL→CV flip.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_precise_pose.py -v`
Expected: 3 passed (expect ~1–3 min total on CPU). If `test_loop_closes_and_poses_improve` fails on `err_out * 5 < err_in`, the renderer's texture is too smooth for SIFT — lower the gaussian sigma to 1.0 before touching thresholds.

- [ ] **Step 7: Run the whole non-GPU suite**

Run: `uv run pytest -q -m "not gpu"`
Expected: all passed (150 before this task + 3).

- [ ] **Step 8: Commit**

```bash
git add server/pyproject.toml server/uv.lock server/recon3d/recon/precise_pose.py server/tests/fixtures/room_render.py server/tests/test_precise_pose.py
git commit -m "feat(server): precise_pose — pycolmap pose refinement with ARCore priors"
```

---

### Task 2: `build_training_set(c2w_override=)`

**Files:**
- Modify: `server/recon3d/recon/dataset.py:50` (`build_training_set` signature and the pose/depth loop)
- Test: `server/tests/test_dataset.py`

**Interfaces:**
- Produces: `build_training_set(cap: Capture, profile: Profile, device: str = "cpu", c2w_override: np.ndarray | None = None) -> TrainingSet`. With an override, `ts.c2w[i] = c2w_override[i]` and `ts.depth_c2w[i] = c2w_override[i] @ inv(c2w_arcore_cv[i]) @ depth_c2w_arcore_cv[i]` for frames that have a depth pose (None stays None). Wrong shape → `ValueError`.

- [ ] **Step 1: Write the failing tests**

```python
def test_c2w_override_replaces_poses_and_corrects_depth_pose(tmp_path):
    dp = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.3, 0.0, 0.05, 1]
    z = make_capture(tmp_path, n_frames=3, format_version=2, depth_poses=[dp, None, dp], with_depth=True)
    cap = load_capture(z, tmp_path / "w")
    base = build_training_set(cap, PROFILES["laptop"])
    override = base.c2w.numpy().astype(np.float64).copy()
    shift = np.eye(4); shift[:3, 3] = [1.0, 2.0, 3.0]
    override[0] = shift @ override[0]
    ts = build_training_set(cap, PROFILES["laptop"], c2w_override=override)
    assert torch.allclose(ts.c2w, torch.tensor(override, dtype=torch.float32))
    assert ts.depth_c2w[1] is None
    expected0 = torch.tensor(override[0], dtype=torch.float32) @ torch.linalg.inv(base.c2w[0]) @ base.depth_c2w[0]
    assert torch.allclose(ts.depth_c2w[0], expected0, atol=1e-5)
    assert torch.allclose(ts.depth_c2w[2], base.depth_c2w[2], atol=1e-6)   # frame 2 unchanged

def test_c2w_override_bad_shape_raises(tmp_path):
    z = make_capture(tmp_path, n_frames=2)
    cap = load_capture(z, tmp_path / "w")
    with pytest.raises(ValueError):
        build_training_set(cap, PROFILES["laptop"], c2w_override=np.zeros((3, 4, 4)))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_dataset.py -k override -v`
Expected: FAIL with `TypeError: build_training_set() got an unexpected keyword argument 'c2w_override'`

- [ ] **Step 3: Implement**

Validate `c2w_override.shape == (len(cap.frames), 4, 4)` up front. In the loop keep `arcore = gl_to_cv_c2w(fr.pose)`; `pose = c2w_override[i] if c2w_override is not None else arcore`; when a depth pose exists and an override is given, compute the corrected depth pose in float64 before converting to the tensor.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add server/recon3d/recon/dataset.py server/tests/test_dataset.py
git commit -m "feat(server): build_training_set c2w_override with depth-pose correction"
```

---

### Task 3: Pipeline / worker mode plumbing with stage messages and fallback

**Files:**
- Modify: `server/recon3d/jobs/pipeline.py`, `server/recon3d/jobs/worker.py:12-16,62-88`, `server/recon3d/main.py:17-19`
- Modify: `server/recon3d/jobs/store.py:78-84` (`JobStatus.mode`), `:147-150` (`_commit(job_id, size, mode)`), `:152-160` (`get` defaults missing `mode` to `"fast"`)
- Test: `server/tests/test_pipeline.py`, `server/tests/test_jobs.py`, `server/tests/test_api.py` (update the fake pipelines)

**Interfaces:**
- Consumes: `precise_pose`, `PrecisePoseFailed` (Task 1); `build_training_set(..., c2w_override)` (Task 2).
- Produces:
  ```python
  # pipeline.py
  POSE_SHARE = 0.3
  TRAIN_SHARE = 0.9
  STAGE_EXTRACT = "位姿优化 (特征提取)"; STAGE_MATCH = "位姿优化 (特征匹配)"; STAGE_MAPPER = "位姿优化 (全局优化)"
  FALLBACK_NOTE = "完成（高精位姿失败，已按快速模式重建）"
  OnProgress = Callable[[float, str | None], None]          # (progress, message or None = default text)
  def run_pipeline(job_dir: Path, profile: Profile, on_progress: OnProgress, mode: str = "fast") -> str | None
      # returns the final message to use instead of "完成" (FALLBACK_NOTE) or None
  # worker.py
  Pipeline = Callable[[Path, str, OnProgress], str | None]   # (job_dir, mode, on_progress)
  # store.py
  @dataclass(frozen=True) class JobStatus: job_id, state, progress, message, created_at, mode: str = "fast"
  JobStore._commit(self, job_id: str, size: int, mode: str = "fast") -> None   # writes mode into status.json
  ```
  The worker passes `store.get(job_id).mode` to the pipeline; `on_progress(p, msg)` sets `message = msg` when `msg` is not None, else `训练中` / `导出中` by `p` as today. On a `str` return the worker uses it as the final `done` message.

- [ ] **Step 1: Write the failing tests**

`tests/test_jobs.py` (replace every `lambda d, p:` fake with `lambda d, m, p:` and `def pipe(d, p)` with `def pipe(d, m, p)`; same in `tests/test_api.py::_fast` → `def _fast(d, m, p)` and the `client_slow` lambda), then add:

```python
def test_status_mode_defaults_to_fast_for_old_status_json(store):
    j = store.create(io.BytesIO(b"x"))
    d = store.job_dir(j)
    raw = json.loads((d / "status.json").read_text(encoding="utf-8")); raw.pop("mode")
    (d / "status.json").write_text(json.dumps(raw), encoding="utf-8")
    assert store.get(j).mode == "fast"

def test_worker_passes_mode_and_stage_messages(store, make_worker):
    j = store.create(io.BytesIO(b"x"))
    store.update(j, mode="precise")
    seen = []
    def pipe(d, mode, on_progress):
        seen.append(mode)
        on_progress(0.1, "位姿优化 (特征提取)")
        seen.append(store.get(j).message)
        on_progress(0.5, None)
        seen.append(store.get(j).message)
        return "完成（高精位姿失败，已按快速模式重建）"
    w = make_worker(pipe); w.start()
    wait_until(lambda: store.get(j).state == "done")
    assert seen == ["precise", "位姿优化 (特征提取)", "训练中"]
    assert store.get(j).message == "完成（高精位姿失败，已按快速模式重建）"
    assert store.get(j).progress == 1.0

def test_worker_default_done_message_when_pipeline_returns_none(store, make_worker):
    j = store.create(io.BytesIO(b"x"))
    w = make_worker(lambda d, m, p: None); w.start()
    wait_until(lambda: store.get(j).state == "done")
    assert store.get(j).message == "完成"
```

`tests/test_pipeline.py`:

```python
def test_pipeline_fast_mode_unchanged_progress(tmp_path):     # replaces the body of test_pipeline_end_to_end's progress asserts
    ... run_pipeline(job, profile, lambda p, m: seen.append((p, m))) ...
    ps = [p for p, _ in seen]
    assert ps == sorted(ps) and ps[-1] == 1.0 and any(abs(p - 0.9) < 1e-9 for p in ps)
    assert all(m is None for _, m in seen)

def test_pipeline_precise_uses_refined_poses_and_stage_messages(tmp_path, monkeypatch):
    import recon3d.jobs.pipeline as pl
    job = tmp_path / "job"; job.mkdir()
    shutil.copy(make_capture(tmp_path, n_frames=4), job / "capture.zip")
    captured = {}
    def fake_precise(cap, workdir, on_progress):
        on_progress(0.2); on_progress(0.6); on_progress(1.0)
        c2w = np.stack([gl_to_cv_c2w(f.pose) for f in cap.frames]); c2w[:, 0, 3] += 1.0
        return PrecisePoseResult(c2w_cv=c2w, registered=np.ones(4, bool), stats={})
    real_bts = pl_dataset.build_training_set
    def spy_bts(cap, profile, device="cpu", c2w_override=None):
        captured["override"] = None if c2w_override is None else c2w_override.copy()
        return real_bts(cap, profile, device, c2w_override)
    monkeypatch.setattr(pl, "precise_pose", fake_precise, raising=False)   # pipeline imports it lazily: patch where it looks it up
    monkeypatch.setattr(pl_dataset, "build_training_set", spy_bts)
    seen = []
    note = run_pipeline(job, replace(PROFILES["laptop"], iterations=50), lambda p, m: seen.append((p, m)), mode="precise")
    assert note is None
    assert captured["override"] is not None and captured["override"][0, 0, 3] != 0
    msgs = [m for _, m in seen if m]
    assert msgs[0] == "位姿优化 (特征提取)" and "位姿优化 (特征匹配)" in msgs and "位姿优化 (全局优化)" in msgs
    ps = [p for p, _ in seen]
    assert ps == sorted(ps) and max(p for p, m in seen if m) <= 0.3 + 1e-9

def test_pipeline_precise_falls_back_on_failure(tmp_path, monkeypatch):
    ... fake_precise raises PrecisePoseFailed(0.42) ...
    note = run_pipeline(job, profile, lambda p, m: None, mode="precise")
    assert note == "完成（高精位姿失败，已按快速模式重建）"
    assert (job / "scene.splat").stat().st_size > 0
    log = (job / "log.txt").read_text(encoding="utf-8")   # pipeline appends the reason itself
    assert "0.42" in log or "42%" in log

def test_precise_import_error_falls_back(tmp_path, monkeypatch):
    ... monkeypatch so that importing recon3d.recon.precise_pose raises ImportError (monkeypatch.setitem(sys.modules, "recon3d.recon.precise_pose", None)) ...
    note = run_pipeline(job, profile, lambda p, m: None, mode="precise")
    assert note == "完成（高精位姿失败，已按快速模式重建）"
    assert "ImportError" in (job / "log.txt").read_text(encoding="utf-8")
```

Stage mapping inside `run_pipeline` for precise: the pipeline wraps `precise_pose`'s `on_progress(q)` as `on_progress(POSE_SHARE * q, stage(q))` with `stage(q) = STAGE_EXTRACT if q < 0.35 else STAGE_MATCH if q < 0.7 else STAGE_MAPPER`; training then maps `p → POSE_SHARE + (TRAIN_SHARE - POSE_SHARE) * p` with message None. Fast mode: unchanged mapping `TRAIN_SHARE * p`, message None. The fallback log line is appended to `job_dir / "log.txt"` with the same `[YYYY-mm-dd HH:MM:SS]` prefix format `JobStore.log` uses (pipeline has no store; write the line directly).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_jobs.py tests/test_pipeline.py -v -x`
Expected: FAIL — first on `JobStatus.__init__() got an unexpected keyword argument 'mode'` / wrong-arity fake pipelines.

- [ ] **Step 3: Implement** `JobStatus.mode`, `_commit(mode)`, `get` default, worker `Pipeline` arity + message handling + return value, `main.py` wrapper `pipeline(job_dir, mode, on_progress) -> run_pipeline(job_dir, profile, on_progress, mode)`, `run_pipeline` precise branch (lazy import of `precise_pose` inside the branch, `except (PrecisePoseFailed, Exception)` → log + fall back; the ImportError case is the same except branch).

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q -m "not gpu"`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add server/recon3d/jobs server/recon3d/main.py server/tests/test_jobs.py server/tests/test_pipeline.py server/tests/test_api.py
git commit -m "feat(server): job mode plumbing — precise pipeline branch, stage messages, fallback note"
```

---

### Task 4: `mode` in the API, `POST /jobs/{id}/rerun`, status page column, docs

**Files:**
- Modify: `server/recon3d/jobs/store.py` (`begin_resumable(size, sha256, mode="fast")`, `ResumableInfo.mode`, `finish_resumable` → `_commit(..., info.mode)`, new `rerun(job_id, mode) -> str`), `server/recon3d/api/resumable.py:42-66` (`UploadCreate.mode`), `server/recon3d/api/upload.py` (legacy multipart `mode` field → `store.create(stream, mode)`), `server/recon3d/api/app.py:66-113` (`mode` in GET bodies, new route), `server/recon3d/api/status_page.py:30,35,60-73`
- Modify: `docs/api.md`
- Test: `server/tests/test_jobs.py`, `server/tests/test_api.py`

**Interfaces:**
- Consumes: `JobStatus.mode`, `_commit(job_id, size, mode)` (Task 3).
- Produces:
  ```python
  class JobStore:
      def begin_resumable(self, size: int, sha256: str, mode: str = "fast") -> ResumableInfo   # mode persisted in upload.json
      def create(self, stream: BinaryIO, mode: str = "fast") -> str
      def rerun(self, job_id: str, mode: str) -> str
          # KeyError when job_id dir/capture.zip missing; UploadInProgress (new exception) when the dir has upload.json and no status.json
          # new dir with capture.zip hard-linked (os.link, OSError -> shutil.copyfile), log first line
          # f"created from {job_id} (rerun, mode={mode})", status queued with mode, enqueued
  MODES = ("fast", "precise")   # store.py; ValueError for anything else
  ```
  API: `POST /uploads {size, sha256, mode?}`; `POST /jobs` multipart optional field `mode`; `POST /jobs/{id}/rerun` JSON `{"mode": "fast"|"precise"}` → `201 {"job_id"}`, `404 {"detail":"任务不存在"}`, `409 {"detail":"上传未完成"}`, `422` on bad mode; `GET /jobs/{id}` and every `GET /jobs` job row include `"mode"` (uploading rows: the mode declared at `POST /uploads`).

- [ ] **Step 1: Write the failing store tests**

```python
def test_create_and_resumable_store_mode(tmp_path):
    s = JobStore(tmp_path / "d")
    j = s.create(io.BytesIO(b"x"), mode="precise")
    assert s.get(j).mode == "precise"
    data = b"abc"
    info = s.begin_resumable(len(data), hashlib.sha256(data).hexdigest(), mode="precise")
    assert s.resumable(info.upload_id).mode == "precise"
    s.append_resumable(info.upload_id, 0, [data])
    assert s.get(s.finish_resumable(info.upload_id)).mode == "precise"
    with pytest.raises(ValueError):
        s.create(io.BytesIO(b"x"), mode="turbo")

def test_rerun_hardlinks_capture_and_queues(store):
    src = store.create(io.BytesIO(b"zipbytes"))
    new = store.rerun(src, "precise")
    assert new != src
    d = store.job_dir(new)
    assert (d / "capture.zip").read_bytes() == b"zipbytes"
    assert os.stat(d / "capture.zip").st_ino == os.stat(store.job_dir(src) / "capture.zip").st_ino  # hard link (same fs in tmp)
    s = store.get(new)
    assert (s.state, s.mode, s.progress) == ("queued", "precise", 0.0)
    assert (d / "log.txt").read_text(encoding="utf-8").splitlines()[0].endswith(f"created from {src} (rerun, mode=precise)")
    assert store.queue.get_nowait() == src and store.queue.get_nowait() == new

def test_rerun_missing_or_uploading_source(tmp_path):
    s = JobStore(tmp_path / "d")
    with pytest.raises(KeyError):
        s.rerun("000000000000", "fast")
    info = s.begin_resumable(3, "0" * 64)
    with pytest.raises(UploadInProgress):
        s.rerun(info.upload_id, "fast")

def test_rerun_copies_when_link_fails(store, monkeypatch):
    src = store.create(io.BytesIO(b"zipbytes"))
    monkeypatch.setattr(os, "link", lambda a, b: (_ for _ in ()).throw(OSError("EXDEV")))
    new = store.rerun(src, "fast")
    assert (store.job_dir(new) / "capture.zip").read_bytes() == b"zipbytes"
```

- [ ] **Step 2: Write the failing API tests**

```python
def test_mode_in_upload_create_and_job_bodies(client):
    data = b"m" * 10
    r = client.post("/api/v1/uploads", json={"size": 10, "sha256": hashlib.sha256(data).hexdigest(), "mode": "precise"})
    uid = r.json()["upload_id"]
    assert client.get("/api/v1/jobs").json()["jobs"][0]["mode"] == "precise"
    _put(client, uid, 0, data); client.post(f"/api/v1/uploads/{uid}/finish")
    wait_until(lambda: client.get(f"/api/v1/jobs/{uid}").json()["state"] == "done")
    assert client.get(f"/api/v1/jobs/{uid}").json()["mode"] == "precise"
    jid = client.post("/api/v1/jobs", files={"capture": ("c.zip", b"zip")}).json()["job_id"]
    assert client.get(f"/api/v1/jobs/{jid}").json()["mode"] == "fast"          # default
    jid2 = client.post("/api/v1/jobs", files={"capture": ("c.zip", b"zip")}, data={"mode": "precise"}).json()["job_id"]
    assert client.get(f"/api/v1/jobs/{jid2}").json()["mode"] == "precise"

def test_bad_mode_is_422(client):
    assert client.post("/api/v1/uploads", json={"size": 1, "sha256": "0" * 64, "mode": "turbo"}).status_code == 422
    assert client.post("/api/v1/jobs", files={"capture": ("c.zip", b"zip")}, data={"mode": "turbo"}).status_code == 422
    jid = client.post("/api/v1/jobs", files={"capture": ("c.zip", b"zip")}).json()["job_id"]
    assert client.post(f"/api/v1/jobs/{jid}/rerun", json={"mode": "turbo"}).status_code == 422
    assert client.get("/api/v1/jobs").json()["jobs"][0]["job_id"] == jid       # nothing else created

def test_rerun_endpoint(client):
    jid = client.post("/api/v1/jobs", files={"capture": ("c.zip", b"zip")}).json()["job_id"]
    wait_until(lambda: client.get(f"/api/v1/jobs/{jid}").json()["state"] == "done")
    r = client.post(f"/api/v1/jobs/{jid}/rerun", json={"mode": "precise"})
    assert r.status_code == 201
    new = r.json()["job_id"]
    wait_until(lambda: client.get(f"/api/v1/jobs/{new}").json()["state"] == "done")
    assert client.get(f"/api/v1/jobs/{new}").json()["mode"] == "precise"
    assert client.post("/api/v1/jobs/000000000000/rerun", json={"mode": "fast"}).json() == {"detail": "任务不存在"}
    uid = _begin(client, b"abc")
    r = client.post(f"/api/v1/jobs/{uid}/rerun", json={"mode": "fast"})
    assert r.status_code == 409 and r.json() == {"detail": "上传未完成"}

def test_rerun_while_source_training_queues(client_slow):
    jid = client_slow.post("/api/v1/jobs", files={"capture": ("c.zip", b"zip")}).json()["job_id"]
    wait_until(lambda: client_slow.get(f"/api/v1/jobs/{jid}").json()["state"] == "training")
    r = client_slow.post(f"/api/v1/jobs/{jid}/rerun", json={"mode": "precise"})
    assert r.status_code == 201
    assert client_slow.get(f"/api/v1/jobs/{r.json()['job_id']}").json()["state"] == "queued"

def test_status_page_has_mode_column(client):
    assert "模式" in client.get("/").text
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_jobs.py tests/test_api.py -v -x -k "mode or rerun"`
Expected: FAIL (`unexpected keyword argument 'mode'` / 404 on `/rerun`).

- [ ] **Step 4: Implement** store (`MODES`, `UploadInProgress`, `ResumableInfo.mode` with `upload.json` default `"fast"` when absent, `rerun`), resumable `UploadCreate.mode: Literal["fast", "precise"] = "fast"`, legacy multipart `mode` (parse the field alongside `capture` in `recon3d/api/upload.py`; invalid → 422 `{"detail": "mode 只能是 fast 或 precise"}`), `app.py` route with a pydantic body `RerunBody(mode: Literal["fast","precise"])`, `mode` in both GET bodies, status page: header `<th>模式</th>` after 状态, cell `${j.mode === "precise" ? "高精" : "快速"}`.

- [ ] **Step 5: Docs** — `docs/api.md`: add `mode` to `POST /jobs`, `POST /uploads`, `GET /jobs`, `GET /jobs/{id}` examples; new section `## POST /jobs/{id}/rerun` (body, 201/404/409/422, hard-link note); status page 模式 column note.

- [ ] **Step 6: Run tests**

Run: `uv run pytest -q -m "not gpu"`
Expected: all passed.

- [ ] **Step 7: Commit**

```bash
git add server/recon3d server/tests docs/api.md
git commit -m "feat(server): job mode in API, POST /jobs/{id}/rerun, status page mode column"
```

---

### Task 5: Offline tooling — `eval_scene.py --mode`, `scripts/pose_drift.py`

**Files:**
- Modify: `server/scripts/eval_scene.py` (`--mode fast|precise`, default fast; `_train` passes `mode` to `run_pipeline`; `evaluate` renders at the refined poses when precise — compute them once via `precise_pose` into `out_dir / "poses_cv.npy"` and reuse via `build_training_set(..., c2w_override)`; prints the stats line)
- Create: `server/scripts/pose_drift.py` — `python scripts/pose_drift.py <job_dir> [--out DIR]`: unpack `capture.zip` to a temp dir, run `precise_pose`, write `pose_drift.csv` (`index,name,registered,shift_m,rot_deg`) and `trajectory.png` (top view x/z, ARCore vs refined, start marker, every 50th index annotated; matplotlib `Agg`), print the stats dict.
- Test: none beyond `uv run python scripts/pose_drift.py --help` and `uv run python scripts/eval_scene.py --help` both exiting 0 (these scripts are GPU/real-data tools; the spike already validated the flow). Add `matplotlib` to the `dev` dependency group.

- [ ] **Step 1: Implement both scripts.**
- [ ] **Step 2: Verify**

Run: `uv run python scripts/pose_drift.py --help && uv run python scripts/eval_scene.py --help | grep -- --mode`
Expected: both print usage; the second shows `--mode {fast,precise}`.

- [ ] **Step 3: Commit**

```bash
git add server/scripts/eval_scene.py server/scripts/pose_drift.py server/pyproject.toml server/uv.lock
git commit -m "feat(server): eval_scene --mode precise, pose_drift.py trajectory report"
```

---

### Task 6: Android data + API: `CaptureRecord.mode`, `Api.upload(mode)`, `Api.rerunJob`

**Files:**
- Modify: `android/app/src/main/java/dev/recon3d/rooms/RoomRepository.kt:30-52` (`val mode: String = "fast"`)
- Modify: `android/app/src/main/java/dev/recon3d/net/ApiClient.kt:50-73` (interface), `:91` (upload), `:313` (`UploadCreateBody(size, sha256, mode)`), new `rerunJob`
- Modify: `android/app/src/main/java/dev/recon3d/AppGraph.kt:52-62` (`SettingsBackedApi` delegates)
- Modify: `android/app/src/test/java/dev/recon3d/upload/Fakes.kt` (`FakeApi.upload` records `lastMode`; `rerunJob` records `rerunCalls: MutableList<Pair<String,String>>`, returns `rerunResult = "j2"`, throws `rerunFailure` once if set)
- Test: `android/app/src/test/java/dev/recon3d/net/ApiClientTest.kt`, `rooms/RoomRepositoryTest.kt`

**Interfaces:**
- Produces:
  ```kotlin
  data class CaptureRecord(..., val mode: String = "fast", ...)
  interface Api {
      suspend fun upload(zip: File, uploadId: String?, mode: String, listener: UploadListener): String
      /** POST /jobs/{id}/rerun → new job id. 404 → ApiException(MSG_JOB_GONE-equivalent server detail, code 404); 409 → detail, code 409. */
      suspend fun rerunJob(jobId: String, mode: String): String
  }
  const val MODE_FAST = "fast"; const val MODE_PRECISE = "precise"   // in RoomRepository.kt next to CaptureRecord
  ```

- [ ] **Step 1: Write the failing tests**

`RoomRepositoryTest`:
```kotlin
@Test fun modePersistsAndDefaultsToFast() {
    val room = repo.createRoom("客厅")
    repo.addCapture(room.id, CaptureRecord(id = "cap-1", zipPath = "/x.zip", mode = "precise"))
    assertEquals("precise", RoomRepository(dir).capture(room.id, "cap-1").mode)
    assertEquals("fast", CaptureRecord(id = "c", zipPath = "z").mode)
}
```
`ApiClientTest` (next to `uploadCreatesPutsChunksAndFinishes`; also change every existing `client.upload(zip, …, l)` call to pass `"fast"` as the third argument):
```kotlin
@Test fun uploadSendsModeInCreateBody() = runTest {
    val zip = tmpZip(10)
    server.enqueue(created()); server.enqueue(received(10)); server.enqueue(finished())
    client.upload(zip, null, "precise", RecordingListener())
    val body = Json.parseToJsonElement(server.takeRequest().body.readUtf8()).jsonObject
    assertEquals("precise", body["mode"]!!.jsonPrimitive.content)
}

@Test fun rerunJobPostsModeAndReturnsNewId() = runTest {
    server.enqueue(MockResponse().setResponseCode(201).setBody("""{"job_id":"bbbbbbbbbbbb"}"""))
    assertEquals("bbbbbbbbbbbb", client.rerunJob("aaaaaaaaaaaa", "precise"))
    val req = server.takeRequest()
    assertEquals("POST", req.method); assertEquals("/api/v1/jobs/aaaaaaaaaaaa/rerun", req.path)
    assertEquals("precise", Json.parseToJsonElement(req.body.readUtf8()).jsonObject["mode"]!!.jsonPrimitive.content)
}

@Test fun rerunJobErrorsCarryServerDetailAndCode() {
    server.enqueue(MockResponse().setResponseCode(404).setBody("""{"detail":"任务不存在"}"""))
    val e = apiError { client.rerunJob("aaaaaaaaaaaa", "fast") }
    assertEquals("任务不存在", e.messageZh); assertEquals(404, e.code)
    server.enqueue(MockResponse().setResponseCode(409).setBody("""{"detail":"上传未完成"}"""))
    assertEquals(409, apiError { client.rerunJob("aaaaaaaaaaaa", "fast") }.code)
}
```

- [ ] **Step 2: Run to verify failure**

Run: `.\gradlew.bat :app:testDebugUnitTest --offline --tests "dev.recon3d.net.ApiClientTest" --tests "dev.recon3d.rooms.RoomRepositoryTest"`
Expected: compilation FAIL (`mode` / `rerunJob` unresolved).

- [ ] **Step 3: Implement** record field, interface + `ApiClient` (`rerunJob` posts JSON `{"mode": …}` via the existing `call`/`httpError` helpers so 4xx bodies map to `ApiException(detail, code)`), `SettingsBackedApi`, `FakeApi`, fix every existing call site of `api.upload(...)` (the view model passes `record.mode` — Task 7 refines; for now pass `repo.capture(...).mode`).

- [ ] **Step 4: Run the full Android suite**

Run: `.\gradlew.bat :app:testDebugUnitTest --offline`
Expected: BUILD SUCCESSFUL, 181 tests (177 + 4).

- [ ] **Step 5: Commit**

```bash
git add android/app/src
git commit -m "feat(app): CaptureRecord.mode, Api.upload(mode), Api.rerunJob"
```

---

### Task 7: `UploadViewModel` — `start(mode)`, `rerun`, status text

**Files:**
- Modify: `android/app/src/main/java/dev/recon3d/upload/UploadScreen.kt:97-116` (start/retry), new `rerun`, `:188-238` (upload uses the record's mode), `:323-345` (`captureStatusText`), `:347-351` (`retryButtonLabel`)
- Test: `android/app/src/test/java/dev/recon3d/upload/UploadViewModelTest.kt`

**Interfaces:**
- Consumes: Task 6.
- Produces:
  ```kotlin
  fun start(roomId: String, captureId: String, mode: String)        // writes mode into the record before uploading
  fun retry(roomId: String, captureId: String)                        // unchanged signature; re-upload uses record.mode
  fun rerun(roomId: String, captureId: String, mode: String)          // requires record.jobId != null else fail(MSG_JOB_GONE, jobGone = true)
  fun captureStatusText(record: CaptureRecord): String                // precise: suffix " · 高精" on 上传中/排队中/训练中/导出中/完成;
                                                                      // training + message startsWith "位姿优化" -> that message (+ suffix)
  fun retryButtonLabel(record: CaptureRecord): String                 // jobId != null -> "重试"; uploadId != null -> "继续上传"; else -> "重新上传" (unchanged; the two-button re-upload UI is Task 8)
  ```
  `rerun`: `api.rerunJob(jobId, mode)` → `editCapture { copy(mode = mode, jobId = new, serverIds = serverIds.plusDistinct(new), state = "queued", progress = 0f, message = null) }` (scenePath untouched) → `track(roomId, captureId, new)`. On `ApiException` → `fail(roomId, captureId, e.messageZh, jobGone = e.code == 404)` and the record keeps its old `scenePath`.

- [ ] **Step 1: Write the failing tests**

```kotlin
@Test fun startRecordsModeAndPassesItToApi() = runTest {
    vm().start(room, cap, "precise"); advanceUntilIdle()
    assertEquals("precise", api.lastMode)
    assertEquals("precise", repo.capture(room, cap).mode)
}

@Test fun retryReuploadsWithRecordedMode() = runTest {
    val vm = vm()
    api.failUploadOnce(ApiException("boom"))
    vm.start(room, cap, "precise"); advanceUntilIdle()
    assertEquals("failed", repo.capture(room, cap).state)
    vm.retry(room, cap); advanceUntilIdle()
    assertEquals(listOf("precise", "precise"), api.modes)
}

@Test fun rerunSwapsJobKeepsSceneUntilDownloaded() = runTest {
    val vm = vm()
    vm.start(room, cap, "fast"); advanceUntilIdle()
    val oldScene = repo.capture(room, cap).scenePath!!
    api.rerunResult = "j2"
    api.jobQueue += JobStatus("j2", JobState.TRAINING, 0.1f, "位姿优化 (特征提取)")
    vm.rerun(room, cap, "precise"); runCurrent()
    val mid = repo.capture(room, cap)
    assertEquals(listOf("j" to "precise"), api.rerunCalls)
    assertEquals("j2", mid.jobId); assertEquals("precise", mid.mode)
    assertEquals(listOf("u1", "j", "j2"), mid.serverIds)
    assertEquals(oldScene, mid.scenePath)
    advanceTimeBy(3_100); runCurrent()
    assertEquals("位姿优化 (特征提取) · 高精", captureStatusText(repo.capture(room, cap)))
    advanceUntilIdle()
    val done = repo.capture(room, cap)
    assertEquals("done", done.state); assertEquals(oldScene, done.scenePath)
}
```
```kotlin
@Test fun rerunFailureKeepsOldScenePath() = runTest {
    val vm = vm()
    vm.start(room, cap, "fast"); advanceUntilIdle()
    val oldScene = repo.capture(room, cap).scenePath!!
    api.rerunFailure = ApiException("上传未完成", code = 409)
    vm.rerun(room, cap, "precise"); advanceUntilIdle()
    val rec = repo.capture(room, cap)
    assertEquals("failed", rec.state); assertEquals("上传未完成", rec.message)
    assertEquals(oldScene, rec.scenePath); assertEquals("j", rec.jobId)   // 409 is not "job gone"
}

@Test fun rerunWithoutJobIdFails() = runTest {
    vm().rerun(room, cap, "precise"); advanceUntilIdle()
    assertEquals("failed", repo.capture(room, cap).state)
    assertTrue(api.rerunCalls.isEmpty())
}

@Test fun statusTextPreciseSuffixAndPoseStage() {
    val r = CaptureRecord(id = "c", zipPath = "z", mode = "precise")
    assertEquals("本地", captureStatusText(r))
    assertEquals("上传中 42% · 高精", captureStatusText(r.copy(state = "uploading", progress = 0.42f)))
    assertEquals("排队中 · 高精", captureStatusText(r.copy(state = "queued")))
    assertEquals("训练中 50% · 高精", captureStatusText(r.copy(state = "training", progress = 0.5f)))
    assertEquals("位姿优化 (特征匹配) · 高精", captureStatusText(r.copy(state = "training", progress = 0.2f, message = "位姿优化 (特征匹配)")))
    assertEquals("导出中 · 高精", captureStatusText(r.copy(state = "exporting")))
    assertEquals("完成 · 高精", captureStatusText(r.copy(state = "done")))
    assertEquals("失败：x", captureStatusText(r.copy(state = "failed", message = "x")))
    assertEquals("仅云端", captureStatusText(r.copy(state = "remote")))
    assertEquals("训练中 50%", captureStatusText(r.copy(mode = "fast", state = "training", progress = 0.5f)))
}
```
Update every existing `vm.start(room, cap)` call in the test file to `vm.start(room, cap, "fast")`. `FakeApi` gains `val modes = mutableListOf<String>()` (append in `upload`), `lastMode`, `rerunCalls`, `rerunResult`, `rerunFailure`.

- [ ] **Step 2: Run to verify failure**

Run: `.\gradlew.bat :app:testDebugUnitTest --offline --tests "dev.recon3d.upload.UploadViewModelTest"`
Expected: compilation FAIL (`start` arity / `rerun` unresolved).

- [ ] **Step 3: Implement.** Note the retry suffix rule: the existing `retrying`/`reconnecting` branches build on the base text, so put the ` · 高精` suffix in the base `when` result for the five states only, before those branches append their own ` · …`.

- [ ] **Step 4: Run the full Android suite**

Run: `.\gradlew.bat :app:testDebugUnitTest --offline`
Expected: BUILD SUCCESSFUL, 187 tests.

- [ ] **Step 5: Commit**

```bash
git add android/app/src
git commit -m "feat(app): UploadViewModel start(mode), rerun, precise status text"
```

---

### Task 8: Android UI — two upload buttons, rerun menu, wiring

**Files:**
- Modify: `android/app/src/main/java/dev/recon3d/capture/PreviewScreen.kt:33-37,55-72` (`onUpload: ((mode: String) -> Unit)?`; buttons 重拍 | 快速上传 (OutlinedButton) | 高精上传 (Button, primary, rightmost))
- Modify: `android/app/src/main/java/dev/recon3d/upload/UploadScreen.kt:361-400` (`CaptureRow(record, onUpload: (mode: String) -> Unit, onRetry, onRerun: (mode: String) -> Unit, …)`)
- Modify: `android/app/src/main/java/dev/recon3d/rooms/RoomListScreen.kt:141-152,198-206`, `android/app/src/main/java/dev/recon3d/nav/AppNav.kt:110-122,191-197`
- Test: `android/app/src/test/java/dev/recon3d/upload/UploadViewModelTest.kt` (pure helper tests; Compose itself is not unit-tested in this project)

**Interfaces:**
- Produces (pure helpers in `UploadScreen.kt`, used by `CaptureRow`):
  ```kotlin
  /** Buttons for a row's right side: list of (label, action). */
  sealed interface RowAction { data class Upload(val mode: String) : RowAction; object Retry : RowAction; object View : RowAction; object Redownload : RowAction }
  fun rowActions(record: CaptureRecord): List<Pair<String, RowAction>>
      // local  -> [("快速上传", Upload(fast)), ("高精上传", Upload(precise))]
      // failed -> jobId != null || uploadId != null -> [(retryButtonLabel(record), Retry)]
      //           else -> [("快速重传", Upload(fast)), ("高精重传", Upload(precise))]
      // done   -> [("查看 3D", View)]; remote -> [("重新下载", Redownload)]; else -> []
  fun rerunMenuLabel(record: CaptureRecord): String?   // done && mode == fast -> "改用高精重算"; done && precise -> "改用快速重算"; else null
  fun rerunMenuMode(record: CaptureRecord): String     // the opposite mode
  ```
  `CaptureRow` renders `rowActions` (the last one as `Button`, others `OutlinedButton`) and, when `rerunMenuLabel != null`, a `DropdownMenuItem` before 删除 that calls `onRerun(rerunMenuMode(record))`. `RoomScreen` gains `onRerun: (captureId: String, mode: String) -> Unit`, `onUpload` becomes `(captureId: String, mode: String) -> Unit`; `AppNav` wires `onUpload = { id, mode -> withServer { graph.uploader.start(roomId, id, mode) } }`, `onRerun = { id, mode -> withServer { graph.uploader.rerun(roomId, id, mode) } }`, and the preview route `onUpload = { mode -> …start(roomId, captureId, mode)… }`.

- [ ] **Step 1: Write the failing helper tests**

```kotlin
@Test fun rowActionsPerState() {
    val r = CaptureRecord(id = "c", zipPath = "z")
    assertEquals(listOf("快速上传", "高精上传"), rowActions(r).map { it.first })
    assertEquals(RowAction.Upload("precise"), rowActions(r).last().second)
    assertEquals(listOf("重试"), rowActions(r.copy(state = "failed", jobId = "j")).map { it.first })
    assertEquals(listOf("继续上传"), rowActions(r.copy(state = "failed", uploadId = "u")).map { it.first })
    assertEquals(listOf("快速重传", "高精重传"), rowActions(r.copy(state = "failed")).map { it.first })
    assertEquals(listOf("查看 3D"), rowActions(r.copy(state = "done")).map { it.first })
    assertEquals(listOf("重新下载"), rowActions(r.copy(state = "remote")).map { it.first })
    assertTrue(rowActions(r.copy(state = "training")).isEmpty())
}

@Test fun rerunMenuOnlyWhenDone() {
    val r = CaptureRecord(id = "c", zipPath = "z", state = "done")
    assertEquals("改用高精重算", rerunMenuLabel(r)); assertEquals("precise", rerunMenuMode(r))
    assertEquals("改用快速重算", rerunMenuLabel(r.copy(mode = "precise"))); assertEquals("fast", rerunMenuMode(r.copy(mode = "precise")))
    assertNull(rerunMenuLabel(r.copy(state = "training")))
    assertNull(rerunMenuLabel(r.copy(state = "remote")))
}
```

- [ ] **Step 2: Run to verify failure** — `.\gradlew.bat :app:testDebugUnitTest --offline --tests "dev.recon3d.upload.UploadViewModelTest"` → compilation FAIL (`rowActions` unresolved).
- [ ] **Step 3: Implement helpers, `CaptureRow`, `PreviewScreen`, `RoomScreen`, `AppNav` wiring.** Remove the old single 上传 button paths entirely.
- [ ] **Step 4: Full suite + build the APK**

Run: `.\gradlew.bat :app:testDebugUnitTest --offline` → BUILD SUCCESSFUL, 189 tests. Then `.\gradlew.bat :app:assembleDebug --offline` → BUILD SUCCESSFUL (Compose code is only compiled, not unit-tested).

- [ ] **Step 5: Commit**

```bash
git add android/app/src
git commit -m "feat(app): 快速上传/高精上传 buttons, 改用高精重算 menu, rerun wiring"
```

---

### Task 9: Docs — setup, acceptance checklist

**Files:**
- Modify: `docs/setup/01-server-laptop.md` (pycolmap installs with `uv sync`; CPU only, no extra steps)
- Modify: `docs/acceptance/phase1.md` §4: new `### 10) 高精模式` checklist (预览页两按钮; 房间页 local 两按钮; 状态文字 `位姿优化 (特征提取) · 高精` → `训练中 x% · 高精` → `完成 · 高精`; ⋮ 改用高精重算 → 新任务、旧"查看 3D"消失、完成后模型替换; 状态页模式列; 回退文案 `完成（高精位姿失败，已按快速模式重建）` 出现在手机和状态页; `curl -X POST …/rerun`); §5: paste the §8/§9 spike numbers of the spec (190 帧 2.5–3 min, 1123 帧 ≈ 45 min, PSNR 16.1 → 22.7 dB).

- [ ] **Step 1: Write the docs.**
- [ ] **Step 2: Commit**

```bash
git add docs/setup/01-server-laptop.md docs/acceptance/phase1.md
git commit -m "docs: precise mode setup note and acceptance checklist"
```

---

## Self-review notes

- Spec coverage: §3.1 → Task 3 (status mode) + Task 4 (API, rerun, page); §3.2 → Task 1; §3.3 → Tasks 2–3; §3.4 → Task 5; §4 → Tasks 6–8; §5 → tests inside each task; §6 → Tasks 4 (api.md), 9; §9 adjustments → constants in Task 1.
- Type consistency: `OnProgress = (float, str | None)` is introduced in Task 3 and used by `main.py`, worker, test fakes; `PrecisePoseResult.c2w_cv` feeds `c2w_override` (both float64 [N,4,4]); Android `mode` strings are `"fast"`/`"precise"` everywhere, constants `MODE_FAST`/`MODE_PRECISE` defined in Task 6.
- Known cost: Task 3 changes the fake-pipeline arity in two existing test files (mechanical).
