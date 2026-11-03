# Floater Round 2 (steps 0–2b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut floaters in Recon3D's 3DGS output with contribution/visibility pruning, an end-of-training outlier filter, opacity/scale regularisers and an MCMC strategy option, each switchable from `Profile` and measurable with `eval_scene.py`.

**Architecture:** All work is in the server (`server/recon3d/recon/`). Two new pure modules (`contrib.py` for per-Gaussian contribution via the all-ones-feature gradient trick, `outliers.py` for kNN statistical outlier removal) feed small hooks in `trainer.py`; regularisers are two loss terms; MCMC is a branch on `profile.strategy`. `scripts/eval_scene.py` gains an off-trajectory contact sheet and a dump of pruned Gaussians so pruning thresholds can be judged visually.

**Tech Stack:** Python 3.10, torch, gsplat 1.5.3 (`rasterization`, `DefaultStrategy`, `MCMCStrategy`, `strategy.ops.remove`), scipy cKDTree, numpy, opencv-python-headless (scripts only). Tests: pytest, GPU tests marked `@pytest.mark.gpu`, run in WSL:
`cd /mnt/d/projects/Recon3D/server && export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/recon3d UV_LINK_MODE=copy && uv run pytest -q <target>`.

**Spec:** `docs/superpowers/specs/2026-11-03-recon3d-floaters-v2-design.md` §2 steps 0, 1, 2a, 2b and §3.

## Global Constraints

- User-visible strings (log lines, CLI help, printed text) in Chinese; code identifiers and comments in English.
- Every new behaviour is a `Profile` field with the spec's default and `0` (or `"default"`) meaning off; existing jobs must train unchanged when the new fields are set to off.
- No new runtime dependency for the server package (opencv is already a scripts-only `--with` extra).
- Culling/pruning never empties the Gaussian set (same guard as `_cull_outside`).
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- The full server suite must stay green: `uv run pytest -q` (the timing test `test_trains_with_depth_from_own_pose` needs an idle GPU; if another job is training, deselect it and say so in the report).

## Review Focus

1. A training set whose image size is not divisible by `downscale` (e.g. 959×539): `compute_contrib` must still render (floor the size, scale K by the actual ratio) and return N-length tensors. Test in Task 2.
2. Fewer than `k+1` Gaussians at the end of training: `sor_mask` must return all-True instead of raising. Test in Task 3.
3. A `nviews` threshold that would remove every Gaussian (e.g. 2 training views): the prune must be skipped, not applied. Test in Task 4.
4. MCMC with contribution pruning: Gaussians must be *re-opacified* (logit 1e-4), never `remove`d, because MCMC relocates them. Test in Task 6.
5. Off-trajectory sheet when there are fewer views than requested tiles: use all views without index repetition or IndexError. Test in Task 1.

---

### Task 1: eval_scene off-trajectory contact sheet

**Files:**
- Create: `server/recon3d/recon/offtrack.py`
- Modify: `server/scripts/eval_scene.py`
- Test: `server/tests/test_offtrack.py`

**Interfaces:**
- Produces: `raise_cameras(c2w: torch.Tensor, dy_m: float = 0.5) -> torch.Tensor` — copy of `[N,4,4]` c2w with world-Y translation increased by `dy_m` (ARCore world is gravity-aligned, +Y up).
- Produces: `pick_views(n: int, k: int) -> list[int]` — `min(n, k)` evenly spaced distinct indices in `[0, n)`.
- Produces: `contact_sheet(imgs: list[np.ndarray], cols: int = 4) -> np.ndarray` — uint8 HxWx3 grid, row-major, blank (black) tiles pad the last row.

- [ ] **Step 1: Write the failing tests**

```python
def test_raise_cameras_moves_world_y_only():
    c2w = torch.eye(4).repeat(3, 1, 1); c2w[:, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    out = raise_cameras(c2w, 0.5)
    assert torch.allclose(out[:, :3, 3], torch.tensor([1.0, 2.5, 3.0]).expand(3, 3))
    assert torch.equal(out[:, :3, :3], c2w[:, :3, :3]) and torch.equal(c2w[:, 1, 3], torch.full((3,), 2.0))

def test_pick_views_even_and_distinct():
    assert pick_views(100, 4) == [0, 33, 66, 99]
    assert pick_views(3, 12) == [0, 1, 2]
    assert pick_views(0, 12) == []

def test_contact_sheet_pads_last_row():
    tiles = [np.full((2, 3, 3), v, np.uint8) for v in (10, 20, 30, 40, 50)]
    sheet = contact_sheet(tiles, cols=4)
    assert sheet.shape == (4, 12, 3)
    assert sheet[0, 0, 0] == 10 and sheet[0, 11, 0] == 40
    assert sheet[2, 0, 0] == 50 and sheet[2, 11, 0] == 0
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest -q tests/test_offtrack.py` → ImportError.
- [ ] **Step 3: Implement the three functions in `recon3d/recon/offtrack.py`** (pure torch/numpy, no cv2).
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Wire into `eval_scene.py`**: add `--offtrack-m` (float, default `0.5`; `0` disables) and `--offtrack-views` (int, default `12`). After PSNR, render `pick_views(N, k)` views from `raise_cameras(ts.c2w, m)` one at a time via `render(splats, view)`, convert to uint8 BGR, write `contact_sheet(..., cols=4)` to `<out_dir>/offtrack.png` with `cv2.imwrite`; add `"offtrack_png"` to the result dict and print `偏离轨迹（抬高 0.5 m）渲染：<path>`. Import cv2 lazily inside the branch (scripts run with `uv run --with opencv-python-headless`).
- [ ] **Step 6: Smoke test** on an existing job: `uv run --with opencv-python-headless python scripts/eval_scene.py data/jobs/9f6d49647dfe` → prints the offtrack line, `data/jobs/9f6d49647dfe/offtrack.png` exists and opens (check size > 10 KB).
- [ ] **Step 7: Commit** — `feat(eval): off-trajectory contact sheet (+0.5 m) in eval_scene`.

### Task 2: Per-Gaussian contribution and view count

**Files:**
- Create: `server/recon3d/recon/contrib.py`
- Test: `server/tests/test_contrib.py` (GPU)

**Interfaces:**
- Produces: `compute_contrib(params: Mapping[str, torch.Tensor], viewmats: torch.Tensor, Ks: torch.Tensor, W: int, H: int, downscale: int = 2, min_px: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]` — `total [N] float32` (summed blending weight in pixels over all views at the downscaled size), `nviews [N] int32` (views where the per-view weight `> min_px`). `params` has raw `means`, `quats`, `scales` (log), `opacities` (logit). Renders one view at a time with `gsplat.rasterization(..., colors=feat [N,1] requires_grad, sh_degree=None, packed=False, width=W//downscale, height=H//downscale, Ks scaled by (W//downscale)/W and (H//downscale)/H)`, gradient of `render.sum()` w.r.t. `feat`.
- Produces: `prune_mask(total: torch.Tensor, nviews: torch.Tensor, min_views: int, min_px: float) -> torch.Tensor` — bool `[N]`, True = prune, `(nviews < min_views) | (total < min_px)`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.gpu
def test_contrib_counts_visible_views_only(synthetic_ts):
    tr = synthetic_ts[0]
    xyz, rgb = init_points(tr)                       # points on the synthetic object
    n = len(xyz)
    means = torch.cat([torch.from_numpy(xyz).cuda(), torch.tensor([[0.0, 0.0, 1e4]], device="cuda")])  # one far behind everything
    params = {"means": means, "quats": torch.tensor([[1.0, 0, 0, 0]], device="cuda").repeat(n + 1, 1),
              "scales": torch.full((n + 1, 3), math.log(0.02), device="cuda"),
              "opacities": torch.full((n + 1,), 2.0, device="cuda")}
    H, W = tr.images.shape[1:3]
    total, nviews = compute_contrib(params, torch.linalg.inv(tr.c2w.cuda()), tr.K.cuda(), W, H, downscale=2)
    assert total.shape == (n + 1,) and nviews.dtype == torch.int32
    assert nviews[-1] == 0 and total[-1] == 0
    assert (nviews[:-1] >= 3).float().mean() > 0.8   # object points are seen by most of the 24 cameras

@pytest.mark.gpu
def test_contrib_handles_size_not_divisible_by_downscale(synthetic_ts):
    tr = synthetic_ts[0]
    params = {...same 5-Gaussian params on the object...}
    total, nviews = compute_contrib(params, viewmats, tr.K.cuda(), 255, 191, downscale=2)  # 127x95 render
    assert total.shape == (5,)

def test_prune_mask_thresholds():
    total = torch.tensor([0.1, 5.0, 5.0, 0.6]); nviews = torch.tensor([10, 2, 3, 3])
    assert prune_mask(total, nviews, min_views=3, min_px=0.5).tolist() == [True, True, False, False]
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement `recon3d/recon/contrib.py`.** `torch.enable_grad()` inside even if called under `no_grad`; detach params; `feat = torch.ones(N, 1, requires_grad=True)`; accumulate in float32.
- [ ] **Step 4: Run to verify pass**; also time it: print seconds for 24 views in the test log (expect < 2 s).
- [ ] **Step 5: Commit** — `feat(recon): per-Gaussian contribution and view count via feature gradient`.

### Task 3: kNN statistical outlier mask

**Files:**
- Create: `server/recon3d/recon/outliers.py`
- Test: `server/tests/test_outliers.py` (CPU)

**Interfaces:**
- Produces: `sor_mask(xyz: np.ndarray, k: int = 8, sigma: float = 3.0) -> np.ndarray` — bool `[M]`, True = outlier: mean distance to `k` nearest *other* points `> mean + sigma * std` over all points. All-False when `M <= k` or `sigma <= 0`. Uses `scipy.spatial.cKDTree(..., workers=-1)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_sor_flags_lone_far_point():
    rng = np.random.default_rng(0)
    cloud = rng.normal(0, 0.05, size=(2000, 3)).astype(np.float32)
    xyz = np.concatenate([cloud, [[3.0, 0, 0]]]).astype(np.float32)
    out = sor_mask(xyz, k=8, sigma=3.0)
    assert out[-1] and out[:-1].mean() < 0.02

def test_sor_noop_when_too_few_or_disabled():
    xyz = np.zeros((5, 3), np.float32)
    assert not sor_mask(xyz, k=8).any()
    assert not sor_mask(np.random.default_rng(0).normal(size=(100, 3)).astype(np.float32), sigma=0.0).any()
```

- [ ] **Step 2: Run to verify failure.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `feat(recon): kNN statistical outlier mask`.

### Task 4: Pruning hooks in the trainer + pruned dump

**Files:**
- Modify: `server/recon3d/config.py` (Profile fields)
- Modify: `server/recon3d/recon/trainer.py`
- Modify: `server/recon3d/jobs/pipeline.py`, `server/scripts/eval_scene.py` (`--dump-pruned`)
- Test: `server/tests/test_trainer.py`

**Interfaces:**
- Consumes: Task 2 `compute_contrib`, `prune_mask`; Task 3 `sor_mask`; existing `_cull_outside`, `gsplat.strategy.ops.remove`, `export_ply(Splats, path)`.
- Produces: `Profile.contrib_min_views: int = 3`, `Profile.contrib_min_px: float = 0.5`, `Profile.sor_sigma: float = 3.0` (contribution pruning off when `contrib_min_views <= 0 and contrib_min_px <= 0`; SOR off when `sor_sigma <= 0`).
- Produces: `train(ts, profile, on_progress, device="cuda", pruned_out: Path | None = None) -> Splats` — when `pruned_out` is given, every Gaussian removed by contribution pruning or SOR (not bounds culling) is appended and written once at the end with `export_ply` (skip writing when nothing was pruned).
- Produces: `_prune_contrib(params, optimizers, state, viewmats, Ks, W, H, profile) -> tuple[int, torch.Tensor | None]` — removes `prune_mask(...)` Gaussians (never all), returns `(n_removed, removed_means_or_None)`; schedule: steps `iters // 2`, `3 * iters // 4`, and after the loop. `_prune_sor(params, optimizers, state, sigma) -> tuple[int, ...]` after the loop, after the final contribution prune.
- `run_pipeline` passes `pruned_out=job_dir / "pruned.ply"` only when the env var `RECON3D_DUMP_PRUNED=1`; `eval_scene.py --dump-pruned` sets that env var before training and prints `被剪高斯已写入 …/pruned.ply`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.gpu
def test_training_prunes_by_contribution_and_logs(synthetic_ts, caplog, tmp_path):
    ts = synthetic_ts[0]
    p = dataclasses.replace(PROFILES["laptop"], iterations=400, max_gaussians=20000)
    with caplog.at_level(logging.INFO, logger="recon3d.recon.trainer"):
        splats = train(ts, p, on_progress=lambda p: None, pruned_out=tmp_path / "pruned.ply")
    msgs = [r.getMessage() for r in caplog.records]
    assert any("贡献度剪枝" in m for m in msgs) and any("离群点滤波" in m for m in msgs)
    assert splats.means.shape[0] > 0
    # pruned.ply exists iff something was pruned; either way training must not crash
    assert (tmp_path / "pruned.ply").exists() == any("移除" in m and "贡献度剪枝" in m and " 0 个" not in m for m in msgs) or True

@pytest.mark.gpu
def test_contribution_prune_never_empties(synthetic_ts):
    ts = synthetic_ts[0]
    p = dataclasses.replace(PROFILES["laptop"], iterations=400, max_gaussians=20000, contrib_min_views=10_000)
    assert train(ts, p, on_progress=lambda p: None).means.shape[0] > 0

@pytest.mark.gpu
def test_pruning_off_matches_old_behaviour(synthetic_ts):
    ts = synthetic_ts[0]
    p = dataclasses.replace(PROFILES["laptop"], iterations=300, max_gaussians=20000,
                            contrib_min_views=0, contrib_min_px=0.0, sor_sigma=0.0)
    n = train(ts, p, on_progress=lambda p: None).means.shape[0]
    assert n > 0  # and no 贡献度剪枝 / 离群点滤波 log lines
```

(Replace the last assert of the first test with two explicit cases: run once with defaults and assert the file exists when the log reports a non-zero removal; the implementer picks the exact log format `贡献度剪枝（步 %d）：移除 %d 个高斯（<%d 个视角可见或贡献<%.1f px），剩余 %d` and `离群点滤波（k=8，%.1fσ）：移除 %d 个高斯，剩余 %d`.)

- [ ] **Step 2: Run to verify failure** (unknown Profile field).
- [ ] **Step 3: Add the Profile fields** with the comment block style used for `far_m`.
- [ ] **Step 4: Implement `_prune_contrib`, `_prune_sor` and the schedule in `train`**; collect removed Gaussians as `Splats` slices (all six tensors) for `pruned_out`. Contribution pruning runs at full training resolution with `downscale=2`.
- [ ] **Step 5: `run_pipeline` / `eval_scene.py --dump-pruned`** as in Interfaces.
- [ ] **Step 6: Run `tests/test_trainer.py` and `tests/test_pipeline.py`** → pass.
- [ ] **Step 7: Commit** — `feat(recon): contribution/visibility pruning and end-of-training outlier filter`.

### Task 5: Opacity and scale regularisers

**Files:**
- Modify: `server/recon3d/config.py`, `server/recon3d/recon/trainer.py`
- Test: `server/tests/test_trainer.py`

**Interfaces:**
- Produces: `Profile.opacity_reg: float = 0.01`, `Profile.scale_reg: float = 0.01`.
- Produces: `reg_loss(params, opacity_reg: float, scale_reg: float) -> torch.Tensor` — `opacity_reg * sigmoid(opacities).mean() + scale_reg * exp(scales).mean()`; returns a 0-dim zero tensor on the params' device when both weights are 0. Added to `loss` every step before `backward()`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.gpu
def test_reg_loss_values():
    params = {"opacities": torch.zeros(4, device="cuda"), "scales": torch.zeros(4, 3, device="cuda")}
    assert torch.isclose(reg_loss(params, 0.01, 0.01), torch.tensor(0.01 * 0.5 + 0.01 * 1.0, device="cuda"))
    assert reg_loss(params, 0.0, 0.0).item() == 0.0
```

- [ ] **Step 2–4: fail → implement → pass.** Also assert in `test_trains_synthetic_scene` that PSNR is still ≥ its current threshold with the default regs on.
- [ ] **Step 5: Commit** — `feat(recon): opacity and scale regularisers`.

### Task 6: MCMC strategy option

**Files:**
- Modify: `server/recon3d/config.py`, `server/recon3d/recon/trainer.py`, `server/scripts/eval_scene.py`
- Test: `server/tests/test_trainer.py`

**Interfaces:**
- Produces: `Profile.strategy: str = "default"` (`"default"` | `"mcmc"`); `eval_scene.py --strategy {default,mcmc}` override (train-only, like the others).
- Trainer, when `profile.strategy == "mcmc"`: `MCMCStrategy(cap_max=profile.max_gaussians, refine_start_iter=min(500, iters // 10), refine_stop_iter=iters // 2, refine_every=100, min_opacity=0.005)`; `state = strategy.initialize_state()`; `step_post_backward(params, optimizers, state, step, info, lr=means_sched.get_last_lr()[0])`; **skip** `_enforce_cap` and the manual `reset_opa`; `_cull_outside` still removes (bounds are hard). Contribution pruning under MCMC sets `params["opacities"].data[mask] = logit(1e-4)` instead of `remove` and logs `（MCMC：压低不透明度）`. SOR at the very end may `remove` (training is over).
- Unknown strategy name → `ValueError("未知的训练策略：…")` at the start of `train`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.gpu
def test_trains_with_mcmc_within_cap(synthetic_ts):
    tr, te = synthetic_ts
    p = dataclasses.replace(PROFILES["laptop"], iterations=1500, max_gaussians=30000, strategy="mcmc")
    splats = train(tr, p, on_progress=lambda p: None)
    assert 0 < splats.means.shape[0] <= 30000
    assert psnr(render(splats, te), te.images) > 20.0

@pytest.mark.gpu
def test_mcmc_contrib_prune_lowers_opacity_instead_of_removing(synthetic_ts):
    # call _prune_contrib directly with a MCMCStrategy state and mcmc=True; the count of Gaussians is unchanged
    # and the pruned ones have sigmoid(opacity) < 1e-3

def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="未知的训练策略"):
        train(..., dataclasses.replace(PROFILES["laptop"], strategy="foo"), on_progress=lambda p: None)
```

(The unknown-strategy check must run before any GPU work so the test can be CPU-only with a tiny fake `TrainingSet`; if that is impractical, mark it `gpu`.)

- [ ] **Step 2–4: fail → implement → pass.**
- [ ] **Step 5: Run the full suite** `uv run pytest -q` → green (deselect the timing test only if the GPU is busy; say so).
- [ ] **Step 6: Commit** — `feat(recon): MCMC strategy option`.

### Task 7: Baselines and A/B on saved captures (controller runs this; not a subagent task)

**Files:**
- Modify: `docs/acceptance/phase1.md` §5 (new table), `docs/superpowers/specs/2026-11-03-recon3d-floaters-v2-design.md` (results note).

Captures: `cfd1968452b2` (office), `2fe7b607c543`, `5cba40b51c1e`, `9f6d49647dfe`. Each run ≈ 10 min on an idle 4060; check `http://127.0.0.1:18765/api/v1/jobs` has no active job first. Run from `server/` with `uv run --with opencv-python-headless python scripts/eval_scene.py data/jobs/<id> --train --tag <tag> …`:

| tag | overrides |
|---|---|
| `r2-base` | `--far-m 0 --bbox-margin-m 0 --contrib-min-views 0 --contrib-min-px 0 --sor-sigma 0 --opacity-reg 0 --scale-reg 0` (round-1 behaviour) |
| `r2-bounds` | pruning + regs off, bounds default |
| `r2-prune` | regs off, strategy default |
| `r2-regs` | everything default (`default` strategy) |
| `r2-mcmc` | `--strategy mcmc` |

(Task 4/5 add the matching `eval_scene.py` overrides `--contrib-min-views --contrib-min-px --sor-sigma --opacity-reg --scale-reg`; follow the existing override pattern.)

- [ ] Record per capture: PSNR median, floater score, Gaussians, minutes; attach the `offtrack.png` paths; look at `pruned.ply` of one capture (`--dump-pruned`) in SuperSplat or the viewer to confirm pruned Gaussians are floaters, not room edges.
- [ ] Decision rule from the spec: a step's default stays on only if floater score drops on all captures and PSNR median drops < 0.2 dB. Set `Profile` defaults accordingly (MCMC becomes default only if it beats `r2-regs`).
- [ ] Commit — `docs: floater round-2 A/B results and default decision`.

## Self-review notes

- Spec coverage: step 0 → Task 1 (+ Task 7 baselines); step 1 → Tasks 2, 3, 4; step 2a → Task 5; step 2b → Task 6; §3 evaluation → Tasks 1, 7. Steps 3–4 of the spec are intentionally out of scope.
- Review Focus items 1–5 are pinned by tests in Tasks 2, 3, 4, 6, 1 respectively.
- Names used consistently: `compute_contrib`, `prune_mask`, `sor_mask`, `_prune_contrib`, `_prune_sor`, `reg_loss`, `raise_cameras`, `pick_views`, `contact_sheet`, Profile fields `contrib_min_views`, `contrib_min_px`, `sor_sigma`, `opacity_reg`, `scale_reg`, `strategy`.
