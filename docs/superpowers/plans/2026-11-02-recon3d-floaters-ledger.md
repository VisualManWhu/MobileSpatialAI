# SDD ledger — plan: docs/superpowers/plans/2026-11-02-recon3d-floaters.md

Branch: floaters (from main a0eb827+..., plan commit 2a13a9e). Spec: docs/superpowers/specs/2026-11-01-recon3d-design.md §5.3.

## Pre-flight scan

| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1 → T2 | `Frame.confidence`, `Frame.depth_kind` → `build_training_set` reads them | consistent |
| T1 → T6 | server accepts v2 fields exactly as the writer emits (`depth_kind`, `confidence`, `format_version: 2`, conf 8-bit PNG) | consistent; T6 fixture `android_v2.zip` closes the loop |
| T2 → T3 | `TrainingSet.confs` (0..1, range mask included) → `init_points(conf_min)` | consistent; T3 must drop its own MAX_DEPTH check (plan says so) |
| T2 → T4 | `confs[i]` as depth-loss weight; `depths[i]` target | consistent |
| T3/T4 → T5 | `init_points` for floater score; `render` unchanged; needs `load_ply` | T5 adds `load_ply` to export_splat.py — not a conflict |
| T4 ↔ Global "world frame unchanged" | pose deltas regularised, `render()` uses original poses | consistent (Review Focus 5) |
| T6 ↔ T1 | conf PNG 8-bit greyscale | consistent |
| T7 | single constant + test | self-consistent |
| T2 self | `TrainingSet.confs` default: plan says `field(default_factory=list)`; existing direct constructors in `tests/fixtures/synthetic.py` | implementer must grep constructors; noted in brief |
| T4 self | GPU test requires Profile ctor with new fields: `Profile` is frozen dataclass → add fields with defaults at the END so positional `Profile("laptop", 960, 15000, 1_500_000, 1)` still works | Ruling R1 below |
| T5 self | `--train` reuses `jobs/pipeline.py` | pipeline signature unknown to brief; implementer reads it |

Rulings:
- Ruling R1: `Profile.depth_lambda` and `Profile.pose_lr` get defaults (0.05, 1e-4) and are appended after `sh_degree` — keeps positional construction in PROFILES/tests valid — cost if wrong: none.
- Ruling R2: `SUPPORTED_VERSION` constant stays (=2, latest) alongside new `SUPPORTED_VERSIONS=(1,2)` only if something imports it; otherwise rename — implementer checks grep — cost: none.

## Tasks
Task 1: complete (commits 2a13a9e..38526cf, review clean; trailer verified by controller)
Task 1: minor (deferred): loader accepts confidence/depth_kind in v1 files (harmless); no path-traversal test for `confidence`; make_capture writes conf/ files even for format_version=1
Task 2: complete (commits 38526cf..e7b0af8, review clean; trailer verified)
Task 2: minor (deferred): heuristic penalises a neighbour that is finite but > MAX_DEPTH_M (acts as edge, negligible); no NaN/inf or invalid-neighbour test; 0.05 m step test tolerance tight for float32
Task 2: carried to Task 3: tests/test_init_points.py TrainingSet helper lacks confs — Task 3 adds it first
Task 3: fix round 1/5 (2 addressed, 0 open — far-drop test now uses depth_confidence + |z|<5 m assertion; imports hoisted; commits ec4049a..c3fd84c)
Task 3: complete (commits e7b0af8..c3fd84c, review clean)
Task 3: minor (deferred): no test for confs None with non-None depth, nor custom conf_min
Note (process): Tasks 2-3 were verified with -m "not gpu"; two GPU tests in test_trainer.py broke (depths replaced without confs) and were only caught in Task 4. From Task 4 on, implementers run the full suite incl. GPU.
Task 4: concern to carry to Task 5: on a synthetic scene with exact poses the refinement still drifts up to 0.84 deg / 1.7 cm in a 300-iter run — evaluate on the office capture; raise POSE_REG if real-capture stats look like noise absorption.
Task 4: complete (commits c3fd84c..18dbe10, review clean)
Task 4: minor (deferred): SMALL_ANGLE=1e-6 tight for float32 (cosmetic); depth loss switches off silently when sizes don't divide (add one-time warning); per-step host sync `if n == 0` and non-non_blocking depth/conf copies; log wording "旋转 中位数" → "旋转中位数"
- Ruling R3: pose refinement drift on exact synthetic poses (0.84°/1.7 cm @300 it) is a design property (Adam + mean-reg + lr 1e-4 ≈ 10× gsplat default). Task 5 runs three variants on the office capture — old model (baseline), new defaults, new with pose_lr=0 — and reports PSNR/floater for each; controller rules the shipped default from those numbers. — cost if wrong: one extra ~10 min training run.
Task 5 results (office capture, laptop profile): baseline PSNR 22.39 / floater 0.1415 / 9.1 min; new (depth 0.05 + pose 1e-4) 20.19 / 0.1181 / 13.6 min; nopose (depth 0.05) 22.28 / 0.0923 / 9.1 min.
- Ruling R4: default `Profile.pose_lr` becomes 0 (pose refinement opt-in; depth_lambda stays 0.05) — on real data refinement lost 2.2 dB, removed fewer floaters and cost +50% time; the PSNR comparison is pessimistic for the refined model (eval renders at original poses) but the floater score and time are not. Applied in Task 8 (config one-liner + docs). — cost if wrong: one config value.
Task 5: complete (commits 18dbe10..e362680, review clean)
Task 5: minor (deferred): docs wording on worst frames (#188–191 not accurate for "new"; #189/#190/#55 common); load_ply no size check / bare ValueError on missing end_header; _train reuses stale capture.zip in a reused retrain dir; per-view .item() sync
Task 6: complete (commits e362680..9fcc58f, review clean; trailer amended by controller — metadata only)
Task 6: minor (deferred): confRowStride default 0 → plane8 should require rowStride >= width; confidence plane pixelStride unchecked; "stale" also counts "no raw image"; JPEG fixture determinism is per-JDK
Tasks 7+8: batched into one dispatch (same-shape small edits + docs), with ruling R4 (pose_lr default 0) applied in config.py + docs.
Task 7: complete (commits 9fcc58f..571853b, review clean)
Task 8: complete (commits 571853b..9aecf86, review clean; includes ruling R4 config change)
Task 7: minor (deferred): no exact-30°/s boundary test (strict >)
Final review (opus, 60caafb..9aecf86): With fixes. Important: (1) depth loss ignores depth_K — mis-registered when depth intrinsics ≠ scaled image K (plan at fault: signature has no K); (2) depth loss switches off silently. Must-fix minors: phase1.md worst-frames sentence false. One fix wave dispatched with the one-liners (plane8 require, eval min_points=0, init c>0, log wording, floater-score caveat, capture-format doc pointer).
- Ruling R5 (final-review "declined" items): depth_kind stays informational; no gauge fixing / Adam-vs-reg concerns accepted since pose refinement is off by default; ED low-alpha semantics accepted (Huber + conf bound it); up-to-3 ARCore images per frame accepted (per-frame catch exists) — device checklist is the evidence. — cost if wrong: none now; revisit if pose refinement is ever enabled by default.
Final fix wave: complete (commits 9aecf86..2a99e3e, scoped re-review clean). Residual minor (not blocking): in-loop skip warning text slightly misleading when depth_loss returns None for "no usable confidence pixels".
Final review: clean after fixes. Verification on HEAD 2a99e3e: viewer 23/23, Android 123/0 + assembleDebug OK, server suite pending.
