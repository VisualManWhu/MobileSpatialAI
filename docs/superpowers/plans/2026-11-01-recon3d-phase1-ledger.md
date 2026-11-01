# SDD ledger — plan: docs/superpowers/plans/2026-11-01-recon3d-phase1.md
Spec: docs/superpowers/specs/2026-11-01-recon3d-design.md
Branch: phase1 (created from main d83b094)

## Pre-flight scan
| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1→T2,T3,T6,T7 | load_capture(zip, workdir)->Capture; make_capture(tmp_path, **overrides) | T1 tests need overrides format_version, drop_image, frame_count; T2 needs image_width,image_height,fx. make_capture must accept all. OK |
| T2→T3,T4 | TrainingSet, Profile dataclass, PROFILES | T4 uses dataclasses.replace → Profile must be (frozen) dataclass. OK |
| T4→T5,T6,T7 | Splats, train(), render() | T7 e2e reuses T4 synthetic scene → fixture must live in tests/fixtures shared module |
| T5→T6 | export_splat + scene.ply | T5 says "训练结束同时保存 scene.ply" but export_splat signature only writes .splat → ruling R3 |
| T6 tests use torch.cuda.OutOfMemoryError | torch added T4 | OK ordering |
| T8→T12 | ApiClient | T12 extracts `Api` interface from ApiClient.kt, file not listed in T12 → ruling R4 |
| T9→T10 | PoseMath | OK |
| T10→T1 server test | android_v1.zip fixture | cross-language; ruling R5 |
| T7,T8,T11,T12,T13,T14 | human steps (firewall/.wslconfig, Android Studio install, phone testing) | ruling R6 |
| each task self-consistency | T1..T14 | tests match signatures; T13 has no automated test (manual only) — acceptable for WebView UI |

## Rulings
- Ruling R1: work on branch `phase1` in place instead of worktree — repo is brand-new, no other work; WSL paths stay simple — cost if wrong: none, merge to main at end.
- Ruling R2: all server code runs/tests inside WSL Ubuntu 24.04 (`wsl -d Ubuntu -- bash -lc "cd /mnt/d/projects/Recon3D/server && ..."`), uv-managed Python 3.11, venv via UV_PROJECT_ENVIRONMENT=$HOME/.venvs/recon3d (off /mnt/d for speed); commits via Windows git; `.gitattributes` forces LF for *.sh/*.py so bash scripts work in WSL — cost if wrong: re-path venv.
- Ruling R3: T5 adds `export_ply(splats, path)` alongside export_splat; T6 pipeline writes both — spec needs full model kept for phase 2 — cost: small extra function.
- Ruling R4: T12 may modify net/ApiClient.kt to add `interface Api` that ApiClient implements — cost: none.
- Ruling R5: T10 adds a JUnit test that writes a sample package to `server/tests/fixtures/android_v1.zip` (relative path from android/ project), committed; server test loads it — cost: fixture regen manual.
- Ruling R6: human-in-loop steps (Android Studio install, .wslconfig/firewall, phone tests) are batched: controller asks user when reached; automated parts proceed first — cost: pause at T7/T8.
- Ruling R7: T1 reviewer-minor ".gitignore BOM" promoted to Important — BOM breaks `.superpowers/` ignore, risk of committing scratch workspace — cost: trivial fix.
Task 1: minor (deferred): extra error paths untested; FileNotFoundError reported as bad zip; manifest/frames errors lack field names/line numbers; format_version true passes; no manifest↔frames consistency checks; make_capture short poses IndexError; no zip-bomb guard (note for upload endpoint T6)
Task 1: minor (deferred→T6): docs/api.md scene.splat not-done error unspecified — T6 plan says 409; T6 should document it
Task 1: fix round 1/5 (4 addressed, 0 open; commits f9e67a9..f96796c)
Task 1: minor (deferred): non-numeric pose elements → raw TypeError at format.py:68; NUL byte in frame path → ValueError from resolve() in _inside()
Task 1: complete (commits d83b094..f96796c, review clean)
- Ruling R8: T2 needs torch for TrainingSet tensors (plan adds torch in T4) → T2 adds torch, pinned to a version+CUDA combo for which gsplat 1.5.x prebuilt wheels exist (verify at docs.gsplat.studio/whl), so T4 doesn't reinstall — cost: re-pin in T4 if wrong.
- Ruling R9: if gsplat 1.5.x prebuilt wheel for pt24cu124 exists only for cp310, T4 switches project Python to 3.10 (plan said 3.11; spec requires no user CUDA compile, which outranks Python minor version) — cost: re-lock, trivial.
- Ruling R10: T2 Important "gsplat cp310 vs py3.11" and "torch2.4.1 vs numpy2.4 ABI risk" are T4 install decisions, not T2 defects → carried into T4 dispatch (apply R9; pin numpy<2 if gsplat/torch ABI issues appear) — cost: T4 re-lock.
Task 2: minor (deferred): JPEG/depth size not checked vs manifest (silent stretch, wrong K); all images float32 on device (memory, consider uint8); pose tests lack non-trivial rotation case
Task 2: complete (commits f96796c..90ba913, review clean)
Task 3: minor (deferred): plane test only checks z (no x/y extent, no rotated c2w); filter test doesn't assert 9m/0m dropped; half-pixel color offset; np.unique/add.at slow for multi-M points
Task 3: complete (commits 90ba913..20cd7fb, review clean)
- Ruling R11: T4 minor #4 (hook order before opt.step skips update on refine steps) folded into fix round 1 — same lines as Important #3, cheap — cost: none.
Task 4: minor (deferred): PSNR test easy (GT depth init); random-init (no depth) collapsed to 399 gaussians — watch in T7/T14; cap overshoot transient memory; SSIM padding differs; opacities stored [N] (T5 must handle)
Task 4: fix round 1/5 (4 addressed, 0 open; commits 8b36803..1a55c7a)
Task 4: minor (deferred): unpinned CPU images → sync per-step copy (~1ms); gsplat 1.5.3 PACKED/strategy-state coupling — re-check on bump; reset would double-fire if gsplat fixes its bug (idempotent)
Task 4: complete (commits 20cd7fb..1a55c7a, review clean)
Task 5: minor (deferred): export_ply N=0 raises (reshape(n,-1,3)); rot-order test circular (add explicit (1,0,0,0)→(255,128,128,128)); stray noqa/blank lines; no NaN/Inf guard
Task 5: complete (commits 1a55c7a..80a2ed1, review clean)
- Ruling R12: T6 minors #2 (OOM detect by MRO + "out of memory" substring), #3 (gc + empty_cache between jobs), #6 (Worker.start idempotent) folded into fix round 1 — Review Focus #5 (OOM → next job succeeds) depends on #2/#3; cheap — cost: none.
Task 6: minor (deferred): 413 test doesn't exercise header pre-check; Starlette MultiPartException 400 English; get() swallows OSError (Linux-only, moot); pre-lifespan-queued job edge case; worker stop() 5s timeout → training job marked failed on next start; no capture.zip cleanup; unknown routes English 404
Task 6: fix round 1/5 (4 addressed, 0 open; commits 7d57678..bf3dd90)
Task 6: minor (deferred): upload disconnect race (abort() while parser thread mid-write — set abandon_on_cancel=False); filename*= form → 422; per-chunk thread hop unbenchmarked on multi-100MB upload
Task 6: complete (commits 80a2ed1..bf3dd90, review clean)
Task 8: pending-human — Android Studio not installed (checked Program Files, LOCALAPPDATA, SDK, adb: none). User asked to install while T7 runs.
Task 7: pending-human — Wi-Fi 'FRITZ!Box 7583 VA' category is Public; firewall rule (-Profile Private) ineffective until user sets network to Private (Settings → Network → Wi-Fi → network properties → Private). Docs must cover this.
- Ruling R13: T7 minor "UV_LINK_MODE=copy" folded into fix round 1 (same file as Important #2, removes noisy warnings for non-dev user) — cost: slower first sync, negligible.
Task 7: minor (deferred): curl-missing hint; docs assume repo on D:; screenshot placeholders text-only; run.sh URL list may include VPN/Hyper-V adapters; e2e uint16 double-convert redundant
Task 7: fix round 1/5 (3 addressed, 0 open; commits ce83beb..29ac29d)
Task 7: minor (deferred): "第 3 步（续）" heading nesting cosmetic
Task 7: complete (commits bf3dd90..29ac29d, review clean; Step 5 phone health check pending-human)
## Human gate before Task 8 (2026-11-01)
Blocked on user: Android Studio + SDK not installed (Gradle needs JDK; cannot build/test android/). Also pending: Wi-Fi→Private, firewall 8765 rule, Hyper-V inbound allow, phone health check (T7 step 5).
- Ruling R14: user has no phone access for now; T7 step 5 + all on-device manual steps (T8 step 4, T11 step 4, T12 step 4, T13 step 3) deferred to a 'phone session' checklist in ledger; tasks proceed build+unit-test only. User chose Tailscale (phone+laptop, 100.x addr) over Wi-Fi Private profile; firewall rule becomes -Profile Any; docs to get a Tailscale section (fold into T8 docs step or T14). — cost: on-device bugs surface later in a batch.
Task 8: unblocked — Android Studio 2026.2.2 at 'C:\Program Files\Android\Android Studio' (JBR openjdk 25), SDK at %LOCALAPPDATA%\Android\Sdk (platforms android-37.0, build-tools 36.0.0, adb). No ANDROID_HOME/JAVA_HOME env. BASE=29ac29d
Task 8: minor (deferred, implementer-flagged): all IOException → "连不上计算端" (incl. disk write fail in downloadScene); GPU label strips only NVIDIA/GeForce prefix; upload progress restarts on OkHttp retry; "Unable to strip .so" info (no NDK) harmless
Task 8: pending-human: install APK on Pixel + 测试连接; 03-phone.md screenshots; Tailscale phone side
- Ruling R15: app label "Recon3D" stays Latin — product name, not UI copy; spec's 中文 rule applies to interface text — cost: none.
- Ruling R16: normalizeBaseUrl default port 8765 when none given (friendlier for non-dev user) folded into Task 12 (which touches net/ per R4) — cost: none.
Task 8: minor (deferred): local file I/O errors reported as 连不上计算端 (→ "保存文件失败"); URL saved before health check; "http://" → "http:" harmless; no test for .part cleanup mid-stream or Content-Length header; broad IAE catch; placeholder app icon; hardcoded Chinese strings bypass strings.xml
Task 8: complete (commits 29ac29d..9c9a551, review clean; device install pending-human)
Task 9: minor (implementer-flagged, pending review): SPIN hint merged '先完成原地转圈：站在房间中央，慢慢向右转'; SWEEP hint switches after up reached; fast-rotation frames still count coverage (KeyframeSelector T10 is the drop point)
- Ruling R17: KeyframeSelector additionally rejects frames whose angular velocity vs the previous seen frame exceeds 60°/s (same threshold as the guide's warning) so motion-blurred frames never enter the capture package, even though CaptureGuide still counts them as coverage — cost: a few fewer keyframes during fast moves.
Task 9: minor (deferred): no test for ±180 yaw crossing in velocity; pitch-only warning; resume-after-loss no warning; ROAM anchor=mean; ROAM LEFT/RIGHT >45° mapping; roamArrow test near-tautological; path length jitter; skipStage drops lost-tracking warning
Task 9: complete (commits 9c9a551..18f89ff, review clean)
Task 10: minor (implementer-flagged, pending review): fixture android_v1.zip rewritten with new bytes every Gradle run (zip timestamps → git churn); fixture path tries ../../server then ../server; finish() DEFLATED level 1 not STORED (fine); CapturePackageWriter not thread-safe (T11 must serialize addFrame on one IO coroutine)
- Ruling R18: T10 minors #3 (staging cleanup on finish() failure, double-finish guard), #4 (>= thresholds), #5 (reset seenPose on tracking loss + test), #7 (drop wrong ../server fallback) folded into fix round 1 — same two files as the Importants — cost: none.
Task 10: minor (deferred): writer test doesn't check depth PNG content / JPEG verbatim / depth_width null; selector tests lack first-frame velocity exemption and slow-drift cases; appendText reopens frames.jsonl per frame
Task 10: fix round 1/5 (6 addressed, 0 open; commits 0af0590..2521bf5)
Task 10: minor (deferred): manifest write outside try/catch in finish(); no rotation boundary test; no failure-injection test for cleanup; fixture bytes depend on JDK deflate impl (write-if-differs would dirty tree on other JDK)
Task 10: complete (commits 18f89ff..2521bf5, review clean)
Task 11: implementer DONE_WITH_CONCERNS — GL/ARCore never run on device (R14); CaptureRenderer.kt + CapturePipeline.kt added beyond plan file list (acceptable split); record button also pauses; portrait lock; zip retrieval needs `adb exec-out run-as dev.recon3d cat`
Task 11: SUSPECT (controller-induced): dispatch told implementer to mask DEPTH16 with 0x1FFF; if ARCore acquireDepthImage16Bits uses full 16 bits for mm, mask WRAPS >8.19 m depths (9 m → 0.8 m) — reviewer asked to verify against ARCore docs
- Ruling R19 (REVERSES controller instruction in T11 dispatch): acquireDepthImage16Bits uses full 16 bits = mm 0..65535 (ARCore Frame docs); NO 0x1FFF mask. Store raw uint16. — cost if wrong: none (docs explicit).
- Ruling R20: T11 minors #3 (depth size unknown at first keyframe → depth lost for whole capture), #5 (broad catch around buildJob on GL thread), #6 (camera error never cleared), #7 (pipeline dead after failed finish) folded into fix round 1 — all first-real-run robustness in the same 3 files; no device available to catch them later — cost: larger fix diff.
Task 11: minor (deferred): dropped keyframe still advances selector (#4); drop threshold off-by-one (#8); permanent camera-permission denial not routed to app settings (#9)
- Ruling R21: T11 implementer additions accepted pending re-review — 10 s no-depth fallback (record without depth if sensor never delivers; avoids blocking user on devices/lighting where depth fails) and millisecond captureId (fresh staging dir after reset) — cost: a capture may silently lack depth; server falls back to random init (T3).
Task 11: fix round 1/5 (6 addressed, 0 open; commits eabe919..2e481a1)
Task 11: minor (deferred): host.finish() catches CancellationException; possible blank preview after error recovery (unverified); persistent non-NotYetAvailable depth exception skips 10s timeout; silent no-depth fallback not surfaced to user; GL-thread probe/fallback untested
Task 11: complete (commits 2521bf5..2e481a1, review clean; device verification pending-human)
Task 13 prep: Node v24.15.0 + npm 11.12.1 at C:\Program Files\nodejs (Windows); viewer-web bundle can be built locally.
Task 12: implementer concerns: 404 detected by message text (ApiException lacks status code); server failed → jobId cleared → retry re-uploads; text-glyph icons ⚙ ←, nested Scaffold insets unchecked on device; uploader plain class in AppGraph, restarts polling at app start
- Ruling R22: T12 Important #1 (disk write per progress tick) resolved by cross-task fact: ApiClient (T8) throttles progress to ~1% steps → ≤100 writes/upload; still add integer-percent guard in the callback in fix round (trivial). #2b (fail() may throw inside catch → uncaught → crash) and #2 (missing download-failure + concurrent-start tests) enter fix round 1.
Task 12: minor (deferred): 404 detected by message text; transient poll error fails capture (spec-conformant); main-thread fsync in createRoom/addCapture; addCapture duplicate-id throws uncaught; start() no state guard; download phase detected via message string; unknown JobState leaks raw string; new ApiClient (+OkHttpClient?) per poll — cache per baseUrl; bare IPv6/"http://" garbage-in; RoomScreen blank on missing id; settings via gear lacks back arrow; orphan upload if process dies before jobId saved; no foreground service
Task 13 prep: npm @sparkjsdev/spark 2.4.0 (published 2026-10-20, peer three>=0.180.0), three 0.187.0, esbuild 0.28.2 — pin these.
Task 12: fix round 1/5 (4 addressed, 0 open; commits 1955855..e91adec)
Task 12: minor (deferred): CoroutineExceptionHandler wiring untested (tests inject TestScope); track() poll writes every 3 s regardless of change; lastPercent plain var on OkHttp thread; import ordering
Task 12: complete (commits 2e481a1..e91adec, review clean; device upload test pending-human)
- Ruling R23: T13 initial camera computed from splat bounds (center + fit distance) instead of the brief's 'first-frame camera pose' — the .splat carries no camera poses; adding a sidecar is phase-2 work. Also: built viewer assets committed so Gradle needs no Node; rebuild via documented npm step. — cost: initial view less 'natural' than the first frame; revisit in T14 if acceptance finds it confusing.
Task 13: implementer notes — coordinate frame: splat world = ARCore world (+Y up) = three.js, NO Y flip (overrides brief's 'Y 轴取反'; reviewer asked to confirm); Spark API SparkRenderer({renderer,onDirty}) + SplatMesh({fileBytes,fileType}); bundle 3.11 MiB ESM; webkit 1.18.0; concerns: never run in real WebView; no CSP (network blocked Kotlin-side 403/404); whole file in JS memory; overlapping loadScene not serialized
Task 14 prep: full server suite re-run at HEAD 1393d62 → [exited with code 0]
Task 14 prep: server suite = 65 passed in 79.09s (GPU incl.)
- Ruling R24 (refines R23): viewer home view = ARCore world origin (first-frame camera position) looking toward −Z at level pitch, target at bounds-centre depth; fall back to bounds-fit only if origin lies outside the trimmed bounds. Matches brief's "first-frame camera" without a pose sidecar. Also fold minor #1 (controls.maxDistance clamp) into fix round 1. — cost if wrong: an odd initial view; device check in T14.
- Confirmed by reviewer (poses.py/dataset.py/recon grep): splat world = ARCore world (+Y up) → NO Y flip; brief's "Y 轴取反" was wrong. Ruling R25: brief text overridden.
Task 13: minor (deferred): flyTo throws sync on bad coords instead of rejecting; damping drift after flyTo/setCameraHome; addJavascriptInterface not origin-scoped (safe, could use addWebMessageListener); no CSP connect-src (defence-in-depth, device-verify first); home-view aspect not recomputed on resize; concurrent loadScene unserialised
Task 13: fix round 1/5 (3 addressed, 0 open; commits 1393d62..37d690b)
Task 13: minor (deferred): fail-fast guard may false-fail on a stray early worker rejection (device-check); late onLoaded after watchdog flips state (benign); fetch has no JS timeout (Kotlin watchdog covers)
Task 13: complete (commits e91adec..37d690b, review clean; device verification pending-human)
Task 14: implementer DONE (docs only, 974742e): phase1.md criteria table with 待测 cells + automated results (server 65 / Android 105 + APK / viewer-web 23) + 8-part phone checklist; README completed; 01-server-laptop heading fix; ~3 min laptop training labelled extrapolated estimate. Review pending.
Task 14: minor (deferred): none beyond fix round (date-line wording folded in)
Task 14: fix round 1/5 (3 addressed, 1 open — WSL check command not runnable from PowerShell quoting; commits 974742e..0c07c6d)
Task 14: fix round 2/5 (1 addressed, 0 open; commits 0c07c6d..e24534e)
Task 14: complete (commits 37d690b..e24534e, review clean; real-room acceptance measurements pending-human)
## All 14 tasks complete at e24534e. Final whole-branch review dispatched (MERGE_BASE d83b094).
## Final review (opus) at e24534e: With fixes. 0 Critical, 5 Important, minors triaged (must-fix: T2 rotated-pose test → folded in #4; T11 #9 permission dead-end; T12 transient poll → #3).
- Ruling R26 (overrides firewall part of R14): accept reviewer — scope Hyper-V to a per-port rule (New-NetFirewallHyperVRule TCP 8765, VMCreatorId WSL) instead of VM-wide DefaultInboundAction Allow; Tailscale variant adds -RemoteAddress 100.64.0.0/10 to both rules. Docs + acceptance checklist updated; user has NOT yet run the old commands. — cost: none.
- Ruling R27 (refines R21): no-depth fallback must be surfaced to the user ("未获取到深度，本次拍摄将不含深度" hint) — cost: none.
- Ruling R28: Important #2 (depth intrinsics alignment) handled additively: record camera.textureIntrinsics per frame as OPTIONAL frames.jsonl fields depth_fx/fy/cx/cy (format stays v1; docs note optional fields); server uses them when present, else falls back to scaled image K; skip depth init with a log if depth/image aspect differ >1% and no depth intrinsics; log both intrinsics on phone for the device session. — cost if wrong: none (additive).
- Ruling R29: Important #3: track() tolerates up to 10 consecutive unreachable polls with backoff before failing; button label "重试" when jobId != null, else "重新上传".
Final fix wave: commits 3ac34b3 (server) 827954d (android) 86232f6 (docs); server 70 passed -rs no skips; Android 113/0 + assembleDebug; fixture deterministic. Implementer concerns (→ deferred minors): Tailscale rules use distinct names (Recon3D server 8765 Tailscale / Recon3D8765TS); depth intrinsics assume depth FOV == texture FOV (device logcat to confirm); aspect-skip logged to stderr not job log.txt; failure message hardcodes data/jobs path; poll backoff ≈3.5 min before fail; F6/F7 UI untested (no device). Scoped re-review dispatched.
Final fix wave re-review: all 8 addressed, no new Critical/Important. Residual minors parked:
- parked: init_points colour lookup assumes depth↔image alignment when depth_K present — Ruling: init RGB only, training refines colour; defer to phase 2.
- parked: retry() with jobId while offline shows no "正在重试" text until failure (~3 min) — Ruling: cosmetic; set state queued at resume start in a later polish.
- parked: Android 11+ dialog-dismiss misread as permanent denial — Ruling: wrong message only, "打开设置" still works.
## Final review complete at 86232f6 (30 commits on phase1). Branch ready to merge; on-device acceptance pending-human.
