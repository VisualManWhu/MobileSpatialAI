# Recon3D 删除设计（拍摄 / 房间，手机 + 云端）

日期：2026-11-05。状态：待实现。

## 1. 为什么

现在手机端只能加拍摄、加房间，不能删。拍摄包 40–200 MB、模型 20–45 MB，试几次手机就满了；服务端 `data/jobs/` 也只增不减，失败或重复的任务只能登录笔记本手动 `rm`。用户要两种删法：

- **彻底删除**：手机和服务端都删干净。
- **手机本地删除**：只删手机上的大文件（拍摄包、点云、模型），列表里保留一条"仅云端"记录，以后还能从服务端重新下载模型。

房间也要能删，同样两种。

## 2. 目标 / 非目标

目标：
- 每条拍摄、每个房间都有"删除"入口，弹框二选一，文案写清删什么。
- 彻底删除：云端任务目录（拍摄包、结果、日志、半截上传）和手机上的一切都没了。
- 本地删除：手机释放空间，记录留着，"重新下载"用现有断点续传拿回模型。
- 服务端训练中 / 导出中的任务不能删（GPU 进程正往目录里写），手机给出明确提示。
- 服务端新接口有测试；手机端 ViewModel 逻辑有 JVM 测试。

非目标：
- 取消正在训练的任务（要给 pipeline 加中断点，另做）。
- 批量多选删除。
- 回收站 / 撤销。删了就是删了，靠二次确认。
- 云端只删结果保留拍摄包之类的部分删除。

反目标（做成这样算失败）：
- 本地删除之后"重新下载"下不回来——记录必须保留 `jobId`，状态 `done` 且云端有 `scene.splat` 才允许本地删除。
- 彻底删除把云端删了、手机上却因为中途出错留下一条指向不存在任务的记录——顺序是**先云端、后手机**，云端 404 视为已删继续。
- 房间彻底删除删了一半停下、用户不知道哪些删了——剩余拍摄和房间保留在列表里，提示失败原因，再点一次从剩下的继续。

## 3. 服务端：`DELETE /api/v1/jobs/{id}`

| 情形 | 响应 | 动作 |
|---|---|---|
| 任务 `done` / `failed` / `queued` | `204` | `shutil.rmtree(job_dir)` |
| 断点续传上传中（有 `upload.json`，无 `status.json`） | `204` | 同上；若正有 PUT 在写（上传锁被占）→ `409 {"detail":"该上传正在接收中"}`。手机先取消上传再删，但服务端要等到察觉断线才放锁，所以手机对这个 409 短暂重试（见 §4.2） |
| `training` / `exporting` | `409 {"detail":"任务正在训练，完成后再删"}` | 不动 |
| id 格式非法或目录不存在 | `404 {"detail":"任务不存在"}` | 不动 |

实现要点（`recon3d/jobs/store.py` + `recon3d/api/app.py`）：
- `JobStore.delete(job_id) -> None`：raise `KeyError`（不存在）、`JobActive`（训练/导出中）、`UploadBusy`（上传锁被占）。删除在 `_upload_lock(job_id)` 和 `_lock` 下做，避免与 `update()` 的状态写入交错。
- `queued` 任务删掉后 id 还在 `queue.Queue` 里：`Worker._run` 开头检查 `store.get(job_id) is None` 就跳过（现在直接 `update` 会 raise `KeyError`，被 `_loop` 吞掉并打印堆栈——改成静默跳过）。
- 删除写一行服务端日志（`log.info`），任务自己的 `log.txt` 随目录一起没了。
- `GET /api/v1/jobs` 的状态页：删掉的任务下次刷新自然消失，页面不加删除按钮（电脑端删除用 `curl -X DELETE` 即可，文档写上）。

## 4. 手机端

### 4.1 入口与对话框

- **拍摄行**（`CaptureRow`）右侧加 ⋮ 图标按钮（contentDescription "更多"），菜单一项"删除"。
- **房间页**（`RoomScreen`）右上角 ⋮，菜单一项"删除房间"。
- 点"删除"弹 `AlertDialog`，标题"删除「{名字}」"，正文说明两项各删什么，按钮按情形出现：

| 情形 | 按钮 |
|---|---|
| 拍摄从未上传（`jobId == null && uploadId == null`） | 「删除」（= 删手机）、「取消」 |
| 拍摄 `done` 且 `scenePath != null` | 「彻底删除」「仅删手机文件」「取消」 |
| 拍摄其他状态（uploading / queued / failed / 仅云端） | 「彻底删除」「取消」 |
| 房间无拍摄 | 「删除」「取消」 |
| 房间有拍摄 | 「彻底删除」「仅删手机」「取消」 |

- 「彻底删除」按钮用 `MaterialTheme.colorScheme.error` 着色。
- 删除进行中（等服务端响应）按钮禁用、显示小进度条；成功后对话框关、行 / 房间消失；失败对话框关、Toast 显示原因。

### 4.2 语义

**拍摄 · 彻底删除**
1. 若正在上传 / 轮询 / 下载（`uploader.isRunning`）：`UploadViewModel.cancel(roomId, captureId)`（suspend，`cancelAndJoin`），等协程结束。
2. 若 `jobId ?: uploadId` 非空：`DELETE /jobs/{id}`。`204` 或 `404` 继续；`409` 且 detail 为"该上传正在接收中"→ 隔 1 s 重试，最多 5 次（服务端察觉断线后才放锁），仍 409 才中止；其他 `409` 抛 `ApiException(detail)` 中止，记录原样不动；网络错误同样中止（"连不上计算端"）。
3. 云端 `204` 之后的本地步骤不会失败：文件删不掉忽略，`removeCapture` 对不存在的记录是 no-op。
4. 删本地文件：`zipPath`、`pointsPath`、`scenePath` 及其 `.part`、`scenesDir/<roomId>/<captureId>/` 目录。
5. `repo.removeCapture`。

**拍摄 · 仅删手机文件**（只在 `done` 且有 `scenePath` 时提供）
1. 若 `home == null`，先从 zip 算 `HomeView.fromCaptureZip` 存进记录（zip 一删就算不出来了）；算不出来就留 null，查看器用默认机位。
2. 删 `zipPath`、`pointsPath`、`scenePath`，不碰服务端。
3. 记录改为 `state = "remote"`，`zipPath = ""`、`pointsPath = null`、`scenePath = null`、`progress = 0`、`message = null`；`jobId`、`home`、`frameCount`、`pointCount` 保留。
4. 行显示状态"仅云端"，按钮「重新下载」。

**重新下载**：`UploadViewModel.redownload(roomId, captureId)` = 现有 `track(roomId, captureId, jobId)`：服务端 `done` → 下载 `scene.splat` → `state = "done"`。期间状态文案"正在下载模型…"（与现有一致，state 临时为 `exporting`）。服务端 404（云端被别处删了）→ `failed`，消息"任务不存在"，此时按钮为「重试」但 zip 已没了，`upload()` 会以 `MSG_ZIP_MISSING` 失败——可接受，用户只能彻底删除这条。

**房间 · 彻底删除**
1. 任一拍摄在服务端训练中（`state == training`，或 `state == exporting` 且 `message` 不以"正在下载模型"开头——`exporting` 也被借用作下载中的状态）→ 直接 Toast "有任务正在训练，完成后再删"，不做任何事（本地状态判断，不打服务端）。正在下载的拍摄不算：删它时先取消下载。
2. 按列表顺序对每条拍摄执行"拍摄 · 彻底删除"。某条失败 → 停止，Toast 原因；已删的已从列表消失，剩余保留，房间保留。
3. 全部成功 → 删 `filesDir/captures/<roomId>/`、`scenesDir/<roomId>/`，`repo.removeRoom(roomId)`，返回房间列表。中途崩溃留下空房间也无妨：空房间的删除对话框只有「删除」。

**房间 · 仅删手机**
1. 任一拍摄 `state in (uploading, queued, training, exporting)` → Toast "有任务在进行中，完成后再删"，不做任何事：云端任务还在跑却没了手机记录，就再也找不回来了。`done` / `failed` 的云端任务留在服务端是用户要的（状态页可见，电脑上可取可删）。
2. 删每条拍摄的本地文件 + 两个房间目录，`repo.removeRoom(roomId)`。云端任务不碰。

### 4.3 代码落点

- `rooms/RoomRepository.kt`：`removeRoom(roomId)`；`CaptureRecord.state` 新值 `"remote"`。
- `upload/UploadScreen.kt`（`UploadViewModel`）：`cancel()`、`redownload()`；`captureStatusText` 加 `"remote" -> "仅云端"`；`CaptureRow` 加 ⋮ 与「重新下载」按钮；`resumeInterrupted` 对 `remote` 不做事。
- 新文件 `rooms/DeleteViewModel.kt`（纯 JVM，可测）：`deleteCapture(roomId, captureId, mode)`、`deleteRoom(roomId, mode)`，`mode ∈ { CLOUD, LOCAL }`，依赖 `Api.deleteJob(id)`、`RoomRepository`、`UploadViewModel`（取消用）、`filesDir`、`scenesDir`。返回 `Result<Unit>`，失败消息中文。
- `net/ApiClient.kt`：`Api.deleteJob(jobId: String)`：`204`/`404` 返回，`409` 抛 `ApiException(detail, code = 409)`，其余沿用现有 `httpError`。
- `nav/AppNav.kt`：接线；删房间成功后 `nav.popBackStack()`。
- `AppGraph.kt`：`SettingsBackedApi.deleteJob` 转发；持有 `DeleteViewModel`。

### 4.4 文案（全部中文）

- 菜单：删除 / 删除房间
- 对话框标题：删除「{captureId 或房间名}」
- 对话框正文（拍摄，done）："彻底删除：手机和计算端上的拍摄包与模型都删掉，不可恢复。\n仅删手机文件：释放手机空间，保留记录，之后可从计算端重新下载模型。"
- 对话框正文（拍摄，其他）："手机和计算端上的拍摄包与模型都删掉，不可恢复。"
- 对话框正文（未上传）："删除手机上的拍摄包，不可恢复。"
- 对话框正文（房间）："彻底删除：这个房间的 {n} 条拍摄在手机和计算端都删掉，不可恢复。\n仅删手机：删掉手机上的房间和文件，计算端的任务保留。"
- 按钮：彻底删除 / 仅删手机文件 / 仅删手机 / 删除 / 取消 / 重新下载
- 状态："仅云端"
- 错误：服务端 409 原文（"任务正在训练，完成后再删" / "该上传正在接收中"）；房间级预检 "有任务正在训练，完成后再删"（彻底删除）、"有任务在进行中，完成后再删"（仅删手机）；网络 "连不上计算端"。

## 5. 测试

服务端（`tests/test_jobs.py`、`tests/test_api.py`）：
- `delete` 对 done / failed / queued 任务返回 204 且目录消失；再删 404。
- training 中 409、目录原样。
- 上传中（有 upload.json）204；锁被占 409。
- 非法 id 404。
- queued 任务被删后 worker 不崩、后续任务正常跑。

手机端（`UploadViewModelTest.kt`、新 `DeleteViewModelTest.kt`，FakeApi 加 `deleteJob` 记录 + 可注入响应）：
- 彻底删除：调用顺序 deleteJob → 文件删 → 记录删；404 照样删本地；409 什么都不删、消息透传；网络错误什么都不删。
- 上传中彻底删除：协程被取消、`deleteJob(uploadId)`；前几次返回 409 "该上传正在接收中" 时重试后成功；5 次仍 409 → 失败，记录不动。
- 下载中彻底删除：下载协程取消后云端删除；房间预检对下载中的拍摄放行。
- 仅删手机：`home == null` 时先算好再删 zip。
- 仅删手机：文件删、记录变 remote、jobId 保留；`captureStatusText` = "仅云端"。
- 重新下载：走 track，结束 state done、scenePath 非空。
- 房间彻底删除：有 training 直接拒；第二条失败时第一条已删、第二三条和房间保留；全成功房间目录与记录消失。
- 房间仅删手机：云端 deleteJob 调用数 0；有 uploading / queued 拍摄时拒绝。

## 6. 文档

- `docs/api.md`：新增 `DELETE /jobs/{id}` 一节；状态页说明加"电脑上删任务：`curl -X DELETE http://localhost:8765/api/v1/jobs/<id>`"。
- `docs/acceptance/phase1.md` §4：加删除验收项（彻底删除后服务端目录消失；本地删除后"重新下载"成功）。

## 7. 我拍板的小决定

- 用 ⋮ 菜单而不是长按：Compose 长按没有视觉提示，用户发现不了。
- "仅云端"是新状态 `remote` 而不是复用 `done` + 空 scenePath：状态文案和按钮判断都靠 state，塞进 done 会让所有 `state == "done"` 的判断多一个分支。
- 房间彻底删除的训练预检只看本地状态：手机每 3 s 轮询，与服务端最多差一次轮询；真撞上服务端 409 也会在逐条删时被拦住。
- 删除本地文件失败（极少见）不算失败：`File.delete()` 返回 false 就忽略，记录照常处理，避免用户卡在删不掉。
- 评审者提出、我拍板的：上传锁 409 短暂重试而不是直接报错；房间"仅删手机"遇到进行中的任务拒绝而不是把云端任务变成孤儿；本地删除前先算好查看器机位；下载中的拍摄不算"训练中"。
