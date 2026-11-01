# 手机端 3D 查看器

## 是什么

拍摄并在计算端重建完成后，App 会把 `scene.splat` 下载到本地。在房间页点"查看"，就会打开查看器：
一个全屏 WebView，里面跑打包进 APK 的 three.js + Spark（3D Gaussian Splatting 渲染器），
在手机上实时渲染这个场景。

- 单指拖动：旋转；双指：缩放和平移（three.js `OrbitControls`，带阻尼）。
- 完全离线：页面和脚本都在 APK 的 `assets/viewer/` 里，不访问 CDN 或任何网络地址。
- WebView 只开放两个地址（`WebViewAssetLoader`，域名 `appassets.androidplatform.net`）：
  - `/assets/...`：APK 内置资源（页面和脚本）；
  - `/scene/scene.splat`：**只**返回当前这次拍摄的场景文件，其他任何路径一律 404，其他域名一律 403。
- 页面通过 `Android.onLoaded(count)` / `Android.onError(消息)` 回调 App，App 据此收起"正在加载场景…"遮罩或显示错误。
- 省电：只在手指操作、阻尼收尾、镜头飞行或 Spark 排序完成时渲染，静止时不渲染。

## 坐标系

`.splat` 中的坐标就是 ARCore 的世界坐标系（重力对齐、**+Y 朝上**、右手系）。
计算端只把**相机**轴从 OpenGL 约定换成 OpenCV 约定（`server/recon3d/capture/poses.py` 右乘），世界坐标系没变。
three.js 同样是 +Y 朝上的右手系，所以查看器里**不做任何坐标转换**（不翻转 Y）。

初始视角：`.splat` 里没有相机位姿，所以初始相机由场景范围计算——看向场景中心（按 2%–98% 分位数去掉漂浮点），
从 +Z 一侧、抬高 30° 往 -Z 方向看（ARCore 会话开始时手机大致朝 -Z），距离刚好装下整个场景。
这是对"放在第一帧相机位置"的简化，后续任务可以再改。

## JS 接口（第二期会用）

在 Chrome 远程调试（`chrome://inspect`，仅 debug 包开启）控制台或 `WebView.evaluateJavascript` 里调用：

| 函数 | 说明 |
| --- | --- |
| `loadScene(url)` | 加载一个 `.splat`，替换当前场景，返回 `Promise<高斯数量>`；失败时 reject。 |
| `flyTo({x,y,z}, {x,y,z}, durationMs)` | 镜头平滑（ease-in-out）飞到第一个点，看向第二个点。坐标为场景世界坐标（与 `.splat` 相同）。返回 `Promise`，飞完 resolve。 |
| `setCameraHome()` | 立即回到初始视角。 |

例：`flyTo({x:0, y:1.5, z:2}, {x:0, y:1, z:0}, 1500)`

## 重新打包

源码在 `android/viewer-web/`（`src/viewer.js`、`src/index.html`、`src/scene-math.js`）。
打包产物 `android/app/src/main/assets/viewer/{index.html,viewer.js,THIRD_PARTY_LICENSES.txt}` **已提交到仓库**，
所以构建 APK 不需要 Node。改了网页源码后需要手动重新打包并提交产物：

```
cd android/viewer-web && npm ci && npm run build
npm test   # 纯函数单元测试（node:test）
```

依赖版本钉死（`package.json` 精确版本 + `package-lock.json`）：
`@sparkjsdev/spark@2.4.0`、`three@0.187.0`、`esbuild@0.28.2`（仅开发）。

## 许可证

Spark（World Labs）和 three.js 都是 MIT 许可证，可以随 App 分发；
`npm run build` 会把两者的 LICENSE 原文写到 `assets/viewer/THIRD_PARTY_LICENSES.txt`，随 APK 一起分发。
