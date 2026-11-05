# 03 把 Recon3D App 装到手机（Pixel / 支持 ARCore 的 Android 手机）

目标：在电脑上编译 App，用 USB 线装到手机，并让 App 连上第 01 篇搭好的计算端。
适用：Android 13 及以上、支持 ARCore 的手机（Pixel 系列都支持）。全程约 20–30 分钟，首次编译要下载几百 MB。

> 前提：已按 [01-server-laptop.md](01-server-laptop.md) 跑起服务端，并且手机浏览器能打开 `.../api/v1/health`。

## 第 0 步：安装 Android Studio（电脑上，一次即可）

1. 打开 https://developer.android.com/studio ，下载 Windows 版 Android Studio 并安装（一路"下一步"即可）。
2. 第一次启动会弹出 "Setup Wizard"：选 **Standard**，接受许可协议，等待它下载 Android SDK。
   Android Studio 自带 Java（JDK），不需要另外安装。
3. 装完后，确认命令行能用（可选）：PowerShell 运行

   ```powershell
   $env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'
   cd D:\projects\Recon3D\android
   .\gradlew.bat --version
   ```

   能看到 `Gradle 9.8.2` 字样即可。第一次会自动下载 Gradle，需要几分钟。

## 第 1 步：在手机上打开"开发者选项"

1. 手机打开 **设置 → 关于手机**。
2. 找到 **版本号**（Build number），**连续点 7 次**，直到提示"您现在处于开发者模式"（可能要输入锁屏密码）。
3. 返回设置，进入 **设置 → 系统 → 开发者选项**（Pixel 上的位置；其他品牌可在设置里搜索"开发者选项"）。

（截图位置：关于手机页面的"版本号"。）

## 第 2 步：开启 USB 调试

1. 在 **开发者选项** 里打开 **USB 调试**，在弹窗里点"确定"。
2. 用 USB 线（能传数据的线，不是只能充电的线）把手机连到电脑。
3. 手机会弹出"允许 USB 调试吗？"，勾选 **"一律允许使用这台计算机进行调试"**，点 **允许**。
4. 若手机上方提示"USB 用于充电"，可以不用管；调试连接不依赖文件传输模式。

检查是否连上（可选）：PowerShell 运行

```powershell
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" devices
```

列表里出现一行设备序列号且后面是 `device` 就成功了；如果是 `unauthorized`，看手机屏幕上的授权弹窗。

（截图位置：手机上的"允许 USB 调试吗？"弹窗。）

## 第 3 步：确认手机装有 ARCore

App 需要 "Google Play 服务（AR）"（Google Play Services for AR）。Pixel 通常已预装。
打开 Google Play 搜索 **"Google Play Services for AR"**，显示"更新"就点更新，显示"打开"或已安装即可。
（第一次打开 App 时如果没装，App 也会提示去安装。）

## 第 4 步：在 Android Studio 里运行 App

1. 打开 Android Studio，选 **Open**，选择文件夹 `D:\projects\Recon3D\android`（注意是 `android` 子文件夹，不是整个项目）。
2. 第一次打开会自动 "Gradle Sync"（右下角有进度条），需要下载依赖，等它完成（几分钟）。
   如果顶部出现 "Trust project?"，选 **Trust Project**。
3. 顶部工具栏的设备下拉框里选择你的手机（显示手机型号，例如 "Google Pixel 8"）。
4. 点绿色的 **▶ Run 'app'** 按钮。
5. 编译完成后 App 会自动在手机上打开，图标名字是 **Recon3D**。

不用 Android Studio 也可以（命令行方式）：

```powershell
$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'
cd D:\projects\Recon3D\android
.\gradlew.bat :app:installDebug
```

装好后在手机桌面找 Recon3D 图标打开。编译出的安装包位于 `android\app\build\outputs\apk\debug\app-debug.apk`。

（截图位置：Android Studio 顶部的设备下拉框和 Run 按钮。）

## 第 5 步：在 App 里填计算端地址并测试连接

1. 确认电脑上的服务端正在运行（01 篇第 5 步的窗口没关）。
2. 打开 App，在 **计算端地址** 里填：
   - 同一 Wi-Fi 方式：笔记本的 IPv4 地址加端口，例如 `192.168.1.23:8765`；
   - Tailscale 方式（推荐，家里外面都一样）：笔记本 `tailscale ip -4` 得到的地址，例如 `100.101.102.103`（端口可省，默认 8765）。上传和模型下载都支持断点续传：断网后 App 显示"网络中断，正在重连 (n)"并自动从断点继续，杀掉 App 再打开也会接着传。
   不用写 `http://`，App 会自动补上。
3. 点 **测试连接**：
   - 成功：显示类似 **"已连接：RTX 4060 Laptop GPU（笔记本档）"**，地址会被记住，下次打开不用再填。
   - 显示 **"连不上计算端，请检查 Wi-Fi 和地址"**：
     先用手机浏览器打开 `http://<同一个地址>/api/v1/health`。浏览器也打不开，按 01 篇"打不开怎么办"排查网络；
     浏览器能打开但 App 不行，检查地址是否填错（端口 `8765` 不能省）。
   - 显示 **"计算端返回错误：HTTP …"**：地址指向了别的服务，确认端口是 `8765`。

（截图位置：App 设置页显示"已连接"。）

## USB 方式（Wi-Fi 太慢或不通时）

手机和笔记本之间的 Wi-Fi 很慢时（2026-11-02 实测 2.4 GHz 下不到 40 KB/s，上传总是中断），可以直接走数据线：

1. 手机用数据线连笔记本，USB 调试已允许（见第 2 步）。
2. 双击 `android\scripts\usb-route.cmd`（或在 PowerShell 运行 `& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" reverse tcp:8765 tcp:8765`）。
3. App 设置页的计算端地址填 `127.0.0.1:8765`，测试连接。

**注意：每次拔掉再插上数据线，这条通道就失效**，App 会提示"连不上计算端"；重新运行第 2 步即可，不用改地址。拍摄时可以拔线走动，上传前插回来再跑一次脚本。

## 上传和训练期间的注意事项

- 上传和训练期间保持应用在前台、屏幕常亮（设置里可临时把屏幕超时调长，例如 设置 → 显示 → 屏幕超时 → 30 分钟）。
  切到后台或锁屏后，系统可能暂停 App 的网络连接，轮询会中断。
- 网络短暂中断时 App 会自动重试（状态里显示"网络暂时不可用，正在重试 (n/10)"）；连续 10 次失败才显示失败，此时点"重试"会接着查原来的任务，不用重新上传。

## 常见问题

- **Android Studio 里看不到手机**：换一根数据线或 USB 口；手机重新插拔后再看授权弹窗；开发者选项里关掉再打开 USB 调试。
- **"INSTALL_FAILED_..." 安装失败**：手机上已装过旧版签名不同的 Recon3D，先在手机上卸载再运行。
- **Gradle Sync 很慢或失败**：首次需要下载几百 MB，确认网络畅通后点 **File → Sync Project with Gradle Files** 重试。
