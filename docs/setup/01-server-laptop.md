# 01 在笔记本上搭建重建服务（Windows + WSL2 + NVIDIA 显卡）

目标：让笔记本跑起 Recon3D 服务端，并让同一 Wi-Fi 下的手机能访问它。
适用：Windows 11，NVIDIA 显卡（如 RTX 4060 8 GB）。全程约 20–40 分钟，大部分是等待下载。

> 带 `PowerShell` 的命令在 Windows 的 PowerShell 里敲；带 `Ubuntu` 的命令在 Ubuntu（WSL）终端里敲。
> 在 Windows 开始菜单搜索 "PowerShell" 或 "Ubuntu" 即可打开。

## 第 0 步：确认已安装 WSL 的 Ubuntu

在 PowerShell 运行：

```powershell
wsl -l -v
```

- 列表里有 `Ubuntu`，且 VERSION 为 `2`：跳到第 1 步。
- 没有 Ubuntu：打开 "Microsoft Store"，搜索 "Ubuntu 24.04"（或 "Ubuntu"），点"获取"安装；
  装完从开始菜单打开 Ubuntu，按提示设置用户名和密码（输入密码时屏幕不显示字符，属正常）。
- VERSION 是 `1`：运行 `wsl --set-version Ubuntu 2`。

（截图位置：Microsoft Store 搜索 Ubuntu 的结果页。）

## 第 1 步：更新 NVIDIA 驱动，并在 WSL 里验证

1. 在 **Windows** 上更新显卡驱动：打开 https://www.nvidia.cn/drivers/ ，选你的显卡型号下载并安装（或用 GeForce Experience / NVIDIA App 更新）。
   **只在 Windows 装驱动，不要在 Ubuntu 里装显卡驱动**，WSL 会自动使用 Windows 的驱动。
2. 装完后在 PowerShell 运行 `wsl --shutdown`，再重新打开 Ubuntu。
3. 在 **Ubuntu** 运行：

```bash
nvidia-smi
```

能看到一个表格，里面有你的显卡型号（如 `NVIDIA GeForce RTX 4060 Laptop GPU`）就成功了。
如果报"command not found"或看不到显卡，说明驱动太旧或没装好，回到第 1 步重做。

（截图位置：nvidia-smi 的正常输出表格。）

## 第 2 步：设置 WSL 镜像网络，让手机能访问

默认情况下 WSL 在一个虚拟网络里，手机连不上。"镜像网络模式"会让 WSL 和 Windows 共用同一个 IP。

前提：需要 **Windows 11 22H2 或更新版本**，且 WSL 版本够新。检查方法：

- 按 `Win+R` 输入 `winver` 回车，看版本号是否为 22H2 及以上。
- 在 PowerShell 运行 `wsl --version`，能显示版本号即可；若报错或版本很旧，先运行 `wsl --update` 升级。

1. 在 PowerShell 运行 `notepad $env:UserProfile\.wslconfig`（文件不存在时会提示新建，选"是"）。
2. **检查**里面是否已有下面两行；没有就**补上**（已有则不用改）：

```ini
[wsl2]
networkingMode=mirrored
```

3. 保存后，在 PowerShell 运行 `wsl --shutdown`，再重新打开 Ubuntu 使其生效。

## 第 3 步：放行 Windows 防火墙的 8765 端口（需要你本人操作）

服务端使用 TCP 8765 端口。Windows 防火墙默认会挡住手机的访问，需要放行一次。
这是系统安全设置，**请你自己执行，不要让工具代劳**：

1. 开始菜单搜索 "PowerShell"，**右键 → 以管理员身份运行**。
2. 粘贴下面这一行并回车：

```powershell
New-NetFirewallRule -DisplayName "Recon3D server 8765" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow -Profile Private
```

含义：新增一条"入站规则"，只允许 TCP 8765 端口、只在"专用网络"下生效（家里 / 实验室确认过的 Wi-Fi）。
注意你的 Wi-Fi 必须被 Windows 标记为"专用网络"：设置 → 网络和 Internet → WLAN → 点你的 Wi-Fi → 网络配置文件类型选"专用网络"。
想撤销时，管理员 PowerShell 运行：`Remove-NetFirewallRule -DisplayName "Recon3D server 8765"`。

## 第 3 步（续）：放行 WSL 的 Hyper-V 防火墙入站（同样由你本人执行）

镜像网络模式下，WSL 自己还有一层 "Hyper-V 防火墙"，默认会拦截手机发往 WSL 的入站连接，
只放行上面的 Windows 规则还不够。仍在**管理员 PowerShell** 里运行：

```powershell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
```

说明：`{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}` 是 WSL 的 Hyper-V 防火墙配置的固定标识；
`-DefaultInboundAction Allow` 表示允许入站连接进入 WSL。
可用 `Get-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'` 查看，`DefaultInboundAction` 显示 `Allow` 即已生效。
想撤销（恢复默认拦截），管理员 PowerShell 运行：

```powershell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction NotConfigured
```

## 第 4 步：安装服务端

在 **Ubuntu** 里（项目在 `D:\projects\Recon3D`，对应 WSL 路径 `/mnt/d/projects/Recon3D`）：

```bash
cd /mnt/d/projects/Recon3D
bash server/scripts/install.sh
```

首次会下载约 3 GB（PyTorch 等），需几分钟，请保持网络畅通。最后出现 `[成功] 安装完成，GPU 可用。` 即成功。
出现 `[失败]` 时，按屏幕上的中文提示处理，然后可以直接重新运行同一条命令（脚本可重复执行）。

## 第 5 步：启动服务

```bash
bash server/scripts/run.sh
```

启动后屏幕会列出几个形如 `http://192.168.x.x:8765/api/v1/health` 的地址，并出现
`Uvicorn running on http://0.0.0.0:8765`。**保持这个窗口不要关**。

- 停止服务：在这个窗口按 `Ctrl+C`。
- 数据位置：上传的拍摄包和重建结果在 `D:\projects\Recon3D\server\data\jobs\` 下，每个任务一个文件夹。可以手动删除旧任务的文件夹来腾空间。
- 笔记本休眠或合盖会中断服务；重建期间请接电源并保持亮屏或设置"从不休眠"。

## 第 6 步：用手机验证

1. 手机连接与笔记本**同一个 Wi-Fi**。
2. 先在笔记本 Windows 上确认自己的 IP：PowerShell 运行 `ipconfig`，找 "无线局域网适配器 WLAN" 下的 `IPv4 地址`（如 `192.168.1.23`）。
3. 手机浏览器打开 `http://<笔记本IP>:8765/api/v1/health`，例如 `http://192.168.1.23:8765/api/v1/health`。
4. 看到类似下面的文字就成功了：

```json
{"version":"...","gpu":"NVIDIA GeForce RTX 4060 Laptop GPU","profile":"laptop"}
```

（截图位置：手机浏览器显示 health JSON。）

### 打不开怎么办

- 手机和笔记本是不是同一个 Wi-Fi？有些 Wi-Fi 开了"AP 隔离/客户端隔离"（学校、实验室常见），设备之间互相访问不了，可换用手机热点让笔记本连接。
- 第 3 步的防火墙规则是否已创建？Wi-Fi 是否是"专用网络"？
- 第 2 步 `.wslconfig` 是否生效？改完需 `wsl --shutdown` 后重开 Ubuntu。
- 第 3 步（续）的 Hyper-V 入站设置是否已执行？这是手机连不上的常见原因：用上面的 `Get-NetFirewallHyperVVMSetting` 命令检查 `DefaultInboundAction` 是否为 `Allow`。
- Windows 版本是否 22H2 及以上、`wsl --version` 是否正常（见第 2 步前提）？
- 服务窗口是否还在运行？
- 先在笔记本自己的浏览器打开 `http://localhost:8765/api/v1/health`，能打开说明服务本身正常，问题在网络。

## Tailscale 方式（推荐；不要求同一个 Wi-Fi）

Tailscale 会在手机和笔记本之间建一条加密的虚拟局域网，两边不在同一个 Wi-Fi（甚至一个用流量）也能互连，
也绕开了 "AP 隔离" 的问题。上面的同一 Wi-Fi 方式仍然可用，二选一即可。

1. **笔记本（Windows）**：打开 https://tailscale.com/download 下载 Windows 版并安装，登录一个账号。
2. **手机**：在 Google Play 安装 "Tailscale"，用**同一个账号**登录，按提示允许建立 VPN。
3. **WSL 里不用装 Tailscale**：第 2 步的镜像网络模式下，WSL 直接共用 Windows 的 Tailscale 网卡。
4. **防火墙规则改用 `-Profile Any`**：Windows 通常把 Tailscale 网卡归为"公用网络"，第 3 步的 `-Profile Private` 规则对它不生效。
   在**管理员 PowerShell** 里（由你本人执行）：
   - 还没建过第 3 步的规则：
     ```powershell
     New-NetFirewallRule -DisplayName "Recon3D server 8765" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow -Profile Any
     ```
   - 已经建过：
     ```powershell
     Set-NetFirewallRule -DisplayName "Recon3D server 8765" -Profile Any
     ```
   撤销方法与第 3 步相同：`Remove-NetFirewallRule -DisplayName "Recon3D server 8765"`。
5. **第 3 步（续）的 Hyper-V 入站设置不变**，照常执行一次即可。
6. **查笔记本的 Tailscale 地址**：PowerShell 运行

   ```powershell
   tailscale ip -4
   ```

   会输出一个 `100.` 开头的地址，例如 `100.101.102.103`。
7. 手机浏览器打开 `http://100.101.102.103:8765/api/v1/health`（换成你的地址），看到 health JSON 即成功。
   在 App 的设置页里填 `100.101.102.103:8765` 即可（见 03-phone.md）。

连不上时：两边 Tailscale App 是否都显示 "Connected"、是否同一账号；在笔记本上 `tailscale status` 能否看到手机；其余同上面的"打不开怎么办"。
