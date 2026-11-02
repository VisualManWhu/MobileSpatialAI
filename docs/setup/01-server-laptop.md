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

## 第 2 步：让手机能访问 WSL 里的服务（两条管理员命令，需要你本人操作）

WSL 在一个虚拟网络里，手机直接连不到它。做法是：服务端在 WSL 里监听 **18765** 端口（启动脚本在 WSL 下会自动这样做），
Windows 把自己的 **8765** 端口转发给它，手机只需要知道笔记本的 IP 和 8765。

> 实测 2026-11-02：WSL 的"镜像网络模式"（`networkingMode=mirrored`）在这台笔记本上不稳定——配置全对的情况下手机时通时不通，
> 还会把 8765 端口劫持走。**不要用镜像模式**。如果你的 `%UserProfile%\.wslconfig` 里有 `networkingMode=mirrored`，
> 把它改成 `networkingMode=NAT`（或删掉这一行），然后 `wsl --shutdown`。

这两条是系统安全设置，**请你自己执行，不要让工具代劳**：开始菜单搜索 "PowerShell"，**右键 → 以管理员身份运行**，依次粘贴：

1. 端口转发（Windows 原生功能，重启后仍有效）：

```powershell
netsh interface portproxy add v4tov4 listenport=8765 listenaddress=0.0.0.0 connectport=18765 connectaddress=127.0.0.1
```

2. 放行 Windows 防火墙的 8765 端口：

```powershell
New-NetFirewallRule -DisplayName "Recon3D server 8765" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow -Profile Private
```

含义：只允许 TCP 8765 端口、只在"专用网络"下生效（家里 / 实验室确认过的 Wi-Fi）。
你的 Wi-Fi 必须被 Windows 标记为"专用网络"：设置 → 网络和 Internet → WLAN → 点你的 Wi-Fi → 网络配置文件类型选"专用网络"；
或在管理员 PowerShell 运行 `Set-NetConnectionProfile -InterfaceAlias "WLAN" -NetworkCategory Private`。

检查：`netsh interface portproxy show v4tov4` 能看到 `0.0.0.0 8765 → 127.0.0.1 18765` 一行；
`Get-NetFirewallRule -DisplayName "Recon3D server 8765"` 显示 `Enabled : True`。

撤销：`netsh interface portproxy delete v4tov4 listenport=8765 listenaddress=0.0.0.0` 和
`Remove-NetFirewallRule -DisplayName "Recon3D server 8765"`。

> 旧版说明里的 `New-NetFirewallHyperVRule ... Recon3D8765` 规则不再需要；留着无害，想删：`Remove-NetFirewallHyperVRule -Name Recon3D8765`。

## 第 3 步：给笔记本固定 IP（建议）

路由器可能给笔记本换 IP（实测重置 WSL 网络后从 `.10` 变成了 `.16`），换了之后手机 App 里填的地址就失效。
在路由器管理页里给笔记本"始终分配同一个 IP"（Fritz!Box：Heimnetz → Netzwerk → 笔记本 → 勾选 "Diesem Netzwerkgerät immer die gleiche IPv4-Adresse zuweisen"）。
不固定也能用，只是 IP 变了要在 App 设置里改一次。

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

在 WSL 下脚本自动使用 18765 端口（配合第 2 步的转发），屏幕会出现
`Uvicorn running on http://0.0.0.0:18765`，并提示手机应访问的地址（笔记本 IP + 8765）。**保持这个窗口不要关**。

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
- 笔记本 IP 是不是变了？PowerShell 运行 `ipconfig` 重新看一眼（见第 3 步）。
- 第 2 步的两条命令是否都执行了？Wi-Fi 是否是"专用网络"？
- `.wslconfig` 里是否还有 `networkingMode=mirrored`？有则改成 NAT 并 `wsl --shutdown`。
- 服务窗口是否还在运行？
- 在笔记本自己的浏览器依次打开：`http://127.0.0.1:18765/api/v1/health`（服务本身）、`http://127.0.0.1:8765/api/v1/health`（端口转发）、
  `http://<笔记本IP>:8765/api/v1/health`（防火墙）。哪一个打不开，问题就在对应那层。

## Tailscale 方式（推荐；不要求同一个 Wi-Fi）

Tailscale 会在手机和笔记本之间建一条加密的虚拟局域网，两边不在同一个 Wi-Fi（甚至一个用流量）也能互连，
也绕开了 "AP 隔离" 的问题。上面的同一 Wi-Fi 方式仍然可用，二选一即可。

1. **笔记本（Windows）**：打开 https://tailscale.com/download 下载 Windows 版并安装，登录一个账号。
2. **手机**：在 Google Play 安装 "Tailscale"，用**同一个账号**登录，按提示允许建立 VPN。
3. **WSL 里不用装 Tailscale**：第 2 步的端口转发监听在 Windows 的所有网卡上，Tailscale 网卡也包括在内。
4. **防火墙规则：只放行 Tailscale 地址段**。Windows 通常把 Tailscale 网卡归为"公用网络"，第 2 步的 `-Profile Private` 规则对它不生效。
   这里改用一条 `-Profile Any`（任何网络类型都生效）的规则，但加上 `-RemoteAddress 100.64.0.0/10`：
   这是 Tailscale 给设备分配的地址段，**只有登录了你 Tailscale 账号的设备能连进来**；
   即使笔记本连着咖啡馆、机场等公共 Wi-Fi，同一 Wi-Fi 上的陌生设备也访问不了 8765 端口。
   在**管理员 PowerShell** 里（由你本人执行）：
   ```powershell
   New-NetFirewallRule -DisplayName "Recon3D server 8765 Tailscale" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow -Profile Any -RemoteAddress 100.64.0.0/10
   ```
   第 2 步那条 `-Profile Private` 的 Wi-Fi 规则可以保留（两种方式都能用），也可以不建。
   撤销：`Remove-NetFirewallRule -DisplayName "Recon3D server 8765 Tailscale"`。
5. **查笔记本的 Tailscale 地址**：PowerShell 运行

   ```powershell
   tailscale ip -4
   ```

   会输出一个 `100.` 开头的地址，例如 `100.101.102.103`。
6. 手机浏览器打开 `http://100.101.102.103:8765/api/v1/health`（换成你的地址），看到 health JSON 即成功。
   在 App 的设置页里填 `100.101.102.103:8765` 即可（见 03-phone.md）。

连不上时：两边 Tailscale App 是否都显示 "Connected"、是否同一账号；在笔记本上 `tailscale status` 能否看到手机；其余同上面的"打不开怎么办"。
