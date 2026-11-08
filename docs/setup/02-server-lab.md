# 02 在实验室服务器上搭建重建服务（Ubuntu + NVIDIA 显卡）

目标：在实验室 Ubuntu 服务器（如 RTX A6000 Ada 48 GB）上运行同一套服务端，速度和画质档位更高（`server` 档位）。
所用脚本与笔记本完全相同，只是启动时选 `server` 档位。

> 需要能登录服务器终端（SSH 或直接接显示器）。不确定怎么登录请先问实验室管理员。

## 第 1 步：确认显卡驱动

在服务器终端运行：

```bash
nvidia-smi
```

能看到显卡型号（如 `RTX 6000 Ada Generation`）就行。报错则联系管理员安装 NVIDIA 驱动（需支持 CUDA 12.4 及以上）。
不要自己随意升级驱动，服务器可能还有别人在用。

## 第 2 步：获取代码

把项目放到服务器上，例如：

```bash
cd ~
git clone <项目仓库地址> Recon3D     # 或者用 scp / U 盘把项目文件夹拷过去
cd Recon3D
```

## 第 3 步：安装

```bash
bash server/scripts/install.sh
```

脚本会自动安装 uv、Python 3.10 和所有依赖（约 3 GB），**不需要管理员权限，也不需要装 CUDA 工具包**。
最后出现 `[成功] 安装完成，GPU 可用。` 即成功。失败时按屏幕中文提示处理后重跑即可。

## 第 3.5 步：预下载物体识别的模型权重（约 20 GB，建议先做）

"识别物体"要用三个模型（Grounding DINO base、SAM 2.1 large、Qwen2.5-VL-7B）。不预下载的话，第一次点"识别物体"时才从
Hugging Face 下载，手机上会等很久。实验室网络可直连 Hugging Face：

```bash
bash server/scripts/predownload_models.sh server
```

看到 `[成功] 模型权重已就绪。` 即可。权重放在 `~/.cache/huggingface`。

## 第 4 步：启动（server 档位）

```bash
bash server/scripts/run.sh server
```

**显卡和别人共用时**：先 `nvidia-smi` 看哪张卡空闲，用 `CUDA_VISIBLE_DEVICES=<编号>` 指定，例如：

```bash
CUDA_VISIBLE_DEVICES=1 bash server/scripts/run.sh server
```

识别开始时若该卡空闲显存不足 30 GB，服务端会自动改为逐个加载模型（慢一些，但不会因显存不够失败）；
再不够则任务显示"显存不足"，错开时间重试即可。

屏幕会列出可访问的地址。想让服务在你断开 SSH 后继续运行，可用 tmux：

```bash
tmux new -s recon3d
bash server/scripts/run.sh server
# 按 Ctrl+B 然后按 D 离开；以后用 tmux attach -t recon3d 回来
```

- 停止服务：回到该窗口按 `Ctrl+C`。
- 数据位置：`<项目目录>/server/data/jobs/`，每个任务一个文件夹，可手动删除旧任务来腾空间。
- 服务器多人共用 GPU 时，若显存被占用，重建可能报"显存不足"，请错开时间。

## 第 5 步：确认实验室网络允许手机访问 8765 端口

**这一步需要先确认**：很多实验室 / 校园 Wi-Fi 会开启"客户端隔离"，或只开放少数端口，手机可能访问不到服务器的 8765 端口。

1. 手机连上实验室 Wi-Fi。
2. 手机浏览器打开 `http://<服务器IP>:8765/api/v1/health`（服务器 IP 见启动时屏幕列出的地址，或运行 `hostname -I`）。
3. 看到类似下面的内容就说明可用：

```json
{"version":"...","gpu":"NVIDIA RTX 6000 Ada Generation","profile":"server"}
```

如果打不开：

- 服务器上先自测 `curl http://localhost:8765/api/v1/health`，通了说明服务正常，是网络问题。
- 服务器防火墙：运行 `sudo ufw status`，若是 active，需要管理员执行 `sudo ufw allow 8765/tcp`（请先征得实验室管理员同意）。
- 询问管理员：实验室 Wi-Fi 是否允许手机访问服务器的 8765 端口，或是否有客户端隔离。
- 实在不行，退回方案：使用笔记本方案（见 `01-server-laptop.md`）。手机先连笔记本、电脑再转发到实验室服务器的功能暂不支持，等第二期。

## 第 6 步（推荐）：用 Tailscale 让手机在任何网络都能访问

校园网常有客户端隔离，最省事的办法是让服务器、手机、笔记本都加入同一个 Tailscale 网络，手机直接用服务器的
Tailscale 地址（形如 `100.x.y.z`）。

1. 服务器上安装并登录（需要 sudo；按提示在浏览器里登录同一个 Tailscale 账号）：
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   tailscale ip -4      # 记下这个地址
   ```
2. 手机装 Tailscale App 并登录同一账号。
3. 手机 App 设置里把服务器地址改为 `http://<服务器的 Tailscale 地址>:8765`，打开「查看 3D」前先在手机浏览器试
   `http://<地址>:8765/api/v1/health`。
4. 让服务常驻：按第 4 步的 tmux 方式启动即可；服务器重启后要重新 `tmux` + `run.sh`。
- （仅供开发者调试）能登录服务器所在网络的电脑上，可以用命令行直接上传一个已取出的拍摄包：
  `curl -F capture=@cap.zip http://<服务器地址>:8765/api/v1/jobs`。这样生成的结果不会自动回到手机 App 里，普通使用不要走这条路。
