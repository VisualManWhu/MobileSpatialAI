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

## 第 4 步：启动（server 档位）

```bash
bash server/scripts/run.sh server
```

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
- 实在不行，退回方案：手机拍完后在同一网络的电脑上传，或使用笔记本方案（见 `01-server-laptop.md`）。
