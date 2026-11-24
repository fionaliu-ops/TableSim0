# TableSim Docker 镜像迁移与服务器部署指南

本文对应 **2026-09-14 已完成并验证的镜像** `tablesim-complete:20260914`，指南更新于 2026-09-15。按下面的步骤，可以在新服务器上导入环境、启动 SAM3、运行 Scene Graph → Reasoning Graph 推理并保存结果。

镜像已包含项目、依赖、数据及模型权重；服务器无需重新安装项目 Python 环境或下载 SAM3 权重。本文的部署命令针对 Ubuntu 22.04 / 24.04 的 x86_64 服务器，默认使用 `sudo docker`。

## 1. 镜像在哪里，需要拿走哪些文件

原机器上的镜像文件绝对路径：

```text
/home/ubuntu/Desktop/Liutong/TableSim_migration_20260914/tablesim-complete-20260914.tar
```

| 项目 | 内容 |
| --- | --- |
| 交付目录 | `/home/ubuntu/Desktop/Liutong/TableSim_migration_20260914/` |
| 文件大小 | 89,829,225,472 字节，约 83.66 GiB |
| 导入后的标签 | `tablesim-complete:20260914` |
| 平台 | `linux/amd64`，即 Linux x86_64 |
| 镜像内 Ubuntu | 24.04 用户空间 |
| 项目快照提交 | `be918c181d206da1e779cf7ab32f60f102603902`，另含打包时的本地修改 |
| 打包时 Docker / Compose | Docker 29.4.1 / Compose 5.1.3 |

文件 SHA-256：

```text
6e85edad66a12fe15cb93de72845ee3e29e29ddd3d7050f7ae2d16ffb67b5c95
```

建议一并复制交付目录里的 `SHA256SUMS`、`PACKAGE_INFO.json`、`README.md`、`load.sh`、`run.sh`、`compose.yaml`。`validation/` 与 `metadata/` 可用于核对验证记录和环境清单。构建暂存目录不需要复制。

**只拿到 `.tar` 文件也可以部署**：本文提供校验、导入、Compose 配置和运行命令。GitHub 上上传的是这份指南，83.66 GiB 的镜像文件仍在上述原机器路径，需要单独传输。

镜像是固定快照，GitHub 后续提交不会自动进入其中。本文推理参数针对镜像内版本；不要把最新仓库的默认行为当成镜像内行为。

## 2. 服务器条件

| 条件 | 要求或建议 |
| --- | --- |
| CPU 架构 | `uname -m` 应输出 `x86_64`；此镜像未提供 ARM 版本 |
| 系统 | Linux；下面安装命令适用于 Ubuntu 22.04 / 24.04 |
| GPU | 兼容镜像内 CUDA / PyTorch 的 NVIDIA GPU；源机器验证使用 RTX 4090 |
| 仿真 | SAPIEN / Isaac Sim 还需要对应的图形、Vulkan / RTX 能力，不能仅凭 CUDA 可用判断仿真可用 |
| 磁盘 | 建议至少 400 GB 可用空间，覆盖 tar、Docker 镜像层、展开文件及新增输出；这是部署预留建议 |
| 内存 | 按所运行任务配置；下面设置 16 GiB 共享内存上限，并非整套环境的内存需求上限 |
| Docker | 推荐使用与打包时相近的现代 Docker Engine，安装 Compose 插件 |
| Compose | `gpus: all` 需要 Compose 2.30.0 或更高版本，包含 Compose 5.x |
| 网络 | 安装系统工具时需访问软件源；镜像导入后的规则推理流程已经断网验证 |

Compose GPU 配置要求见 [Docker 官方说明](https://docs.docker.com/reference/compose-file/services/#gpus)。GPU 换代、驱动差异或显存差异可能需要额外适配，本次未在你的目标服务器上执行验收。

镜像包含 Ubuntu 用户空间，不包含原机器内核、启动分区、正在运行的进程或 NVIDIA 内核驱动。服务器需要自己的系统和驱动，不需要在宿主机重新安装 Conda、SAM3、SAPIEN 或 CUDA Toolkit。

在**目标服务器**检查：

```bash
uname -m
cat /etc/os-release
nvidia-smi
df -h
free -h
```

除了放 tar 的磁盘，还要检查 Docker 实际使用的磁盘。安装 Docker 后可执行：

```bash
sudo docker info --format '{{.DockerRootDir}}'
sudo du -sh /var/lib/docker /var/lib/containerd 2>/dev/null
df -h /var/lib/docker /var/lib/containerd 2>/dev/null
```

如果使用了自定义存储路径，应检查对应挂载点；不要只看 `/data` 有空间就开始向空间不足的系统盘导入。

## 3. 安装服务器基础环境

已有可用 Docker、Compose、驱动和 GPU 容器支持的服务器，可以跳过对应安装步骤，直接到第 4 节。

### 3.1 NVIDIA 驱动

先执行 `nvidia-smi`。若能正常显示显卡，先保留现有驱动，后面用容器内实际计算判断兼容性。

新装 Ubuntu 服务器尚无驱动时，可使用发行版工具选择驱动：

```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers list --gpgpu
sudo ubuntu-drivers install --gpgpu
```

安装后在可维护时段重启服务器，再执行 `nvidia-smi`。如果只缺少该命令，安装与驱动分支匹配的 `nvidia-utils` 包。驱动选择、Secure Boot 和计算服务器安装细节见 [Ubuntu 官方驱动指南](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/)。不要为匹配某个示例版本去替换服务器上已正常工作的驱动。

### 3.2 Docker Engine 与 Compose

以下适用于尚未安装 Docker 的 Ubuntu 服务器。已有其他 Docker/containerd 安装时，先按 [Docker Ubuntu 安装说明](https://docs.docker.com/engine/install/ubuntu/) 处理包冲突，避免直接覆盖现有容器服务。

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg rsync
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker version
sudo docker compose version
```

### 3.3 NVIDIA Container Toolkit

该工具让容器使用宿主机 GPU。以下仓库及配置方式来自 [NVIDIA 官方安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes \
      -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

最后一步重启 Docker，服务器已有业务容器时应安排维护时段。后面的 GPU 检查直接使用已导入的 TableSim 镜像，无需另外下载测试镜像。

完全离线的服务器需提前准备对应发行版的软件包或内部软件源；TableSim tar 不包含 Docker Engine 和宿主机驱动安装包。

## 4. 把文件传到服务器

下面把部署目录统一设为 `/data/tablesim-migration`；可以更换路径，但后续命令应保持一致。

先在**目标服务器**执行，当前登录用户应为实际接收文件的用户：

```bash
sudo mkdir -p /data/tablesim-migration
sudo chown "$(id -u):$(id -g)" /data/tablesim-migration
```

然后在**原机器**执行，替换登录用户名和 IP：

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim_migration_20260914
rsync -ah --partial --append-verify --info=progress2 \
  tablesim-complete-20260914.tar SHA256SUMS PACKAGE_INFO.json \
  README.md load.sh run.sh compose.yaml \
  YOUR_USER@YOUR_SERVER_IP:/data/tablesim-migration/
```

中断后重新执行同一条命令可以续传；首次续传后的完整性以第 5 节 SHA-256 为准。SSH 非默认端口时给 `rsync` 增加 `-e 'ssh -p 2222'`。镜像层已压缩，不需要额外使用 `rsync -z`。

若文件已经在你的电脑、移动硬盘或其他存储上，直接把 tar 复制到上述目标目录即可，不要求服务器能连接原机器，也不要求克隆 GitHub 仓库。

## 5. 校验并导入

以下及后续未特别标注的命令均在**目标服务器**执行：

```bash
cd /data/tablesim-migration
sha256sum -c SHA256SUMS
sudo docker load -i tablesim-complete-20260914.tar
sudo docker image inspect tablesim-complete:20260914 \
  --format '{{.Id}} {{.Os}}/{{.Architecture}}'
```

如果只有 tar，没有 `SHA256SUMS`，先创建该文件再执行以上命令：

```bash
cd /data/tablesim-migration
printf '%s  %s\n' \
  '6e85edad66a12fe15cb93de72845ee3e29e29ddd3d7050f7ae2d16ffb67b5c95' \
  'tablesim-complete-20260914.tar' > SHA256SUMS
```

校验应显示 `OK`，导入应显示 `Loaded image: tablesim-complete:20260914`。两步都需要完整读大文件，时间由磁盘与 Docker 存储决定。校验失败应先重新传输文件。

源机器使用相同 Docker 存储后端重导入验证的镜像 ID 是：

```text
sha256:b36c90bf24075efb925e22faa8ef19a96c1e17101e33c323077f4f8073980c22
```

不同 Docker 版本可能以不同方式显示镜像索引和配置 ID；文件 SHA-256、`linux/amd64`、镜像内清单与运行检查共同用于验收。

如果同时拿到了 `load.sh`，可以用 `sudo bash load.sh` 替代本节手工校验及导入。不要使用 `docker import`，不要解开 tar 后执行构建，也不需要 `docker pull`。

确认 GPU 透传和真实 CUDA 计算：

```bash
sudo docker run --rm --network none --gpus all \
  tablesim-complete:20260914 nvidia-smi
sudo docker run --rm --network none --gpus all \
  tablesim-complete:20260914 python -c \
  'import torch; assert torch.cuda.is_available(); x=torch.arange(8, device="cuda"); print(torch.__version__, torch.cuda.get_device_name(0), x.sum().item())'
```

第二条应输出设备名和求和结果 `28`。若出现 CUDA 架构不支持、驱动不兼容或 OOM，先处理 GPU 环境再启动长期服务。

## 6. 启动服务，并让新结果直接保存在服务器磁盘

### 6.1 创建输出目录

镜像默认以 UID/GID `1000:1000` 运行。为本次部署新建一个输出目录并赋予对应写权限：

```bash
cd /data/tablesim-migration
sudo install -d -o 1000 -g 1000 -m 0775 outputs
```

### 6.2 创建独立的部署配置

下面创建 `compose.server.yaml`，与原交付的 `compose.yaml` 分开。这个配置只把新增输出映射到宿主机 `./outputs`；镜像内已有代码、权重和历史结果仍使用镜像快照。

```bash
cat > compose.server.yaml <<'YAML'
name: tablesim-migration
services:
  tablesim:
    image: tablesim-complete:20260914
    pull_policy: never
    command: ["sam3"]
    gpus: all
    init: true
    shm_size: 16gb
    environment:
      NVIDIA_DRIVER_CAPABILITIES: compute,utility,graphics,display
      SAM3_PRELOAD: "1"
    ports:
      - "127.0.0.1:18003:8003"
    volumes:
      - ./outputs:/outputs
    healthcheck:
      test: ["CMD", "/usr/local/bin/tablesim-entrypoint", "host-python", "-c", "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8003/health')); assert d['status']=='ok' and d['model_loaded']"]
      interval: 15s
      timeout: 5s
      start_period: 120s
      retries: 12
    restart: unless-stopped
YAML

sudo docker compose -f compose.server.yaml config --quiet
sudo docker compose -f compose.server.yaml up -d
sudo docker compose -f compose.server.yaml ps
sudo docker compose -f compose.server.yaml logs --tail=100 tablesim
```

后续一直显式使用 `-f compose.server.yaml`，避免误用项目开发环境的 Compose 文件。首次启动会加载权重，等待状态变成 `healthy`。

```bash
curl -fsS http://127.0.0.1:18003/health
```

响应应包含 `"status":"ok"`、`"model_loaded":true`，JSON 中可能有空格和其他字段。容器内部地址为 `http://127.0.0.1:8003`，服务器宿主机地址为 `http://127.0.0.1:18003`。SAM3 是接口服务，不提供项目网页界面。

不要把服务器上的空项目目录挂到 `/workspace/TableSim` 或原项目绝对路径，否则会遮住镜像内已经打包的代码和环境。

## 7. 单个场景完整推理

在目标服务器部署目录执行，SAM3 服务保持运行：

```bash
cd /data/tablesim-migration
batch_id="migration_$(date +%Y%m%d_%H%M%S)"
sudo docker compose -f compose.server.yaml exec -T tablesim \
  tablesim-entrypoint host-python -m high_level_planner.demo \
  --image image_collectors/scenes_constrained_v2/scene_003.png \
  --instruction 'Clean the desk.' \
  --output-dir "/outputs/planner_outputs/$batch_id/scene_003" \
  --sam3-endpoint http://127.0.0.1:8003 \
  --sam3-prompts 'notebook,pen,monitor,keyboard,cup,mouse,phone,folder,stapler' \
  --reasoning-backend rule_based --reasoning-scene-state json \
  --reasoning-max-depth 8 --reasoning-beam-width 3 --reasoning-num-branches 5

ls -lh "outputs/planner_outputs/$batch_id/scene_003"
cat "outputs/planner_outputs/$batch_id/scene_003/ucs_path_output/optimal_path.json"
```

这是与镜像打包时验证一致的规则推理配置，无需在线 LLM API。原机器断网测试返回 `status: solved`、`edge_count: 2`；目标机器应以实际输出为准。

结果位于**服务器磁盘**：

```text
/data/tablesim-migration/outputs/planner_outputs/<批次名>/scene_003/
├── scene_graph.json
├── reasoning_graph.json
├── sam3_input_crop.png
├── ucs_path_output/
│   └── optimal_path.json
└── vla_request.json
```

`optimal_path.json` 中的 `status` 表示是否找到路径。`vla_request.json` 是下游策略请求数据；生成它不等于已执行机械臂动作。

需要交互式检查时：

```bash
sudo docker compose -f compose.server.yaml exec tablesim bash
```

容器工作目录是 `/home/ubuntu/Desktop/Liutong/TableSim`，`/workspace/TableSim` 是它的别名。容器内调用宿主 Python 环境应使用 `tablesim-entrypoint host-python`，以设置正确的模块路径。

## 8. 批量推理 scene_000 到 scene_009

下面串行处理 10 张图片，每次创建新的批次子目录；单个场景执行失败会记录状态并继续其他场景。需要先完成第 6 节服务启动。

```bash
cd /data/tablesim-migration
sudo docker compose -f compose.server.yaml exec -T tablesim bash -s <<'BASH'
set -uo pipefail
batch_dir="/outputs/test_all_output/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$batch_dir"
printf 'scene\texit_code\n' > "$batch_dir/status.tsv"
failed=0
for i in $(seq 0 9); do
  scene=$(printf 'scene_%03d' "$i")
  mkdir -p "$batch_dir/$scene"
  echo "Running $scene -> $batch_dir/$scene"
  tablesim-entrypoint host-python -m high_level_planner.demo \
    --image "image_collectors/scenes_constrained_v2/$scene.png" \
    --instruction 'Clean the desk.' \
    --output-dir "$batch_dir/$scene" \
    --sam3-endpoint http://127.0.0.1:8003 \
    --sam3-prompts 'notebook,pen,monitor,keyboard,cup,mouse,phone,folder,stapler' \
    --reasoning-backend rule_based --reasoning-scene-state json \
    --reasoning-max-depth 8 --reasoning-beam-width 3 --reasoning-num-branches 5 \
    > "$batch_dir/$scene/run.log" 2>&1
  code=$?
  printf '%s\t%s\n' "$scene" "$code" >> "$batch_dir/status.tsv"
  if [ "$code" -ne 0 ]; then failed=1; fi
done
cat "$batch_dir/status.tsv"
echo "Container output: $batch_dir"
exit "$failed"
BASH
```

服务器结果路径为 `/data/tablesim-migration/outputs/test_all_output/batch_<时间>/`。批次时间由容器时区决定，以命令实际打印目录为准。

`status.tsv` 的 `exit_code=0` 表示程序正常完成；是否成功求解还要读取各场景 `ucs_path_output/optimal_path.json` 的 `status`。搜索深度、场景状态及目标约束可能导致无可达路径，不应把“不可达”直接视为环境部署失败。本节使用深度 8 做部署检查，需要更深搜索时可调整 `--reasoning-max-depth`，同时预留更多时间和内存。

## 9. 结果下载、历史数据导出和服务管理

新增结果已直接写入服务器 `outputs/`。在**你的电脑**下载：

```bash
rsync -ah --partial --info=progress2 \
  YOUR_USER@YOUR_SERVER_IP:/data/tablesim-migration/outputs/ \
  ./tablesim_results/
```

若要取出**镜像内历史结果**，在服务器执行：

```bash
cd /data/tablesim-migration
sudo docker compose -f compose.server.yaml cp \
  tablesim:/workspace/TableSim/planner_outputs ./historical_planner_outputs
sudo docker compose -f compose.server.yaml cp \
  tablesim:/workspace/TableSim/test_all_output ./historical_test_all_output
sudo chown -R "$(id -u):$(id -g)" historical_planner_outputs historical_test_all_output
```

服务管理：

```bash
sudo docker compose -f compose.server.yaml logs -f --tail=100 tablesim
sudo docker compose -f compose.server.yaml stop
sudo docker compose -f compose.server.yaml start
sudo docker compose -f compose.server.yaml restart
```

`stop/start` 保留容器内部改动。`down` 会删除容器；以后 `up` 从镜像重新创建。挂载的 `outputs/` 会保留，但容器内部其他位置的新增文件或代码修改必须先导出。镜像 tar 本身不会随推理运行而更新。

需要从本地访问远程 SAM3 接口时，可在**你的电脑**建立 SSH 隧道：

```bash
ssh -N -L 18003:127.0.0.1:18003 YOUR_USER@YOUR_SERVER_IP
```

保持该终端运行，另一终端访问 `http://127.0.0.1:18003/health`。默认服务仅绑定服务器回环地址。

## 10. 其他已打包环境的入口

以下命令在 `sudo docker compose -f compose.server.yaml exec tablesim bash` 进入容器后执行。

| 功能 | 入口或检查命令 |
| --- | --- |
| SAM3 / SAPIEN / cuRobo | `python`，对应 Python 3.10 环境 |
| 图推理 Python 3.12 | `tablesim-entrypoint host-python` |
| pi05 Python 3.11 | `tablesim-entrypoint pi05-python` |
| Isaac Sim Python | `tablesim-entrypoint isaacsim-python` |
| Conda | `source /home/ubuntu/miniconda3/etc/profile.d/conda.sh` 后 `conda activate RoboTwin` 等 |
| SAPIEN 场景初始化 | `python script/check_sim.py --task arrange_scene_000 --config arrange_scene_000 --setup` |
| MuJoCo | `/home/ubuntu/miniconda3/envs/mujoco_aloha/bin/python script/smoke_mujoco.py` |
| Isaac Sim 启动检查 | `tablesim-entrypoint isaacsim-python script/smoke_isaacsim.py` |
| 既有几何规划测试 | `tablesim-entrypoint host-python -m unittest high_level_planner.tests.test_relational_geometry_planning` |

pi05 已包含权重、原虚拟环境、补齐的 Python 3.11.15 和 PaliGemma tokenizer。运行时可在 `policy/pi05/` 使用 `uv run --no-sync ...`，或给现有脚本设置 `UV_NO_SYNC=1`，避免自动重建环境。完整训练、策略执行及所有任务没有逐项验证。

需要窗口显示时，还需针对服务器配置 X11 / Wayland / 远程显示；本指南使用无窗口推理和仿真检查。服务器若没有适合仿真的图形 GPU，可以先单独验收 SAM3 和图推理。

## 11. 常见问题

| 现象 | 检查与处理 |
| --- | --- |
| `sha256sum` 不匹配 | 确认文件名、大小和传输是否完整，续传或重传后重新校验 |
| `docker load` 长时间没有输出 | 大文件导入可能较慢；另开终端用 `df -h`、`sudo docker system df` 查看空间与状态 |
| `no space left on device` | 检查 Docker/containerd 所在磁盘和 inode，而不仅是 tar 所在目录；先解决空间问题再导入 |
| Compose 不认识 `gpus` | 更新 Compose 插件到 2.30.0 或更高版本；使用 `docker compose` 命令 |
| `could not select device driver` / 找不到 GPU | 先检查宿主机 `nvidia-smi`，再核对 Toolkit 配置并重启 Docker |
| 驱动不支持、`no kernel image` | 宿主驱动或 GPU 架构与镜像内 CUDA/PyTorch 不兼容；换兼容设备或单独适配环境 |
| GPU OOM | 检查 `nvidia-smi`，避免多份 SAM3 抢占显存，串行推理；不同显卡容量需实测 |
| `unhealthy` / SAM3 连接失败 | 查看服务日志和权重加载状态；容器内端口是 8003，宿主机是 18003 |
| 18003 被占用 | 在 `compose.server.yaml` 中只改宿主端口，例如 `127.0.0.1:18004:8003`；健康检查和容器内推理仍用 8003 |
| `/outputs` 无写权限 | 镜像 UID/GID 为 1000:1000，检查新建输出目录的属主和挂载权限，不需要改整个项目权限 |
| 推理无解但退出码为 0 | 查看最优路径 JSON 的 `status`、目标和搜索配置；程序完成与求解成功是两件事 |
| Python 包路径错误 | 使用本指南指定入口；不要随意替换镜像内 Python、移动虚拟环境或挂载空目录遮住项目 |
| 在服务容器内再次运行 `planner` 报端口占用 | `planner` 会再启动一份 SAM3；常驻服务已运行时用第 7 节 `host-python -m ...` 调用 |

## 12. 打包范围与已验证结果

包含 Ubuntu 24.04 系统用户空间、项目快照、资产/视频/数据/历史推理输出、SAM3 权重、CUDA 12.1 运行栈、SAPIEN、cuRobo、Isaac Sim 5.1.0（含扩展缓存）、ROS、pi05，以及 Miniconda base、RoboTwin、mujoco_aloha、depthanything3、labelme、xanylabel 环境。

访问令牌、SSH 密钥和原项目 `.env` 未打包。本文规则推理无需这些凭据；在线 LLM 推理需按镜像内版本的接口配置另行传入，不要把凭据写入 GitHub 文档或提交到仓库。

原机器完成的验证：

- 镜像 tar 导出后实际 `docker load` 成功，恢复的镜像 ID 与导出前一致。
- 13 项环境检查全部通过，包括实际 CUDA 计算、pi05 GPU/tokenizer、6 个 Conda 解释器、SAPIEN 场景初始化、MuJoCo 渲染、Isaac Sim 离线启动。
- 18 个已有几何规划测试通过，Python 3.10 开发头文件和链接检查通过。
- 无网络、无原项目目录挂载条件下，SAM3 → Scene Graph → Reasoning Graph → UCS 完整执行通过，scene_003 找到 2 步路径。

验证清单在镜像 `/opt/migration/manifest.json`，详细记录在 `/opt/migration/validation/`，原交付目录也有 `PACKAGE_INFO.json`、`validation/` 和日志。查看清单：

```bash
sudo docker compose -f compose.server.yaml exec -T tablesim \
  tablesim-entrypoint host-python -m json.tool /opt/migration/manifest.json
```

保留的源项目问题：`envs/scene_003.py` 已有语法错误，因此仿真检查使用 `arrange_scene_000`；它不影响 `scene_003.png` 的图推理。旧 README 的 `stack_plate` 任务已不存在。可选的 `depthanything3` 环境引用了原机器上已缺失的外部源码，本次保留环境文件，未恢复不存在的源码。

完成目标服务器验收的标志是：文件校验通过、GPU 容器真实计算通过、服务健康、scene_003 推理生成完整 JSON，并能在服务器 `outputs/` 目录直接看到结果。
