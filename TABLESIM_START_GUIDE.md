# TableSim Docker 环境启动与数据采集

## 0. 完整镜像部署与运行命令（2026-09-15 更新）

本节从新服务器安装环境开始，依次完成镜像导入、SAM3 启动、Scene Graph → Reasoning Graph → UCS 推理、仿真检查、数据采集和结果下载。除注明“原机器”或“本地电脑”的步骤外，命令均在**目标服务器的 Bash 终端**执行；部署目录统一使用 `/data/tablesim-migration`。

使用已经打包的 **`tablesim-complete:20260914`**：包含 Ubuntu 24.04 用户空间、项目代码和资产、SAM3 / pi05 权重及 Python、CUDA、SAPIEN、cuRobo、Isaac Sim 等运行环境。代码是 `be918c1` 时的快照（含打包时本地修改），不会自动同步 GitHub 后续提交。这里的规则推理命令无需在线 API 凭据；新版本代码的 API 默认设置不适用于此快照。

| 项目 | 值 |
| --- | --- |
| 原机器镜像文件 | `/home/ubuntu/Desktop/Liutong/TableSim_migration_20260914/tablesim-complete-20260914.tar` |
| 文件大小 | 89,829,225,472 字节，约 83.66 GiB |
| 目标平台 | Linux x86_64；以下安装命令适用于 Ubuntu 22.04 / 24.04 |
| GPU | NVIDIA GPU；原机器使用 RTX 4090 验证，仿真还需要图形 / Vulkan 能力 |
| 磁盘预留 | 建议至少 400 GB 可用空间，包含 tar、Docker 存储及新输出 |
| 宿主机端口 | `127.0.0.1:18003`；容器内 SAM3 端口为 `8003` |
| 新结果根目录 | 宿主机 `/data/tablesim-migration/outputs` ↔ 容器 `/outputs` |

宿主机仍需安装自己的 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit；镜像不含宿主内核。完整迁移说明和环境验证记录见 [DOCKER_SERVER_DEPLOYMENT.md](DOCKER_SERVER_DEPLOYMENT.md)。

### 0.1 安装宿主机环境

先检查；已有可用组件可跳过对应安装步骤：

```bash
uname -m
cat /etc/os-release
nvidia-smi
docker --version
docker compose version
df -h
free -h
```

**仅在尚未安装 NVIDIA 驱动时**执行以下命令，安装完成后安排重启，再确认 `nvidia-smi` 正常。驱动选择参考 [Ubuntu 官方说明](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/)。

```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers list --gpgpu
sudo ubuntu-drivers install --gpgpu
# 安装完成并确认可以重启后执行：sudo reboot
```

**仅在尚未安装 Docker 时**执行。已有其他 Docker / containerd 安装的服务器先按 [Docker 官方安装说明](https://docs.docker.com/engine/install/ubuntu/) 处理包冲突。

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

安装并配置 GPU 容器支持，命令参考 [NVIDIA 官方安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。最后一步会重启 Docker，已有业务容器时应安排维护时段。

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg rsync
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

完全离线服务器需要提前准备上述系统软件包或内部软件源。后续 Compose 的 `gpus: all` 要求 [Compose 2.30.0 或更新版本](https://docs.docker.com/reference/compose-file/services/#gpus)。导入前确认 Docker 存储所在磁盘有足够空间：

```bash
sudo docker info --format '{{.DockerRootDir}}'
df -h /data /var/lib/docker /var/lib/containerd 2>/dev/null
```

若这些目录尚不存在或 Docker 使用自定义路径，检查其实际所在挂载点。

### 0.2 传输、校验和导入镜像

在**目标服务器**创建接收目录：

```bash
sudo mkdir -p /data/tablesim-migration
sudo chown "$(id -u):$(id -g)" /data/tablesim-migration
```

在**原机器**执行，替换服务器用户名和 IP；中断后可执行同一命令续传：

```bash
rsync -ah --partial --append-verify --info=progress2 \
  /home/ubuntu/Desktop/Liutong/TableSim_migration_20260914/tablesim-complete-20260914.tar \
  YOUR_USER@YOUR_SERVER_IP:/data/tablesim-migration/
```

回到**目标服务器**校验并导入；仅校验成功才导入。已经导入过同一镜像时可以跳过这一块：

```bash
cd /data/tablesim-migration
printf '%s  %s\n' \
  '6e85edad66a12fe15cb93de72845ee3e29e29ddd3d7050f7ae2d16ffb67b5c95' \
  'tablesim-complete-20260914.tar' > SHA256SUMS
sha256sum -c SHA256SUMS && \
  sudo docker load -i tablesim-complete-20260914.tar
sudo docker image inspect tablesim-complete:20260914 \
  --format '{{.Id}} {{.Os}}/{{.Architecture}}'
```

应显示校验 `OK`、`Loaded image: tablesim-complete:20260914` 和平台 `linux/amd64`。使用 `docker load` 恢复镜像，无需构建或从镜像仓库拉取。

检查容器 GPU 和实际 CUDA 计算：

```bash
sudo docker run --rm --network none --gpus all \
  tablesim-complete:20260914 nvidia-smi
sudo docker run --rm --network none --gpus all \
  tablesim-complete:20260914 python -c \
  'import torch; assert torch.cuda.is_available(); x=torch.arange(8, device="cuda"); print(torch.__version__, torch.cuda.get_device_name(0), x.sum().item())'
```

第二条应输出 GPU 名称和求和结果 `28`。

### 0.3 创建 Compose 配置并启动 SAM3

镜像以 UID/GID `1000:1000` 运行，输出目录需要可写。下面创建独立部署文件，后续命令均显式指定它：

```bash
cd /data/tablesim-migration
sudo install -d -o 1000 -g 1000 -m 0775 outputs outputs/data
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
      TZ: Asia/Shanghai
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
sudo docker compose -f compose.server.yaml up -d --wait --wait-timeout 360
sudo docker compose -f compose.server.yaml ps
sudo docker compose -f compose.server.yaml logs --tail=100 tablesim
curl -fsS http://127.0.0.1:18003/health
```

健康响应应包含 `"status":"ok"` 和 `"model_loaded":true`。超时后查看日志，确认服务健康再推理。18003 被占用时只修改端口映射左侧；容器内推理地址仍用 `http://127.0.0.1:8003`。

容器工作目录为 `/home/ubuntu/Desktop/Liutong/TableSim`，`/workspace/TableSim` 是别名。不要挂载空目录覆盖这些路径，否则会遮住打包的代码和环境。SAM3 提供 HTTP 接口，没有网页界面。

### 0.4 运行 scene_003 完整推理并生成图像

在同一终端执行下面整块命令，每次创建一个批次目录。显式选择规则推理，使用镜像打包时的部署验证参数：

```bash
cd /data/tablesim-migration
batch_id="migration_$(date +%Y%m%d_%H%M%S)"
scene_output="/outputs/planner_outputs/$batch_id/scene_003"
sudo docker compose -f compose.server.yaml exec -T tablesim \
  tablesim-entrypoint host-python -m high_level_planner.demo \
  --image image_collectors/scenes_constrained_v2/scene_003.png \
  --instruction 'Clean the desk.' \
  --output-dir "$scene_output" \
  --sam3-endpoint http://127.0.0.1:8003 \
  --sam3-prompts 'notebook,pen,monitor,keyboard,cup,mouse,phone,folder,stapler' \
  --reasoning-backend rule_based --reasoning-scene-state json \
  --reasoning-max-depth 8 --reasoning-beam-width 3 --reasoning-num-branches 5 && \
sudo docker compose -f compose.server.yaml exec -T tablesim \
  tablesim-entrypoint host-python -m script.visualize_planner_graphs \
  --input-dir "$scene_output" --output-dir "$scene_output"

ls -lh "outputs/planner_outputs/$batch_id/scene_003"
cat "outputs/planner_outputs/$batch_id/scene_003/ucs_path_output/optimal_path.json"
```

服务器实际结果目录：`/data/tablesim-migration/outputs/planner_outputs/<批次>/scene_003/`。主要文件包括 `scene_graph.json`、`reasoning_graph.json`、`sam3_input_crop.png`、`ucs_path_output/optimal_path.json`、`vla_request.json` 和可视化 PNG / SVG。

`optimal_path.json` 中 `status: solved` 表示找到了路径；规则搜索完成也可能返回不可达。深度 8 用于部署检查，更深搜索可调整 `--reasoning-max-depth`。`vla_request.json` 是策略请求，生成它不代表已经执行机械臂动作。

### 0.5 批量推理 scene_000 到 scene_009

每次在 `test_all_output` 下新建批次子目录，保存每个场景的 JSON、可视化和日志；单个失败后继续其余场景：

```bash
cd /data/tablesim-migration
sudo docker compose -f compose.server.yaml exec -T tablesim bash -s <<'BASH'
set -uo pipefail
batch_dir="/outputs/test_all_output/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$batch_dir"
printf 'scene\tinference_exit\tvisualization_exit\n' > "$batch_dir/status.tsv"
failed=0
for i in $(seq 0 9); do
  scene=$(printf 'scene_%03d' "$i")
  out="$batch_dir/$scene"
  mkdir -p "$out"
  echo "Running $scene -> $out"
  tablesim-entrypoint host-python -m high_level_planner.demo \
    --image "image_collectors/scenes_constrained_v2/$scene.png" \
    --instruction 'Clean the desk.' --output-dir "$out" \
    --sam3-endpoint http://127.0.0.1:8003 \
    --sam3-prompts 'notebook,pen,monitor,keyboard,cup,mouse,phone,folder,stapler' \
    --reasoning-backend rule_based --reasoning-scene-state json \
    --reasoning-max-depth 8 --reasoning-beam-width 3 --reasoning-num-branches 5 \
    > "$out/run.log" 2>&1
  inference_code=$?
  visualization_code=skipped
  if [ "$inference_code" -eq 0 ]; then
    tablesim-entrypoint host-python -m script.visualize_planner_graphs \
      --input-dir "$out" --output-dir "$out" > "$out/visualization.log" 2>&1
    visualization_code=$?
    if [ "$visualization_code" -ne 0 ]; then failed=1; fi
  else
    failed=1
  fi
  printf '%s\t%s\t%s\n' "$scene" "$inference_code" "$visualization_code" \
    >> "$batch_dir/status.tsv"
done
cat "$batch_dir/status.tsv"
echo "Container output: $batch_dir"
exit "$failed"
BASH
```

服务器结果在 `/data/tablesim-migration/outputs/test_all_output/batch_<时间>/`。`status.tsv` 的退出码表示命令是否正常完成；是否求解成功仍需查看每个 `optimal_path.json` 的 `status`。

### 0.6 SAPIEN 仿真检查与数据采集

当前快照存在的示例任务是 `arrange_scene_000`。先检查完整场景初始化：

```bash
cd /data/tablesim-migration
sudo docker compose -f compose.server.yaml exec -T tablesim \
  python script/check_sim.py \
  --task arrange_scene_000 --config arrange_scene_000 --setup
```

输出 `Full simulation setup OK: arrange_scene_000 actors=...` 表示场景初始化成功。图推理只读取 `scene_003.png`；镜像中 `envs/scene_003.py` 的已有语法错误不影响图片推理。

需要采集数据时，以下命令启动一次性容器，将仿真数据目录单独映射到宿主机 `outputs/data`。原配置为无窗口运行、采集 2 个成功 episode；规划失败时可能尝试多个种子，场景初始化通过不保证所有采集动作成功。

```bash
cd /data/tablesim-migration
sudo docker compose -f compose.server.yaml run --rm --no-deps \
  -v /data/tablesim-migration/outputs/data:/home/ubuntu/Desktop/Liutong/TableSim/data \
  tablesim bash collect_data.sh arrange_scene_000 arrange_scene_000 0
```

采集结果在 `/data/tablesim-migration/outputs/data/arrange_scene_000/arrange_scene_000/`，包括 `seed.txt`、`_traj_data/` 和 `data/*.hdf5`。再次运行会复用已有数据进行断点续跑。显存紧张时可先执行下一节的 `stop` 暂停 SAM3，采集完成后再 `start`。

其他环境入口（按需执行）：

```bash
cd /data/tablesim-migration
# 进入常驻容器
sudo docker compose -f compose.server.yaml exec tablesim bash
# 以下在容器内执行
tablesim-entrypoint host-python --version
tablesim-entrypoint pi05-python --version
tablesim-entrypoint isaacsim-python script/smoke_isaacsim.py
/home/ubuntu/miniconda3/envs/mujoco_aloha/bin/python script/smoke_mujoco.py
exit
```

### 0.7 日志、停止恢复、结果下载

以下管理命令按需单独执行：

```bash
cd /data/tablesim-migration
sudo docker compose -f compose.server.yaml ps
sudo docker compose -f compose.server.yaml logs -f --tail=100 tablesim
# Ctrl+C 退出日志跟踪，服务继续运行
sudo docker compose -f compose.server.yaml stop
sudo docker compose -f compose.server.yaml start
sudo docker compose -f compose.server.yaml restart
```

`stop/start` 保留容器内部改动；如需删除部署容器，可执行 `sudo docker compose -f compose.server.yaml down`。宿主机 `outputs/` 会保留，容器内部其他位置的新增文件需先导出。镜像 tar 不会随运行结果更新。

在**本地电脑**下载新增结果：

```bash
rsync -ah --partial --info=progress2 \
  YOUR_USER@YOUR_SERVER_IP:/data/tablesim-migration/outputs/ \
  ./tablesim_results/
```

需要导出镜像中的历史推理结果时，在**目标服务器**执行：

```bash
cd /data/tablesim-migration
sudo docker compose -f compose.server.yaml cp \
  tablesim:/workspace/TableSim/planner_outputs ./historical_planner_outputs
sudo docker compose -f compose.server.yaml cp \
  tablesim:/workspace/TableSim/test_all_output ./historical_test_all_output
sudo chown -R "$(id -u):$(id -g)" historical_planner_outputs historical_test_all_output
```

部署验收应确认：文件校验成功、容器 CUDA 计算通过、SAM3 健康、推理生成 JSON 和图像、宿主机输出目录可直接看到结果。目标服务器的 GPU、驱动与磁盘情况以实际运行检查为准。

---

以下保留原有开发环境与数据采集说明。原文中的 `stack_plate` 是历史任务示例，当前仓库和完整镜像已不包含该任务；直接部署运行请使用上面的第 0 节。

原文对应 `stack_plate` 任务，使用 Docker Compose 服务
`tablesim`、镜像 `tablesim:cuda12.1` 和 0 号 GPU。项目在容器中的路径为
`/workspace/TableSim`。

## 1. 文件与参数对应关系

- 任务实现：`envs/stack_plate.py`
- 任务配置：`task_config/stack_plate.yml`
- 启动脚本：`collect_data.sh`
- 采集入口：`script/collect_data.py`
- 默认输出：`data/stack_plate/stack_plate/`

采集命令的三个参数依次为：

```text
bash collect_data.sh <任务名> <配置名> <GPU 编号>
```

因此当前任务使用：

```bash
bash collect_data.sh stack_plate stack_plate 0
```

注意：任务名和配置名均为 `stack_plate`。

## 2. 启动前检查

在宿主机执行：

```bash
nvidia-smi
docker --version
docker compose version
```

`nvidia-smi` 应能看到目标 GPU。Docker 还需要安装 NVIDIA Container
Toolkit，使容器可以使用 `--gpus all`。

进入项目目录：

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim
```

如果需要打开 SAPIEN 可视化窗口，先确认宿主机存在 `DISPLAY`：

```bash
echo "$DISPLAY"
xhost +local:docker
```

纯后台采集时，`task_config/stack_plate.yml` 中的 `render_freq: 0` 不会打开
交互式 Viewer，不需要操作窗口。

## 3. 启动 TableSim 容器

TableSim 镜像复用本机已经验证过的 `robotwin:cuda12.1` 依赖层，但会生成独立
的 `tablesim:cuda12.1` 镜像，并将用户、HOME、缓存和工作目录全部切换为
TableSim。先确认基础镜像存在：

```bash
docker image inspect robotwin:cuda12.1 >/dev/null
docker compose build tablesim
```

该派生构建不会重复下载 PyTorch、SAPIEN、mplib、cuRobo 或 PyTorch3D。若基础
镜像使用了其他标签，可先设置
`TABLESIM_BASE_IMAGE=<镜像名:标签>` 再构建。

已有镜像时直接启动：

```bash
docker compose up -d tablesim
```

如果镜像不存在或 Dockerfile 已修改，则构建并启动：

```bash
docker compose up -d --build tablesim
```

检查容器状态和 GPU：

```bash
docker compose ps
docker compose exec tablesim nvidia-smi
```

检查 Python、SAPIEN、PyTorch、Warp 和兼容入口：

```bash
docker compose exec tablesim bash -lc '
cd /workspace/TableSim
python - <<"PY"
import sapien
import torch
import warp

print("SAPIEN:", sapien.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Warp:", warp.__version__)
print("warp.torch available:", hasattr(warp, "torch"))
PY
'
```

当前 `compose.yaml` 会把 `compat/` 加入 `PYTHONPATH`。其中的
`sitecustomize.py` 为镜像内的 cuRobo 提供 Warp 1.15 兼容命名空间，因此不应
再依赖容器 `/tmp` 中的临时补丁。

## 4. 检查 SAPIEN 仿真环境

运行统一检查入口：

```bash
docker compose exec tablesim python script/check_sim.py
```

它会检查 GPU、PyTorch、SAPIEN、Warp、renderer、核心基类、配置、机器人路径
以及任务动态导入。输出 `Render Well` 和 `Task pipeline OK` 表示基础环境正常。
某些驱动组合会先打印 `Failed to find Vulkan ICD file` 警告；只要随后出现
`Render Well`，说明 SAPIEN 已成功使用可用的 ICD，可忽略该警告。

进一步初始化机器人、相机、任务物体和运动规划器：

```bash
docker compose exec tablesim \
  python script/check_sim.py --task stack_plate --config stack_plate --setup
```

输出 `Full simulation setup OK: stack_plate actors=...` 表示完整场景初始化成功。
还可执行 `docker compose exec tablesim python script/collect_data.py --help`，它会
检查 renderer 并显示 `task_name task_config` 用法，不会创建数据。

## 5. 新建任务脚本

新任务遵循“文件名 = 类名 = 采集命令中的任务名”的约定。例如新建 `envs/pick_book.py` 时，文件中必须定义：

```python
from ._base_task import Base_Task
from .utils import *


class pick_book(Base_Task):
    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        ...

    def play_once(self):
        ...

    def check_success(self):
        return ...
```

复制 `task_config/_config_template.yml` 为 `task_config/pick_book.yml` 后按任务修改。无需编辑 `envs/__init__.py`；采集器会动态导入 `envs.pick_book`，并检查其中是否存在 `pick_book` 类。

```bash
docker compose exec tablesim python -c "from script.collect_data import class_decorator; class_decorator('pick_book'); print('task import OK')"
bash collect_data.sh pick_book pick_book 0
```

## 6. 配置采集规模与数据内容

编辑 `task_config/stack_plate.yml`。常用字段如下：

- `episode_num`：成功样本数量，当前为 50。
- `use_seed`：`false` 表示先规划并筛选成功种子；`true` 表示直接使用已有
  `seed.txt`。
- `collect_data`：`true` 表示种子规划结束后继续回放轨迹并写 HDF5。
- `render_freq`：`0` 为后台运行；非零值用于调试 Viewer。
- `save_freq`：图像/状态保存频率。
- `camera.collect_head_camera`：是否采集顶部相机。
- `camera.collect_wrist_camera`：是否采集腕部相机。
- `data_type.rgb`、`depth`、`pointcloud`、`endpose`、`qpos`：控制各类数据。
- `save_path`：输出根目录，当前为 `./data`。

为了快速调试，可复制一份配置并将 `episode_num` 改成 1、
`collect_data` 改成 `false`。不要直接用小规模配置覆盖正式配置。

## 7. 处理旧采集结果

采集器支持断点续跑：如果输出目录已有 `seed.txt`，它会从现有种子数量继续。
但是修改任务动作代码后，旧 `_traj_data/*.pkl` 与新工作流不兼容。此时应先保留
旧目录，再开始一次干净采集：

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim
mv data/stack_plate/stack_plate \
  data/stack_plate/stack_plate_backup_$(date +%Y%m%d_%H%M%S)
```

如果目录不存在，可忽略该步骤。确认旧数据不再需要时，也可以自行删除备份。

## 8. 开始正式数据采集

推荐从宿主机一条命令启动。脚本会自动创建一次性 TableSim Compose 容器：

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim
bash collect_data.sh stack_plate stack_plate 0
```

若要复用已启动的常驻容器：

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim
docker compose exec tablesim bash -lc '
cd /workspace/TableSim
bash collect_data.sh stack_plate stack_plate 0
'
```

也可以先进入容器再运行：

```bash
docker compose exec tablesim bash
cd /workspace/TableSim
bash collect_data.sh stack_plate stack_plate 0
```

程序分为两个阶段：

1. `Start Seed and Pre Motion Data Collection`：逐个尝试随机种子，只保存规划
   和物理成功的轨迹。
2. `Start Data Collection`：按成功种子回放轨迹，采集相机与机器人状态并写入
   HDF5。

单个随机种子失败是正常现象，程序会继续尝试，直到成功数量达到
`episode_num`。看到以下形式的日志才表示该样本通过：

```text
simulate data episode N success! (seed = S)
```

当前 `stack_plate` 的成功条件包括：移动盘准确叠到左侧盘堆、碗底位于
桌面高度附近，并且碗的平面投影不与任一盘子重叠。碗不要求落在固定 XY
坐标。

## 9. 输出目录说明

默认目录结构为：

```text
data/stack_plate/stack_plate/
├── seed.txt
├── _traj_data/
│   ├── episode0.pkl
│   └── ...
├── data/
│   ├── episode0.hdf5
│   └── ...
└── scene_info.json
```

- `seed.txt`：已通过的仿真随机种子。
- `_traj_data/episodeN.pkl`：规划阶段生成的左右臂关节轨迹。
- `data/episodeN.hdf5`：正式回放阶段采集的数据。
- `scene_info.json`：每个 episode 的对象与任务信息。

查看进度：

```bash
wc -w data/stack_plate/stack_plate/seed.txt
find data/stack_plate/stack_plate/data -name 'episode*.hdf5' | wc -l
```

## 10. 停止、恢复与日志排查

前台运行时按 `Ctrl+C` 可停止。只要保留输出目录，再次执行相同命令就会根据
`seed.txt` 和已有 HDF5 继续。

检查是否仍有采集进程：

```bash
docker compose exec tablesim bash -lc \
  'pgrep -af "collect_data.py|collect_data.sh" || true'
```

查看容器日志：

```bash
docker compose logs --tail=200 tablesim
```

停止容器但保留数据和缓存：

```bash
docker compose stop tablesim
```

重新启动：

```bash
docker compose start tablesim
```

如果出现 cuRobo 找不到 URDF 或碰撞球配置，检查以下文件中的路径是否为
`/workspace/TableSim/...`：

```text
assets/embodiments/aloha-agilex/curobo_left.yml
assets/embodiments/aloha-agilex/curobo_right.yml
```

正常情况下不需要手工修改：`collect_data.py` 会在每次运行前根据当前项目根
目录，从 `curobo*_tmp.yml` 自动重新生成这些配置。

如果修改了任务动作流程，应重新生成种子和 `_traj_data`，不要用新代码回放旧
轨迹。
