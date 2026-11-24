# TableSim 在 QZ 平台的启动与功能测试

本文适用于 QZ 平台上的原生环境部署，不依赖 Docker。项目根目录中已经存在以下三个环境：

```text
.venv-sim             Python 3.10，SAPIEN / cuRobo / PyTorch3D / TableSim
.venv-sam3            Python 3.10，SAM3 感知服务
policy/pi05/.venv     Python 3.11，OpenPI / pi0.5
```

本文所有命令都从 TableSim 仓库根目录执行。每打开一个新终端，都先设置公共变量：

```bash
cd /inspire/hdd/global_user/liutong-CZXS25230163/TableSim

export TABLESIM_ROOT="$PWD"
export PYTHONPATH="$TABLESIM_ROOT"
export TABLESIM_SAM3_ENDPOINT="http://127.0.0.1:8003"
```

`compat/` 中的 `sitecustomize.py` 会导入仿真环境专用的 Warp。不要把
`$TABLESIM_ROOT/compat` 全局加入 `PYTHONPATH`；只在运行 SAPIEN/cuRobo 的
终端中按下文设置，否则 `.venv-tools`、SAM3 和 pi0.5 环境会因找不到 `warp`
而在 Python 启动时报警。

如果仓库位于其他路径，只需修改第一条 `cd`。不要把 API Key 或密码写入本文、Git 配置或命令行参数。

## 1. 双卡分配

推荐固定如下：

| 物理 GPU | 进程 | 原因 |
| --- | --- | --- |
| GPU 0 | pi0.5 模型服务 | pi0.5 长时间占用模型显存，独占一张卡便于控制 JAX 显存 |
| GPU 1 | SAM3、SAPIEN、cuRobo | SAM3 已验证峰值约 5.1 GiB，可与单套仿真共享一张大显存卡 |

`CUDA_VISIBLE_DEVICES` 会重新编号进程内部可见设备。例如设置 `CUDA_VISIBLE_DEVICES=1` 后，程序内部的 `cuda:0` 实际对应物理 GPU 1。

查看两张卡：

```bash
nvidia-smi
```

持续观察显存、利用率和进程：

```bash
watch -n 1 nvidia-smi
```

## 2. 环境快速验收

### 2.1 仿真环境

```bash
source "$TABLESIM_ROOT/.venv-sim/bin/activate"
export PYTHONPATH="$TABLESIM_ROOT/compat:$TABLESIM_ROOT"

python - <<'PY'
import torch
import sapien
import mplib
import warp
import pytorch3d
import curobo

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("SAPIEN:", sapien.__version__)
print("MPLib:", mplib.__version__)
print("Warp:", warp.__version__)
print("PyTorch3D: OK")
print("cuRobo: OK")
PY

deactivate
```

在 GPU 实例中，`CUDA available` 应为 `True`。CPU 实例中为 `False` 属于正常结果。

### 2.2 SAM3 环境

```bash
source "$TABLESIM_ROOT/.venv-sam3/bin/activate"

python - <<'PY'
import torch
import sam3

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("SAM3 import: OK")
PY

test -s "$TABLESIM_ROOT/.cache/sam3/sam3.pt"
sha256sum "$TABLESIM_ROOT/.cache/sam3/sam3.pt"

deactivate
```

权重的预期 SHA-256 为：

```text
9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

### 2.3 pi0.5 环境

```bash
cd "$TABLESIM_ROOT/policy/pi05"

UV_CACHE_DIR="$TABLESIM_ROOT/.cache/uv" \
  .venv/bin/python - <<'PY'
import jax
import torch
import openpi

print("JAX:", jax.__version__)
print("JAX devices:", jax.devices())
print("PyTorch:", torch.__version__)
print("OpenPI import: OK")
PY

cd "$TABLESIM_ROOT"
```

pi0.5 的真实推理还需要训练后的 checkpoint。项目自定义执行入口要求 checkpoint 目录结构如下：

```text
policy/pi05/checkpoints/
└── <TRAIN_CONFIG>/
    └── <MODEL_NAME>/
        └── <CHECKPOINT_ID>/
            ├── assets/
            └── params/
```

检查已有 checkpoint：

```bash
find policy/pi05/checkpoints -maxdepth 5 -type d 2>/dev/null | sort
```

## 3. 单独测试 TableSim 仿真模块

仿真模块提供以下功能：

- 创建 SAPIEN 渲染器与物理场景；
- 加载双臂机器人、相机、任务物体和桌面资产；
- 初始化 cuRobo / MPLib 运动规划器；
- 执行专家轨迹并采集 RGB、关节状态、末端位姿和 HDF5 数据。

### 3.1 基础渲染和任务导入测试

此测试验证 Python/CUDA、SAPIEN 无窗口渲染以及任务配置能否导入，但不创建完整机器人场景：

```bash
source "$TABLESIM_ROOT/.venv-sim/bin/activate"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$TABLESIM_ROOT/compat:$TABLESIM_ROOT"
export SAPIEN_NO_DISPLAY=1
export SAPIEN_RENDERER=opengl

python script/check_sim.py \
  --task stack_blocks_two \
  --config stack_plate
```

通过标志包括：

```text
CUDA available: True
Render Well
Task pipeline OK
```

### 3.2 完整场景初始化测试

此测试会加载机器人、相机、物体和运动规划器，但不保存采集数据：

```bash
python script/check_sim.py \
  --task stack_blocks_two \
  --config stack_plate \
  --setup
```

通过标志为：

```text
Full simulation setup OK
```

### 3.3 单独运行专家数据采集

此功能执行任务自带的专家策略，生成轨迹数据，可用于检查物理任务、规划器、相机和数据保存流程：

```bash
export CUDA_VISIBLE_DEVICES=1
bash collect_data.sh stack_plate stack_plate 1
```

在当前 QZ 容器中，`collect_data.sh` 会直接运行 Python；第三个参数 `1` 表示物理 GPU 1。输出默认写入 `data/`。采集数量、保存频率、相机和随机化参数由 `task_config/stack_plate.yml` 控制。

停止前台任务使用 `Ctrl+C`。

## 4. 单独启动和测试 SAM3 模块

SAM3 模块提供以下功能：

- 根据文本类别在单张 RGB 图片中检测并分割物体；
- 返回标签、置信度、边界框、中心点和压缩 mask；
- 为 Scene Graph 构建提供对象级感知结果；
- 提供视频 mask memory 的创建、传播、重定位和关闭接口。

### 4.1 启动 SAM3 服务

打开一个独立终端，从项目根目录执行：

```bash
export TABLESIM_ROOT="$PWD"
export PYTHONPATH="$TABLESIM_ROOT"
export CUDA_VISIBLE_DEVICES=1
export SAM3_DEVICE=cuda
export SAM3_CHECKPOINT="$TABLESIM_ROOT/.cache/sam3/sam3.pt"
export SAM3_PRELOAD=1
export SAM3_MAX_PROMPTS=64

exec "$TABLESIM_ROOT/.venv-sam3/bin/python" \
  -m high_level_planner.perception.sam3_server \
  --host 127.0.0.1 \
  --port 8003 \
  --preload
```

`--preload` 会在服务监听前加载权重，因此启动可能需要一段时间。终端出现以下内容表示已监听：

```text
[sam3-server] listening on 127.0.0.1:8003
```

### 4.2 健康检查

在另一个终端执行：

```bash
curl -fsS http://127.0.0.1:8003/health | python -m json.tool
```

应看到：

```json
{
  "status": "ok",
  "backend": "sam3",
  "model_loaded": true,
  "device": "cuda"
}
```

健康检查只验证服务和模型已加载，不验证检测质量。

### 4.3 对一张真实图片进行检测

设置图片路径并运行客户端：

```bash
export TEST_IMAGE="/绝对路径/桌面图片.png"

source "$TABLESIM_ROOT/.venv-sim/bin/activate"
python - <<'PY'
import json
import os
from high_level_planner.perception import Sam3PerceptionClient

client = Sam3PerceptionClient(
    endpoint="http://127.0.0.1:8003",
    prompt_catalog=["book", "cup", "mouse", "keyboard", "phone", "pen"],
    prompt_catalog_path=None,
    max_prompts=6,
    confidence_threshold=0.35,
    timeout=300,
)
result = client.detect(os.environ["TEST_IMAGE"], instruction="organize the desk")
print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
PY
```

该测试会执行一次真实 GPU 推理。输出中的 `detections` 是后续 Scene Graph 使用的对象列表。

## 5. 单独测试高层规划模块

高层规划模块提供以下功能：

- 从初态图片调用 SAM3 构建初态 Scene Graph；
- 调用外部图像/视觉模型生成目标布局并读取目标 Scene Graph；
- 根据初态和目标状态生成 Reasoning Graph；
- 使用 UCS 搜索最低成本动作路径；
- 生成下一条 pi0.5 `vla_request.json`。

### 5.1 不调用 SAM3/API 的离线图规划测试

如果已经有一对初态和目标 Scene Graph，可以只测试 Reasoning Graph、UCS 和 VLA 请求生成：

```bash
source "$TABLESIM_ROOT/.venv-sim/bin/activate"

python -m script.generate_reasoning_graph \
  --scene-graph /绝对路径/initial_scene_graph.json \
  --target-scene-graph /绝对路径/terminal_scene_graph.json \
  --instruction "整理桌面" \
  --backend rule_based \
  --max-actions 8 \
  --beam-width 2 \
  --num-branches 2 \
  --output-dir "planner_outputs/offline_test_$(date +%Y%m%d_%H%M%S)"
```

该测试适合验证本地规划逻辑，不需要 GPU、SAM3 或 API Key。

### 5.2 API Key 的安全输入

完整图片规划需要图像模型和推理模型凭据。在运行命令的同一个终端输入：

```bash
read -rsp '请输入图像 API Key: ' TABLESIM_IMAGE_API_TOKEN
echo
export TABLESIM_IMAGE_API_TOKEN

read -rsp '请输入推理 API Key: ' TABLESIM_REASONING_API_TOKEN
echo
export TABLESIM_REASONING_API_TOKEN
```

如果两个服务使用同一个 Key，也应分别设置两个环境变量。不要把真实 Key 写入 `config/goal_generator.example.json`。

## 6. 单独启动和测试 pi0.5 模块

pi0.5 模块提供以下功能：

- 读取三路相机图像、14 维机器人状态和语言指令；
- 生成双臂动作序列；
- 按 `pi0_step` 分块输出动作；
- 可对重叠动作块执行 temporal ensemble；
- 通过 TCP 模型服务与仿真进程分离运行。

以下命令需要真实训练 checkpoint。先设置与磁盘目录一致的值：

```bash
export PI05_TRAIN_CONFIG="pi05_base_aloha_lora"
export PI05_MODEL_NAME="你的模型目录名"
export PI05_CHECKPOINT_ID="5000"

test -d \
  "$TABLESIM_ROOT/policy/pi05/checkpoints/$PI05_TRAIN_CONFIG/$PI05_MODEL_NAME/$PI05_CHECKPOINT_ID/assets"
```

如果 `test` 返回非零状态，说明 checkpoint 名称、步数或权重文件尚未准备好，不能执行真实 pi0.5 推理。

### 6.1 启动项目使用的 pi0.5 TCP 服务

打开一个独立终端：

```bash
cd /inspire/hdd/global_user/liutong-CZXS25230163/TableSim
export TABLESIM_ROOT="$PWD"

export PI05_TRAIN_CONFIG="pi05_base_aloha_lora"
export PI05_MODEL_NAME="你的模型目录名"
export PI05_CHECKPOINT_ID="5000"

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export PYTHONPATH="$TABLESIM_ROOT:$TABLESIM_ROOT/policy/pi05/src"

exec "$TABLESIM_ROOT/policy/pi05/.venv/bin/python" \
  script/policy_model_server.py \
  --port 9999 \
  --config policy/pi05/deploy_policy.yml \
  --overrides \
  --policy_name pi05 \
  --train_config_name "$PI05_TRAIN_CONFIG" \
  --model_name "$PI05_MODEL_NAME" \
  --checkpoint_id "$PI05_CHECKPOINT_ID" \
  --pi0_step 20 \
  --temporal_ensemble True \
  --temporal_ensemble_decay 0.5
```

服务完成模型加载后应输出：

```text
loading model success!
Model server started on localhost:9999
```

端口检查：

```bash
ss -ltn | grep ':9999'
```

### 6.2 使用仿真执行 pi0.5 评测

在另一个终端运行。仿真固定使用 GPU 1，模型服务固定使用 GPU 0：

```bash
cd /inspire/hdd/global_user/liutong-CZXS25230163/TableSim
export TABLESIM_ROOT="$PWD"

export PI05_TRAIN_CONFIG="pi05_base_aloha_lora"
export PI05_MODEL_NAME="你的模型目录名"
export PI05_CHECKPOINT_ID="5000"

source .venv-sim/bin/activate

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$TABLESIM_ROOT/compat:$TABLESIM_ROOT"
export SAPIEN_NO_DISPLAY=1
export SAPIEN_RENDERER=opengl

python script/eval_policy_client.py \
  --port 9999 \
  --config policy/pi05/deploy_policy.yml \
  --overrides \
  --policy_name pi05 \
  --task_name stack_plate \
  --task_config stack_plate \
  --train_config_name "$PI05_TRAIN_CONFIG" \
  --model_name "$PI05_MODEL_NAME" \
  --checkpoint_id "$PI05_CHECKPOINT_ID" \
  --ckpt_setting "$PI05_MODEL_NAME-$PI05_CHECKPOINT_ID" \
  --seed 0 \
  --instruction_type unseen \
  --pi0_step 20 \
  --temporal_ensemble True \
  --temporal_ensemble_decay 0.5
```

该入口会创建完整 SAPIEN 任务，持续向端口 `9999` 发送相机图像与机器人状态，接收动作块并在仿真中执行。评测视频和成功率写入：

```text
eval_result/stack_plate/pi05/...
```

当前脚本内部将评测目标数量固定为 100，属于完整评测，不是几秒钟的 smoke test。可以用 `Ctrl+C` 提前终止功能验证。

## 7. 完整静态规划 pipeline

该 pipeline 是当前仓库已经连通的图片规划链路：

```text
初态图片
  → SAM3 初态感知
  → 外部模型生成目标布局图
  → 读取目标 Scene Graph
  → Reasoning Graph
  → UCS
  → vla_request.json
```

运行前保持 SAM3 服务在 GPU 1 的独立终端中运行，并确认 `/health` 返回 `model_loaded: true`。

在新终端执行：

```bash
cd "$TABLESIM_ROOT"
source .venv-sim/bin/activate

export TABLESIM_SAM3_ENDPOINT="http://127.0.0.1:8003"
export INPUT_IMAGE="/绝对路径/桌面初态图片.png"
export RUN_DIR="planner_outputs/full_$(date +%Y%m%d_%H%M%S)"

read -rsp '请输入图像 API Key: ' TABLESIM_IMAGE_API_TOKEN
echo
export TABLESIM_IMAGE_API_TOKEN

read -rsp '请输入推理 API Key: ' TABLESIM_REASONING_API_TOKEN
echo
export TABLESIM_REASONING_API_TOKEN

python -m script.generate_reasoning_graph \
  --image "$INPUT_IMAGE" \
  --instruction "整理桌面" \
  --config config/goal_generator.example.json \
  --sam3-endpoint "$TABLESIM_SAM3_ENDPOINT" \
  --output-style layout \
  --backend api \
  --max-actions 64 \
  --beam-width 3 \
  --num-branches 5 \
  --output-dir "$RUN_DIR"
```

成功后检查：

```bash
python -m json.tool "$RUN_DIR/run_summary.json"
python -m json.tool "$RUN_DIR/vla_request.json"
ls -lah "$RUN_DIR"
```

主要输出：

| 文件 | 内容 |
| --- | --- |
| `initial_scene_graph.json` | SAM3 从初态图片构建的图 |
| `terminal_scene_graph.json` | 目标布局对应的图 |
| `reasoning_graph.json` | 候选动作及状态转移图 |
| `ucs_path_output/optimal_path.json` | UCS 选择的最低成本路径 |
| `vla_request.json` | 下一条动作转换后的 pi0.5 请求 |
| `run_summary.json` | 是否找到目标、路径长度、成本和模型执行摘要 |

`goal_reached: true` 表示规划器找到了通向目标图的动作路径；它不表示机器人已经在物理仿真中执行成功。

## 8. 三个模块同时运行

按以下顺序开启三个终端：

1. 终端 A：GPU 0 启动 `script/policy_model_server.py`，端口 `9999`。
2. 终端 B：GPU 1 启动 SAM3 服务，端口 `8003`。
3. 终端 C：运行完整静态规划，或使用 GPU 1 运行 `eval_policy_client.py`。

统一检查：

```bash
curl -fsS http://127.0.0.1:8003/health | python -m json.tool
ss -ltn | grep -E ':8003|:9999'
nvidia-smi
```

正常分配应表现为：

- GPU 0 上存在 pi0.5/JAX Python 进程；
- GPU 1 上存在 SAM3 Python 进程；
- 运行仿真时，GPU 1 上还会出现 SAPIEN/cuRobo Python 进程。

## 9. 当前端到端边界

当前代码会把高层规划结果写成 `vla_request.json`，也能独立运行 pi0.5 仿真评测，但 `script/eval_policy_client.py` 目前从任务生成器取得语言指令，没有读取 `vla_request.json`，也没有逐条消费 Reasoning Graph 的子任务。

因此，当前已经能够直接运行的是：

1. 图片到 VLA 请求的完整静态规划 pipeline；
2. 语言任务到 pi0.5 动作再到仿真结果的完整策略评测 pipeline。

“图片规划 → 逐条 pi0.5 执行 → 新观测 → Graph Memory 更新 → 下一子任务”的闭环仍需要一个执行协调器。不能仅仅同时启动三个进程就认为该闭环已经接通。

## 10. 常见问题

### `ModuleNotFoundError: pkg_resources`

SAPIEN 依赖旧接口，在仿真环境中固定 setuptools：

```bash
source .venv-sim/bin/activate
pip install "setuptools==80.9.0"
```

### `CUDA available: False`

先确认当前是 GPU 实例，再检查：

```bash
nvidia-smi
echo "$CUDA_VISIBLE_DEVICES"
```

### SAM3 `/health` 可用但 `model_loaded` 为 `false`

服务未使用 `--preload`，或权重尚未加载。调用一次加载接口：

```bash
curl -fsS -X POST \
  -H 'Content-Type: application/json' \
  -d '{}' \
  http://127.0.0.1:8003/v1/load | python -m json.tool
```

### pi0.5 找不到 `assets`

确认以下目录存在，并且其中包含训练时保存的归一化统计：

```bash
ls -lah \
  "policy/pi05/checkpoints/$PI05_TRAIN_CONFIG/$PI05_MODEL_NAME/$PI05_CHECKPOINT_ID/assets"
```

### GPU 显存不足

先确认进程没有分配到同一张卡：

```bash
nvidia-smi
```

pi0.5 的 JAX 显存比例可从 `0.80` 降为 `0.60`：

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
```

SAM3 和仿真共享 GPU 1 时，应只启动一个 SAM3 服务实例。

### 停止服务

前台服务使用 `Ctrl+C`。检查是否仍有残留进程：

```bash
pgrep -af 'sam3_server|policy_model_server|eval_policy_client|collect_data.py'
```
