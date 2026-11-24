# TableSim SAM3 Object-Level Perception 部署与使用指南

本文档说明如何在不修改 TableSim 主仿真环境的前提下，启动 SAM3 object-level
perception，并用它生成 Scene Graph。

SAM3 官方仓库：

- [https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- ModelScope 权重镜像：[https://www.modelscope.cn/models/facebook/sam3](https://www.modelscope.cn/models/facebook/sam3)
- Hugging Face 原始权重：[https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)

## 1. 当前接入架构

SAM3 没有直接安装进 `tablesim:cuda12.1` 的 Python 环境，而是作为同一 Docker
Compose 项目中的可选 sidecar：

```text
tablesim container
  HighLevelPlanner
    └─ Sam3PerceptionClient（Pillow + Python 标准库）
          │ HTTP/JSON，Compose 内网
          ▼
sam3 container
  ├─ Sam3Runtime：单图语义检测
  │   └─ 常驻 GPU 的 SAM3 image model
  └─ Sam3VideoMemoryRuntime：PNG 序列 session
      ├─ 常驻 GPU 的 SAM3 video model
      ├─ stable Scene Graph ID ↔ numeric track ID
      └─ 每个 session 独立的 mask memory
```

这样有以下特性：

- 不升级、不卸载 TableSim 主容器里的 Torch、SAPIEN、cuRobo 或其他依赖。
- SAM3 服务可以单独构建、启动、停止和重建。
- image model 和 video model 都只懒加载一次并常驻 GPU。
- 5 Hz PNG 序列使用同一个 tracking session，不再把每帧当作无关请求。
- 主进程不会导入 `torch` 或 `sam3`。
- SAM3 是 Scene Graph 的唯一感知来源；服务错误会直接上抛。
- Scene Graph 构造不读取场景 `manifest.json`，也不包含图像启发式 fallback。

当前 SAM3 image model 已在本机的 RTX 4090 上完成无权重结构初始化验证：

```text
SAM3 source commit: 46957e47805eaa273f4aa7bbbd25a88bca9108ce
Python:             3.10.12
PyTorch:            2.4.1+cu121
CUDA runtime:       12.1
Image parameters:   840,509,750
Full image/text forward: passed
Peak allocated VRAM:     about 5.1 GiB
```

这是为固定源码 commit 验证过的 TableSim 兼容栈。官方仓库的新版本可能提高最低
版本要求，因此更新 submodule 后必须重新执行本文档的检查命令。

官方当前 README 推荐 Python 3.12、PyTorch 2.7+ 和 CUDA 12.6+。本项目没有为了
满足该推荐值而升级主仿真环境，而是对固定 commit 的 image detector 和
mask-memory video tracker 都完成了实际推理验证。若启用 SAM3 训练、
FlashAttention，或升级官方源码，应改用官方推荐栈并继续保持独立 sidecar。

## 2. 相关文件

```text
compose.yaml
docker/sam3.Dockerfile
script/download_sam3_modelscope.sh
script/generate_scene_graph_video.py
high_level_planner/
├── perception/
│   ├── base.py
│   ├── mask_codec.py
│   ├── robot_exclusion.py
│   ├── sam3_client.py
│   ├── sam3_tracking_client.py
│   ├── video_memory.py
│   └── sam3_server.py
├── scene_graph/constructor.py
└── third_party/
    └── sam3/                 # 官方 Git submodule
```

各文件职责：

| 文件                                     | 职责                                                                        |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| `docker/sam3.Dockerfile`               | 在独立 venv 中安装 SAM3 image inference 依赖                                |
| `compose.yaml` 的 `sam3` service     | GPU、cache、端口和环境变量配置                                              |
| `script/download_sam3_modelscope.sh`   | 从 ModelScope 断点续传并校验`sam3.pt`                                     |
| `script/extract_video_frames.py` | 先按指定频率把视频提取为 PNG 图片序列和帧清单                              |
| `script/generate_scene_graph_video.py` | 从 PNG 帧清单更新 Scene Graph，保存快照并合成左右对照视频                  |
| `perception/base.py`                   | 检测结果、错误类型和 backend protocol                                       |
| `perception/mask_codec.py`             | 压缩传输和解码 SAM3 实例布尔 mask                                           |
| `perception/robot_exclusion.py`        | 用独立机器人辅助查询过滤机械臂区域误检，不让机器人参与 Scene Graph 和可见性 |
| `perception/sam3_client.py`            | 主 planner 使用的轻量 HTTP 客户端                                           |
| `perception/sam3_tracking_client.py`   | session 创建、seed、track、re-anchor、新节点追加和关闭客户端                 |
| `perception/video_memory.py`           | image detector、mask memory、定期校正和 robot 排除的混合时序 backend         |
| `perception/sam3_server.py`            | image/video 模型、跨 prompt mask 聚类、多标签候选、session 和 HTTP API       |
| `scene_graph/constructor.py`           | 把 SAM3 detection 转换为`SceneObject` 和关系                              |
| `third_party/sam3`                     | 固定版本的 Meta 官方源代码                                                  |

## 3. 初始化官方源码

第一次 clone TableSim，或 submodule 目录为空时执行：

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim
git submodule update --init --depth 1 high_level_planner/third_party/sam3
```

检查固定版本：

```bash
git -C high_level_planner/third_party/sam3 rev-parse HEAD
```

预期：

```text
46957e47805eaa273f4aa7bbbd25a88bca9108ce
```

不要直接复制或修改 `third_party/sam3` 中的文件。TableSim 自己的适配逻辑放在
`high_level_planner/perception/`，这样可以保持上游代码、许可证和本项目修改之间
的清晰边界。

## 4. 从 ModelScope 获取权重

本项目默认使用 ModelScope 的公开 `facebook/sam3` 镜像，不依赖 Hugging Face
gated model 审批。执行：

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim
./script/download_sam3_modelscope.sh
```

脚本会：

1. 启动或复用独立的 `sam3` sidecar；
2. 只下载 image inference 所需的 `sam3.pt`；
3. 使用 `.part` 文件支持断点续传；
4. 检查文件大小；
5. 检查完整 SHA-256；
6. 原子替换为最终 checkpoint。

当前固定的权重版本：

```text
ModelScope model: facebook/sam3
Revision:         f8ee5d16d5c30d0e87b1d22cbd358ce97761df56
File:             sam3.pt
Size:             3,450,062,241 bytes
SHA-256:          9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

权重位于 Docker named volume 内：

```text
/home/tablesim/.cache/huggingface/modelscope/facebook/sam3/sam3.pt
```

该路径是容器内路径，Compose 默认把 `SAM3_CHECKPOINT` 指向这里。权重不会写入
TableSim Git 工作区，也不会随代码提交。

重复运行下载脚本是安全的：如果现有文件大小和 SHA-256 均正确，脚本只校验，
不会重新下载。

如需使用其他本地 checkpoint，可在启动 Compose 前覆盖：

```bash
export SAM3_CHECKPOINT=/container/path/to/sam3.pt
```

自定义路径必须对 `sam3` 容器可见；必要时同时在 `compose.yaml` 中添加只读 volume
mount。

## 5. 构建独立 SAM3 镜像

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim
docker compose --profile sam3 build sam3
```

当前镜像名称：

```text
tablesim-sam3:cuda12.1
```

它从本地 `tablesim:cuda12.1` 派生，但所有新增包都位于
`/opt/sam3-venv`。构建不会修改父镜像，也不会停止正在运行的 TableSim 容器。

检查镜像版本：

```bash
docker image inspect tablesim-sam3:cuda12.1 \
  --format '{{json .Config.Labels}}'
```

`org.opencontainers.image.revision` 应等于固定的 SAM3 commit。

## 6. 启动服务

下载 ModelScope checkpoint 后启动或重建：

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim
docker compose --profile sam3 up -d --no-deps --force-recreate sam3
```

该命令只操作 `sam3` 服务，不会重启 `tablesim`。

默认服务端口：

```text
Compose 内部：http://sam3:8003
宿主机：      http://127.0.0.1:8003
```

模型权重保存在 named volume：

```text
tablesim_tablesim-sam3-cache
```

容器重建后不会重复下载。

`HF_TOKEN` 仅作为可选的 Hugging Face 回退方式保留；默认 ModelScope 本地
checkpoint 加载流程不读取也不需要该 token。

## 7. 检查服务和模型

### 7.1 健康检查

```bash
curl -fsS http://127.0.0.1:8003/health
```

服务刚启动、模型尚未加载时：

```json
{
  "status": "ok",
  "backend": "sam3",
  "model_loaded": false,
  "load_attempted": false,
  "device": null,
  "checkpoint": "local"
}
```

### 7.2 显式加载模型

```bash
curl -fsS \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{}' \
  http://127.0.0.1:8003/v1/load
```

成功后 `model_loaded` 应为 `true`，`device` 应为 `cuda`。

如果返回 checkpoint 不存在或 SHA-256 错误：

1. 重新执行 `./script/download_sam3_modelscope.sh`；
2. 确认脚本最终输出 `SAM3 checkpoint is ready`；
3. 使用 `--force-recreate` 重建 sidecar，使本地路径进入容器；
4. 再调用 `/v1/load`。

检查 GPU：

```bash
docker compose exec sam3 python -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

检查日志：

```bash
docker compose --profile sam3 logs --tail=200 sam3
```

## 8. 用 SAM3 生成 Scene Graph

在 TableSim 主容器中执行：

```bash
docker compose exec tablesim python -m high_level_planner.demo \
  --image image_collectors/scenes/scene_000.png \
  --instruction "Pick up the notebook from the cluttered desk." \
  --output-dir planner_outputs/sam3_scene_000 \
```

该入口始终使用 SAM3，不需要额外的 strict 开关。

成功时命令输出应包含：

```text
perception_source: sam3
```

生成的 `scene_graph.json` 中：

```json
{
  "metadata": {
    "source": "sam3",
    "resolution": [640, 480],
    "image_geometry": {
      "coordinate_frame": "image_normalized",
      "origin": "top_left",
      "x_axis": "right",
      "y_axis": "down",
      "xy": "bbox_center_divided_by_image_size",
      "footprint": "bbox_size_divided_by_image_size",
      "image_size": [640, 480]
    },
    "object_level_perception": {
      "device": "cuda",
      "prompts": ["notebook", "pen", "cup"]
    }
  },
  "objects": [
    {
      "name": "notebook",
      "state": "visible on desk",
      "attributes": {
        "rigidity": 0.9,
        "fragility": 0.2,
        "visibility": "fully_visible",
        "movable": true,
        "portable": true,
        "confidence": 0.91,
        "bbox": [100.0, 120.0, 260.0, 310.0],
        "xy": [0.28125, 0.447917],
        "footprint": [0.25, 0.395833],
        "extra": {
          "detector": "sam3",
          "mask_area": 18342,
          "mask_centroid": [181.5, 219.2],
          "xy_source": "bbox_center",
          "footprint_source": "bbox_size",
          "geometry_coordinate_frame": "image_normalized"
        }
      }
    }
  ]
}
```

具体数值取决于图片和 checkpoint。

每个对象固定输出以下五项语义属性：

| 属性           | 类型             | 含义                                       |
| -------------- | ---------------- | ------------------------------------------ |
| `rigidity`   | `0.0`–`1.0` | 刚性程度，值越大越刚硬                     |
| `fragility`  | `0.0`–`1.0` | 易碎程度，值越大越脆弱                     |
| `visibility` | 枚举             | `fully_visible` 或 `partially_visible` |
| `movable`    | bool             | 是否允许改变物体位置                       |
| `portable`   | bool             | 是否便于执行常规 pick-and-place            |

`movable=true` 不等于 `portable=true`。例如灯和植物可以改变位置，但由于尺寸、
形状或脆弱性，不被视为便于移动。`confidence`、`bbox`、`xy` 等继续作为感知和
几何数据保留。

SAM3 路径会从像素坐标 bbox 自动计算 `xy` 和 `footprint`。
对图片尺寸 `W × H` 和 bbox `[x1, y1, x2, y2]`：

```text
xy = [
  ((x1 + x2) / 2) / W,
  ((y1 + y2) / 2) / H
]

footprint = [
  abs(x2 - x1) / W,
  abs(y2 - y1) / H
]
```

二者均是 `[0, 1]` 尺度的二维图像平面量，原点位于图片左上角，x 向右、y 向下。
它们不是桌面坐标系中的米制位置或真实物理尺寸。坐标约定保存在
`metadata.image_geometry`，每个物体的推导来源保存在
`attributes.extra.xy_source`、`footprint_source` 和
`geometry_coordinate_frame`。

SAM3 本身没有输出“全部可见/部分可见”的离散标签。本项目把检测框接触图像边界，
或与另一实例显著重叠的对象标记为 `partially_visible`；其他对象标记为
`fully_visible`。判定依据写入 `attributes.extra.visibility_evidence`。

以上是通用单图构造器的保守默认值。视频脚本在跨帧合并完成后执行
`on_relation` 可见性校正：一个物体只有在存在其他 Scene Graph 物体指向它的
`on` 关系时才是 `partially_visible`，否则一律为 `fully_visible`。视频图中不允许
`invisible`/`not_visible` 状态；机械臂辅助检测不进入图，也不参与可见性判断。
完整流程见
[8.3 节](#83-连续视频-scene-graph-更新与对照视频)。

Scene Graph 会根据 detection bbox 补充：

- `left_of`、`right_of`；
- `behind_image`、`in_front_image`；
- `next_to`；
- `overlaps`。

每条关系边都会写入：

```json
{
  "relation": "next_to",
  "distance": [0.125, 0.04],
  "direction": "right"
}
```

其中 `distance` 不是距离大小，而是
`[abs(target.x-source.x), abs(target.y-source.y)]`，使用 Object 中心的归一化
`xy` 坐标。只有 `next_to` 增加 `direction=left/right`，表示 target 相对
source 位于左侧或右侧；其他关系保持原名称。Scene Graph PNG 和展示视频把边标签
显示为 `relationship: (delta_x, delta_y)`，例如
`next_to_right: (0.125, 0.040)`。

对于 bbox 明显重叠的实例，构造器还会根据交叠区域的可见 mask 归属推断
`source --on--> target`。该关系是二维遮挡启发式结果，不等价于深度或真实物理
接触真值；`blocks` 仍应由仿真真值、深度图、连续帧或后续关系模型补充。

### 8.1 已验证样例：`scene_006`

以下命令已使用 ModelScope `sam3.pt` 在本机完成严格模式验证：

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim

docker compose exec -T tablesim python -m high_level_planner.demo \
  --image image_collectors/scenes/scene_006.png \
  --instruction "Pick up the apple from the cluttered desk." \
  --output-dir planner_outputs/sam3_scene_006_visualized \
  --sam3-prompts \
    "notebook,pen,cup,mouse,keyboard,phone,laptop,paper,scissors,monitor,book,box,pencil,blackboard,lamp,plant,apple,glasses" \
  --sam3-max-prompts 18 \
  --sam3-threshold 0.35 \
```

实测输出：

```text
objects:           17
relations:         214
perception_source: sam3
next_subtask:      pick apple
```

其中 Scene Graph 检测到的类别实例包括：

```text
apple, blackboard, book, cup, keyboard, lamp, laptop, monitor,
paper, pen, phone, plant, scissors
```

部分类别存在多个实例。关系统计为：

| 关系               | 数量 |
| ------------------ | ---: |
| `right_of`       |   75 |
| `left_of`        |   44 |
| `next_to`        |   60 |
| `overlaps`       |   18 |
| `in_front_image` |   10 |
| `behind_image`   |    7 |

使用新属性模式重新推理后的可见性统计：

| `visibility`        | 对象数 |
| --------------------- | -----: |
| `fully_visible`     |      2 |
| `partially_visible` |     15 |

不便于常规移动的对象为 plant、lamp、monitor 和 blackboard；其中 monitor 和
blackboard 同时为 `movable=false`，plant 和 lamp 为“可以移动但不便携”。
17 个对象均已验证包含 `rigidity`、`fragility`、`visibility`、`movable` 和
`portable`，且不再输出旧的 `graspable`、`weight`、`occluded`。

完整 Scene Graph 保存在：

```text
planner_outputs/sam3_scene_006_visualized/scene_graph.json
```

JSON 中 `metadata.source` 必须为 `sam3` 或 SAM3 视频 memory backend 的来源名。

### 8.2 绘制 Scene Graph 和 Reasoning Graph

使用现有可视化脚本：

```bash
docker compose exec -T tablesim python \
  script/visualize_planner_graphs.py \
  --input-dir planner_outputs/sam3_scene_006_visualized \
  --max-spatial-edges 48 \
  --dpi 180
```

生成：

```text
planner_outputs/sam3_scene_006_visualized/
├── scene_graph.json
├── scene_graph.png
├── scene_graph.svg
├── scene_graph_spatial.png
├── scene_graph_spatial.svg
├── reasoning_graph.json
├── reasoning_graph.png
├── reasoning_graph.svg
└── vla_request.json
```

`scene_graph.json` 始终保留全部对象和关系。为了防止高密度图中标签完全重叠，
`--max-spatial-edges` 只限制 PNG/SVG 中绘制的空间关系数量，不会删除或修改 JSON
数据。本次图片绘制了 108 条代表性关系，完整 214 条关系仍在 JSON 中。

`planner_outputs/` 默认被 `.gitignore` 排除。这些文件会保留在本机，但不会自动
进入 Git；如需长期归档，应复制到专门的评估产物目录或对象存储。

### 8.3 连续视频 Scene Graph 更新与对照视频

Scene Graph 不直接接收视频。流程首先由 `script/extract_video_frames.py` 按 5 Hz
把视频提取为 PNG 图片和 `frames_manifest.json`，然后
`script/generate_scene_graph_video.py` 逐张读取 PNG，运行 object-level
perception、Scene Graph 构建和跨帧更新。原 MP4 只在最后合成左右对照视频时读取，
不作为 SAM3 或 Scene Graph 的输入。该流程不生成 Reasoning Graph，也不生成单独
的 spatial 图。生成脚本已将 `--frames-manifest` 设为必填参数，并移除
`--video` Scene Graph 输入接口；直接向该脚本传视频会被命令行解析器拒绝。
第一次观测由 `SceneGraphConstructor` 建图。当前视频流程已经恢复多帧初始化与
节点锁定：前 20 秒内，未匹配 detection 必须满足不同帧重复观测、置信度、位置
距离和最大间隔条件才能成为新节点；20.2 秒起节点集合固定，只更新属性和关系。
Object inventory 校验器代码仍保留，但不参与当前生成流程。

SAM3 感知本身默认使用同一个 mask-memory session：

```text
全部裁剪后的 5 Hz PNG
  → POST /v1/tracking/sessions/start（只传有序图片目录）
  → 第 0 帧运行 image detector
  → detection mask 与首帧 SceneObject.id 一一映射
  → POST /seed，把稳定 Scene Graph ID 写入 Object memory
  → 第 1..N 帧 POST /track，严格按顺序传播已有 mask
  → 每 5 帧重新运行 image detector
      ├─ 与 tracker mask 按 IoU 匹配
      ├─ 修正 bbox、mask 和语义类别
      └─ POST /reanchor，刷新条件 memory、抑制漂移
  → 未匹配 detector 实例只成为新 Object 候选
  → 初始化窗口内经多帧确认后 POST /add
  → 初始化窗口结束后不再向 Object memory 增加新 ID
  → POST /close 释放该序列的 session state
```

`--sam3-video-memory` 默认开启；`--sam3-redetect-interval 5` 表示在 5 Hz 输入下
每 1 秒校正一次；`--sam3-tracker-match-iou 0.25` 控制 detector mask 与 track
mask 的匹配阈值。`--no-sam3-video-memory` 仅保留用于对比旧的逐图片独立检测
流程。session 仍以 PNG 目录为输入，服务端没有接收 MP4。

SAM3/robot exclusion 输出进入固定的 6 帧标签滑动窗口：

```text
窗口 = 当前帧 + 当前帧之前 5 帧
前 5 帧历史不足：
  不修正，只保存原始 SAM3 标签
从第 6 帧开始：
  按 stable_scene_object_id 优先、bbox 位置相似度兜底关联同一目标
  → 统计该位置前 5 帧的原始 SAM3 标签
  → 唯一最高频标签 != 当前标签：覆盖当前标签
  → 最高频并列：保留当前标签
```

历史保存的是原始 SAM3 标签，不是修正后的输出，这样 detector 如果连续改变判断，
新标签在滑动窗口滚动后仍可成为众数，不会被早期标签永久锁死。默认阈值为
归一化中心距离 `0.08`、bbox IoU `0.20`、bbox 尺寸相似度 `0.50`。
逐目标投票、匹配依据、修正前后标签会写入 detection metadata；逐帧汇总位于
`object_level_perception.label_sliding_window` 和中间 trace JSON。

滑动窗口之后，mask-memory detection 携带 `stable_scene_object_id` 时，
`SceneGraphUpdater.merge()` 首先按这个 ID
直接匹配；因此物体大幅移动或定期 detector 修正了语义类别，也不会变成新节点。
没有稳定 memory ID 的候选才回退到 `(name, source, model_id)` 与最近位置匹配。
匹配后更新 bbox、xy、footprint、属性和关系。当前帧没有
匹配到的旧节点保留最后一次信息，不降置信度、不设为 invisible。无法匹配已有
节点的 detection 在初始化窗口内交给 `MultiFrameSceneGraphInitializer`；确认后
加入 Scene Graph 和 SAM3 mask memory，不经过 inventory 校验。窗口结束后的
未匹配 detection 只进入审计信息。锁定阶段节点既不能增加也不能删除。每次更新的
`metadata.node_update_policy` 会记录：

- `node_ids_before` 和 `node_ids_after`；
- `node_ids_unchanged`；
- `current_observation_object_ids`；
- `retained_without_current_detection_ids`；
- `new_object_ids`；
- `removed_object_ids`，始终为空；
- `multi_frame_initialization_enabled=true`；
- `node_locking_enabled=true`；
- `object_addition_validator_enabled=false` 和
  `object_addition_validator_code_retained=true`。

完整标签窗口配置与累计修正次数写入
`manifest.json:update_config.label_sliding_window`；初始化与锁定配置位于
`update_config.node_update_policy`。校验器字段继续保留，但固定为未启用、
未尝试。

使用 `test0.mp4` 的已验证两阶段命令：

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim

# 阶段一：按 5 Hz 提取 PNG，30 FPS 源视频每隔 6 帧采一张
docker compose exec -T tablesim python \
  script/extract_video_frames.py \
  --video image_collectors/videos/test0.mp4 \
  --output-dir image_collectors/videos/test0_frames_5hz \
  --frequency 5.0 \
  --overwrite

# 阶段二：Scene Graph 只读取图片清单
docker compose exec -T tablesim python \
  script/generate_scene_graph_video.py \
  --frames-manifest \
    image_collectors/videos/test0_frames_5hz/frames_manifest.json \
  --output-dir image_collectors/videos/test0_scene_graph \
  --sam3-crop-roi "0,0.125,1,0.9375" \
  --prompts-file image_collectors/prompts.json \
  --max-prompts 24 \
  --label-window-size 6 \
  --label-window-match-distance 0.08 \
  --label-window-min-bbox-iou 0.20 \
  --label-window-min-size-similarity 0.50 \
  --initialization-window 20.0 \
  --initialization-min-observations 2 \
  --initialization-min-confidence 0.50 \
  --initialization-match-distance 0.08 \
  --initialization-max-gap 3.0 \
  --threshold 0.25 \
  --min-mask-area 50 \
  --max-relations 18 \
  --sam3-video-memory \
  --sam3-redetect-interval 5 \
  --sam3-tracker-match-iou 0.25 \
  --robot-exclusion \
  --robot-prompts "robotic arm,robot arm,robot gripper,robot" \
  --robot-match-iou 0.65 \
  --robot-mask-containment 0.65 \
  --infer-on-from-masks \
  --on-bbox-overlap-threshold 0.25 \
  --on-mask-ownership-margin 0.10 \
  --visibility-policy on_relation \
  --overwrite
```

本例直接从 `image_collectors/prompts.json` 读取有序查询词，并受
`--max-prompts 24` 限制。`manifest.json` 不再复制 prompt 列表，只在
`update_config.query_catalog` 中保存 catalog 路径、实际查询数量和数量上限。
prompt 是发送给 SAM3 的文本查询词，不等于检测结果。为避免混淆，每条 update
分别记录：

- `sam3_detection_labels_after_robot_filter`：机械臂过滤后的 SAM3 detection；
- `sam3_detection_labels_after_sliding_window`：6 帧窗口修正后、实际用于建图的
  detection；
- `matched_existing_node_labels`：成功匹配 Scene Graph 节点的 detection；
- `initialization_confirmed_new_object_ids`：初始化窗口内该帧确认的新节点；
- `detected_labels`：为兼容旧产物而保留，等同于
  `matched_existing_node_labels`。

视频脚本默认 `--prompts ""`，因此直接读取 `--prompts-file`。显式传入
`--prompts "cup,book,pen"` 才会覆盖 catalog。任务 instruction 明确提到的类别会
被移到查询队列前部。

#### 多帧初始化、节点锁定与保留的校验器

当前组合如下：

- `scene_graph/initializer.py`：已启用，负责 0–20 秒内的多帧候选确认；
- `SceneGraphUpdater.merge()`：已启用，20.2 秒起固定节点集合；
- `scene_graph/addition_validator.py`：代码保留但未启用。

标签修正先于初始化发生。默认确认条件为至少 2 个不同更新帧、置信度不低于
`0.50`、归一化位置距离不超过 `0.08`、观测间隔不超过 3 秒。确认的新 ID 直接
加入 SAM3 memory，不经过 inventory 校验。`--initialization-*` 参数会实际影响
视频输出，manifest 会记录固定起点与确认的新 ID。

#### 新方法：基于 `on` 关系校正 visibility

SAM3 sidecar 会把每个实例的布尔 mask 以 `packbits + zlib + base64` 形式压缩后
传给主容器。mask 只在内存中参与关系推断，不写进最终 Scene Graph JSON。

`on` 关系的推断顺序为：

1. 计算两个 bbox 的交集面积；
2. `交集面积 / 较小 bbox 面积 >= 0.25` 才视为明显重叠；
3. 分别统计两个实例在 bbox 交集内的可见 mask 像素；
4. 对每个物体计算
   `交集内 mask 像素 / 该物体全部可见 mask 像素`；
5. 两个得分相差至少 `0.10` 且胜者至少占有 5 个交叠像素时，得分更高者作为
   `source`，生成 `source --on--> target`；
6. 同类别的整体 mask 与嵌套局部 mask 如果互包含比例都超过 `0.65`，按同一物体
   的重复检测处理，不生成自己叠在自己上的关系。

关系中的 bbox 交集、两个 mask 得分、像素数和 ownership margin 均保存在
`relation.metadata`；图级统计位于 `metadata.on_relation_inference`。

每次构造或更新 Scene Graph 后，对全部固定节点执行 visibility 后处理：

```text
若存在 relation="on" 且 target=当前物体.id：
  设置 attributes.visibility=partially_visible
否则：
  设置 attributes.visibility=fully_visible
```

这里的关系方向是 `source --on--> target`。例如杯子在书上时，应存在
`cup --on--> book`；因此书可以保持 `partially_visible`。

`attributes.visibility` 只允许 `fully_visible` 和 `partially_visible`。
当前帧是否成功匹配节点单独记录在
`metadata.node_update_policy.current_observation_object_ids`，不会被编码成第三种
可见性或 `not_visible` state。

视频默认启用 robot auxiliary exclusion。机械臂、夹爪和机器人硬件 detection
只用于过滤同区域的普通物体误检，不会成为 SceneObject、`on`/`overlaps` 关系或
visibility evidence。机械臂造成短时漏检时，旧节点保持原属性。
原始检测判断会保存在
`attributes.extra.visibility_before_on_relation_policy` 和
`visibility_evidence_before_on_relation_policy` 中，便于追踪；校正后的依据写入
`attributes.extra.visibility_evidence`，值为 `no_incoming_on_relation` 或
`incoming_on:<source_object_id>`。图级汇总位于
`metadata.visibility_policy`。

mask ownership 只能说明二维画面中哪个实例占据交叠前景，不能证明两个物体在
三维空间真实接触。因此对物理精度要求较高时，仍建议用仿真真值、深度或连续帧
关系模型校验 `on`。

#### 机械臂排除

以下 robot auxiliary 配置已经是视频脚本默认值；命令中显式写出便于复现：

```bash
docker compose exec -T tablesim python \
  script/generate_scene_graph_video.py \
  --frames-manifest \
    image_collectors/videos/test0_frames_5hz/frames_manifest.json \
  --output-dir image_collectors/videos/test0_scene_graph \
  --robot-exclusion \
  --robot-prompts "robotic arm,robot arm,robot gripper,robot" \
  --robot-match-iou 0.65 \
  --robot-mask-containment 0.65 \
  --visibility-policy on_relation \
  --overwrite
```

该方法将普通物体和 robot auxiliary 分开处理，并同时执行实例过滤和 memory
隔离：

1. object bbox 与 robot bbox 的 IoU 不小于 `--robot-match-iou`；
2. object 可见 mask 落在 robot mask 内的比例不小于
   `--robot-mask-containment`。

两条规则满足任意一条即过滤。第二条可以发现 bbox 仅部分重合、但物体 mask
实质位于机械臂上的误检。此外，保留下来的 object mask 会减去 robot mask
区域，再用于 seed、re-anchor 或新节点追加。这保证 robot mask 永远不会作为
Object mask 的一部分进入 SAM3 memory。机器人辅助 detection 只保存在 metadata，
不会成为 SceneObject、track ID、`on` 关系或 visibility evidence。

启用 `--sam3-video-memory` 时必须同时启用 robot exclusion；脚本会拒绝
`--sam3-video-memory --no-robot-exclusion`，避免产生不受约束的 Object memory。

#### SAM3 前置裁剪

桌面是场景支撑面，不应作为可操作物体进入 Scene Graph。当前方案不再对完整图片
执行额外的 desk/table 查询，而是在任何 object 或 robot SAM3 请求发生之前，先把
原图裁剪到桌面物体操作区域：

```bash
--sam3-crop-roi "0,0.125,1,0.9375"
```

参数顺序是归一化的 `left,top,right,bottom`。对当前 640×480 图片，该 ROI
转换为像素框 `[0, 60, 640, 450]`，生成 640×390 的 PNG。这个范围保留左右桌面
边缘、远端桌面边缘和完整操作区，仅切掉顶部多余墙面以及底部露出的 ALOHA 底座。
进入操作区的左右机械臂不能用矩形 ROI 移除，仍由独立 robot mask 排除。

裁剪是 SAM3 视频入口的必经步骤，不提供跳过开关。执行顺序为：

```text
原始 5 Hz PNG
  → 按归一化 ROI 裁剪并落盘
  → cropped PNG 发送给 object SAM3 query
  → 同一 cropped PNG 发送给 robot auxiliary query
  → robot exclusion
  → SceneGraphConstructor
```

`high_level_planner.demo --perception sam3` 的单图入口也执行相同裁剪，并在输出
目录保存 `sam3_input_crop.png`。如相机视角或图片分辨率改变，可以修改
`--sam3-crop-roi`；归一化坐标会自动映射到新分辨率。

裁剪后，Scene Graph 内的 `bbox` 使用 640×390 裁剪图坐标，`xy` 和
`footprint` 也相对裁剪图宽高归一化，而不是相对 320×240 原图。每帧 JSON 的
`metadata.sam3_input_crop` 和 trace 都保存原图路径、原图尺寸、像素裁剪框、
裁剪图路径及尺寸；`manifest.json.frame_input.sam3_input_crop` 保存全序列配置。
原始图片仍通过 `source_image` 索引，实际感知输入通过 `observation_image` 索引。

若处理其他场景，可通过 `--prompts "cup,book,pen"` 覆盖；将 `--prompts ""`
设为空字符串时，脚本会从 `--prompts-file` 指定的有序列表中取前
`--max-prompts` 类。Scene Graph 更新周期来自 `frames_manifest.json`；本例为
5 Hz，即每 0.2 秒一张输入图片。脚本会为每张输入图片保存一份完整 JSON 和一份
拓扑图 PNG。

输出结构：

```text
image_collectors/videos/test0_frames_5hz/
├── frames/
│   ├── frame_000000.png
│   ├── frame_000006.png
│   └── ...                         # 共 207 张 5 Hz PNG
└── frames_manifest.json            # 源帧号、时间戳和图片路径

image_collectors/videos/test0_scene_graph/
├── sam3_cropped_inputs/
│   ├── sam3_crop_0000_frame_000000.png
│   └── ...                         # 共 207 张 640×390 SAM3 实际输入
├── sam3_intermediate/
│   ├── object_query/               # robot 过滤前的物体 bbox + mask
│   ├── robot_query/                # 独立 robot auxiliary bbox + mask
│   ├── filtered/                   # robot exclusion 后的最终 detection
│   ├── pipeline/                   # 四宫格完整感知过滤管线
│   └── json/                       # 每帧 detection 与节点匹配审计
├── updates/
│   ├── scene_graph_update_0000.json
│   ├── scene_graph_update_0000.png
│   └── ...                         # 共 207 份 JSON 和 207 张 PNG
├── manifest.json                   # 帧号、时间戳、参数和产物索引
└── test0_scene_graph_demo.mp4       # 左操作视频，右对应最新 Scene Graph
```

输出 `manifest.json.frame_input` 明确记录
`mode=image_sequence`、`sampling_frequency_hz=5.0`、
`perception_input_type=png_image` 和
`source_video_used_for_perception=false`，并记录裁剪 ROI、像素框和裁剪尺寸。

#### SAM3 中间检测结果

脚本会为 207 张输入图片逐帧保存实际裁剪输入和以下诊断产物：

1. `sam3_cropped_inputs/sam3_crop_NNNN_*.png`：实际传入 SAM3 的 640×390
   裁剪图，不是仅用于显示的副本。
2. `object_query/sam3_object_query_NNNN.png`：SAM3 object query 经置信度阈值、
   最小 mask 面积和服务端 mask 聚类后的结果；红色 `REMOVED` 表示该 detection
   随后被 robot exclusion 删除。
3. `robot_query/sam3_robot_query_NNNN.png`：独立的 robot arm、robot gripper 和
   robot auxiliary 查询结果。
4. `filtered/sam3_filtered_NNNN.png`：机械臂过滤完成后，实际交给
   `SceneGraphConstructor` 的 bbox 和半透明实例 mask。
5. `pipeline/sam3_pipeline_NNNN.png`：裁剪输入、object query、robot query 和
   最终输入的四宫格对照图。
6. `json/sam3_trace_NNNN.json`：完整的 label、score、bbox、mask 面积、mask
   编码类型、过滤依据、匹配节点 ID、未匹配 detection 和保留节点列表。为控制
   文件体积，压缩 mask 本体不重复写入 JSON；PNG 中保留 mask 的半透明可视化。

每条 `manifest.json.updates[]` 都包含实际裁剪图和以上五个诊断文件的相对路径。
定位问题时按照以下顺序检查：

```text
sam3_cropped_inputs 中目标已被切掉
  → 调整 --sam3-crop-roi

object_query 中没有目标
  → SAM3 query / 阈值 / mask 聚类问题

object_query 中有目标，但 filtered 中消失并标为 REMOVED
  → robot exclusion 的 IoU 或 mask containment 过滤问题

filtered 中有目标，但 trace.scene_graph_update.matched_node_ids 中没有
  → Scene Graph 稳定 ID / 位置匹配或初始化确认与锁定问题

scene_graph_input_after_robot_exclusion 标签正确，但
six_frame_label_sliding_window 标签错误
  → 检查该目标前 5 帧的位置匹配、标签计数和并列策略

matched_node_ids 中已有目标，但最终关系错误
  → bbox 关系、mask ownership on 推断或 updater 关系更新问题
```

`object_query` 不是阈值前的模型原始 proposal；当前 sidecar 只返回经过阈值、
最小面积和 mask 聚类后的 detection。服务端阈值前候选数量仍可从 trace 的
`raw_detection_count` 查看。

右侧 Scene Graph 使用跨帧固定的拓扑网格位置。蓝色和橙色节点分别表示
fully visible 和 partially visible，不存在灰色 invisible 节点；节点中同时显示
置信度、刚性、易碎性、可见性、可移动性和
portable，并显示由 bbox 推导的归一化 `xy` 和 `footprint`。`--max-relations`
只限制 PNG 和视频中显示的边数，JSON 仍保留该次更新的完整关系。紫色状态行显示
当前启用 `on_relation` 可见性规则，以及本次更新被校正为 fully visible 和因
incoming `on` 保持 partially visible 的节点数量。

以下 `test0.mp4` 表格是切换到滑动窗口之前的历史旧流程结果，仅用于复现，不代表
当前视频入口的初始化、锁定或校验策略：

| 项目                               | 结果                                        |
| ---------------------------------- | ------------------------------------------- |
| 输入                               | 320×240，30 FPS，1237 帧，41.233 秒        |
| 图片采样                           | 5 Hz，207 张 PNG，源帧间隔 6 帧             |
| SAM3 实际输入                      | ROI `[60,55,260,205]`，207 张 200×150 PNG  |
| Scene Graph 更新                   | 207 次，时间戳 0–41.2 秒，间隔 0.2 秒       |
| 感知来源                           | 207/207 均为 `sam3`，无 fallback            |
| 普通物体 + robot 两查询推理         | 0.673–0.707 秒，平均约 0.685 秒              |
| 首帧 object nodes                  | 11 个                                       |
| 初始化确认                         | 第 18.6 秒新增 glasses 和第二把 scissors     |
| 固定阶段                           | 第 20.2 秒开始，后续始终为相同 13 个 ID       |
| 初始化新增 / 全程删除节点           | 2 / 0                                       |
| 当前帧匹配节点数                   | 1–11 个；未匹配旧节点仍保持 visible         |
| 被忽略的新 detection               | 累计 285 个，全部保存在审计 metadata         |
| robot auxiliary detections         | 累计 1302 个，不进入 Scene Graph            |
| robot 区域普通物体误检过滤         | 累计 231 个；其中 35 个仅由 mask containment 发现 |
| 支撑面节点                         | 0 个 desk/table/tabletop/desktop             |
| 大面积最终 detection               | 0 个宽或高超过裁剪图 80% 的 detection        |
| 节点最大归一化 footprint           | 宽 0.313，高 0.513                           |
| 不同关系集合                       | 147 组，说明关系随画面持续更新               |
| visibility 取值                    | 仅 `fully_visible` / `partially_visible`    |
| invisible / not_visible            | 0                                           |
| SAM3 裁剪输入                      | 207 张 PNG，全部为 200×150                  |
| 更新图片                           | 207 张 PNG                                  |
| 更新数据                           | 207 份 JSON                                 |
| SAM3 中间结果                      | 4×207 张诊断 PNG + 207 份 trace JSON        |
| 合成视频                           | 1920×720，H.264，30 FPS，1237 帧           |
| spatial / reasoning 图             | 均未生成                                    |

上表是早期逐图片检测基线。以下 `test1.mp4` 表格是加入 session memory、但仍
使用旧固定节点策略时的历史验证：

| 项目 | session 结果 |
| --- | --- |
| 5 Hz PNG / Scene Graph 更新 | 206 / 206 |
| 首帧 image detection | 1 帧 |
| mask-memory propagation | 164 帧 |
| 定期 image detector 校正 | 41 帧，每 5 帧一次 |
| ROI | 默认 `[0,60,640,450]`，640×390 PNG |
| re-anchor | 累计 233 个实例 |
| 节点数 | 全程 11 |
| 不同节点 ID 集合 | 1 组 |
| 删除节点 | 0 |
| 合成视频 | 1920×720、30 FPS、1235 帧、41.1667 秒 |
| session 清理 | 完成后 `active_session_count=0` |

产物位于 `image_collectors/videos/test1_scene_graph/`。其
`manifest.json.temporal_perception_backend` 为 `sam3_video_memory`，
`frame_input.all_png_images_share_one_tracking_session=true`，完整 session 配置
位于 `update_config.sam3_video_memory`。

当前“6 帧滑动窗口 + 多帧初始化 + 节点锁定”已使用 `test2.mp4` 覆盖验证：

| 项目 | 当前 `test2` 结果 |
| --- | --- |
| 原视频 | 640×480、30 FPS、1235 帧、41.1667 秒 |
| 5 Hz PNG / Scene Graph 更新 | 206 / 206 |
| SAM3 裁剪输入 | ROI `[0,60,640,450]`，206 张 640×390 PNG |
| session 模式 | 首帧检测 1、mask propagation 164、定期校正 41 |
| 滑动窗口启用点 | update 0–4 不启用，update 5（第 6 张图）开始启用 |
| 标签修正 | 109 个更新帧、累计 221 个 detection |
| 初始化阶段 | update 0–100，时间戳 0–20.0 秒 |
| 初始化确认 | update 100（20.0 秒）确认 `glasses_sam3` |
| 节点锁定 | update 101（20.2 秒）开始 |
| 节点数 | 初始 11、确认新增 1、锁定及最终均为 12、删除 0 |
| 新增校验器 | 代码保留；尝试、通过、拒绝次数均为 0 |
| 关系边 | 全部包含 `[abs(delta_x), abs(delta_y)]`；`next_to` 全部含 left/right |
| 输出 | 206 JSON、206 Scene Graph PNG、206 trace JSON |
| 合成视频 | 1920×720、H.264、30 FPS、1235 帧、41.1667 秒 |
| session 清理 | 完成后 `active_session_count=0` |

产物位于 `image_collectors/videos/test2_scene_graph/`。20.2 秒之后的 105 次
更新均保持相同的 12 个 Object ID，说明锁定已经生效。由于校验器仍未启用，
初始化确认只依赖多帧次数、置信度、位置和时间间隔；标签滑动窗口负责抑制类别
抖动，但不承担 Object inventory 校验。

`manifest.json` 中的 `spatial_graph_generated` 和
`reasoning_graph_generated` 均应为 `false`。每条 update 记录可通过
`frame_index`、`timestamp_seconds`、`scene_graph_json` 和
`scene_graph_image` 进行精确回放和核对。新规则配置位于
`update_config.visibility_policy`；每次更新还记录
`visibility_corrected_to_full_count` 和
`visibility_partial_due_to_on_count`。节点保留审计信息位于每份 Scene Graph
JSON 的 `metadata.node_update_policy`，可见性校正详情位于
`metadata.visibility_policy`。机械臂排除配置位于
`update_config.robot_exclusion`，裁剪配置位于
`frame_input.sam3_input_crop`。固定节点配置位于
`update_config.node_update_policy`，mask 关系阈值位于
`update_config.on_relation_inference`，每张图的关系推断汇总位于
`metadata.on_relation_inference`。

## 9. Prompt 配置与轻量化

SAM3 是 text-prompted concept segmentation。通用 `Sam3PerceptionClient`、单图
demo 和视频脚本默认从 `image_collectors/prompts.json` 读取有序类别，选择前
24 类，并优先把指令中明确提到的类别移到最前。

使用少量自定义类别可以显著减少延迟：

```bash
docker compose exec tablesim python -m high_level_planner.demo \
  --image image_collectors/scenes/scene_000.png \
  --instruction "Pick up the notebook." \
  --output-dir planner_outputs/sam3_notebook \
  --perception sam3 \
  --sam3-prompts "notebook,pen,cup,mouse,keyboard" \
  --sam3-max-prompts 5 \
  --sam3-threshold 0.50 \
  --sam3-strict
```

推荐调节：

| 参数                    |              默认值 | 作用                      |
| ----------------------- | ------------------: | ------------------------- |
| `--sam3-max-prompts`  |                  24 | 单张图最多查询的概念数    |
| `--sam3-threshold`    |                0.50 | detection/mask 置信度阈值 |
| `--sam3-timeout`      |              300 秒 | 首次权重下载和推理超时    |
| `--sam3-prompts`      |                  空 | 逗号分隔的显式类别        |
| `--sam3-prompts-file` | `prompts.json` | 类别库                    |

服务端还会：

- 对同一张图只执行一次 `processor.set_image()`；
- 逐个复用 image state 执行 text prompt；
- 使用 bfloat16 autocast；
- 过滤面积小于 64 像素的 mask；
- 使用 mask IoU NMS 合并跨 prompt 重复检测；
- 默认最多返回 50 个物体。

## 10. Python API

```python
from high_level_planner.perception import Sam3PerceptionClient
from high_level_planner.scene_graph import SceneGraphConstructor

sam3 = Sam3PerceptionClient(
    endpoint="http://sam3:8003",
    prompt_catalog=["notebook", "pen", "cup", "mouse", "keyboard"],
    prompt_catalog_path=None,
    max_prompts=5,
    confidence_threshold=0.5,
)

constructor = SceneGraphConstructor(
    perception_backend=sam3,
)

scene_graph = constructor.construct(
    "image_collectors/scenes/scene_000.png",
    instruction="Pick up the notebook.",
)
```

SAM3 请求失败时，客户端异常由 `construct()` 直接上抛。

视频序列应优先使用封装好的混合 backend；调用方必须在首帧构图后 seed，并在
`finally` 中关闭 session：

```python
from high_level_planner.perception import (
    LabelSlidingWindowPerceptionBackend,
    Sam3PerceptionClient,
    Sam3TrackingClient,
    Sam3VideoMemoryPerceptionBackend,
)
from high_level_planner.scene_graph import (
    MultiFrameSceneGraphInitializer,
    SceneGraphConstructor,
    SceneGraphUpdater,
)
from high_level_planner.scene_graph.graph import SceneObject, unique_id

objects = Sam3PerceptionClient(
    prompt_catalog=["cup", "book", "mouse"],
    prompt_catalog_path=None,
)
robots = Sam3PerceptionClient(
    prompt_catalog=["robot arm", "robot gripper", "robot"],
    prompt_catalog_path=None,
)
temporal = Sam3VideoMemoryPerceptionBackend(
    image_backend=objects,
    robot_backend=robots,
    tracking_client=Sam3TrackingClient(),
    resource_path="/workspace/TableSim/path/to/cropped_pngs",
    redetect_interval=5,
    new_object_memory_policy="multi_frame_confirmation_without_validator",
)
backend = LabelSlidingWindowPerceptionBackend(
    temporal,
    window_size=6,
)
constructor = SceneGraphConstructor(
    perception_backend=backend,
)
updater = SceneGraphUpdater()
initializer = MultiFrameSceneGraphInitializer(
    min_observations=2,
    min_confidence=0.50,
    max_position_distance=0.08,
    max_gap_observations=15,  # 5 Hz 下对应 3 秒
)

try:
    graph = constructor.construct("path/to/cropped_pngs/frame_0000.png")
    temporal.seed_scene_graph(graph)
    for update_index, image in enumerate(ordered_pngs[1:], start=1):
        observation_graph = constructor.construct(image)
        provisional = updater.merge(graph, observation_graph)
        confirmed_ids = []
        if update_index <= 100:  # 5 Hz，包含 20.0 秒更新
            ignored_ids = {
                item["temporary_id"]
                for item in provisional.metadata[
                    "node_update_policy"
                ]["ignored_unmatched_detections"]
            }
            candidates = [
                SceneObject.from_dict(obj.to_dict())
                for obj in observation_graph.objects
                if obj.id in ignored_ids
            ]
            confirmations = initializer.observe(
                candidates,
                observation_index=update_index,
            )
            if confirmations:
                expanded = graph.copy()
                existing_ids = [obj.id for obj in expanded.objects]
                for confirmation in confirmations:
                    obj = SceneObject.from_dict(
                        confirmation.object.to_dict()
                    )
                    obj.id = unique_id(obj.id, existing_ids)
                    existing_ids.append(obj.id)
                    expanded.add_object(obj)
                    confirmed_ids.append(obj.id)
                graph = updater.merge(expanded, observation_graph)
                temporal.add_graph_objects(
                    graph,
                    confirmed_ids,
                    admission_reason=(
                        "multi_frame_confirmation_without_validator"
                    ),
                )
            else:
                graph = provisional
        else:
            graph = provisional  # 节点集合锁定，只更新已有节点
finally:
    temporal.close()
```

## 11. HTTP API

### `GET /health`

检查服务、模型加载状态和 checkpoint 来源。

### `POST /v1/load`

显式加载模型和权重。

### `POST /v1/detect`

请求字段：

```json
{
  "image_base64": "<jpeg bytes in base64>",
  "prompts": ["notebook", "pen", "cup"],
  "confidence_threshold": 0.5,
  "max_detections": 50,
  "min_mask_area": 64,
  "mask_cluster_iou_threshold": 0.8
}
```

响应字段：

```json
{
  "backend": "sam3",
  "image_size": [640, 480],
  "detections": [
    {
      "label": "notebook",
      "score": 0.91,
      "bbox": [100.0, 120.0, 260.0, 310.0],
      "mask_area": 18342,
      "centroid": [181.5, 219.2],
      "mask": {
        "encoding": "packbits_zlib_base64",
        "size": [480, 640],
        "data": "<base64>"
      },
      "label_candidates": [
        {"rank": 1, "label": "notebook", "score": 0.77},
        {"rank": 2, "label": "laptop", "score": 0.637},
        {"rank": 3, "label": "book", "score": 0.617}
      ]
    }
  ],
  "metadata": {
    "prompt_count": 3,
    "inference_seconds": 2.3,
    "device": "cuda"
  }
}
```

实例 mask 会先按行展平并使用 `packbits + zlib + base64` 压缩，再通过 HTTP
返回。主 planner 解码后只用它判断 bbox 交叠区域的 mask 归属；完整 mask 不写入
Scene Graph JSON，因此不会扩大最终图文件。

不同 prompt 返回的高 IoU mask 不再执行“最高分胜出并丢弃其余类别”的跨 prompt
NMS。服务端现在按 `mask_cluster_iou_threshold` 聚成物理实例，每个 cluster
使用最高分候选提供兼容的主 `label` 和 mask，同时在 `label_candidates` 中保存
该实例的全部去重类别、分数、bbox、mask area 以及它与代表 mask 的 IoU。Scene
Graph 对象会把候选列表保存到 `metadata.label_candidates` 和
`attributes.extra.label_candidates`，供后续 inventory-constrained semantic
resolver 使用。

### Tracking session API

所有 mutation endpoint 都通过同一个 GPU inference lock 串行执行。`frame_index`
必须从 seed 帧开始严格递增，跳帧或倒序会返回 HTTP 400。

| Endpoint | 作用 |
| --- | --- |
| `POST /v1/tracking/sessions/start` | 从只读的有序 PNG 目录创建 session |
| `POST /v1/tracking/sessions/seed` | 把首帧 RLE mask 与稳定 Scene Graph ID 写入 memory |
| `POST /v1/tracking/sessions/track` | 使用 mask memory 传播到下一帧 |
| `POST /v1/tracking/sessions/reanchor` | 用定期 image detector mask 校正已有 ID |
| `POST /v1/tracking/sessions/add` | 加入调用方已经接纳的新稳定 ID；当前视频流程无需 inventory 校验 |
| `POST /v1/tracking/sessions/close` | 释放该 session 的图片、特征和 tracker state |

start 请求示例：

```json
{
  "resource_path": "/workspace/TableSim/path/to/cropped_pngs",
  "offload_video_to_cpu": true,
  "offload_state_to_cpu": false
}
```

seed、re-anchor 和 add 的 `objects` 使用同一格式：

```json
{
  "session_id": "<uuid>",
  "frame_index": 0,
  "objects": [
    {
      "object_id": "cup_sam3",
      "label": "cup",
      "score": 0.91,
      "mask": {
        "encoding": "packbits_zlib_base64",
        "size": [150, 200],
        "data": "<base64>"
      }
    }
  ]
}
```

track 响应的每个 detection metadata 都含
`stable_scene_object_id`、`sam3_track_id`、`source=sam3_mask_memory` 和
`propagated=true`。robot mask 不允许出现在上述 objects 中；该约束由
`Sam3VideoMemoryPerceptionBackend` 在客户端侧保证。

## 12. 错误行为

Scene Graph 没有备用感知算法。SAM3 服务不可用、超时或返回协议错误时，异常直接
传播给调用方；调用方应修复服务状态后重试，而不是使用低可信结果继续规划。

## 13. 验证原 TableSim 环境未被破坏

```bash
docker compose ps

docker compose exec tablesim python -c \
  'import torch; print(torch.__version__, torch.version.cuda)'

docker compose exec tablesim python -m unittest discover \
  -s high_level_planner/tests \
  -v
```

主容器应继续使用原来的 Torch/CUDA，当前测试共 45 项。

## 14. 停止 SAM3

```bash
docker compose --profile sam3 stop sam3
```

只停止 perception sidecar，不影响仿真。

如需删除 sidecar 容器但保留权重 cache：

```bash
docker compose --profile sam3 rm -f sam3
```

不要删除 `tablesim_tablesim-sam3-cache`，否则下次需要重新下载权重。

## 15. 更新官方 SAM3 源码

不要使用未固定版本的 `pip install git+...`。升级时显式选择并记录 commit：

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim/high_level_planner/third_party/sam3
git fetch origin
git checkout <reviewed-sam3-commit>

cd /home/ubuntu/Desktop/Liutong/TableSim
git add high_level_planner/third_party/sam3
```

随后同步：

1. `.gitmodules` 和 `docker/sam3.Dockerfile` 中的 commit；
2. 重新构建 `sam3` 镜像；
3. 执行无权重模型初始化；
4. 执行 `/v1/load`；
5. 用固定图片生成 SAM3 Scene Graph；
6. 执行全部 High-Level Planner 单元测试。
