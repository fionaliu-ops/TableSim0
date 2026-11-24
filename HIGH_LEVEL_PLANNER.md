> 静态入口已改为 Image-first 双图规划，旧目标生成参数不再支持。请先阅读 [当前运行说明](high_level_planner/README.md)。下文保留为旧动态接口的历史参考。

# High-Level Planner Implementation Specification

> 当前实现说明：Reasoning Graph 已升级为 v3。节点可选择 image 或前序模块生成的
> Scene Graph JSON，边是单个 atomic-action subtask，并按
> `image_collectors/GOAL_STATE.md` 判断终态。
> 本文件前半部分保留最初 implementation specification 作为需求背景；当前模块、
> Graph-of-Thoughts operation/controller 架构及 API 配置以
> [`HIGH_LEVEL_PLANNER_ARCHITECTURE.md`](HIGH_LEVEL_PLANNER_ARCHITECTURE.md#5-reasoning-graph-模块)
> 和 [`high_level_planner/reasoning_graph/README.md`](high_level_planner/reasoning_graph/README.md)
> 为准。

## Project Goal

Implement a hierarchical planner for robotic table tidying tasks.

The planner converts:

RGB observation + natural language instruction

into:

structured scene representation,
task reasoning graph,
and executable subtask instructions for VLA policy.

Architecture:

RGB Image
    |
    v
3D Scene Graph Module
    |
    v
Reasoning Graph Generator
    |
    v
Graph Search Planner
    |
    v
Subtask Instruction
    |
    v
VLA Policy

---

# 1. Overall Directory Structure

high_level_planner/

├── scene_graph/
│   ├── constructor.py
│   ├── updater.py
│   ├── graph.py
│   └── attributes.py
│

├── reasoning_graph/
│   ├── generator.py
│   ├── graph.py
│   └── state.py
│

├── planner/
│   ├── ucs.py
│   ├── cost.py
│   └── planner.py
│

├── interface/
│   ├── vla_interface.py
│

├── memory/
│   └── graph_memory.py
│

├── config/
│   └── planner.yaml
│

└── demo.py

---

# 2. Scene Graph Module

## Input

RGB image

High-level instruction

## Output

SceneGraph object

Example:

{
objects:[
 {
 id:1,
 name:"notebook",

 state:"on desk",

 attributes:{
   rigidity:0.9,
   fragility:0.2,
   visibility:"fully_visible",
   movable:true,
   portable:true
 }
 }
],

relations:[
 {
 source:1,
 target:2,
 relation:"left_of"
 }
]

}

---

# 3. Scene Graph Update

Function:

update_scene_graph(
old_graph,
new_observation
)

Requirements:

- preserve object identity
- add newly discovered objects
- update object states
- update relations

Example:

Before:

mug blocks notebook

After moving mug:

notebook becomes visible

Update:

remove blocking relation

---

# 4. Reasoning Graph

Purpose:

Represent task execution states.

Node:

TaskState

Example:

S0:

{
desk:"messy",
notebook:"desk",
mug:"desk"
}

S1:

{
mug:"removed"
}

S2:

{
notebook:"drawer"
}

Edge:

Subtask

Example:

S0
 |
(move mug)
 |
S1

---

# 5. LLM Reasoning Graph Generation

Input:

SceneGraph

Instruction

Output:

JSON graph

Example:

{
nodes:[
{
id:"S0",
description:"desk messy",
completed:true
}
],

edges:[
{
from:"S0",
to:"S1",
action:"move mug away"
cost:1.2
}
]

}

---

# 6. Cost Function

Do NOT use pure distance.

Cost:

C =

alpha * motion_cost

beta * uncertainty

gamma * failure_probability

delta * manipulation_difficulty

Example:

Fragile object:

higher cost

Occluded object:

higher cost

---

# 7. Graph Search

Implement:

Uniform Cost Search

Input:

Reasoning Graph

Output:

next optimal subtask

Algorithm:

initialize priority queue

while queue:

pop minimum cost state

if goal:

return path

expand neighbors

---

# 8. VLA Interface

Output format:

{
instruction:

"Put notebook into drawer",

object_target:

{
source:"notebook_1",
target:"drawer_1"
},

constraints:[
"avoid mug"
]

}

This output will be sent to pi0.5.

---

# 9. Main Execution Loop

while not goal:

capture RGB

update scene graph

update reasoning graph

search next subtask

send instruction to VLA

execute action

check status

repeat

---

# 10. Implementation Priority

Phase 1:

Implement graph data structures.

No LLM.

Use manually defined JSON.

Phase 2:

Implement UCS.

Phase 3:

Connect LLM API.

Phase 4:

Connect VLA policy.

Phase 5:

Real robot experiment.
----------------------

# Coding Requirements

- Use dataclass
- Use NetworkX if helpful
- Every module must have unit tests
- All intermediate graphs must support serialization

---

# 11. 当前实现位置

当前代码实现放在：

```text
high_level_planner/
├── scene_graph/
│   ├── attributes.py
│   ├── constructor.py
│   ├── graph.py
│   └── updater.py
├── reasoning_graph/
│   ├── generator.py
│   ├── graph.py
│   └── state.py
├── planner/
│   ├── cost.py
│   ├── planner.py
│   └── ucs.py
├── interface/
│   └── vla_interface.py
├── memory/
│   └── graph_memory.py
├── config/
│   └── planner.yaml
├── tests/
│   └── test_high_level_planner.py
└── demo.py
```

根目录不再额外维护 `planner/` 兼容包；调用时统一使用
`high_level_planner`。

SAM3 object-level perception 的 Docker sidecar、模型授权、启动检查和 Scene
Graph 调用方式见 [`SAM3_PERCEPTION_GUIDE.md`](SAM3_PERCEPTION_GUIDE.md)。

---

# 12. 根据图片生成 Scene Graph 和 Reasoning Graph

## 12.1 命令行调用

在项目根目录执行：

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim
/usr/bin/python3 -m high_level_planner.demo \
  --image image_collectors/scenes/scene_000.png \
  --instruction "Pick up the notebook from the cluttered desk." \
  --output-dir planner_outputs/scene_000
```

输出：

```text
planner_outputs/scene_000/
├── scene_graph.json
├── reasoning_graph.json
└── vla_request.json
```

`planner_outputs/` 已加入 `.gitignore`，不会作为代码提交。

## 12.2 生成逻辑

`SceneGraphConstructor` 只接受 RGB 图片，并始终通过 SAM3 object-level
perception 生成 Scene Graph。它不读取场景 manifest，也不包含图像启发式 fallback；
SAM3 服务错误会直接上抛。

生成的 `scene_graph.json` 包含：

- `objects`：物体节点，包括 `id`、`name`、`state`、可抓取性、脆弱性、重量、
  位置、尺寸、遮挡状态等属性。
- `relations`：物体关系，包括 `inside`、`overlaps`、`next_to`、`left_of`、
  `right_of`、`in_front_image` 和 `behind_image`。
- `metadata`：SAM3 来源、图片名、分辨率、检测与关系推断记录等。

`ReasoningGraphGenerator` 会根据 scene graph 和自然语言指令生成任务状态图：

- 初始节点 `S0` 和所有后继节点只保存 Simple Scene Graph：对象 ID 列表和
  `{source, target, relation}` 关系；独立的原始 Scene Graph 文件保持不变。
- Reasoning Graph 仅从初始输入提取 ID 和关系，物体属性按需查询
  `image_collectors/object_attributes_with_category.json`，不查询 Real Scene Graph。
- 如果目标被压住或遮挡，会先生成移开 blocker 的子任务。
- Clean Desk 最终节点须满足 GOAL_STATE 全部 14 项规则，且不能再有出边。
- 每条边是只作用于一个 source object 的 atomic-action subtask，带有前置条件、
  效果、动作成本、失败概率、不确定性和操作难度。

例如对 `scene_000.png` 执行：

```bash
/usr/bin/python3 -m high_level_planner.demo \
  --image image_collectors/scenes/scene_000.png \
  --instruction "Pick up the notebook from the cluttered desk." \
  --output-dir planner_outputs/scene_000
```

当前输出的 high-level plan 为：

```text
S0 -> S1: move pen away
S1 -> S2: move cup away
S2 -> S3: pick notebook
```

这是因为 manifest 中存在：

```text
pen on notebook
cup blocks notebook
```

---

# 13. 更新已有 Scene Graph / Reasoning Graph

如果需要在连续观测中保留物体身份，使用 `GraphMemory` 和 `--update`。

第一次观测：

```bash
/usr/bin/python3 -m high_level_planner.demo \
  --image image_collectors/scenes/scene_000.png \
  --instruction "Pick up the notebook from the cluttered desk." \
  --output-dir planner_outputs/scene_000 \
  --memory planner_outputs/scene_000/memory.json
```

后续观测更新：

```bash
/usr/bin/python3 -m high_level_planner.demo \
  --image image_collectors/scenes/scene_001.png \
  --instruction "Pick up the notebook from the cluttered desk." \
  --output-dir planner_outputs/scene_001 \
  --memory planner_outputs/scene_000/memory.json \
  --update
```

更新时会：

- 按 `name + source + model_id` 优先匹配旧物体身份。
- 对缺失但旧图中存在的物体标记为 `not_visible`。
- 更新 object state、relations 和 graph version。
- 重新生成 reasoning graph。

---

# 14. Python API 调用

## 14.1 端到端调用

```python
from high_level_planner import HighLevelPlanner

planner = HighLevelPlanner()
result = planner.process_observation(
    "image_collectors/scenes/scene_000.png",
    "Pick up the notebook from the cluttered desk.",
    update=False,
)

scene_graph = result.scene_graph
reasoning_graph = result.reasoning_graph
next_subtask = result.next_subtask
vla_request = result.vla_request

scene_graph.to_json("planner_outputs/scene_000/scene_graph.json")
reasoning_graph.to_json("planner_outputs/scene_000/reasoning_graph.json")
```

## 14.2 单独构建 Scene Graph

```python
from high_level_planner.scene_graph import SceneGraphConstructor

constructor = SceneGraphConstructor()
scene_graph = constructor.construct(
    "image_collectors/scenes/scene_000.png",
    instruction="Pick up the notebook from the cluttered desk.",
)
```

## 14.3 单独生成 Reasoning Graph

```python
from high_level_planner.reasoning_graph import ReasoningGraphGenerator

generator = ReasoningGraphGenerator()
reasoning_graph = generator.generate(
    scene_graph,
    "Pick up the notebook from the cluttered desk.",
)
```

使用 OpenAI-compatible API 时，在运行时注入 token：

```python
from high_level_planner.reasoning_graph import (
    APIConfig,
    OpenAICompatibleLanguageModel,
    ReasoningGraphGenerator,
)

model = OpenAICompatibleLanguageModel(
    APIConfig(
        model="your-model",
        api_token="your-token",
        base_url="https://your-provider.example/v1",
    )
)
generator = ReasoningGraphGenerator(model=model, beam_width=2, num_branches=2)
reasoning_graph = generator.generate(scene_graph, "整理桌面")
```

也可以不在代码中传 token，改用 `TABLESIM_REASONING_API_TOKEN`。API 只负责生成
候选 action，parser、validator、Scene Graph transition 和 goal evaluator 均在本地。

## 14.4 搜索下一步子任务

```python
from high_level_planner.planner import GraphSearchPlanner

graph_planner = GraphSearchPlanner()
search_result = graph_planner.plan(reasoning_graph)
next_subtask = search_result.next_subtask
```

---

# 15. pi0.5 VLA 调用接口

`Pi05VLAInterface` 会把 high-level subtask 转成 pi0.5 可用的请求。

高层输出格式：

```python
{
    "instruction": "Move pen away from notebook.",
    "object_target": {
        "source": "pen_010_pen_101748",
        "target": "free_space"
    },
    "constraints": [
        "avoid disturbing notebook"
    ],
    "metadata": {
        "action": "move pen away",
        "motion_cost": 1.2,
        "uncertainty": 0.0,
        "failure_probability": 0.178,
        "manipulation_difficulty": 0.65
    }
}
```

构建 remote OpenPI/pi0.5 observation：

```python
import numpy as np
from high_level_planner.interface import Pi05VLAInterface

interface = Pi05VLAInterface()
request = interface.format_subtask(next_subtask)

observation = interface.build_pi05_observation(
    request,
    cam_high=cam_high_rgb,              # H x W x 3
    cam_left_wrist=cam_left_wrist_rgb,  # H x W x 3, 可省略
    cam_right_wrist=cam_right_wrist_rgb,# H x W x 3, 可省略
    state=np.zeros(14, dtype=np.float32),
)
```

生成的 observation 结构为：

```python
{
    "state": state,
    "images": {
        "cam_high": ...,
        "cam_left_wrist": ...,
        "cam_right_wrist": ...
    },
    "prompt": "Move pen away from notebook. Constraints: avoid disturbing notebook."
}
```

如果直接调用本地 `policy/pi05/pi_model.py` 中的 `PI0` 实例：

```python
interface.update_local_pi05_model(
    model,
    request,
    front_rgb=cam_high_rgb,
    left_wrist_rgb=cam_left_wrist_rgb,
    right_wrist_rgb=cam_right_wrist_rgb,
    state=robot_state,
)

action_chunk = model.get_action_chunk()
```

该接口会内部调用：

```python
model.set_language(request.prompt)
model.update_observation_window([front_rgb, right_wrist_rgb, left_wrist_rgb], state)
```

---

# 16. 测试

当前轻量测试不依赖 SAPIEN，只需要 Python 环境中存在 `numpy` 和 `PIL`：

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim
/usr/bin/python3 -m compileall -q high_level_planner
/usr/bin/python3 -m unittest discover -s high_level_planner/tests -v
```

当前已覆盖：

- SAM3 场景图构建与错误传播。
- scene graph identity update。
- reasoning graph 生成，包括 `on` 和 `blocks` blocker 处理。
- UCS 成本搜索。
- pi0.5 payload 格式。
- 端到端 planner 调用。
