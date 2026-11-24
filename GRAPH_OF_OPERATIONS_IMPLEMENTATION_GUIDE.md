# TableSim Graph of Operations 实施指南

> 实现状态：本指南中的核心 Graph of Operations 已接入运行时。默认流程包含 Generate、Validate、Score、KeepBestN、ApplyTransition、AggregateEquivalentStates 和 EvaluateGoal；Improve、候选 Aggregate 与 LLM Score 已实现为可选能力，可通过 `ReasoningGraphGenerator` 的对应开关启用。Operation DAG 可通过 `save_operation_graph(path)` 单独保存。

## 1. 实施目标

在现有 Reasoning Graph 基础上增加 Graph of Operations，使推理流程从 Controller 中硬编码的固定步骤，转变为可配置、可组合的 Operation DAG。

系统中需要同时保留两层图：

```text
Graph of Operations
负责定义“如何推理”

Reasoning Graph
负责记录“推理得到了哪些场景状态和动作路径”
```

推荐的整体架构如下：

```text
Graph of Operations
        │
        │ 执行 Generate / Validate / Score / Select 等操作
        ▼
Thought（运行时推理状态）
        │
        │ ApplyTransition
        ▼
Reasoning Graph
状态节点 --动作边--> 状态节点
```

需要遵守以下约束：

- Reasoning Graph 节点继续使用 Simple Scene Graph。
- Simple Scene Graph 只保存对象 ID 和对象间关系。
- 推理过程不查询 Real Scene Graph。
- 物体静态属性和四类 Zone 按需从 `image_collectors/object_attributes_with_category.json` 查询。
- Graph of Operations 描述推理流程，Reasoning Graph 描述环境状态变化。
- 现有 UCS 路径搜索作为 Reasoning Graph 构造后的独立阶段保留。

---

## 2. 推荐实施顺序

不要一次性重写整个 Reasoning Graph。推荐按以下顺序逐步迁移：

```text
建立 Graph of Operations 基础骨架
→ 用 Operation 包装现有推理逻辑
→ 替换 Controller 调度方式
→ 静态展开多轮 Operation DAG
→ 增加等价状态聚合
→ 增加 Improve 和 LLM Aggregate
```

第一版的目标是改变调度架构，但尽量保持现有推理行为和输出格式不变。

---

## 3. 第一阶段：建立运行时数据模型

### 3.1 新增 Thought

创建文件：

```text
high_level_planner/reasoning_graph/thought.py
```

建议定义：

```python
from dataclasses import dataclass, field
from typing import Any

from .state import SimpleSceneGraph


@dataclass
class Thought:
    thought_id: str
    scene_graph: SimpleSceneGraph
    symbolic_state: dict[str, Any]

    candidate_action: dict[str, Any] | None = None
    reasoning: str | None = None
    score: float | None = None
    valid: bool | None = None
    solved: bool = False

    parent_thought_ids: list[str] = field(default_factory=list)
```

各字段职责如下：

| 字段 | 职责 |
|---|---|
| `thought_id` | 标识一次中间推理结果 |
| `scene_graph` | 可写入 Reasoning Graph 的 Simple Scene Graph |
| `symbolic_state` | 推理期间使用的动态动作状态 |
| `candidate_action` | 当前 Thought 对应的候选动作 |
| `reasoning` | 模型对候选动作的解释 |
| `score` | 候选动作或状态得分 |
| `valid` | 动作是否合法 |
| `solved` | 当前状态是否已经满足目标 |
| `parent_thought_ids` | 推理来源，用于记录分支和聚合关系 |

实现时必须确保：

- 每个候选分支创建新的 Thought。
- Operation 不直接修改输入 Thought。
- `scene_graph` 与 `symbolic_state` 在分支前需要进行安全复制。
- `symbolic_state` 不写入 Simple Scene Graph 节点结构。
- Thought 中不保存 Real Scene Graph。

### 3.2 新增 OperationContext

创建文件：

```text
high_level_planner/reasoning_graph/operation_context.py
```

建议定义：

```python
from dataclasses import dataclass


@dataclass
class OperationContext:
    llm: object
    prompter: object
    parser: object
    attribute_query: object
    transition_model: object
    goal_evaluator: object
    reasoning_graph: object
```

`OperationContext` 用于集中提供 Operation 所需依赖，避免每个 Operation 自行创建模型客户端、属性查询器或状态转移器。

所有物体属性查询都应通过：

```text
context.attribute_query
```

对应属性文件为：

```text
image_collectors/object_attributes_with_category.json
```

属性应当按需查询。例如，处理 `place(cup, box)` 时，只查询 `cup` 和 `box` 所需的 `movable`、`container` 等属性，而不是将完整属性表加入 Prompt。

---

## 4. 第二阶段：实现 Graph of Operations 基础设施

### 4.1 定义 Operation 基类

在以下文件中增加 Operation 抽象及具体实现：

```text
high_level_planner/reasoning_graph/operations.py
```

建议接口：

```python
from abc import ABC, abstractmethod


class Operation(ABC):
    def __init__(self, operation_id: str):
        self.operation_id = operation_id
        self.predecessors = []
        self.successors = []
        self.output_thoughts = []
        self.executed = False

    def is_ready(self) -> bool:
        return (
            not self.executed
            and all(operation.executed for operation in self.predecessors)
        )

    @abstractmethod
    def execute(self, thoughts, context):
        raise NotImplementedError
```

所有 Operation 使用统一的数据接口：

```text
输入：list[Thought]
输出：list[Thought]
```

Operation 不应直接决定下一个 Operation。Operation 间的执行顺序由 GraphOfOperations 的边决定。

### 4.2 实现 GraphOfOperations

创建文件：

```text
high_level_planner/reasoning_graph/graph_of_operations.py
```

建议实现：

```python
class GraphOfOperations:
    def __init__(self):
        self.operations = {}

    def add_operation(self, operation):
        if operation.operation_id in self.operations:
            raise ValueError(
                f"Duplicate operation id: {operation.operation_id}"
            )
        self.operations[operation.operation_id] = operation

    def connect(self, source_id: str, target_id: str):
        source = self.operations[source_id]
        target = self.operations[target_id]

        source.successors.append(target)
        target.predecessors.append(source)

    def roots(self):
        return [
            operation
            for operation in self.operations.values()
            if not operation.predecessors
        ]

    def leaves(self):
        return [
            operation
            for operation in self.operations.values()
            if not operation.successors
        ]
```

第一版只支持有向无环图 DAG，不实现 Operation 循环。多轮推理通过静态展开多个 Round 实现。

---

## 5. 第三阶段：将现有推理逻辑包装为 Operation

第一版建议实现以下七个 Operation：

```text
GenerateActionOperation
ValidateActionOperation
ScoreActionOperation
KeepBestNOperation
ApplyTransitionOperation
AggregateEquivalentStatesOperation
EvaluateGoalOperation
```

### 5.1 GenerateActionOperation

职责：

- 针对每个输入 Thought 单独构造 Prompt。
- 将 Simple Scene Graph、动态符号状态和任务目标传给模型。
- 按需查询候选动作涉及对象的静态属性。
- 调用模型生成多个候选动作。
- 为每个候选动作创建新的 Thought。

数据流：

```text
一个父 Thought
    ↓
多个携带 candidate_action 的子 Thought
```

可以复用现有：

```text
high_level_planner/reasoning_graph/prompter.py
```

模型输出的解析也应封装在该 Operation 内部，或通过 `context.parser` 调用统一解析器。

### 5.2 ValidateActionOperation

职责：

- 检查对象 ID 是否存在。
- 检查动作名称和参数是否合法。
- 查询 `movable`、`container`、`openable` 等必要属性。
- 检查当前符号状态是否满足动作前置条件。
- 设置 `thought.valid`。

第一版可直接过滤无效 Thought：

```python
def execute(self, thoughts, context):
    valid_thoughts = []

    for thought in thoughts:
        thought.valid = self.validate(thought, context)
        if thought.valid:
            valid_thoughts.append(thought)

    return valid_thoughts
```

动作合法性应优先使用确定性规则，不应完全交给 LLM 判断。

### 5.3 ScoreActionOperation

职责：

- 评估动作是否推进目标。
- 评估动作可执行性和安全性。
- 计算动作成本。
- 惩罚重复动作或状态回退。
- 设置 `thought.score`。

第一版建议采用本地确定性评分：

```text
total_score
= goal_progress_score
+ feasibility_score
+ safety_score
- action_cost
- repetition_penalty
```

待基础架构稳定后，再增加 LLM Score Operation。

### 5.4 KeepBestNOperation

职责是在当前整轮 frontier 中执行全局 Top-K：

```python
def execute(self, thoughts, context):
    return sorted(
        thoughts,
        key=lambda thought: (
            thought.score
            if thought.score is not None
            else float("-inf")
        ),
        reverse=True,
    )[:self.keep_n]
```

必须在所有父节点生成的候选之间进行全局选择。不要为每个父节点分别保留 Top-K，否则候选数量仍会随深度快速增长。

### 5.5 ApplyTransitionOperation

该 Operation 复用现有：

```text
high_level_planner/reasoning_graph/transition.py
```

职责：

- 根据 `candidate_action` 生成新的 `symbolic_state`。
- 更新 Simple Scene Graph 中受到动作影响的关系。
- 在 Reasoning Graph 中创建新的状态节点。
- 创建父状态到子状态的动作边。
- 保存新 Thought 与父 Thought 的来源关系。

数据流：

```text
父 Thought
    +
candidate_action
    ↓
新的 symbolic_state
    +
新的 Simple Scene Graph
    ↓
子 Thought
```

每次状态转移都必须基于父状态副本，避免不同候选分支互相修改状态。

### 5.6 AggregateEquivalentStatesOperation

第一版只实现等价状态合并，不调用 LLM。

推荐使用以下信息生成状态标准化键：

```python
def state_key(thought):
    return canonicalize({
        "scene_graph": thought.scene_graph.to_dict(),
        "symbolic_state": thought.symbolic_state,
    })
```

状态标准化时需要：

- 对对象 ID 排序。
- 对关系边按 `source`、`target`、`relation` 排序。
- 对 `symbolic_state` 的字典键排序。
- 移除时间戳、临时 ID 等不影响环境语义的字段。

如果多个 Thought 到达同一个状态：

- Reasoning Graph 中只保留一个状态节点。
- 保留所有父状态到该节点的动作边。
- 选择分数最高的 Thought 继续向后推理。
- 在保留 Thought 中记录所有来源 Thought ID。

不能只比较 Simple Scene Graph，因为仅修改动态属性的动作可能不会改变关系边。

### 5.7 EvaluateGoalOperation

该 Operation 复用现有：

```text
high_level_planner/reasoning_graph/goal.py
```

职责：

- 判断 Thought 是否满足目标。
- 设置 `thought.solved`。
- 将已完成 Thought 加入最终结果集合。
- 将未完成 Thought 传递给下一轮 GenerateActionOperation。

---

## 6. 第四阶段：将 Controller 改为通用 DAG 调度器

修改文件：

```text
high_level_planner/reasoning_graph/controller.py
```

Controller 不再硬编码以下顺序：

```text
生成 → 验证 → 评分 → 剪枝 → 状态转移
```

Controller 只负责：

- 查找已经满足前驱依赖的 Operation。
- 收集所有前驱 Operation 输出的 Thought。
- 调用 Operation。
- 保存输出 Thought。
- 标记 Operation 已执行。
- 收集最终已完成 Thought。

建议实现：

```python
class GraphOfOperationsController:
    def __init__(self, graph, context):
        self.graph = graph
        self.context = context
        self.solved_thoughts = []

    def run(self, initial_thoughts):
        pending = list(self.graph.operations.values())

        while pending:
            ready = [
                operation
                for operation in pending
                if operation.is_ready()
            ]

            if not ready:
                raise RuntimeError(
                    "Operation graph contains a cycle "
                    "or unresolved dependency"
                )

            for operation in ready:
                input_thoughts = self.collect_inputs(
                    operation,
                    initial_thoughts,
                )

                operation.output_thoughts = operation.execute(
                    input_thoughts,
                    self.context,
                )
                operation.executed = True
                pending.remove(operation)

        return self.collect_results()
```

根 Operation 从 `initial_thoughts` 获取输入；其他 Operation 收集全部前驱的 `output_thoughts`。

该阶段的关键验收标准是：

> 改变推理步骤和执行顺序时，只修改 GraphOfOperations 的节点与连接关系，不修改 Controller。

---

## 7. 第五阶段：构建多轮 Operation DAG

修改文件：

```text
high_level_planner/reasoning_graph/generator.py
```

第一版每轮推理结构如下：

```text
Generate
→ Validate
→ Score
→ KeepBestN
→ ApplyTransition
→ AggregateEquivalentStates
→ EvaluateGoal
```

根据 `max_depth` 静态展开：

```text
Round 0:
Generate_0
→ Validate_0
→ Score_0
→ KeepBest_0
→ ApplyTransition_0
→ Aggregate_0
→ Goal_0

Round 1:
Generate_1
→ Validate_1
→ Score_1
→ KeepBest_1
→ ApplyTransition_1
→ Aggregate_1
→ Goal_1

Round 2:
Generate_2
→ Validate_2
→ Score_2
→ KeepBest_2
→ ApplyTransition_2
→ Aggregate_2
→ Goal_2
```

轮次之间连接：

```text
Goal_0 → Generate_1
Goal_1 → Generate_2
```

`EvaluateGoalOperation` 向下一轮只传递 `solved == False` 的 Thought。已完成 Thought 由 Controller 单独收集。

可以通过构建函数生成 DAG：

```python
def build_planning_graph(max_depth: int, beam_width: int):
    graph = GraphOfOperations()
    previous_goal = None

    for depth in range(max_depth):
        generate = GenerateActionOperation(f"generate_{depth}")
        validate = ValidateActionOperation(f"validate_{depth}")
        score = ScoreActionOperation(f"score_{depth}")
        keep_best = KeepBestNOperation(
            f"keep_best_{depth}",
            keep_n=beam_width,
        )
        transition = ApplyTransitionOperation(f"transition_{depth}")
        aggregate = AggregateEquivalentStatesOperation(
            f"aggregate_{depth}"
        )
        goal = EvaluateGoalOperation(f"goal_{depth}")

        operations = [
            generate,
            validate,
            score,
            keep_best,
            transition,
            aggregate,
            goal,
        ]

        for operation in operations:
            graph.add_operation(operation)

        graph.connect(generate.operation_id, validate.operation_id)
        graph.connect(validate.operation_id, score.operation_id)
        graph.connect(score.operation_id, keep_best.operation_id)
        graph.connect(keep_best.operation_id, transition.operation_id)
        graph.connect(transition.operation_id, aggregate.operation_id)
        graph.connect(aggregate.operation_id, goal.operation_id)

        if previous_goal is not None:
            graph.connect(
                previous_goal.operation_id,
                generate.operation_id,
            )

        previous_goal = goal

    return graph
```

---

## 8. 第六阶段：保持现有输出接口

Graph of Operations 执行结束后的处理流程：

```text
Operation Graph
    ↓
生成完整 Reasoning Graph
    ↓
UCS
    ↓
输出最终动作路径
```

现有文件：

```text
high_level_planner/planner/ucs.py
```

可以继续作为后处理路径搜索器，不需要将 UCS 变成 Operation。

建议分别保存：

```text
reasoning_graph.json
operation_graph.json
```

`reasoning_graph.json` 的状态节点继续使用 Simple Scene Graph：

```json
{
  "nodes": ["object_id"],
  "edges": [
    {
      "source": "object_a",
      "target": "object_b",
      "relation": "inside"
    }
  ]
}
```

`operation_graph.json` 用于记录推理流程和执行轨迹：

```json
{
  "operations": [
    {
      "id": "generate_0",
      "type": "GenerateActionOperation",
      "predecessors": [],
      "successors": ["validate_0"],
      "output_thought_ids": ["thought_1", "thought_2"]
    }
  ]
}
```

Operation Graph 的输出不应改变现有 Reasoning Graph 文件格式。

---

## 9. 第七阶段：增加完整 GoT 能力

基础版本运行稳定后，再加入以下能力。

### 9.1 ImproveActionOperation

用途是修复无效或低质量动作：

```text
原动作：place(mug, monitor)
问题：monitor 不是容器
修正：place(mug, tray)
```

为保持 Operation Graph 是 DAG，建议显式展开：

```text
ValidateInitial
→ ImproveInvalid
→ ValidateImproved
```

不要让 `Validate` 和 `Improve` 形成执行环。

### 9.2 AggregateProposalOperation

用于综合多个候选动作的推理信息：

```text
多个候选动作及其分析
    ↓
LLM Aggregate
    ↓
一个新的综合候选或排序判断
```

该操作聚合的是候选信息，不能直接混合两个互不兼容的物理场景状态。

### 9.3 LLMScoreOperation

本地评分无法处理复杂语义目标时，可以加入 LLM 对比评分。建议保留本地评分作为基础分，LLM 只补充语义目标推进程度。

### 9.4 SelectorOperation

Selector 可以根据任务状态决定：

- 哪些 Thought 进入下一轮。
- 哪些 Thought 需要 Improve。
- 哪些 Thought 可以直接终止。
- 是否对某组候选执行 Aggregate。

---

## 10. 文件改造关系

| 文件 | 改造内容 |
|---|---|
| `reasoning_graph/thought.py` | 新增 Thought 运行时状态 |
| `reasoning_graph/operation_context.py` | 新增 Operation 共享依赖 |
| `reasoning_graph/graph_of_operations.py` | 新增 Operation DAG 容器 |
| `reasoning_graph/operations.py` | 实现 Operation 基类和具体操作 |
| `reasoning_graph/controller.py` | 改为通用 DAG 就绪调度器 |
| `reasoning_graph/generator.py` | 构建多轮 Operation Graph 并启动 Controller |
| `reasoning_graph/prompter.py` | 分别提供 Generate、Improve、Aggregate、Score Prompt |
| `reasoning_graph/transition.py` | 由 ApplyTransitionOperation 复用 |
| `reasoning_graph/goal.py` | 由 EvaluateGoalOperation 复用 |
| `reasoning_graph/object_attribute_query.py` | 通过 OperationContext 提供按需属性查询 |
| `reasoning_graph/state.py` | 继续维护 Simple Scene Graph 定义 |
| `planner/ucs.py` | 保持为 Reasoning Graph 构造后的路径搜索器 |

---

## 11. 推荐提交顺序

### 提交 1：基础类型

新增：

```text
thought.py
operation_context.py
graph_of_operations.py
```

在 `operations.py` 中加入 Operation 基类。该提交不改变现有推理行为。

### 提交 2：包装现有操作

实现：

```text
GenerateActionOperation
ValidateActionOperation
ScoreActionOperation
KeepBestNOperation
ApplyTransitionOperation
EvaluateGoalOperation
```

仍使用现有 Controller 执行，用于降低迁移风险。

### 提交 3：通用 Controller

将现有逐轮硬编码 Controller 替换成 DAG 就绪调度器。

### 提交 4：多轮 DAG

在 Generator 中根据 `max_depth` 构建多轮 Operation Graph，确保每个 frontier Thought 都能进入下一轮模型推理。

### 提交 5：状态聚合

实现 `AggregateEquivalentStatesOperation`，支持等价状态去重和多个父动作指向同一个状态节点。

### 提交 6：持久化

增加 `operation_graph.json`，记录 Operation 拓扑、执行状态和 Thought 来源。

### 提交 7：高级 GoT 操作

增加 `ImproveActionOperation`、`AggregateProposalOperation` 和可选的 `LLMScoreOperation`。

---

## 12. 验收标准

满足以下条件后，可以认为已经完成 Graph of Operations 的核心实现：

1. Controller 不再硬编码具体推理步骤。
2. 修改推理流程只需要修改 GraphOfOperations 的连接关系。
3. 每个 Operation 独立接收和输出 `list[Thought]`。
4. 一个 Operation 可以收集多个前驱 Operation 的输出。
5. 每个 frontier Thought 在对应轮次独立调用候选生成逻辑。
6. 存在全局候选评分和 Top-K 剪枝。
7. 存在等价状态聚合，并能够保留多条入边。
8. Reasoning Graph 节点继续使用 Simple Scene Graph。
9. 动态状态保存在运行时 Thought 中，不污染 Simple Scene Graph 格式。
10. 物体属性只从 `object_attributes_with_category.json` 按需查询。
11. 推理过程不反向查询 Real Scene Graph。
12. Operation Graph 和 Reasoning Graph 可以分别保存和检查。
13. UCS 可以继续在生成后的 Reasoning Graph 上提取动作路径。

---

## 13. 最小可行版本

建议首先完成以下最小流程：

```text
GenerateActionOperation
→ ValidateActionOperation
→ ScoreActionOperation
→ KeepBestNOperation
→ ApplyTransitionOperation
→ AggregateEquivalentStatesOperation
→ EvaluateGoalOperation
```

最小版本暂不实现：

- LLM Improve
- LLM Aggregate
- LLM Score
- 动态 Operation 循环
- 复杂并行调度

待最小版本保持现有 Reasoning Graph 输出语义不变后，再逐步加入完整 GoT 能力。

## Region、Zone 与输入隔离约束

- `region` 是 Simple Scene Graph 节点中物体的当前桌面区域，只用于判断该物体是否位于类别要求的 zone。
- zone 映射固定为：`document -> left`、`electronic_device -> center`、`container -> right_upper`、`others -> right_lower`。
- 初始 `region` 只从只读 Real Scene Graph 快照的 bbox 投影一次；后续推理不再查询或更新 Real Scene Graph。
- 每个动作以当前 Simple Scene Graph 为输入，生成新的 Simple Scene Graph 子节点；只允许改变子节点中的 `region` 与 `edges`。
- transition 所需的物体属性按 `object_id` 从 `image_collectors/object_attributes_with_category.json` 查询，临时工作图不包含原始 bbox，也不与原始输入共享对象引用。
- 原始 Real Scene Graph 对象、关系、bbox、metadata 和输入文件在整个 Reasoning Graph 生命周期内保持不变。
