# 对比基线

项目保留 6 个无需训练专用模型权重的 baseline。Full Replan 归入本文下方的 Ablation 部分。MCTS 的 Budgeted UCS 仅作为同图搜索预算对照，不单独计为基线。原 TSMCTS 学习适配版需要训练数据和模型权重，已从可运行入口中移除。

| 方法 | 原参考文献 |
|---|---|
| Direct VLM | 本项目自定义对照，无对应原始论文 |
| 启发式 Tidiness MCTS | [TSMCTS 原论文](https://arxiv.org/html/2502.17235)，仅作方法启发 |
| Goal-conditioned MCTS / Budgeted UCS | 本项目搜索实现与对照，无对应原始论文 |
| Inner Monologue | [原论文](https://arxiv.org/abs/2207.05608) · [项目说明](https://innermonologue.github.io/) |
| SayPlan | [原论文](https://proceedings.mlr.press/v229/rana23a.html) · [项目说明](https://sayplan.github.io/) |
| LERa | [原论文](https://arxiv.org/abs/2507.05135) · [作者实现](https://github.com/AmpiroMax/LERa) |

以上论文方法均为 TableSim 适配版，不代表作者系统的逐项复现或成功率复现。

## Baseline 1：Direct VLM Planner

`DirectVLMPlanner` 每轮直接从当前 RGB、任务指令和最近执行历史选择一个动作。
它不运行 SAM3，不构造视觉 Scene Graph 或 Reasoning Graph，不调用目标生成器、
本地状态转移、几何可行性推理、增量图修复或 UCS。复用的 SkillSpec 只用于满足
现有执行消息的格式契约；没有使用它评估观测或筛选可行动作。

```text
当前 RGB + 指令 + 执行历史 [+ 固定目标图片 / 物体 ID 清单]
    → 一次 VLM 决策 → Subtask → VLARequest → 相同低层执行器
    → 执行结果 + 新 RGB → 下一次 VLM 决策
```

### 接口与使用

```python
from baseline import DirectVLMPlanner
from high_level_planner.reasoning_graph.language_models.openai_compatible import APIConfig

planner = DirectVLMPlanner(api_config=APIConfig.from_env())
output = planner.plan_initial_image("initial.png", "整理桌面")
while output.status == "action":
    # 调用相同低层执行器；execute/observe 由实验 runner 提供。
    success, details = execute(output.vla_request)
    planner.record_execution_result(output.next_subtask, success=success, details=details)
    if success:
        planner.mark_subtask_completed(output.next_subtask)
    output = planner.update_from_image(observe(), "整理桌面")
```

输入支持路径、PIL.Image 和 uint8 H×W×3 RGB 数组，与高层入口一致。
输出复用真实的 `Subtask` / `VLARequest`，包含相同的 `object_target`、
`constraints` 和 `metadata.reasoning_edge`。可直接交给 `Pi05VLAInterface`
或 `script.dynamic_sim.request_action()` 解码。

`plan_initial_image()` 开始新任务；`process_observation(update=False)` 也会重置。
`process_observation()`、`update_from_image()` 和 `update_target_conditioned_image()`
每次产生一个新决策。请求新决策前需要记录上一动作的执行结果；致命执行失败和
默认 50 次决策预算耗尽返回 `blocked`。

输出状态为 `action`、`done`、`blocked`；`done` 是 VLM 自评，必须使用共同的外部
评估器确认任务成功。无效模型回复抛出 `LanguageModelError`，不会伪装成完成。
序列化结果中的 `scene_graph`、`reasoning_graph`、`search_result` 为 null。
这不是完整 `PlannerOutput` 的替代品：现有 `script.dynamic_sim` 主循环依赖图和
图搜索状态，不能直接替换其 planner。应使用上面的逐步调用流程连接共享执行器。

### 单步 CLI

```bash
python -m baseline.demo --image /path/to/desk.png \
  --instruction "整理桌面" --config config/goal_generator.local.json \
  --output-dir planner_outputs/baseline/direct_vlm
```

配置复用 `reasoning` 节，支持 `url` / `base_url`、`model`、`api_token` 及 APIConfig
字段；环境变量沿用 `TABLESIM_REASONING_API_TOKEN`。CLI 只输出一次决策，不执行动作。
`--target-image` 提供固定目标图片，`--object-catalog` 提供 `{物体ID: 可见描述}` JSON。
目标图片在会话开始时固定，不会由基线自动生成。输出目录已被仓库忽略，代码目录可提交。

### 实验口径

为隔离图表示、推理与搜索的贡献，建议两方法使用相同模型、API 参数、任务集合、
初始场景、执行器、物体 ID 清单、动作预算和外部成功判据。目标图片条件也应一致：
若高层方法使用终态布局图，可把同一图片通过 `target_image` 交给基线；仅文字任务的
Direct VLM 作为单独条件报告。ID 清单只提供执行命名，不含位置、关系或隐藏仿真状态。
图像坐标必须由共享执行器标定，不能直接当作世界位姿。

记录任务成功率、执行动作数、执行失败率、模型请求与 token 数、决策耗时、总耗时，
并区分模型自评完成与外部验证完成。这些对比用于检验方法收益，不预设基线表现。
`planner.metrics` 和输出 `metadata.metrics` 提供当前会话的决策数、执行数、执行失败数、
模型调用耗时、请求数和 token 计数；使用的模型适配器需要提供后面三项计数属性，
默认 API 适配器已提供。执行耗时和外部成功率由共同实验 runner 记录。

依赖沿用 `high_level_planner/requirements.txt`，要求 Python 3.10+。

运行离线接口测试：`python -m unittest discover -s baseline/tests -v`。

## Ablation

| 方法 | 消融内容 | 原参考文献 |
|---|---|---|
| Full Replan | 移除推理图的增量复用，每轮完整重建并重新搜索 | 本项目消融方法，无对应原始论文 |

### Full Replan Planner（每轮观测后全量重规划）

`FullReplanPlanner` 继承现有 `HighLevelPlanner`，返回完整的 `PlannerOutput`。
首次 image-first 规划与原方法相同；终态图在任务开始时固定。后续每轮观测使用同一
SceneGraphUpdater 更新场景与实例身份，写入相同的 EvidenceStore，然后无条件调用
完整 `generate()` 重建 Reasoning Graph，从新图根节点运行 UCS。它不调用调度器
选择修复范围，也不调用 `update_reasoning_graph()`，不保留上一轮推理图的节点、边、
执行游标或增量计算结果。即使事实没有变化，仍全量重规划。

```text
首次 RGB + 指令 → 初态感知 + 终态生成 → 固定目标图
每轮新 RGB → SceneGraphUpdater → EvidenceStore
    → 完整构建 Reasoning Graph（同一固定目标）→ 从新根节点 UCS
    → 下一条 Subtask / VLARequest → 执行反馈 → 下一轮观测
```

保留动作前置条件、几何检查、状态转移、代价函数、执行失败证据和致命失败停止。
感知的身份维护和证据账本仍可跨轮保存；消融的是未来推理图的增量复用。
完整重建沿用原方法 `_full_target_rebuild()` 的语义、目标评价器和失败动作抑制，
因此可以把实验差异限定在“按需修复”与“每轮完整重建”。

```python
from baseline import FullReplanPlanner

planner = FullReplanPlanner(
    scene_constructor=shared_scene_constructor,
    reasoning_generator=shared_reasoning_generator,
    graph_planner=shared_graph_search_planner,
)
output = planner.plan_initial_image(
    "initial.png", "整理桌面", goal_generator=shared_goal_generator,
)
# 若实验已有共同的初态图和目标图，可用以下入口替代上面的首次规划：
# output = planner.plan_initial_scene(initial_graph, target_graph, "整理桌面")

while output.next_subtask is not None:
    success, details = execute(output.vla_request)
    planner.record_execution_result(output.next_subtask, success=success, details=details)
    if success:
        planner.mark_subtask_completed(output.next_subtask)
    output = planner.update_target_conditioned_image(observe(), "整理桌面")
```

`process_observation(update=True)`、`update_from_image()`、
`update_target_conditioned_image()` 都进入全量重规划；
`update_target_conditioned_scene()` 可输入缓存的当前观测图。
`plan_initial_image()`、`plan_initial_scene()` 开始新任务并重置统计。
终态不会随观测重新生成；修改任务目标应显式开始新任务。

无下一条子任务时，必须检查 `search_result.goal_state`：为 None 表示没找到路径；
等于 `start_state` 才表示当前观测已经满足规划目标。搜索找到未来目标节点只代表
存在预测路径，物理任务完成仍由统一外部评估器确认。

每轮 `reasoning_graph.metadata.baseline` 保存模式、理由、节点/边数量、证据版本、
规划耗时和轮次历史；`planner.metrics` 汇总规划轮数、全量重规划次数及耗时。
后续轮次的规划计时从已获得当前 Scene Graph 开始，首次图片入口包含初态感知和
目标生成，所以比较在线推理开销应使用后续轮次的 `planning_seconds`；相机、感知、
执行及完整系统耗时由实验 runner 统一记录。模型调用和 token 统计沿用每轮图中
`metadata.model_execution.stages`。

#### 缓存观测重放 CLI

```bash
python -m baseline.full_replan_demo \
  --scene-graph /path/to/initial_scene_graph.json \
  --target-scene-graph /path/to/target_scene_graph.json \
  --observations /path/to/observation_001.json /path/to/observation_002.json \
  --instruction "整理桌面" \
  --output-dir planner_outputs/baseline/full_replan
```

模型设置使用 `--model` / `--api-base` 和现有 `TABLESIM_REASONING_API_TOKEN` 等环境变量。
此入口只重放给定观测并保存每轮 `PlannerOutput`，不执行动作，也不推断历史动作是否成功。
真实闭环实验用上面的 Python 接口回传实际执行结果。两种方法应采用相同模型、
目标、感知、搜索预算、执行器和成功判据，比较成功率、每轮模型调用/token、
推理开销、动作次数和失败恢复行为。
