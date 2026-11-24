# TableSim MCTS 对照基线

保留启发式 Tidiness MCTS 与目标条件化 MCTS 两种方法，均有代码入口。Budgeted UCS 是
目标条件化 MCTS 的同图搜索预算对照。需要训练权重的学习式 TSMCTS 适配已移除。

## 1. 方法边界

| 方法 | 输入 | 价值 / 提议 | 输出 |
|---|---|---|---|
| [`TidinessMCTSPlanner`](https://arxiv.org/html/2502.17235)（TSMCTS 方法启发） | 当前观测 SceneGraph、标定工作区 | 几何启发式 / 网格动作 | 单个 Subtask + VLARequest |
| `GoalConditionedMCTSPlanner` | 与 UCS 完全相同的 ReasoningGraph | 节点目标满足比例 / 相同有效出边 | 原有 SearchResult |

启发式整理方法是无显式终态的：会去除缓存观测中的 task_goals/task_policy，不能输入
终态图来暗中引导整理。指令用于记录；启发式不解析任意自然语言约束。目标条件化方法
在完整 HighLevelPlanner 中注入时，初态图、终态图和 Reasoning Graph 仍由共享流程生成。

这两种方法属于不同实验条件。不要将无目标方法在指定布局任务上的失败解释为搜索器更差。

## 2. 共用 MCTS 与目标条件化对照

`mcts.py` 实现 UCT、按 proposal 权重进行扩展和 rollout、价值/rollout 混合回传，以及
随机种子、搜索深度、转移评估次数和时间预算。默认 lambda=0.3；这些默认参数未在
TableSim 测试集上调优。循环仅按当前搜索路径排除，搜索树不会写回原始观测/ReasoningGraph。

`GoalConditionedMCTSPlanner` 与 UCS 使用相同的边有效性检查和 `CostFunction`。
叶节点价值为已有 goal_checks 中满足条件的比例；没有 checks 时为 0；terminal 为 1。
回传值为 `(1-lambda)*leaf_value + lambda*rollout_terminal - cost_weight*path_cost`。
在预算内发现的所有目标路径中返回代价最小者，包括 rollout 找到的路径；并不保证全局最优。
没有发现目标路径时返回 `goal_state=None, path=[], total_cost=inf`，不会执行未完成的推测路径。
原始当前节点已是目标时返回空路径、零代价和当前节点 ID。

```python
from baseline import GoalConditionedMCTSPlanner, MCTSConfig
from high_level_planner import HighLevelPlanner

searcher = GoalConditionedMCTSPlanner(config=MCTSConfig(
    iterations=256, max_depth=12, rollout_depth=4,
    max_transitions=4096, time_limit_seconds=2., seed=0))
planner = HighLevelPlanner(graph_planner=searcher)
# 正常使用 planner.plan_initial_image / update_target_conditioned_image 等接口。
# searcher.metrics 记录最后一轮搜索指标；实验 runner 每轮保存它。
```

离线公平比较同一张图：

```bash
python -m baseline.mcts_demo --mode compare \
  --reasoning-graph baseline/fixtures/shared_reasoning_graph.json \
  --max-transitions 4096 --seeds 0 1 2 3 4 \
  --output planner_outputs/baseline/mcts/paired_search.json
```

CLI 同时运行 `BudgetedUCSPlanner` 和 MCTS，交替执行顺序，并保存输入图 SHA256。
一次转移评估是检查/应用一条有效边；MCTS 重复检查同一条边也计数。分别报告时间和
转移次数，不能声称一次 MCTS iteration 等于一次 UCS 节点扩展。MCTS 的有限深度、
iteration cap 是额外参数，必须同时披露。若使用时间预算，超时只在调用边界检查，不能
中断正在执行的模型推理或候选生成。图构建、感知和真实执行耗时不包含在这个离线对照中。
完整系统对照需另计共享的图生成和模型成本，使用同一闭环重规划频率。

`fixtures/` 是人为构造的小例子，不是 TableSim 实验结果或论文数据。

## 3. 启发式整理版

仅支持桌面平面内、可见的刚体移动。每个对象必须有稳定 ID、pose、footprint 和
geometry_frame，并显式提供观测/标定得到的 workspace bounds；不把整张图的边缘当桌边。
支撑/包含任务、手中恢复、堆叠和关节操作超出这个基线的范围。旋转的图像 AABB 采用保守
包围框；不会把二维碰撞检查当作真实机器人轨迹证书。

默认评分是 alignment 0.5、compactness 0.25、grouping 0.25 的加权平均。alignment 衡量
朝向是否接近坐标轴；compactness 衡量物体中心包围范围相对工作区对角线的大小；grouping
只使用显式 category，相同有效类别有多个对象时才计算，不使用 others/unknown。
缺失 grouping 项时重新归一化其他权重；缺失朝向不会得到完美 alignment 分数。
当前物体重叠或越界时评分为 0。它没有学习语义或个体偏好，不等同于人类整洁度。

候选为工作区内部网格落点，默认 0/90 度旋转；对所有物体的候选均匀抽取最多 64 条，
避免只保留第一个物体的候选。共享几何/技能可行性检查剔除已知非法动作，未知轨迹检查
仍交给同一低层执行器。存在明确 incoming on/blocks/occludes/attached_to 的源对象不移动。

```bash
python -m baseline.mcts_demo --mode heuristic \
  --scene-graph baseline/fixtures/planar_observation.json \
  --iterations 64 --max-transitions 256 \
  --output planner_outputs/baseline/mcts/heuristic.json
```

`threshold=0.85` 是待校准配置，并非该手工分数的人类验证阈值。`done` 仅表示当前观测
评分达到阈值且当前布局未越界/重叠；预测未来达到阈值不会当作当前完成。
缺少可行动作/预算不足返回 blocked。

Python 逐步接口：

```python
from baseline import TidinessMCTSPlanner

planner = TidinessMCTSPlanner()
output = planner.plan_initial_scene(initial_observed_graph, "整理桌面")
while output.status == "action":
    feedback = execute(output.vla_request)  # 复用同一个低层执行器
    planner.record_execution_result(output.next_subtask,
        success=feedback["success"], details=feedback.get("details", {}))
    output = planner.update_from_scene(observe_scene_graph(), "整理桌面")
```

图像入口需要注入共享 SceneGraphConstructor/SceneGraphUpdater，分别调用
plan_initial_image/update_from_image；多轮身份维护由共享 updater 负责。
失败动作在观测几何/身份/关系未变时不重复；这只是精确动作抑制，不是证据驱动局部修复。
fatal 反馈立即阻止后续规划。新任务显式调用 plan_initial_scene/plan_initial_image 重置。

`mcts_runner.run_tidiness_episode` 提供统一 observe/execute/evaluate 回调循环。外部评价器
必须独立评价当前物理状态；若 scorer 判 done 但外部未成功，记录 scorer_false_positive。
该 runner 支持现有执行器输出消息，不要求复制 dynamic_sim 中依赖 ReasoningGraph 的主循环。

## 4. 参考文献与范围

- 启发式 Tidiness MCTS 的方法灵感来自 TSMCTS：[原论文](https://arxiv.org/html/2502.17235)。本实现是 TableSim 启发式适配，不使用学习模型，也不是作者实现的复现。
- Goal-conditioned MCTS 与 Budgeted UCS 是本项目共享 Reasoning Graph 上的搜索实现/对照，不对应 TSMCTS 的学习式整理方法，也没有声称复现某篇论文。

原先的学习式 TSMCTS 适配及其数据准备、训练、导出和模型加载代码已移除，因为它需要项目当前未提供的训练数据与有效权重。
