# 闭环自动评估

`script.dynamic_sim.run_closed_loop`（包括 Full Replan 传入同一接口的运行）、旧版 `script.direct_dynamic_sim`、MCTS callback runner 和论文基线 runner 在任务结束时生成评估结果。成功、动作上限、无路径和异常结束均保留报告；评估器不以规划器的 `done` 或预测节点作为成功证据。

每个运行目录新增：

- `evaluation_metrics.json`：完整机器可读指标、分母、逐物体误差及测量范围。
- `evaluation_metrics.md`：简洁结果表。

主仿真入口与论文基线入口同时打印评估结果或报告路径。callback runner 将相同内容放在返回值的 `evaluation_metrics` 中；MCTS runner 传入 `output_dir` 才写入文件。

## 指标口径

| 字段 | 定义 |
|---|---|
| `task_success` / `task_success_int` | 新观测满足固定目标全部验收条件、身份可用、物体已释放且没有致命执行事件。异常结束按失败记录。 |
| `goal_progress.initial_fraction` / `final_fraction` | 初态/终态满足的固定目标条件数除以固定条件总数。目标条件来自目标 pose、yaw、关节状态和目标关系；身份与可见性验收不会额外作为进度奖励重复加分。 |
| `goal_progress.net_change` | 终态满足率减初态满足率；允许为负。 |
| `initially_unsatisfied_completion` | 初始未满足、最终满足的条件数除以初始未满足条件数；分母为零时为 null。 |
| `newly_violated_ids` | 初始满足、最终不满足的条件，用来暴露抵消净进展的退步。 |
| `layout.object_pose_match_rate` | 在目标位置/支撑容差内的物体数除以目标物体数。朝向另外通过 yaw 条件及误差报告。 |
| `layout.target_relation_satisfaction` | 目标关系的满足比例；无关系目标时为 null。 |
| `layout.object_errors` | 每个对象的位置与可观测朝向误差，保留坐标系和单位。不同坐标系不混合求均值。朝向遵循当前目标评估器的 360° 有向约定。 |
| `preservation.newly_misplaced_object_rate` | 初始 pose 已正确而最终 pose 错误的物体数 / 初始 pose 已正确的物体数。 |
| `missing_or_unobserved_ids` | 当前未被可靠观测的目标物体；不能解释为物理丢失。 |
| `attempted_actions` / `motion_completed` / `execution_failures` | 实际执行接口的尝试、完成及失败次数。模型输出被拒绝但尚未调用执行器不计动作尝试。 |
| `actions_with_postcondition_check` / `postcondition_verified_actions` | 有明确后置条件反馈的动作数及验证成功数，与控制器完成数分开记录。 |
| `total_seconds` / `execution_seconds` / `phase_seconds` | 评估运行的墙钟耗时、执行接口耗时以及可测的规划/感知阶段耗时。 |
| `model_usage` | 新增或更新 API audit 中的请求尝试（包括重试）、完成次数、已返回的 token 用量和覆盖情况。 |
| `disturbance_recovery` | 显式扰动后的最终固定任务是否完成、扰动后动作数和剩余运行耗时；普通执行失败单独报告。 |

固定目标条件满足率为：

$$Q_t=\frac{1}{K}\sum_{k=1}^{K}\mathbf{1}[g_k(s_t)],\qquad \Delta Q=Q_T-Q_0.$$

身份/可见性是完成的验收门槛；不可用观测不能借用历史位置充当成功。任务内目标变化不会替换自动评估的初始目标，`target_sha256` 和 `target_revision_count` 提供审计信息。

相似度采用对象布局与关系一致性，未加入未经任务验证的整图 SSIM/CLIP 总分。原始完整验收检查仍保存在观测快照逻辑中，进度分母不直接采用全部验收字段数量。某些目标条件存在语义关联，满足率不代表独立子任务的统计比例。

## 数据来源和未测量项

- 主闭环与有 `target_scene` 的 callback runner 采用目标图评估器，读取观测 SceneGraph，绝不读取 Reasoning Graph 的预测后继充当观测。
- 旧 Direct 脚本沿用其原有独立仿真验收、图像/位置/倾斜容差及 Scene 003 执行覆盖策略；报告标记 `legacy_direct_dynamic_sim` 和 `evaluation_source`，不能与纯相机评估不加说明地混合。
- 不提供 `target_scene` 时，callback 的 `evaluate()` 可返回固定的 `goal_checks: dict[str, bool]`；缺少这些条件则进度为 null。callback 必须提供独立 `success`；有条件失败时不能用 `success=True` 覆盖。有固定目标图时还需提供 `payload_released`。
- 物体掉落、物理丢失及真实接触安全没有独立传感器证书时为 null/未认证；不能把漏检当掉落，也不能把无记录当作零事故。
- 无显式扰动时，恢复率为 null。通过 `env.inject_event` 执行的主闭环事件自动计入；callback 可通过执行反馈的 `disturbance` 字段记录已发生的事件。未触发的计划扰动不属于已执行的扰动实验。
- `remaining_episode_seconds` 是扰动到运行结束的时间，不宣称隔离了单个事件的恢复耗时。额外动作的因果增量需要同场景无扰动对照，目前该字段为 null。
- `token_usage_complete=False` 表示 token 总数只是已报告请求的合计；无 audit 时相关用量为 null。计量范围只包含该评估器启动后新增/更新的 audit。
- 阶段计时可能包含嵌套区间，例如初始规划流程包含感知、目标生成和搜索，不能与各阶段直接相加。`initialization` 还包括初态与目标准备。总耗时截至评估报告生成前，不包含关闭环境与视频编码。

## 调用与批量汇总

主闭环原有命令无需新增参数。MCTS callback 使用：

```python
run_tidiness_episode(planner, observe=observe, execute=execute, evaluate=evaluate,
                     target_scene=target, output_dir=run_dir)
```

论文基线物理入口自动传入目标图；它的 callback API 同样接受 `target_scene` 和 `position_tolerance`。

```bash
python -m script.summarize_episode_metrics planner_outputs/my_experiment \
  --output planner_outputs/my_experiment/automatic_summary.json
```

汇总按方法分别报告 episode 成功率、Wilson 95% 区间、平均终态目标满足率、全部运行的平均动作/耗时与发生扰动的 episode 成功率。缺少终态观测的运行保留为失败，并单独报告测量覆盖。区间假设 episode 独立；同场景多种子或重复运行应进一步按场景进行聚类统计。工具不会自动验证不同运行的模型、预算或验收来源是否可比。

## 参考依据

- [Visual Room Rearrangement / 官方评估](https://github.com/allenai/ai2thor-rearrangement#-evaluation)：严格成功、部分完成、新增错放与状态误差。
- [BEHAVIOR 官方评估](https://behavior.stanford.edu/challenge/evaluation.html)：目标条件满足比例和运动效率。
- [Habitat Rearrangement](https://aihabitat.org/challenge/rearrange_2022/)：任务成功、连续状态进展与耗时。
- [Agarwal et al., NeurIPS 2021](https://agarwl.github.io/rliable/)：统计不确定性与可靠评估原则；本工具采用的 Wilson 区间并非该论文指定的唯一方案。

扰动恢复字段是为本项目动态重规划主张设计的指标，未声称直接复制上述基准。人工整齐度与人工偏好不在自动评估中生成。
