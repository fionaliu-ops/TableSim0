# 论文算法迁移基线

这组实现将论文的决策流程迁移到 TableSim 的观测、对象 ID、动作技能和执行器接口。
它们是 **TableSim 适配版**，不是作者原始系统的逐项复现，也不代表复现了论文成功率。
没有调用本项目的 reasoning graph、UCS 搜索或局部修复器来替代论文方法。

## 算法对应关系

| 方法 | 保留的算法流程 | 当前迁移范围与差异 |
| --- | --- | --- |
| [Inner Monologue](https://arxiv.org/abs/2207.05608) | 每次执行后，把场景描述、上一动作和成功检测写入文本历史；语言模型逐步选择下一技能 | 采用 Object + Scene + Success 反馈变体。共享感知器把相机观测转换为文本；规划模型不看 RGB。没有主动问答、人类反馈或原机器人技能策略；提示示例按本项目重新编写 |
| [SayPlan](https://proceedings.mlr.press/v229/rana23a.html) | 折叠场景层级 → expand/contract 语义搜索 → 生成完整计划 → 场景图模拟器检查 → 根据错误迭代修改 → 执行 | 按论文 Algorithm 1 改为桌面支撑/容纳层级，保留已展开节点记忆。没有楼层、房间或移动机器人导航；底层运动交给公共执行器。成功执行时不重新调用模型；执行失败后重新观察、搜索与规划属于此次适配的在线扩展 |
| [LERa](https://arxiv.org/abs/2507.05135) | 执行现有计划；失败触发 Look → Explain → Replan；成功直接推进计划 | Look 接收当前 RGB，Explain/Replan 接收文本、剩余计划与已成功动作。初始计划由独立外部规划步骤生成，或显式传入。没有把本项目控制器的丰富失败原因直接送进 LERa |

LERa 还核对了[作者公开实现](https://github.com/AmpiroMax/LERa)的
`llserver/llserver/models/lera_api/LLERa_API.py`，版本为
`e5d52f243ec970ffd9f5c7343be7b40b63315da5`：只有 Look 阶段使用图像，后两阶段使用文本。
本实现重新编写了动作接口和提示词，没有复制作者代码。
Inner Monologue 同时参考[项目说明](https://innermonologue.github.io/)，SayPlan 同时参考[项目说明](https://sayplan.github.io/)。

## 共同接口和判断边界

- 三种规划器均输出现有 `Subtask` / `VLARequest`，动作沿用 move/place/stack/insert/align/reorient/open/close/attach。
- 对象 ID 必须来自当前任务对象目录。动作解析拒绝未知 ID、非法坐标、非有限数字、空计划和超长计划；必须先反馈当前动作结果才能请求下一步。
- 三种方法使用同一固定目标图的文本事实作为目标说明。LERa 外部初始规划还可看到目标图像。这是显式目标条件下的规划/恢复比较，不能据此证明从“整齐”生成目标的能力。
- 当前场景事实只从共享感知结果投影，过滤无关内部元数据。SayPlan 的模拟只检查图层面的访问性、技能前提和状态转移；不声称证明完整机器人轨迹可行。
- SayPlan 默认仅验证计划可执行性，未额外给它一个精确目标判定器。代码中的可选 `verify_goal_in_simulator=True` 是实验扩展，使用时应单独注明。
- 统一运行器以新观测和控制器返回检查动作结果，以独立 `TargetGraphEvaluator` 检查任务完成。模型声称完成但外部判据不满足，记录为 `false_completion`。
- 默认动作检查器要求源对象在新帧中被观察并且身份明确。完全遮蔽的容器内放置可能被保守判为失败；如果替换检查器，三种方法必须使用同一版本。
- 公共安全恢复可以尝试释放残留抓持；安全故障或恢复失败会停止任务。丰富控制器诊断保留在日志中，不直接作为规划器的额外语义提示。

## 运行实时仿真

在项目已有 SAPIEN/cuRobo、SAM3 服务和模型 API 环境中，从项目根目录运行：

```bash
python -m script.run_paper_baseline \
  --method inner_monologue \
  --scene scene_003 --task-config scene_003 \
  --target-scene-graph /absolute/path/target_scene_graph.json \
  --goal-image /absolute/path/goal.png \
  --api-config /absolute/path/api_config.json \
  --sam3-endpoint http://127.0.0.1:8003 \
  --seed 0 --max-steps 30 --max-model-calls 200 \
  --output-dir output/paper_baselines/inner_monologue/seed_0
```

分别把 `--method` 改为 `sayplan`、`lera`，并使用新的输出目录。
API 配置沿用项目 `load_api_settings` / `dynamic_reasoning_config`，详见根目录 `REASONING_API_SETUP.md`。
目标图和目标图像应来自同一次目标生成，且对象 ID 与实时初始感知对应。
若目标图包含 `derived_from_image_sha256`，入口会校验图像哈希。

本入口复用 `RGBDAdapter`、`RGBDGraspExecutor`、SAM3、视觉标签复核、对象跟踪和录像。
**该执行器沿用项目的理想接触选择/物体跟随机制；它不等同于真实抓取动力学或真机技能策略。**
运行日志因此保留 `robot_execution_verified=False`，即使外部观测已满足目标也不能解释为真机实验成功。
实机实验需要替换共同执行回调，并重新验证技能和传感器标定。

主要输出：

- `run_manifest.json`：种子、目标哈希、模型配置、预算、执行器边界。
- `episode.json`：独立成功判定、动作序列、控制器与动作检查结果、规划器调用量、时间、停止原因。
- `method_audit/`、`model_audit/`：方法阶段、提示词、响应、模型错误及耗时。
- `label_model_audit/`：共享视觉标签复核的调用，独立于规划模型调用统计。
- `observed_scene_*.json`：实际相机观测构建的状态；另保留底层执行证据和录像。

## 缓存观测与反馈回放

无仿真环境时，可以检查规划和恢复流程，但仍需要模型 API：

```bash
python -m baseline.paper_demo \
  --method lera \
  --scene-graph /absolute/path/initial_scene_graph.json \
  --target-scene-graph /absolute/path/target_scene_graph.json \
  --image /absolute/path/initial.png \
  --config /absolute/path/api_config.json \
  --replay /absolute/path/replay.json \
  --output-dir output/paper_baselines/replay_lera
```

回放 JSON 是列表，每项对应上一决策的真实反馈与随后观测：

```json
[
  {"success": false, "image": "after_failure.png", "scene_graph": "after_failure.json"},
  {"success": true, "image": "after_recovery.png", "scene_graph": "after_recovery.json"}
]
```

观测路径相对于回放文件目录，或者使用绝对路径。可附加 `details` 和 `fatal`。
回放不生成假执行状态；`replay_result.json` 不能当作仿真成功率证据。

## 两种比较协议

1. **完整流程比较**：不传入 `--initial-plan`。IM 每步选技能，SayPlan 搜索并模拟整段计划，LERa 先使用外部初始规划器再进行失败恢复。需要报告 LERa 初始规划器的质量和成本，不能把它当作 LERa 自身贡献。
2. **固定初始计划的恢复比较**：三种方法都传入同一 `--initial-plan /absolute/path/actions.json`。LERa 执行该计划；IM 按共同前缀执行直到首次失败，之后使用文本反馈选技能；SayPlan 会先检查该计划的可执行性，必要时提前修正。因此该协议包括 SayPlan 的失败预防差异，不能宣称三者经历了完全相同的首次失败。

初始计划是 ACTION 列表，不能直接传 reasoning graph：

```json
[
  {
    "action_type": "move",
    "source_object_id": "object_1",
    "target": "table",
    "parameters": {"pose": {"frame": "image_normalized", "xy": [0.7, 0.6], "support": "table"}}
  }
]
```

把示例 ID 替换为实际对象 ID。技能的输入约束见 `paper_common.ACTION_FORMAT`。
程序调用可以直接实例化规划器，再使用 `run_paper_episode`：
`observe() -> PaperObservation`、`execute(VLARequest) -> {success, details, fatal}`、
`evaluate(PaperObservation) -> {success: bool, ...}`；可提供共同 `check_action`。

公平比较必须固定初态、目标、种子、对象身份映射、感知/执行器、成功阈值、模型及采样参数、动作与调用预算。
方法原生输入通道不同，应明确报告，不能声称完全相同信息条件。
规划器的 `model_calls` 不包含共享感知模型调用，统计总成本时还须合并 `label_model_audit`。
应分别报告目标达成率、首次失败后恢复率、额外动作数、失败/安全停止次数、调用数、token、决策耗时；不要把单元测试通过率当成任务成功率。

## 本次验证

使用无真实模型的可控响应和场景夹具检查了方法调用顺序、失败恢复、动作格式、模拟错误反馈、独立完成判据、硬预算和审计落盘。
全部 `baseline/tests` 共 64 项，62 项通过，2 项因可选 PyTorch 未安装跳过。
未运行真实 API 的整段仿真或真机实验，尚不能声称达到论文指标或优于当前方法。

```bash
python -m unittest discover -s baseline/tests -v
python -m script.run_paper_baseline --help
python -m baseline.paper_demo --help
```
