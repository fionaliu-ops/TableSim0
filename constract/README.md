# kimi-k3：同一 Scene Graph 目标下的两种规划方式

本实验的推理输入是**初态 scene graph 和终态 scene graph**，不是图片。
两组都实际调用外部 `kimi-k3`，使用相同的输入、目标约束、技能词表、动作规则、模型参数、
状态投影、动作校验、状态转移、终态检查和代价函数。

- `direct`：一次 API 请求输出完整有序动作列表；随后只做离线验证，不反馈、不补动作、不修复。
- `graph`：使用现有 `ReasoningGraphGenerator` 逐状态调用模型生成候选动作，验证、打分、合并状态，最后 UCS 选路。
- 两者均不使用规则后备模型，不调用图像生成或 SAM，不把旧模型结果当作新实验结果。
- 两组允许相同的最大计划长度，但总 API 调用/token 预算不同，报告会公开资源消耗。

## 已确认的输入与目标

`inputs/` 复制自 `planner_outputs/scene_003_kimi_api_unique_goal_20260919/inputs/` 的
`initial_reviewed.json`、`terminal_reviewed.json`，保留完整属性和几何侧信息。
已确认目标包括浅蓝书合上并叠到深蓝书上、杯子在浅蓝书上、订书机闭合，以及各物体的位置和目标关系。
位置容差为归一化图像坐标 `0.04`。
这些目标在实验中是固定约束，不再从图片重新解释；没有添加此前图片分析提出、但 scene graph 未确认的键盘转角。
每次运行会保存目标谓词、完整目标图和输入哈希。可以用 `--initial`、`--terminal` 指定另一对已确认图。

历史 `model_reasoning/` 是 Astra 直接读图的旧记录，不能更改其来源后作为本实验基线。
新的直接推理基线是本目录 `direct` 流程，其输入明确为 scene graph。

## 运行

在仓库根目录使用安装了 `high_level_planner/requirements.txt` 依赖的 Python：

```powershell
python -m constract.compare --config config/goal_generator.local.json
```

配置文件格式见 `config/goal_generator.example.json`。只读取 `reasoning` 配置；不要把密钥提交到 git。
也可以在启动进程之前设置 `TABLESIM_REASONING_API_TOKEN`，然后不带 `--config` 运行。
默认 endpoint 使用项目 `APIConfig` 中的 kimi 服务地址，可通过配置中的 `reasoning.url` 覆盖。
模型固定为 `kimi-k3`；两组同用 temperature、reasoning_effort 和单次输出上限。

如果没有项目 Python 环境，可运行：

```powershell
uv run --cache-dir .cache/uv --with numpy --with pillow --with opencv-python-headless --with scipy python -m constract.compare --config config/goal_generator.local.json
```

只准备公共输入和两份初始提示词、不调用 API：

```powershell
python -m constract.compare --prepare-only
```

默认每次新建 `constract/runs/<时间戳>/`，拒绝覆盖已有运行目录。
可用 `--output-dir constract/runs/my_run` 指定新目录。
默认参数：`--max-actions 16 --max-states 128 --beam-width 3 --num-branches 3 --max-output-tokens 8192`。
每次执行始终保留两个方法的成功或失败，不根据结果修改输入，也不自动重试直到成功。
重复运行可检查模型随机性，但单次结果不能证明统计上的普遍优势。

## 查看结果

批量 constrained-v2 输入使用 `run_scenes_constrained_v2.py`。当前改动尚未运行测试：

- 终态由 `compact_layout.py` 按占用矩形与归一化间隔 0.015 排布，优先靠近后方固定设备；
  固定物体不移动，支撑物体居中放在承载物上。不存在有效布局时，在调用模型前报错。
- 这是 manifest 模拟坐标的平面投影基准，不是照片中的桌面边界检测；保留初始 yaw，
  仅转换规划用的中心及包围框，不修改仿真物体参数。平面布局检查不能证明动作路径可达。
- 不指定预算时，动作上限为 `max(16, 3*可移动物体数 + 2*on关系数 + 4)`，
  状态上限为 `max(128, 4*分支数*动作上限)`；小于场景基本需求的显式预算会被拒绝。
- 默认串行、单次请求上限300秒、最大输出8192 tokens。每场景保存 `planning_budget.json`，
  批次保存 `batch_config.json`。不带 `--run` 仅准备输入；只有显式加 `--run` 才调用模型。
- 支撑位置携带真实的 support 对象 ID；direct 可读取未来支持面动作模板，
  graph 仍须选择当前状态的 grounded candidate。所有动作继续使用原来的验证门槛。
- 初态和目标有别于 v6 的几何换算和固定槽位方案，后续结果应作为新批次保存，不能覆盖旧结果。

- `index.html` / `comparison.md`：并排指标、逐步动作和逐项目标检查。
- `comparison.json`、`manifest.json`：程序可读对比、输入哈希、共享参数和代码版本。
- `inputs/confirmed_goal_constraints.json`：两组共同使用的目标约束。
- `direct/actions.json`、`graph/actions.json`：模型完整计划与 UCS 选出的计划。
- 各组 `replay.json`：每步前置条件、grounding、技能效果、代价、剩余目标、失败位置。
- 各组 `states/S*.json`：有效动作前缀产生的预测状态。
- 各组 `model_audit/`：真实请求、可见回答、返回模型、用量和耗时；不保存认证头。
- `graph/reasoning_graph.json`、`graph/ucs_path_output/`：完整搜索图和最优路径。

`prepared_not_run` / `missing_credentials` 表示尚未调用模型，不能视为实验结论。
`goal_reached=true` 仅表示全部动作通过公共校验且预测终态满足共同目标；不是物理执行成功。
不可达性、碰撞、接触稳定性等 unknown 检查保留在结果中。
搜索失败时保留完整图和失败状态，不把空路径说成成功。

## 离线测试

```powershell
uv run --cache-dir .cache/uv --with numpy --with pillow --with opencv-python-headless --with scipy --with pytest python -m pytest constract/tests -q
```

测试使用固定历史动作夹具或内存回答检查验证器，不会调用外部模型；不作为新 API 实验结果输出。
