# Kimi 推理 API 配置

辅助图片生成使用独立的 [Seedream 生成器](high_level_planner/scene_image_generator/README.md)
和 `config/seedream.json`，不影响本文的 Kimi 推理配置。

Scene Graph → Reasoning Graph 的默认模型已设为 `kimi-k3`，默认 base URL 为
`https://apicz.boyuerichdata.com/v1`。默认 `auto` 协议会发送到
`https://apicz.boyuerichdata.com/v1/chat/completions`，不会强制走 Responses。
目标关系提议和动作提议共用这个推理模型，推理架构不变。

## API Key 输入位置

Key 由运行 Python 的进程从环境变量 `TABLESIM_REASONING_API_TOKEN` 读取，
不需要填写到 Python 源码。在 Linux / Ubuntu 容器内、项目根目录执行：

```bash
read -rsp '请输入该网关的 API Key: ' TABLESIM_REASONING_API_TOKEN
echo
export TABLESIM_REASONING_API_TOKEN
```

提示出现后粘贴 Key 并回车，输入不会回显，也不会将 Key 本身写进 shell 命令历史。
必须在**同一个终端 / 容器会话**运行后续命令；新的会话需要重新输入。
如果在宿主机设置变量而用 `docker compose exec` 运行，需要显式传入
`-e TABLESIM_REASONING_API_TOKEN`，宿主机变量不会自动进入容器。
项目没有自动读取 `.env` 的逻辑，仅编辑 `.env` 不会生效。
不要把真实 Key 提交到 Git，也不要将 Key 放进 `--api-token` 命令行参数。

## 用已有 Scene Graph 快速运行

在已经装好项目 Python 依赖的环境中执行：

```bash
python -m script.generate_reasoning_graph \
  --scene-graph high_level_planner/reasoning_graph/scene_003_scene_graph.json \
  --instruction "Clean the desk." \
  --max-actions 3 --beam-width 3 --num-branches 3 \
  --output-dir "planner_outputs/scene_003_kimi_$(date +%Y%m%d_%H%M%S)"
```

不需要再指定模型或 URL。这里将最大动作深度设为 3，适合小规模试跑，
不保证在 3 步内完成整理；完整运行可增大 `--max-actions`。
结果目录中的 `run_summary.json` 会记录模型、协议和实际请求地址；
`ucs_path_output/optimal_path.json` 保存路径诊断输出。

如果此前设置过旧配置，可先清除覆盖项以使用新的默认值：

```bash
unset TABLESIM_REASONING_MODEL TABLESIM_REASONING_API_BASE
```

配置优先级为：显式 CLI 参数 > 环境变量 > 代码默认值。
仍支持 `--model`、`--endpoint` 和 `--api-mode` 覆盖。
`python -m high_level_planner.demo` 和直接使用默认 `ReasoningGraphGenerator`
也采用同样的模型与 URL 默认值。

## 配置代码与限制

- 默认值及 Key 读取：`high_level_planner/reasoning_graph/language_models/openai_compatible.py`。
- 快捷推理入口：`script/generate_reasoning_graph.py`。
- `--reasoning-effort` 仅对 Responses 请求发送，不是 Kimi 思考模式开关。
- 本次仅配置文本/Scene Graph 推理。图片生成接口及模型保持独立配置，
  不代表 `kimi-k3` 支持图像输入或后继图片生成；上述 JSON 输入试跑不需要图片 API。
- 未提供 Key 时无法确认该账号是否有 `kimi-k3` 权限，也无法验证实际推理质量。
  代码使用用户指定的模型 ID，不自动替换模型或退回规则推理。
  HTTP 401 通常需检查 Key，模型不可用需检查网关模型列表、账号权限和服务端报错。
