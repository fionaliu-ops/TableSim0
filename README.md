# TableSim

Inner Monologue、SayPlan、LERa 的 TableSim 适配基线、统一执行入口和比较协议见
[论文基线迁移说明](baseline/PAPER_BASELINES.md)。这些是论文算法流程的适配实现，尚无任务成功率复现结论。

Reasoning Graph 默认使用 `kimi-k3`，通过指定网关调用；仅需输入 API Key，见
[Kimi 推理 API 配置与试跑命令](REASONING_API_SETUP.md)。

辅助桌面场景图片生成见 [Seedream 独立生成器](high_level_planner/scene_image_generator/README.md)，
配置位于 `config/seedream.json`；仅按需调用，尚未接入验证器。

当前静态架构为 **Image-first 双图条件化规划**：初态照片 → 外部模型生成终态图片 →
SAM3 构建两端图 → Reasoning Graph → UCS → VLA 请求。
目标生成不注入整理规则，旧 graph-first/physical 版本已移除。
使用 `python -m script.generate_reasoning_graph --image ...`，详见
[当前运行说明](high_level_planner/README.md)。动态 image-first 执行已接入统一证据存储、
保守增量修复、执行反馈和显式全量回退。

已有完整环境镜像的服务器迁移步骤见 [Docker 镜像迁移与服务器部署指南](DOCKER_SERVER_DEPLOYMENT.md)，包括镜像文件位置、传输校验、GPU 配置、推理运行和结果持久化。

TableSim 是一个沿用 RoboTwin 任务接口、基于 SAPIEN 与 cuRobo 的桌面双臂仿真和数据采集项目。当前可执行任务为 `stack_plate`（`find_mouse.py` 仍是空占位文件）；通用任务基类、Docker 环境、资产和采集入口均位于本仓库，不依赖 RoboTwin 的项目路径。

## 快速开始

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim
docker compose build tablesim
docker compose up -d tablesim
docker compose exec tablesim python script/check_sim.py --setup
bash collect_data.sh stack_plate stack_plate 0
```

完整的环境准备、仿真检查、新任务约定、数据采集、断点恢复和排错说明见 [TABLESIM_START_GUIDE.md](TABLESIM_START_GUIDE.md)。

High-level planner 的 scene graph / reasoning graph 生成、更新和 pi0.5 payload
调用方式见 [HIGH_LEVEL_PLANNER.md](HIGH_LEVEL_PLANNER.md)。
由 `HighLevelPlanner` 的 VLARequest 驱动、逐步观测并重新规划的仿真入口见
[动态仿真闭环接口](script/DYNAMIC_SIM_README.md)。
各模块职责、端到端数据流和扩展边界见
[HIGH_LEVEL_PLANNER_ARCHITECTURE.md](HIGH_LEVEL_PLANNER_ARCHITECTURE.md)。
SAM3 object-level perception 的隔离部署、权重加载和 Scene Graph 生成方式见
[SAM3_PERCEPTION_GUIDE.md](SAM3_PERCEPTION_GUIDE.md)。

## 主要目录

- `envs/`：通用仿真基类、机器人/相机/工具模块和任务脚本。
- `task_config/`：任务采集配置及机器人、相机配置。
- `assets/`：机器人、物体和背景资产。
- `script/collect_data.py`：Python 数据采集入口。
- `collect_data.sh`：宿主机与容器通用的采集封装。
- `high_level_planner/`：从 RGB 图片和语言指令生成 scene graph、reasoning graph
  和 pi0.5 子任务请求。
- `data/`：采集输出。
- `compose.yaml`、`docker/`：TableSim 独立 Docker 环境。
