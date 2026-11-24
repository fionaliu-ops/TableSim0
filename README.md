# TableSim

Inner Monologue、SayPlan、LERa 的 TableSim 适配基线、统一执行入口和比较协议见
[论文基线迁移说明](baseline/PAPER_BASELINES.md)。这些是论文算法流程的适配实现，尚无任务成功率复现结论。

已有完整环境镜像的服务器迁移步骤见 [Docker 镜像迁移与服务器部署指南](DOCKER_SERVER_DEPLOYMENT.md)，包括镜像文件位置、传输校验、GPU 配置、推理运行和结果持久化。

TableSim 是一个沿用 RoboTwin 任务接口、基于 SAPIEN 与 cuRobo 的桌面双臂仿真和数据采集项目。当前可执行任务为 `stack_plate`（`find_mouse.py` 仍是空占位文件）；本仓库包含通用任务基类、Docker 配置、资产元数据和采集入口。运行前请按资产目录的说明准备模型文件。

## 快速开始

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim
docker compose build tablesim
docker compose up -d tablesim
docker compose exec tablesim python script/check_sim.py --setup
bash collect_data.sh stack_plate stack_plate 0
```

完整的环境准备、仿真检查、新任务约定、数据采集、断点恢复和排错说明见 [TABLESIM_START_GUIDE.md](TABLESIM_START_GUIDE.md)。

高层规划目录仅包含包初始化文件。运行依赖高层规划的基线和动态仿真脚本时，需要完整的规划模块。

## 主要目录

- `envs/`：通用仿真基类、机器人/相机/工具模块和任务脚本。
- `task_config/`：任务采集配置及机器人、相机配置。
- `assets/`：机器人、物体和背景资产。
- `script/collect_data.py`：Python 数据采集入口。
- `collect_data.sh`：宿主机与容器通用的采集封装。
- `data/`：采集输出。
- `compose.yaml`、`docker/`：TableSim 独立 Docker 环境。
