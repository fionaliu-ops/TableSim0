# TableSim Docker 说明

TableSim 从本机 `robotwin:cuda12.1` 依赖层派生独立镜像
`tablesim:cuda12.1`，使用 Compose 服务 `tablesim` 和缓存卷
`tablesim-cache`。仓库挂载到容器内的 `/workspace/TableSim`，`compat/` 会
通过 `PYTHONPATH` 自动加载。

完整的构建、启动、GPU/SAPIEN 仿真检查和数据采集步骤见 [TABLESIM_START_GUIDE.md](TABLESIM_START_GUIDE.md)。
