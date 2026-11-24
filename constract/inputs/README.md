# 固定实验输入

- `initial_scene_graph.json`：复制自 `planner_outputs/scene_003_kimi_api_unique_goal_20260919/inputs/initial_reviewed.json`。
- `terminal_scene_graph.json`：复制自同目录 `terminal_reviewed.json`。

这些是之前已校正、已确认的图，包括书本身份对应和支撑关系；本次不重读图片、不重新生成终态。
每次运行用相同两份图为两组编译同一份 `confirmed_goal_constraints.json`。
完整源文件保持原样，运行时仅去除旧 task policy / planning context，再使用当前代码进行公共投影。
