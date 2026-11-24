# scene_003 整理桌面任务（紧凑布局、移除大书）

用户要求移除浅蓝色大书。当前任务只实例化小书、杯子、键盘、鼠标、手机、订书机和固定显示器；其余物体的模型、缩放、质量、材料和初始平面布局不变。移除支撑物后，小书改为从桌面开始，杯子仍在小书上。

入口为 `envs/organize_table_scene003.py`，接触动作实现为 `envs/utils/scene003_physical.py`。
原始动作文件 `planner_outputs/scene_003_kimi_api_unique_goal_20260919/action_flow.json` 保留作为来源。加载后明确删除大书的合拢和叠放步骤，并将杯子的最终支撑改为小书。保留原步骤编号用于追溯，实际执行七步：

1. 把杯子夹到临时空位。
2. 推动小书到目标位置。
3. 夹取并摆好鼠标。
4. 推动键盘到目标位置。
5. 把杯子夹到小书上。
6. 将订书机夹取到物品组中，检查是否闭合；未闭合时通过接触压合。
7. 夹取并摆好手机。

推动使用短行程与实际位置反馈；反向或切换机械臂前先退让。抓取必须检测双指接触、实际抬升，以及搬运中接触保持。每条运动路径由现有规划器求解，另检查实际末端跟踪误差。运行中禁止附着物体到末端、直接设置物体位姿来搬运或直接修改开合角度。

终态根据小书、杯子、键鼠、订书机与手机四组实际状态评分，每组 25 分；还要求显示器固定、双臂回位及夹爪张开。杯子支撑高度使用小书的实际位置计算。

## 测试入口

```bash
python script/test_organize_table_scene003.py
python script/collect_data.py organize_table_scene003 organize_table_scene003_collect2_compact
```

采集配置要求两个成功 episode，并以 `max_seed_attempts: 2` 限制失败尝试。新输出目录为 `data/organize_table_scene003/organize_table_scene003_collect2_compact/`，与旧场景结果分开保存。

本机运行使用 `tablesim-complete:20260914` Docker 镜像、GPU、仓库挂载及 `PYTHONPATH=/workspace/TableSim/compat:/workspace/TableSim`。

`data/organize_table_scene003/physical_review/no_large_book_01/` 保存移除大书后首次完整物理成功的始末图和逐动作接触日志。

## 历史记录

`organize_table_scene003_collect2` 是早期运动学跟随结果，不能作为物理成功证据。`organize_table_scene003_collect2_physical` 和 `physical_review/attempt_*` 是包含大书时的失败诊断。旧场景的合书清障尝试不参与当前七步任务。

## 紧凑布局要求

终态位置由 `COMPACT_LAYOUT` 指定：小书和杯子在显示器左前方，键盘居中，鼠标在键盘右侧，手机在鼠标前方，订书机在小书前方。临时杯子缓冲位不属于终态。原始图片坐标仅保留作为来源，实际终态覆盖值记录在 `execution.plan_adaptation.final_targets` 和 `grounded_stage_targets` 中。

对实际旋转后的物体占用范围计算紧凑度：最小桌边余量至少 0.17、占用外包矩形面积不超过 0.14、物品中心到显示器中心的平均距离不超过 0.26（仿真坐标单位）。这些检查与物体位置、支撑、机械臂回位一起决定任务成功，不放宽原有物体终态容差。
