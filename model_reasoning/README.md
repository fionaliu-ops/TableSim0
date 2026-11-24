# scene_003 中间动作推导

> 历史读图实验，不是当前 scene graph 对比基线。最新对比已改为初态 scene graph＋终态 scene graph，
> 两组都调用外部 kimi-k3，分别直接生成完整计划和执行 graph 搜索，入口见 [constract](../constract/README.md)。
> 本目录保留原始实验来源，不将旧读图结果重新标注为 scene graph 推理结果。

当前助手根据初始照片和现有 kimi 终态布局图直接制定的候选方案。没有调用外部模型，没有注入额外 rules。

- [动作说明、图像依据与不确定性](scene_003_action_process.md)
- [结构化动作、依赖与 S00–S09 预测状态](scene_003_action_plan.json)
- [初始照片](inputs/initial.png) / [终态布局图](inputs/terminal_layout.png)
- [结构与状态更新检查结果](verification.json)

核心顺序：杯子移至空区 → 移开深蓝书 → 合上浅蓝书 → 浅蓝书叠到深蓝书上 → 确认支撑后放回杯子。键鼠另行调整，手机和订书机仅在相对摆放不符时移动。

杯子的精确支撑关系无法由这张终态示意图唯一确定；计划明确保留视觉确认环节。所有坐标都是图像区域参考，不需要真实尺寸，也不能直接作为机器人位姿。

本目录记录可观察证据、简要动作理由与完整候选动作流程，不包含私有内部思维链。尚未进行物理执行。

复核：在仓库根目录运行 `python3 model_reasoning/verify_plan.py`。
