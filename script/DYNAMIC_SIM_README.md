# 动态仿真闭环接口

`script/dynamic_sim.py` 在现有 TableSim 场景上实现 `reset / observe / step / evaluate`。
每轮从相机获取 RGB，经 `HighLevelPlanner` 产生初态、终态 Scene Graph 和 Reasoning Graph，
执行 UCS 选出的下一条 `VLARequest`。执行结果先写入 `EvidenceStore`，重新观测后保持
当前版本终态图，优先依据证据变化增量重放、重验和修复旧图；纯视觉理想夹持模式下，
目的地受阻或不可达也会触发外部模型重新生成终态布局图（详见文末）。依赖契约缺失、指令变化、
变化范围过大或局部修复无有效路径时，才记录原因并回退完整重建。
只有**新的观测图**满足终态谓词，且新的 Reasoning Graph 将当前起点认证为终态时，
运行结果才是 `certified_terminal`。预测路径到达目标不算执行成功。

默认场景为 `scene_003`。运行入口使用场景的 `setup_demo`、相机、物体、机器人与
`take_dense_action`，不会调用硬编码的 `play_once`。`organize_table_scene003` 自身在
`setup_demo` 中强制加载固定动作链，目前明确拒绝作为这个动态入口的场景；其基础
布局可以用 `scene_003`。其他场景需具备 `objects`、`OBJECT_SPECS`、`_top_down_pose`
及图中对象到仿真对象的可建立映射。

动态入口在首次 SAM3 观测后调用外部模型校正标签，再创建 Scene Graph 的 ID、
静态类别属性和关系。后续帧先通过稳定跟踪 ID、检测框重叠和局部图像外观进行
一对一身份匹配，继承已校正标签；可匹配的移动或遮挡后重现物体不触发模型。
出现无法可靠匹配的候选新物体时，才将这些候选交给外部模型校正，并追加节点。
不单看检测总数，避免将暂时漏检后恢复的旧物体当作新物体。
模型输入仅包含 RGB、编号检测框、局部放大图及通用标签词表，
不提供仿真资产清单、位姿或尺寸。`label_review/` 保存图像与校正审计，
`label_model_audit/` 保存模型调用记录。低置信度、未确定身份或非法响应会停止规划，
不会使用仿真真值猜测标签。SceneGraphUpdater 保留已有身份及漏检节点，允许新增节点。
跨帧外观匹配存在歧义时仍可能触发复核，不保证所有移动和遮挡情况下都能免调用。

经校正的图仅按语义名称或显式绑定表连接执行器，不使用真值几何补齐身份；无法唯一
绑定的可动物体在动作开始前报错。执行器内部仍可使用仿真状态完成控制和相机坐标
标定，但实际位姿与放置误差只保留在诊断文件中，不传入规划器执行反馈。
模型可见的反馈详情只包含观测路径、证据来源和错误类型。已有目标图必须与校正后的
对象 ID 一致；旧的错误标签目标图不会被静默套用。

在项目 GPU 容器内启动 SAM3，然后运行：

API 配置默认读取仓库内的 `config/goal_generator.local.json`，不受启动目录影响。
在该文件的 `reasoning.api_token` 和 `image.api_token` 中分别填写令牌即可，
以后无需在终端输入或导出令牌。两项使用同一服务密钥时可填写相同值。
该本地文件已被 Git 忽略；新环境可复制 `config/goal_generator.example.json` 后填写。
`--api-config` 可以显式指定其他配置文件；默认文件不存在时保留环境变量方式。

```bash
docker compose --profile sam3 up -d sam3
docker compose exec tablesim python -m script.dynamic_sim \
  --scene scene_003 --task-config scene_003 \
  --instruction '整理桌面' --sam3-endpoint http://sam3:8003 \
  --output-dir planner_outputs/dynamic_scene003 --max-steps 20
```

可传 `--initial-scene-graph` 和 `--target-scene-graph` 复用已有两张图，跳过目标图片生成；
这两个参数必须同时使用。可传 `--object-map mapping.json` 指定 `{场景图对象ID: 仿真对象名}`。
自动映射只接受唯一对象名称。未映射的动作对象会报错，不会猜测或移动其他物体。

使用 `--events events.json` 可在指定高层动作之后注入物体移动，再获取新的相机观测并
重新规划。例如：

```json
[
  {"after_step": 2, "type": "move_object", "object_id": "cup_sam3", "delta_xy": [0.08, -0.04]}
]
```

`after_step` 从 1 开始，指“第几条 VLARequest 执行完毕后”。扰动记录进入 trace。

执行器读取 `VLARequest.metadata.reasoning_edge` 的动作类型、源对象和位姿。
对 `pick`，机械臂先到物体上方、闭合夹爪，然后附着物体；对 `move/place/stack`
等放置动作，物体随末端移动，夹爪张开后解除附着。
`--graph-validation` 的搬运同样要求实际完成接近、抓取姿态和闭合夹爪，
随后按抓取瞬间的末端到物体变换连续跟随。释放使用实际到达位姿，不能将
物体直接设置到指令目标。任一运动失败会保留实际状态（包括尚未释放的抓取），
停止本轮并记录 `graph_validation_execution_failed`；不回滚物体或机器人位置。
该模式使用辅助附着验证高层动作，不代表真实摩擦抓取已通过验证。
叠加 `--hybrid-execution` 后，执行器按当前物体尺寸、夹爪开度、动态刚体能力
筛选顶部夹取候选。可尝试的对象先进行双指接触和小幅试抬；只有实际接触、
物体随试抬升高均通过时，才记为 `physical_contact`，此时不启用物体附着。
候选不可用或接触试验失败时，重新接近当前位置、闭爪，再启用 `assisted_follow`。
这是有边界的顶部夹取测试，不是穷举所有侧向夹取方式。控制器使用关节目标插值、
停稳和分段位姿命令；不直接设置关节状态。该模式启用双臂路径和运行时间距检查。

`--execution-control-dir <目录>` 在每个高层动作前写入 `request_NNN.json`，等待
本地监督者写入同编号 `decision_NNN.json`。例如
`{"execute": true, "grasp_mode": "auto", "arm": "right", "reason": "检查当前视觉后执行"}`。
允许选择执行臂和抓取模式，不允许在该接口重写模型目标或动作顺序；未收到决策
不执行。运行记录保留决策、`hybrid_execution.jsonl`、视觉观测及完整双视角视频。
重复观测使用独立修订文件名，失败截图不覆盖动作前图像。

当任务明确允许物品自行移动以隔离高层规划验证时，可使用
`--graph-validation --object-motion-validation`（与 `--hybrid-execution` 互斥）。
该模式从实时视觉重新调用外部模型生成到 GOAL 的路径，保持其动作顺序和图像目标，
在低层使用相机标定、物体高度和支撑状态生成连续抬升／平移／下降轨迹，绕过机器人 IK。
物品按每个仿真步的位置更新移动，速度有界；视频记录整个过程。它不是实际夹爪搬运
或接触物理验证，录像 manifest 中 `has_robot_control_steps=false`。

为保证留下完整动作链录像，视觉定位不确定不会提前截断此模式；原始错误和节点匹配
结果全部保留，不将预测状态冒充视觉观测。仅在所有节点匹配且最终视觉目标检查通过时
报告 `object_motion_visually_verified`，否则报告
`object_motion_video_complete_needs_visual_review`。后者表示视频完整，不表示高层闭环已通过。
桌面目标坐标由初态图中至少三个非共线且身份匹配的物体拟合到仿真 XY。
拟合残差超过 6.5 cm 会拒绝执行。此映射只适用于当前相机和场景，不是通用相机标定。
支持目标默认对准目标物体中心；需要精确相对位姿的操作应进一步添加几何求解器。
`open/close` 仅支持有实际关节的物体；其余缺少执行器的技能会明确失败。

每轮保存 `observation_*.png`、`scene_graph_*.json`、`reasoning_graph_*.json`、
`evidence_store.json` 和 `dynamic_trace.json`。trace 中记录每次 repair decision、变化事实、
复用/失效边、执行证据和全量回退原因。异常也写入 trace。运行结果区分 `certified_terminal`、
`no_executable_path`、`step_limit` 和 `execution_or_replan_error`。

当前 image-first 路径保存持久化的 target-conditioned generator，并调用统一的证据条件化
修复入口。依赖覆盖仍是 `conservative_full_planning_context`，所以这是保守正确性优先的
闭环版本，还不能声称运行时最小依赖或最少重算。该运行器可用于 `move_object` 扰动下的
基础闭环验证；主动换视角、更多扰动类型和自动消融批量 runner 仍需扩展。

机械臂象征性抓取验证使用 `--graph-validation --symbolic-grasp-validation`，
与 `--hybrid-execution`、`--object-motion-validation` 互斥。
机械臂通过实际关节驱动执行接近、下探、闭爪、抬升、搬运、下降、张爪和撤离。
仅当实际指端到达抓取点且闭爪完成后才建立理想抓持约束；物品位置跟随实际指端，
朝向随实际腕部转动，张爪后解除约束。执行失败时保留现场，不独立移动物品或回滚位姿。
该模式允许低层读取仿真位姿和相机标定求解 IK，外部高层仍使用视觉图及目标图。
为隔离抓持摩擦问题，物品采用运动学刚体并关闭物品碰撞；因此验证的是可见抓取动作
与规划动作链，不是接触物理抓取成功率。双臂路径预检和运行时间距检查仍启用。
`symbolic_grasp.jsonl` 记录实际到达误差、闭爪后附着和张爪后释放事件；完整双视角
视频和节点视觉匹配保留在输出目录。`symbolic_grasp_video_complete_needs_visual_review`
只说明动作链录像完成，不代表视觉终态验证通过。

象征性抓取模式现使用 `script/symbolic_feedback.py` 的单步反馈闭环，
不再通过 `run_video_path` 盲目执行完整旧路径。每步执行后由实时 RGB 定位图节点；
匹配时复用剩余图，失败或视觉后置条件不满足时，记录执行证据并调用持久化的
目标条件规划器修复／重建 reasoning graph。遮挡反馈包含失败阶段、源物体与遮挡物 ID，
不向高层注入仿真世界坐标。遮挡物发生视觉位置变化后的候选节点允许重新抓取，
执行前仍需重新检查。观测或重规划失败时停止，不假定旧节点已经完成。
搬运途中失败且仍持物时，尝试由机械臂沿已执行路径返回抓取点、张爪释放，再观测重规划；
无法安全返回则保留持物状态并停止后续抓取。真实双臂碰撞／运行时间距违规仍为硬停止。

### RGB-D 视觉执行模式

`--visual-only-execution` 配合 `--graph-validation --symbolic-grasp-validation`，使用现有头部相机 RGB/深度/标定进行抓取与放置定位。不得同时提供 `--object-map` 或物品扰动事件。该路径封锁物品真实位姿/几何绑定接口，不使用旧执行器的位姿回滚或仿真成功判定。

```bash
PYTHONPATH="$PWD:$PWD/envs/curobo/src" \
/home/ubuntu/miniconda3/envs/RoboTwin/bin/python -u script/dynamic_sim.py \
  --scene scene_003 --task-config scene_003 \
  --goal-image dynamic_test_output/scene_003_assistant_goal/terminal_image_v2.png \
  --output-dir planner_outputs/my_fresh_rgbd_run \
  --graph-validation --symbolic-grasp-validation --visual-only-execution \
  --max-steps 24
```

令牌仍从本地 API 配置读取。每次运行使用新的输出目录，初始图与 reasoning graph 均重新构建。视频为 `operation_head_camera.mp4`、`operation_front_camera.mp4`；执行反馈在 `dynamic_trace.json`，传感器数据在同名观测 `.png/.npz`，接口拦截记录在 `truth_access_audit.json`。

此模式用于隔离验证高层规划：物品在闭合接触前保持静止，接触后的理想跟随由物理刚体速度积分实现，释放不写目标位姿。原生物理接触用于选择被夹持刚体，未使用语义资产映射。搬运时关闭载荷碰撞响应，因此视频不构成真实摩擦夹持或完整载荷碰撞安全的证明。终态必须由新观测核验。

## Feedback-driven terminal layout revisions

In `--visual-only-execution --symbolic-grasp-validation` runs, a supplied
`--goal-image` specifies the initial desired layout. Destination occupancy,
obstruction, trajectory rejection or motion failure can trigger a new external
KIMI layout illustration using the current camera frame, observed scene node,
failure details and preceding goal image. Failed temporary buffers only repair
the route. Grasp occlusion first repairs ordering, unless its blockers are already
at their terminal poses. No target is revised while recovery still holds an object.

SAM regions in each new picture are bound to identities by the external model;
positions are measured from those image regions. A model point outside its
selected region triggers a visual reread; unresolved bindings are rejected.
Each candidate is read back
into a target scene graph and checked for inventory
and original support/containment relationships. A successful revision builds a
new Reasoning Graph from the current observation while retaining execution
evidence. Failed candidates leave the active target unchanged. All revisions
(up to three activated revisions and nine candidate attempts per run) are retained
in `goal_revisions/`; the trace
identifies the active version. Generated pictures are planning targets, never
proof of execution. Fresh observed images must satisfy the active target.
