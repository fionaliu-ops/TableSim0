
# TableSim 资产新增与 Isaac Sim 场景搭建指南

本文档说明四件事：

1. 如何在 `assets/objects/` 中新增可被 TableSim/SAPIEN 正常加载、可被 clutter
   随机桌面流程发现的物品资产。
2. 如何用 `image_collectors/messy_desk.json` 和现有资产生成
   `image_collectors/scenes/` 下的随机桌面场景图片与 manifest。
3. 如何从 `image_collectors/scenes/manifest.json` 中选取一组场景参数，构建新的
   `envs/<task>.py` 任务，并运行 `collect_data` 生成数据。
4. 如何把指定资产、桌面、相机、灯光和物理属性组装成 Isaac Sim 场景，并导出
   或复用为 USD 场景。

本文档中的命令默认在项目根目录执行：

```bash
cd /home/ubuntu/Desktop/Liutong/TableSim
```

容器内项目路径为：

```bash
/workspace/TableSim
```

## 1. 当前资产体系

`assets/objects/` 里目前主要有三类资产。

### 1.1 编号 GLB 刚体资产

这是新增普通桌面物品的推荐格式。典型结构如下：

```text
assets/objects/121_paper/
├── collision/
│   └── base0.glb
├── visual/
│   └── base0.glb
├── model_data0.json
└── points_info.json
```

如果同一类别有多个模型变体，则编号对应扩展：

```text
assets/objects/001_bottle/
├── collision/base0.glb
├── collision/base1.glb
├── visual/base0.glb
├── visual/base1.glb
├── model_data0.json
├── model_data1.json
└── points_info.json
```

加载路径：

- `envs.utils.create_actor.create_actor(...)`
- `envs.utils.rand_create_cluttered_actor.get_all_cluttered_objects()`
- `script/render_messy_desk_objects.py`
- `script/scence_generator.py`

### 1.2 编号 URDF 资产

部分物品来自 SAPIEN/PartNet 风格数据，结构通常如下：

```text
assets/objects/010_pen/
├── 101685/
│   ├── mobility.urdf
│   ├── model_data.json
│   ├── textured_objs/
│   └── ...
└── points_info.json
```

加载路径：

- `envs.utils.create_actor.create_sapien_urdf_obj(...)`

如果物体有关节、抽屉、盖子、门等结构，优先使用 URDF 资产；如果只是桌面静态
杂物，优先使用 GLB 刚体资产。

### 1.3 Objaverse clutter 资产

Objaverse 资产集中在：

```text
assets/objects/objaverse/
├── list.json
└── <category>/<model_id>/model.urdf
```

加载路径：

- `envs.utils.rand_create_cluttered_actor.create_cluttered_urdf_obj(...)`

这类资产需要维护 `objaverse/list.json` 中的 `list_of_items`、`radius`、
`z_offset` 和 `z_max`。普通新增物品不建议优先走这条路径，除非资产来源本身已经
整理为 Objaverse URDF。

## 2. 新增普通物品的推荐流程

推荐新增为编号 GLB 刚体资产。当前最大编号为 `135_magazine`，所以下一个普通
新增资产建议从 `136_<object-name>` 开始。目录名必须满足：

```text
三位数字_英文小写名称
```

示例：

```text
136_usb-drive
137_mouse-pad
138_envelope
```

命名建议：

- 使用英文小写。
- 多词用 `-` 连接，例如 `paper-clip`。
- 不要用空格。
- 不要随意改已有目录名；已有脚本和 JSON 可能引用这些目录。

### 2.1 准备视觉模型

视觉模型放在：

```text
assets/objects/<NNN_name>/visual/base0.glb
```

要求：

- 单位使用米。
- 模型大小接近真实桌面物体。
- 材质应能被 GLTF/SAPIEN 渲染器读取，优先使用 PBR base color 或纹理。
- 可以有较多细节，但不应过度复杂，否则批量渲染和仿真会变慢。

TableSim 编号 GLB 资产沿用 RoboTwin 的局部坐标约定：局部 `Y` 轴为竖直方向。
创建到 SAPIEN 世界时，clutter 流程会使用近似如下旋转，把局部 Y-up 转为世界
Z-up：

```python
qpos = [0.707107, 0.707107, 0.0, 0.0]
```

因此，如果你用 Blender 或 trimesh 建模，建议让物体在局部坐标中：

- `X`：左右宽度
- `Y`：高度
- `Z`：前后深度

### 2.2 准备碰撞模型

碰撞模型放在：

```text
assets/objects/<NNN_name>/collision/base0.glb
```

要求：

- 必须闭合、有限、没有 NaN/Inf 顶点。
- 尽量低多边形。
- 对薄片、中空、细线结构使用简化实体碰撞。
- 对 `create_actor(..., convex=True)` 友好。

常见建模取舍：

| 物品         | 视觉模型                   | 碰撞模型              |
| ------------ | -------------------------- | --------------------- |
| 纸张、贴纸   | 极薄平面或薄盒，有印刷细节 | 稍微加厚的闭合盒体    |
| 眼镜、回形针 | 管状/环状结构              | 几个盒体或胶囊近似    |
| 剪刀         | 两片刀刃和指环             | 刀刃盒体 + 指环外接盒 |
| 杯子         | 中空杯壁                   | 简化圆柱或多凸包      |

### 2.3 新增 `model_data0.json`

`model_data0.json` 是 TableSim 自动发现和任务逻辑使用的核心配置。最小可用结构：

```json
{
    "center": [0.0, 0.00075, 0.0],
    "extents": [0.21, 0.0015, 0.297],
    "scale": [1.0, 1.0, 1.0],
    "transform_matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ],
    "target_pose": [
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0015],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    ],
    "contact_points_pose": [],
    "functional_matrix": [],
    "orientation_point": [],
    "contact_points_group": [],
    "contact_points_mask": [],
    "target_point_description": ["Center point on the top surface."],
    "contact_points_description": [],
    "functional_point_description": [],
    "orientation_point_description": [],
    "stable": true
}
```

字段说明：

| 字段                    | 作用                                              |
| ----------------------- | ------------------------------------------------- |
| `center`              | 碰撞模型局部包围盒中心，未乘`scale`             |
| `extents`             | 碰撞模型局部包围盒尺寸，未乘`scale`             |
| `scale`               | 加载视觉和碰撞模型时使用的统一缩放                |
| `transform_matrix`    | 资产局部补偿变换；普通刚体可用单位矩阵            |
| `target_pose`         | 任务可用目标点，通常给顶面中心                    |
| `contact_points_pose` | 抓取接触点；杂物资产可先留空                      |
| `functional_matrix`   | 功能点；普通杂物可先留空                          |
| `orientation_point`   | 方向语义点；普通杂物可先留空                      |
| `stable`              | 必须为`true`，否则 clutter 自动发现会跳过该模型 |

注意：`get_all_cluttered_objects()` 会读取 `center`、`extents`、`scale` 和
`stable`。如果缺字段或 `stable` 为 `false`，资产不会进入随机桌面物体池。

### 2.4 新增 `points_info.json`

普通桌面杂物可使用空结构：

```json
{
    "contact_points": [],
    "functional_points": []
}
```

如果后续要用于机器人抓取任务，应补充抓取点、功能点，并保持这些点与
`model_data0.json` 中的矩阵定义一致。

## 3. 用脚本生成新资产

已有的程序化资产生成器是：

```text
script/generate_messy_desk_assets.py
```

它已经提供了常用 primitive：

- `box(...)`
- `cylinder_between(...)`
- `cylinder_y(...)`
- `sphere(...)`
- `cone_between(...)`
- `torus_y(...)`
- `polyline_tubes(...)`

推荐扩展方式：

1. 复制一个已有 builder 函数，例如 `paper()`、`pencil()`、`scissors()`。
2. 改成新物体的视觉 mesh 列表和碰撞 mesh 列表。
3. 在 `ASSETS` 列表中新增：

```python
Asset(136, "usb-drive", 0.01, usb_drive)
```

4. 运行生成器。

生成命令：

```bash
docker compose up -d tablesim
docker compose exec tablesim bash -lc '
cd /workspace/TableSim
python script/generate_messy_desk_assets.py
'
```

如果你只是新增资产，不建议直接对已有 `121-135` 批次使用 `--force`，因为
`--force` 会替换该脚本声明的编号目录。需要批量重生成时，先确认脚本中的
`ASSETS` 列表只包含你准备覆盖的资产。

## 4. 手动导入外部 GLB/OBJ 资产

如果资产来自 Blender、Objaverse、ShapeNet 或扫描模型，可以手动整理为
TableSim GLB 刚体资产。

### 4.1 目录结构

```bash
mkdir -p assets/objects/136_usb-drive/visual
mkdir -p assets/objects/136_usb-drive/collision
```

放入文件：

```text
assets/objects/136_usb-drive/visual/base0.glb
assets/objects/136_usb-drive/collision/base0.glb
assets/objects/136_usb-drive/model_data0.json
assets/objects/136_usb-drive/points_info.json
```

### 4.2 生成包围盒参数

可以在容器中用 trimesh 读取碰撞模型，计算 `center` 和 `extents`：

```bash
docker compose exec tablesim bash -lc '
cd /workspace/TableSim
python - <<"PY"
from pathlib import Path
import trimesh

path = Path("assets/objects/136_usb-drive/collision/base0.glb")
scene = trimesh.load(path, force="scene")
mesh = trimesh.util.concatenate(tuple(scene.dump()))
print("center:", mesh.bounding_box.centroid.tolist())
print("extents:", mesh.extents.tolist())
print("watertight:", mesh.is_watertight)
PY
'
```

把输出填入 `model_data0.json`。若模型尺寸不是米制，可以：

- 在建模软件中直接缩放到米制后导出；这是推荐方式。
- 或在 `model_data0.json` 的 `scale` 中设置缩放，例如 `[0.01, 0.01, 0.01]`。

注意：如果使用 `scale`，`center` 和 `extents` 仍填写原始模型坐标下的包围盒。
TableSim 加载时会把 `scale` 同时应用到视觉和碰撞模型。

## 5. 新资产验证

### 5.1 文件与 GLB 基础检查

可参考：

```text
script/check_messy_desk_assets.py
```

这个脚本目前固定检查 `121-135` 资产。新增资产后，可以临时把你的目录名加入
`ASSET_NAMES`，或另写一个小检查脚本。最低限度要确认：

- `visual/baseX.glb` 存在且非空。
- `collision/baseX.glb` 存在且非空。
- `model_dataX.json` 存在且包含必要字段。
- `points_info.json` 存在且键为 `contact_points`、`functional_points`。
- 碰撞 GLB 顶点有限。
- 碰撞 GLB 尽量为 watertight。

### 5.2 SAPIEN 加载检查

在容器中运行：

```bash
docker compose exec tablesim bash -lc '
cd /workspace/TableSim
python - <<"PY"
import sapien
from envs.utils.create_actor import create_actor

scene = sapien.Scene()
scene.set_timestep(1 / 250)
scene.add_ground(0)

wrapper = create_actor(
    scene,
    sapien.Pose([0, 0, 0.1], [0.707107, 0.707107, 0, 0]),
    "136_usb-drive",
    model_id=0,
    convex=True,
)
assert wrapper is not None

for _ in range(250):
    scene.step()

pose = wrapper.get_pose()
print("loaded:", wrapper.get_name())
print("pose:", pose.p.tolist(), pose.q.tolist())
PY
'
```

判断标准：

- `wrapper is not None`
- 仿真步进不崩溃
- 位姿不是 NaN/Inf
- 物体没有穿透地面或飞走

### 5.3 clutter 自动发现检查

```bash
docker compose exec tablesim bash -lc '
cd /workspace/TableSim
python - <<"PY"
from envs.utils.rand_create_cluttered_actor import get_all_cluttered_objects

info, names, same = get_all_cluttered_objects()
name = "136_usb-drive"
print(name in names)
print(info.get(name))
PY
'
```

如果输出 `False`，重点检查：

- 目录名是否是 `NNN_name` 格式。
- 是否存在 `model_data0.json`。
- `model_data0.json` 是否包含 `center` 和 `extents`。
- `stable` 是否为 `true`。

### 5.4 单物体渲染检查

如果该物体需要用于 messy desk JSON 或图片采集，还要更新：

```text
image_collectors/object_asset_map.json
```

然后运行：

```bash
docker compose exec tablesim bash -lc '
cd /workspace/TableSim
python script/render_messy_desk_objects.py --only 136_usb-drive --poses 3
'
```

输出应在：

```text
image_collectors/objects/136_usb-drive/
```

## 6. 更新映射与同类排斥关系

### 6.1 `image_collectors/object_asset_map.json`

如果新增资产要参与 messy desk 场景生成，需要在该文件加入映射：

```json
{
  "output": "136_usb-drive",
  "source": "136_usb-drive",
  "kind": "rigid",
  "objects": ["usb drive"]
}
```

字段说明：

| 字段        | 含义                                             |
| ----------- | ------------------------------------------------ |
| `output`  | 图片输出目录名，通常等于资产目录名               |
| `source`  | `assets/objects/` 下的资产目录名               |
| `kind`    | `rigid`、`sapien_urdf` 或 `objaverse_urdf` |
| `objects` | messy desk JSON 中的类别名，可放多个别名         |

### 6.2 `assets/objects/same.json`

`same.json` 用于 clutter 随机放置时避免同类或近似同类重复出现在同一个任务场景。
如果新资产和已有资产语义重复，应补充互斥关系。

示例：

```json
"136_usb-drive": ["usb_drive"],
"usb_drive": ["136_usb-drive"]
```

如果只是全新类别，可以暂不更新 `same.json`。

## 7. 构建 `image_collectors/scenes` 随机桌面场景

本节说明如何基于 `image_collectors/messy_desk.json`、资产映射表和
`assets/objects/` 里的现有资产，批量生成随机 messy desk 场景图片，并把每个
场景的对象、位姿、关系和遮挡记录到 manifest 中。

当前项目脚本名保留了历史拼写：

```text
script/scence_generator.py
```

注意是 `scence_generator.py`，不是 `scene_generator.py`。

### 7.1 输入文件

随机场景生成主要依赖三个输入。

```text
image_collectors/messy_desk.json
image_collectors/object_asset_map.json
assets/objects/<object_asset>/
```

它们的职责分别是：

| 文件/目录                                  | 作用                                                             |
| ------------------------------------------ | ---------------------------------------------------------------- |
| `image_collectors/messy_desk.json`       | 提供桌面物体类别频率、共现关系和空间关系。                       |
| `image_collectors/object_asset_map.json` | 把 JSON 里的语义类别映射到`assets/objects/` 中的具体资产目录。 |
| `assets/objects/<object_asset>/`         | 提供视觉模型、碰撞模型、`model_data*.json` 和 URDF。           |

`messy_desk.json` 里需要至少包含：

```json
{
  "objects": {
    "notebook": 0.72,
    "pen": 0.68
  },
  "co-occurence": [
    {
      "object_a": "keyboard",
      "object_b": "mouse",
      "frequency": 0.64
    }
  ],
  "relationship": [
    {
      "subject": "pen",
      "predicate": "on",
      "object": "notebook",
      "frequency": 0.42
    }
  ]
}
```

`object_asset_map.json` 中的 `objects` 字段必须覆盖 `messy_desk.json` 中要采样的
语义类别。例如：

```json
{
  "output": "092_notebook",
  "source": "092_notebook",
  "kind": "rigid",
  "objects": ["notebook"]
}
```

如果 `messy_desk.json` 里有某个类别，但 `object_asset_map.json` 没有映射，或者
对应资产目录下没有可加载模型，这个类别不会进入随机场景候选池。

### 7.2 生成脚本的行为

`script/scence_generator.py` 会执行以下流程：

1. 读取 `messy_desk.json` 的 `objects`、`co-occurence`、`relationship`。
2. 读取 `object_asset_map.json`，筛出当前项目真实存在的资产。
3. 基于物体频率和共现关系采样一组桌面物品。
4. 优先应用 `relationship` 中的空间关系。
5. 默认强制至少成功放置一个 `on` 关系。
6. 默认强制构造一组从相机视角可见的前景遮挡关系。
7. 用 SAPIEN 渲染桌面场景图片。
8. 写出 `image_collectors/scenes/manifest.json`。

默认桌面尺寸和相机在脚本顶部定义：

```text
TABLE_LENGTH = 1.35
TABLE_WIDTH = 0.86
TABLE_SURFACE_Z = 0.74
```

桌面物体数量默认约为 10 类：

```text
--min-objects 9
--max-objects 11
--min-placed 9
```

如果要调节 clutter 程度，优先改命令行参数，而不是直接改脚本常量：

- 更拥挤：提高 `--min-objects`、`--max-objects`、`--min-placed`。
- 更稀疏：降低这些参数。
- 更稳定：适当降低 `--min-placed`，避免少数大物体导致放置失败。

当前布局策略把 `laptop`、`monitor`、`printer` 视为大型桌面设备。每张图最多
采样一个大型设备；若存在，它会被放在桌面中间靠后、以 `yaw=0°` 面向负 Y
（正前方相机），其他物体只能出现在它的左前、正前或右前区域。未通过放置约束或
曝光检查的候选会自动重试，直到达到 `--scenes` 指定的成功图片数或耗尽
`--max-attempts`。

### 7.3 生成 10 个随机桌面场景

推荐在 Docker 内运行，保证 SAPIEN、渲染器和资产路径一致。

```bash
docker compose up -d tablesim
docker compose exec -T tablesim bash -lc '
cd /workspace/TableSim
python script/scence_generator.py \
  --scenes 10 \
  --seed 20260720 \
  --width 1024 \
  --height 768 \
  --min-objects 9 \
  --max-objects 11 \
  --min-placed 9 \
  --render-steps 4
'
```

输出目录：

```text
image_collectors/scenes/
├── manifest.json
├── scene_000.png
├── scene_001.png
├── ...
└── scene_009.png
```

如果你需要完全覆盖旧图片，建议先备份或清空旧 scenes 输出目录。不要只生成更少
数量的场景后假设旧图片自动消失，因为旧的 `scene_010.png`、`scene_011.png`
之类文件可能仍留在目录中。

安全备份方式：

```bash
mv image_collectors/scenes image_collectors/scenes_backup_$(date +%Y%m%d_%H%M%S)
mkdir -p image_collectors/scenes
```

然后重新运行 `script/scence_generator.py`。

### 7.4 `on` 关系和遮挡控制

当前生成器默认比较严格：

- 没有成功放置 `on` 关系时，该 scene 会失败。
- 没有成功构造遮挡关系时，该 scene 会失败。

这能保证生成的桌面不是全部 `next to`，且至少有物体叠放和遮挡。但如果你只是想
快速调试渲染流程，可以临时放宽：

```bash
python script/scence_generator.py \
  --scenes 3 \
  --allow-missing-on \
  --allow-missing-occlusion
```

正式用于构建 task 或数据集时，不建议加这两个 flag。

### 7.5 检查 manifest

生成结束后，先检查 manifest summary：

```bash
docker compose exec -T tablesim bash -lc '
cd /workspace/TableSim
python - <<"PY"
import json
from pathlib import Path

p = Path("image_collectors/scenes/manifest.json")
m = json.loads(p.read_text(encoding="utf-8"))
print(json.dumps(m["summary"], ensure_ascii=False, indent=2))
for i, scene in enumerate(m["scenes"][:3]):
    print(
        i,
        scene["image"],
        "objects=", len(scene["placed_objects"]),
        "on=", len(scene.get("on_relationships", [])),
        "occlusion=", bool(scene.get("occlusion")),
    )
PY
'
```

一个可用于 task 的 scene 至少应满足：

- `placed_objects` 数量足够。
- `on_relationships` 非空。
- `occlusion` 非空。
- `failures` 为空，或者失败数量可接受且不会影响目标 scene。
- 目标物体没有被完全遮挡。
- 目标物体周围留有机械臂可接近的空间。

### 7.6 manifest 中的关键字段

`manifest.json` 顶层结构通常如下：

```json
{
  "generator": "script/scence_generator.py",
  "messy_json": "image_collectors/messy_desk.json",
  "asset_map": "image_collectors/object_asset_map.json",
  "seed": 20260720,
  "table": {
    "length": 1.35,
    "width": 0.86,
    "surface_z": 0.74
  },
  "scenes": [],
  "failures": [],
  "summary": {}
}
```

每个 scene 里最重要的是 `placed_objects`：

```json
{
  "label": "mouse",
  "asset": "047_mouse",
  "source": "047_mouse",
  "kind": "rigid",
  "model_id": "2",
  "xy": [-0.29208, -0.10648],
  "z_offset": 0.0,
  "yaw_degrees": -137.88,
  "footprint": [0.048, 0.087],
  "height": 0.026,
  "relation": null
}
```

字段解释：

| 字段                       | 含义                                                         |
| -------------------------- | ------------------------------------------------------------ |
| `label`                  | messy desk 语义类别，例如`mouse`、`notebook`。           |
| `source`                 | `assets/objects/` 下真实资产目录。                         |
| `kind`                   | 加载方式，常见为`rigid` 或 `sapien_urdf`。               |
| `model_id`               | 对应`model_dataX.json`、`baseX.glb` 或 URDF 子目录编号。 |
| `xy`                     | 桌面平面内位置，世界坐标。                                   |
| `z_offset`               | 相对桌面顶面的额外高度；`on` 关系会让它大于 0。            |
| `yaw_degrees`            | 绕世界 Z 轴旋转角度。                                        |
| `footprint` / `height` | 生成器估算的占地和高度，主要用于调试。                       |
| `relation`               | 该对象被某条空间关系约束时记录原始关系。                     |

注意：`z_offset` 不是 SAPIEN 里刚体最终 pose 的 `z`。对编号 GLB 刚体，TableSim
资产沿用局部 Y-up 约定，加载时还要结合 `model_dataX.json` 中的 `center`、
`extents` 和 `scale` 计算底面偏移。

### 7.7 从 manifest 随机选取一个 scene

如果要基于随机场景构建任务，可以先抽取一组 scene 参数：

```bash
docker compose exec -T tablesim bash -lc '
cd /workspace/TableSim
python - <<"PY"
import json
import random
from pathlib import Path

m = json.loads(Path("image_collectors/scenes/manifest.json").read_text(encoding="utf-8"))
scene = random.choice(m["scenes"])
print("image:", scene["image"])
print("placed_objects:", len(scene["placed_objects"]))
print("on_relationships:", json.dumps(scene.get("on_relationships", []), ensure_ascii=False))
print("occlusion:", json.dumps(scene.get("occlusion", {}), ensure_ascii=False))
print(json.dumps(scene["placed_objects"], ensure_ascii=False, indent=2))
PY
'
```

选择 task scene 时，不要只看图片是否 clutter。还要检查目标物体：

- 是否是轻量、可抓取的小物体。
- 是否不是静态场景大物体，例如显示器、台灯、黑板。
- 是否没有被完全压住或完全遮挡。
- 是否在机械臂工作空间内。
- 周围是否有足够接近方向。

实践中，`mouse`、`pen`、`pencil`、`eraser`、`sticky note`、`small book` 这类目标
通常比 `monitor`、`lamp`、`plant` 更适合作为第一版 pick task。

### 7.8 批量生成与自动筛选

QZ 原生服务器有三套虚拟环境，但场景生成只使用 `.venv-sim`。在服务器上先执行：

```bash
cd /inspire/hdd/global_user/liutong-CZXS25230163/TableSim
export TABLESIM_ROOT="$PWD"
source "$TABLESIM_ROOT/.venv-sim/bin/activate"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$TABLESIM_ROOT/compat:$TABLESIM_ROOT"
export SAPIEN_NO_DISPLAY=1
export SAPIEN_RENDERER=opengl
```

直接生成 100 个场景时，可沿用单批生成器：

```bash
python script/scence_generator.py \
  --scenes 100 \
  --seed 20260919 \
  --output-dir image_collectors/scenes_new
```

生成器会对每个候选执行旋转包围盒、垂直区间和最终 SAPIEN world bounds 检查。
穿透桌面、悬空、意外相交、越出桌面或 `on` 关系高度不合理的候选不会计入这
100 个成功场景，而会作为失败 attempt 记录到 manifest。遮挡物与被遮挡物会沿
相机视线排列，但在桌面平面保留真实间隙，不再靠几何穿模制造遮挡。

需要生成较大的候选池时，使用批处理入口：

```bash
python script/batch_generate_scenes.py \
  --batches 20 \
  --scenes-per-batch 10 \
  --seed 20260919 \
  --output-dir image_collectors/scenes_new \
  --selected-count 50 \
  --resume
```

该脚本逐批调用 `script/scence_generator.py`。每个批次使用独立、可复现的 seed，
并写入独立目录；某一批失败不会覆盖其他批次。`--resume` 会复用场景数量和 seed
都匹配的完整批次，并重新运行不完整批次。先检查命令而不启动 SAPIEN 时可使用：

```bash
python script/batch_generate_scenes.py \
  --batches 20 \
  --scenes-per-batch 10 \
  --seed 20260919 \
  --output-dir image_collectors/scenes_new \
  --dry-run
```

默认自动筛选规则与本指南前面的可用 scene 建议保持一致：

- 至少成功放置 9 个物体，且跳过的物体不超过 2 个。
- 至少有一条 `on` 关系和一条明确的遮挡关系。
- 新版 manifest 中的 `layout_policy` 必须有效且没有违规项。
- PNG 必须存在、可解码，分辨率与 manifest 一致，且没有明显全黑、全白或低信息量。
- 至少有一个轻量目标位于 ALOHA 默认工作区
  `x=[-0.36, 0.36]`、`y=[-0.27, 0.22]`。
- 候选目标不能是明确的遮挡目标、不能承托另一个 `on` 物体，并需保留默认
  `0.015 m` 的近似接近间隙。

这些阈值均可通过 `--help` 中的参数调整。自动结果是保守的 manifest/几何/图像
启发式筛选，不等价于机械臂运动规划或抓取成功验证；最终仍应查看 review 拼图，
再用 `script/check_sim.py --setup` 验证选定 task。`--selected-count` 是最多保留的
数量；若远端作业必须产出至少 50 个场景，再同时指定 `--min-selected 50`，不足时
脚本会保留已有结果并以非零状态退出。

输出结构如下：

```text
image_collectors/scenes_new/
├── batches/
│   ├── batch_00000_seed_20260919/
│   │   ├── manifest.json
│   │   └── scene_*.png
│   └── ...
├── generation_run.json       # 每批命令、seed、退出码和运行状态
├── catalog.json              # 所有候选及详细通过/拒绝原因
├── catalog.csv               # 便于人工排序、过滤
├── selected/
│   ├── manifest.json         # 最终选择、目标候选、评分和源路径
│   └── scene_*.png           # 默认复制；可改为 hardlink 或不复制
└── review/
    ├── accepted_*.png        # 入选场景拼图
    └── rejected_*.png        # 拒绝场景及简短原因拼图
```

只重新筛选已有结果而不启动渲染：

```bash
python script/batch_generate_scenes.py \
  --filter-only \
  --input-dir image_collectors/scenes_new \
  --output-dir image_collectors/scenes_new \
  --selected-count 50
```

多台远程服务器分片时，为每台服务器指定不重叠的 `--batch-start-index`，并建议
使用各自的输出目录。例如第一台从 0 开始运行 20 批，第二台从 20 开始运行 20 批。
只生成、不汇总时加 `--generate-only`；把各服务器的 `batches/` 汇总到一个目录后，
再用 `--filter-only` 统一筛选。不要让多台机器同时写同一个
`catalog.json` 或 `selected/` 目录。

## 8. 基于 scenes 构建新 task 并运行数据采集

本节说明如何从 `image_collectors/scenes/manifest.json` 中选一组场景参数，构建
一个新的 `envs/<task_name>.py`，并用 `collect_data` 生成 HDF5、视频和语言指令。

项目里已经有一个可参考实现：

```text
envs/pick_an_object_messy.py
task_config/pick_an_object_messy.yml
description/task_instruction/pick_an_object_messy.json
```

它使用 `image_collectors/scenes/manifest.json` 中的固定 scene 参数，加载一个
cluttered desk，然后抓取指定的 `mouse`。

### 8.1 task 相关文件约定

一个可被 `collect_data.py` 发现的新 task 至少需要三个文件：

```text
envs/<task_name>.py
task_config/<task_name>.yml
description/task_instruction/<task_name>.json
```

命名必须一致：

- Python 文件名：`envs/pick_mouse_messy.py`
- Python 类名：`class pick_mouse_messy(Base_Task)`
- 配置文件：`task_config/pick_mouse_messy.yml`
- 指令模板：`description/task_instruction/pick_mouse_messy.json`
- 运行参数：`pick_mouse_messy pick_mouse_messy`

`script/collect_data.py` 会动态导入 `envs.<task_name>`，并检查其中是否存在同名类。
因此通常不需要手动编辑 `envs/__init__.py`。

### 8.2 创建 task 文件

推荐从现有 messy pick 任务复制：

```bash
cp envs/pick_an_object_messy.py envs/pick_mouse_messy.py
cp task_config/pick_an_object_messy.yml task_config/pick_mouse_messy.yml
cp description/task_instruction/pick_an_object_messy.json \
  description/task_instruction/pick_mouse_messy.json
```

然后至少修改：

- `envs/pick_mouse_messy.py` 中的类名。
- `TARGET_LABEL`、`TARGET_ASSET`、`TARGET_MODEL_ID`。
- `SCENE_OBJECTS`。
- `task_config/pick_mouse_messy.yml` 顶部注释、`episode_num`、采集字段。
- `description/task_instruction/pick_mouse_messy.json` 中的任务描述和模板。

### 8.3 固定 scene 的推荐做法

对第一版 task，建议把选中的 manifest scene 固化到 Python 里，而不是每个
episode 重新随机采样 scene。原因：

- 方便复现实验和调试。
- 失败时可以明确定位是动作规划问题，还是场景采样问题。
- `collect_data` 的 seed 机制会保存规划成功的 seed；如果场景也在运行时随机变，
  旧 seed 和旧轨迹很容易失效。

`SCENE_OBJECTS` 可以直接来自 manifest 的 `placed_objects`，保留这些字段：

```python
SCENE_OBJECTS = [
    {
        "label": "mouse",
        "source": "047_mouse",
        "kind": "rigid",
        "model_id": "2",
        "xy": [-0.29208, -0.10648],
        "z_offset": 0.0,
        "yaw_degrees": -137.88,
    },
    {
        "label": "pen",
        "source": "010_pen",
        "kind": "sapien_urdf",
        "model_id": "101685",
        "xy": [0.12031, 0.14308],
        "z_offset": 0.02633,
        "yaw_degrees": -169.54,
    }
]
```

保留 `label/source/kind/model_id/xy/z_offset/yaw_degrees` 即可；`footprint`、
`height`、`relation` 可作为注释或调试信息，不是加载必需字段。

### 8.4 关键 task 方法

新 task 类应继承 `Base_Task`，并实现这些方法：

| 方法                          | 必须性 | 作用                                                            |
| ----------------------------- | -----: | --------------------------------------------------------------- |
| `setup_demo(self, **kwags)` |   必须 | 初始化环境。固定 messy scene 时，应关闭基础环境的随机 clutter。 |
| `load_actors(self)`         |   必须 | 加载桌面物体、目标物体和禁入区域。                              |
| `play_once(self)`           |   必须 | 定义机械臂动作流程，并返回`self.info`。                       |
| `check_success(self)`       |   必须 | 判断当前 episode 是否成功。                                     |

固定 messy scene 推荐在 `setup_demo` 中关闭 `domain_randomization.cluttered_table`：

```python
def setup_demo(self, **kwags):
    kwags = deepcopy(kwags)
    random_setting = dict(kwags.get("domain_randomization") or {})
    random_setting["cluttered_table"] = False
    kwags["domain_randomization"] = random_setting
    super()._init_task_env_(**kwags)
```

否则基础环境可能会额外生成一批随机 clutter，导致你固化的 scene 变成“双重 clutter”。

### 8.5 从 manifest pose 加载物体

`kind == "rigid"` 的编号 GLB 资产用 `create_actor(...)` 加载。关键点：

- 编号 GLB 资产是局部 Y-up。
- SAPIEN 世界是 Z-up。
- 需要乘一个 Y-up 到 Z-up 的基础四元数。
- 最终 `z` 需要根据 `model_dataX.json` 的底面偏移计算。

参考：

```python
RIGID_Y_UP_TO_Z_UP = [2**-0.5, 2**-0.5, 0.0, 0.0]
```

`kind == "sapien_urdf"` 的资产用 `create_sapien_urdf_obj(...)` 加载。通常不需要
GLB 的 Y-up 修正，但要确认 URDF 模型自身坐标和 `z_offset` 是否合理。

静态 clutter 和目标物体的物理属性应区分：

- 非目标 clutter：建议 `is_static=True` 或 `fix_root_link=True`，减少不必要的物理扰动。
- 目标物体：必须是动态物体，否则机械臂抓取后不会被带起。

在 `load_actors()` 中，建议对非目标物体调用：

```python
self.add_prohibit_area(actor, padding=0.01)
```

这样运动规划会尽量避开这些 clutter。

### 8.6 选择目标物体

第一版 pick task 建议选择轻量目标，例如：

- `047_mouse`
- `010_pen`
- `124_pencil`
- `128_eraser`
- `123_sticky-note`

目标物体需要满足：

- 能被当前机械臂工作空间覆盖。
- 没有被其它物体压住。
- 上方或侧方有可接近空间。
- `check_success()` 能用简单条件判断，例如抬高距离、夹爪闭合、目标位移。

示例成功条件：

```python
def check_success(self):
    target_pose = self.target_actor.get_pose().p
    initial_pose = self.initial_target_pose.p
    lifted = target_pose[2] > initial_pose[2] + 0.045
    return lifted and self.is_left_gripper_close()
```

如果目标物体较薄，例如纸张、便签、文件夹，只用顶抓可能不稳定；这类任务应增加
更明确的接近方向、抓取点或换成更厚的目标。

### 8.7 `play_once()` 和语言指令占位符

`play_once()` 中应设置 `self.info["info"]`。这里的 key 必须和
`description/task_instruction/<task_name>.json` 的占位符一致。

例如：

```python
arm_tag = ArmTag("left")
self.info["info"] = {
    "{A}": "047_mouse/base2",
    "{a}": str(arm_tag),
}
```

对应的 instruction JSON 可以只使用 `{A}` 和 `{a}`：

```json
{
  "full_description": "pick the specified mouse from a cluttered desk scene",
  "schema": "{A} identifies the mouse to be picked; {a} identifies the arm",
  "seen": [
    "Use {a} to pick up {A} from the messy desk."
  ],
  "unseen": [
    "Lift {A} from the cluttered tabletop."
  ]
}
```

如果 JSON 模板里出现了 `{B}`，但 `self.info["info"]` 没有提供 `{B}`，指令生成
阶段会出错或生成不完整内容。反过来，`self.info["info"]` 提供了模板不用的 key
通常没问题，但不建议保留无关字段。

### 8.8 配置 `task_config/<task_name>.yml`

推荐先复制 `_config_template.yml` 或现有 task 配置，然后按目标任务修改。

固定 messy scene 的常用配置：

```yaml
render_freq: 0
episode_num: 5
use_seed: false
save_freq: 15
embodiment: [aloha-agilex]
language_num: 100
domain_randomization:
  random_background: false
  cluttered_table: false
  clean_background_rate: 1
  random_head_camera_dis: 0
  random_table_height: 0
  random_light: false
  crazy_random_light_rate: 0
camera:
  head_camera_type: D435
  wrist_camera_type: D435
  collect_head_camera: true
  collect_wrist_camera: true
data_type:
  rgb: true
  depth: false
  pointcloud: false
  endpose: true
  qpos: true
save_path: ./data
clear_cache_freq: 5
collect_data: true
eval_video_log: true
```

`D435` 的项目级参数定义在 `task_config/_camera_config.yml`。TableSim 当前固定使用
`640×480`；顶部相机、左右腕部相机以及其他选择 `D435` 的静态相机都会读取这一
全局值。新任务应继续在 task config 中选择 `D435`，不要在任务脚本中覆盖宽高，
以保证后续采集始终保持 `640×480`。

调试时建议：

- `episode_num: 1` 或 `2`
- `collect_data: false` 用于只测规划和成功条件
- `eval_video_log: true` 便于检查视频

正式采集时再提高 `episode_num`。

### 8.9 检查 task 能否被导入

先做动态导入检查：

```bash
docker compose exec -T tablesim bash -lc '
cd /workspace/TableSim
python - <<"PY"
from script.collect_data import class_decorator
task = class_decorator("pick_mouse_messy")
print(type(task).__name__)
PY
'
```

如果失败，重点检查：

- `envs/pick_mouse_messy.py` 是否存在。
- 类名是否是 `pick_mouse_messy`。
- 文件内 import 是否正确。
- 是否有语法错误。

### 8.10 检查完整仿真 setup

导入成功后，运行完整 setup 检查：

```bash
docker compose exec -T tablesim bash -lc '
cd /workspace/TableSim
python script/check_sim.py \
  --task pick_mouse_messy \
  --config pick_mouse_messy \
  --setup
'
```

预期日志类似：

```text
Full simulation setup OK: pick_mouse_messy actors=<N>
```

这一步会初始化机器人、相机、场景物体和 motion planner，但不会写正式数据。

### 8.11 运行 `collect_data`

推荐使用项目封装脚本：

```bash
bash collect_data.sh pick_mouse_messy pick_mouse_messy 0
```

或直接在已启动的容器里运行：

```bash
docker compose exec -T -e CUDA_VISIBLE_DEVICES=0 tablesim bash -lc '
cd /workspace/TableSim
python script/collect_data.py pick_mouse_messy pick_mouse_messy
'
```

输出目录：

```text
data/pick_mouse_messy/pick_mouse_messy/
├── _traj_data/
├── data/
├── instructions/
├── video/
├── seed.txt
└── scene_info.json
```

采集分两阶段：

1. `Start Seed and Pre Motion Data Collection`：寻找可成功规划和通过
   `check_success()` 的 seed，并保存 `_traj_data/episodeN.pkl`。
2. `Start Data Collection`：按成功 seed 回放轨迹，保存 HDF5、视频、语言指令和
   `scene_info.json`。

看到类似日志才表示单个 episode 规划成功：

```text
simulate data episode 0 success! (seed = 0)
```

看到 `Successfully Saved Instructions` 表示采集和语言指令生成都完成。

### 8.12 重新采集时处理旧数据

`collect_data.py` 会复用已有 `seed.txt`，并跳过已经存在的 HDF5 文件。因此，如果你
修改了场景、相机、动作逻辑或成功条件，想重新生成一套干净数据，应先备份旧输出：

```bash
OUT=data/pick_mouse_messy/pick_mouse_messy
if [ -d "$OUT" ]; then
  mv "$OUT" "data/pick_mouse_messy/pick_mouse_messy_backup_$(date +%Y%m%d_%H%M%S)"
fi
```

然后重新运行：

```bash
bash collect_data.sh pick_mouse_messy pick_mouse_messy 0
```

不要把新任务逻辑直接回放旧 `_traj_data/*.pkl`，除非你确认动作路径、场景对象和
目标完全兼容。

### 8.13 检查采集结果

采集结束后检查数量：

```bash
OUT=data/pick_mouse_messy/pick_mouse_messy
find "$OUT/data" -name "episode*.hdf5" | wc -l
find "$OUT/video" -name "episode*.mp4" | wc -l
find "$OUT/_traj_data" -name "episode*.pkl" | wc -l
wc -w < "$OUT/seed.txt"
```

检查 HDF5 结构：

```bash
docker compose exec -T tablesim bash -lc '
cd /workspace/TableSim
python - <<"PY"
import h5py
from pathlib import Path

p = Path("data/pick_mouse_messy/pick_mouse_messy/data/episode0.hdf5")
with h5py.File(p, "r") as f:
    def show(name, obj):
        if hasattr(obj, "shape"):
            print(name, obj.shape, obj.dtype)
    f.visititems(show)
PY
'
```

至少应看到：

```text
observation/head_camera/rgb
observation/head_camera/cam2world_gl
observation/head_camera/extrinsic_cv
observation/head_camera/intrinsic_cv
joint_action/vector
endpose/left_endpose
```

如果开启了 wrist camera，也应看到 `left_camera` 和 `right_camera`。

### 8.14 task 构建常见问题

| 问题                         | 常见原因                                              | 处理                                                        |
| ---------------------------- | ----------------------------------------------------- | ----------------------------------------------------------- |
| `Task module not found`    | `envs/<task>.py` 不存在或任务名拼错                 | 保证运行参数和文件名一致。                                  |
| `must define class <task>` | 类名和文件名不一致                                    | 修改 class 名，保持完全一致。                               |
| setup 阶段物体飞走           | 碰撞模型不稳定、初始穿透、目标被压住                  | 检查`z_offset`、碰撞 GLB、静态 clutter 设置。             |
| 规划失败率高                 | 目标太靠边、被遮挡太重、禁入区域太密                  | 换目标、减少周边 clutter、降低`min_placed` 后重采 scene。 |
| 数据没有重新生成             | 旧`seed.txt` 或 HDF5 被复用                         | 备份/移走`data/<task>/<config>/` 后重跑。                 |
| 指令生成失败                 | instruction JSON 占位符和`self.info["info"]` 不匹配 | 保持`{A}`、`{a}` 等 key 一一对应。                      |

### 8.15 `clean_the_desk_009` 长序列整理任务

`clean_the_desk_009` 是基于 `image_collectors/scenes/scene_009.png` 的固定场景
长序列任务，相关文件为：

```text
envs/clean_the_desk_009.py
task_config/clean_the_desk_009.yml
description/task_instruction/clean_the_desk_009.json
```

任务保留参考图中的电脑、键鼠、杯子、文件夹、书本和办公用品，同时把物体中心压缩到
以下 ALOHA 可达工作区：

```text
x: [-0.36, 0.36]
y: [-0.27, 0.22]
```

脚本严格执行以下阶段，`check_success()` 也会检查阶段列表的顺序：

1. 右臂下压并闭合笔记本电脑屏幕。
2. 依次把键盘、鼠标放在闭合电脑的两个独立区域。
3. 把大书上的水杯移到左侧空位。
4. 对齐文件夹。
5. 把大书放到文件夹上，再把小书放到大书上。
6. 依次整理眼镜、铅笔、剪刀、便利贴和订书机。
7. 双臂回到初始位并张开夹爪。

任务会在 `setup_demo()` 内复制 embodiment 配置，再强制把 ALOHA 顶部相机设置为：

```yaml
position: [-0.032, -0.45, 1.8]
forward: [0.03, 0.45, -1.02]
```

该高度比共享配置中的 2.2 m 位置降低 0.4 m，并根据 D435 的 37° 垂直视场角调整
俯视方向，使桌面靠机器人一侧的边缘基本落在 480 像素画面的底边，同时保留桌面
远端边缘和全部任务物体。这不会原地修改共享的
`assets/embodiments/aloha-agilex/config.yml`。同时关闭
`random_head_camera_dis`、`random_table_height` 和额外随机 clutter，确保场景、
相机和已规划轨迹可以复现。

文件夹、键盘、眼镜等自建薄物体没有完整的物理抓取点。该任务仍为每个物体执行真实
的机械臂接近、闭爪、抬升、转移、下降和开爪轨迹，但在闭爪后把静态 actor 的相对
位姿绑定到当前末端执行器，并在控制步内同步。这样可以避免 12 阶段长序列因薄物体
滑落而中断，也能保证回放数据中物体持续跟随机械臂。

完整成功条件包括：

- laptop 关节到达闭合阈值；
- 11 个可移动物体都到达各自目标位姿；
- 键盘与鼠标包围盒之间保留至少 `0.008 m` 间隔；
- 文件夹、大书、小书按指定高度叠放；
- 12 个动作阶段顺序完全匹配；
- 所有物体仍在工作区内；
- 结束时左右夹爪都张开。

运行两个 episode：

```bash
bash collect_data.sh clean_the_desk_009 clean_the_desk_009 0
```

当前 `z=1.8 m` 配置和验证结果：

```text
episode_num: 2
planning failures: 0
episode0: 1235 frames, 320x240, 30 FPS
episode1: 1245 frames, 320x240, 30 FPS
```

数据保存在：

```text
data/clean_the_desk_009/clean_the_desk_009/
├── _traj_data/       # 两个成功规划轨迹
├── data/             # episode0.hdf5、episode1.hdf5
├── instructions/     # 每个 episode 各 100 条 seen 和 100 条 unseen
├── video/            # episode0.mp4、episode1.mp4
├── scene_info.json
└── seed.txt
```

`self.info["info"]` 中使用 `asset/model` 路径作为语言占位符值时，对应文件必须存在于
`description/objects_description/<asset>/<model>.json`。本任务已为 laptop、folder、
glasses、pencil、scissors 和 sticky-note 补齐描述文件；缺少其中任意文件都会使
仿真数据生成成功，但语言指令生成在最后一步退出。

## 9. Isaac Sim 场景搭建目标

TableSim 当前主仿真环境是 SAPIEN；项目内只有一个 Isaac Sim smoke test：

```text
script/smoke_isaacsim.py
```

因此，Isaac Sim 场景建议作为独立 USD 场景搭建，不直接替代
`collect_data.py` 的 SAPIEN 流程。推荐产物形式：

```text
assets/isaac_scenes/<scene_name>/<scene_name>.usd
assets/isaac_scenes/<scene_name>/manifest.json
assets/isaac_scenes/<scene_name>/preview.png
```

建议新增脚本位置：

```text
script/build_isaac_scene.py
```

输入建议使用 JSON/YAML 描述指定场景：

```text
configs/isaac_scenes/messy_desk_demo.json
```

## 10. Isaac Sim 环境检查

先确认当前环境是否能启动 Isaac Sim：

```bash
python script/smoke_isaacsim.py
```

如果在 Docker 里运行：

```bash
docker compose exec tablesim bash -lc '
cd /workspace/TableSim
python script/smoke_isaacsim.py
'
```

如果失败，说明当前 TableSim Docker 镜像没有完整 Isaac Sim 运行时。此时需要使用
安装了 Isaac Sim 的独立容器或宿主环境运行场景搭建脚本。TableSim 资产文件仍可
复用，只是执行环境不同。

常见要求：

- NVIDIA Driver 与 Isaac Sim 版本兼容。
- 容器需要 `--gpus all`。
- headless 渲染需要可用 Vulkan。
- 需要能 import `isaacsim` 或 `omni.isaac.kit`。

## 11. 指定 Isaac Sim 场景描述格式

建议用 JSON 描述“要搭建什么场景”，不要把场景参数硬编码在 Python 里。

示例：

```json
{
  "scene_name": "messy_desk_demo",
  "output_usd": "assets/isaac_scenes/messy_desk_demo/messy_desk_demo.usd",
  "units": "meters",
  "table": {
    "size": [1.35, 0.86, 0.055],
    "position": [0.0, 0.0, 0.74],
    "color": [0.68, 0.59, 0.48]
  },
  "objects": [
    {
      "name": "notebook_0",
      "asset": "assets/objects/092_notebook/visual/base0.glb",
      "collision": "assets/objects/092_notebook/collision/base0.glb",
      "position": [-0.20, -0.05, 0.77],
      "rotation_euler_deg": [90.0, 0.0, 18.0],
      "scale": [1.0, 1.0, 1.0]
    },
    {
      "name": "cup_0",
      "asset": "assets/objects/021_cup/visual/base0.glb",
      "collision": "assets/objects/021_cup/collision/base0.glb",
      "position": [0.18, 0.05, 0.80],
      "rotation_euler_deg": [90.0, 0.0, -12.0],
      "scale": [1.0, 1.0, 1.0]
    }
  ],
  "camera": {
    "position": [0.35, -1.05, 1.35],
    "target": [0.0, 0.0, 0.82],
    "focal_length": 28.0
  },
  "lights": [
    {
      "type": "dome",
      "intensity": 450.0
    },
    {
      "type": "distant",
      "rotation_euler_deg": [-45.0, 0.0, 35.0],
      "intensity": 600.0
    }
  ]
}
```

注意 GLB 资产的姿态：

- TableSim 编号 GLB 是局部 Y-up。
- USD/Isaac Sim 通常使用 Z-up stage。
- 导入这些 GLB 时，通常需要给 `rotation_euler_deg` 的 X 轴加 `90` 度，或者在
  脚本中统一加一个 Y-up 到 Z-up 的修正旋转。

## 12. Isaac Sim 场景搭建脚本骨架

下面是推荐脚本骨架。不同 Isaac Sim 版本的 API 可能略有差异，所以实际接入时
先跑 smoke test，再按本机版本调整 import。

建议文件：

```text
script/build_isaac_scene.py
```

骨架：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from isaacsim import SimulationApp
except Exception:
    from omni.isaac.kit import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    app = SimulationApp({"headless": args.headless})
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
        import omni.usd

        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        output_usd = Path(config["output_usd"])
        output_usd.parent.mkdir(parents=True, exist_ok=True)

        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        # Physics scene.
        physics_scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        physics_scene.CreateGravityMagnitudeAttr(9.81)

        # Table.
        table_cfg = config["table"]
        table = UsdGeom.Cube.Define(stage, "/World/table")
        size = table_cfg["size"]
        table.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], size[2]))
        table.AddTranslateOp().Set(Gf.Vec3d(*table_cfg["position"]))
        UsdPhysics.CollisionAPI.Apply(table.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(table.GetPrim()).CreateRigidBodyEnabledAttr(False)

        # Objects.
        for obj in config["objects"]:
            prim_path = f"/World/objects/{obj['name']}"
            prim = stage.DefinePrim(prim_path, "Xform")
            prim.GetReferences().AddReference(str(Path(obj["asset"]).resolve()))
            xform = UsdGeom.Xformable(prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(*obj["position"]))
            rx, ry, rz = obj.get("rotation_euler_deg", [0, 0, 0])
            xform.AddRotateXYZOp().Set(Gf.Vec3f(rx, ry, rz))
            xform.AddScaleOp().Set(Gf.Vec3f(*obj.get("scale", [1, 1, 1])))
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.RigidBodyAPI.Apply(prim)

        # Lights.
        for index, light in enumerate(config.get("lights", [])):
            if light["type"] == "dome":
                dome = UsdLux.DomeLight.Define(stage, f"/World/lights/dome_{index}")
                dome.CreateIntensityAttr(light.get("intensity", 450.0))
            elif light["type"] == "distant":
                distant = UsdLux.DistantLight.Define(stage, f"/World/lights/distant_{index}")
                distant.CreateIntensityAttr(light.get("intensity", 600.0))
                xform = UsdGeom.Xformable(distant.GetPrim())
                xform.AddRotateXYZOp().Set(Gf.Vec3f(*light.get("rotation_euler_deg", [0, 0, 0])))

        # Camera.
        camera_cfg = config["camera"]
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.AddTranslateOp().Set(Gf.Vec3d(*camera_cfg["position"]))
        camera.CreateFocalLengthAttr(camera_cfg.get("focal_length", 28.0))
        # For production, compute a look-at transform from position + target.
        # A helper should set the full camera transform rather than only translate.

        stage.GetRootLayer().Export(str(output_usd))
        print(f"saved: {output_usd}")
        app.update()
    finally:
        app.close()


if __name__ == "__main__":
    main()
```

这个骨架给出 USD stage、物理场景、桌子、对象、灯光和相机的创建位置。实际落地
时建议补上两个工程化细节：

- `look_at(position, target)`：把相机朝向目标点。
- `add_collision_from_collision_glb(...)`：如果 Isaac Sim 对直接引用 GLB 的碰撞
  支持不符合预期，改为引用视觉 GLB，同时用简化 mesh 或 convex decomposition
  生成碰撞。

## 13. 指定场景搭建步骤

当你要搭建一个指定 Isaac Sim 场景时，按以下顺序做。

### 13.1 明确场景配置

先写配置文件，例如：

```text
configs/isaac_scenes/messy_desk_demo.json
```

必须明确：

- 场景名
- 输出 USD 路径
- 桌子尺寸和高度
- 每个物体的资产路径
- 每个物体的位置、旋转、缩放
- 相机位置和观察目标
- 灯光类型和强度

### 13.2 检查资产路径

每个对象至少要有视觉模型：

```text
assets/objects/<NNN_name>/visual/baseX.glb
```

如果要物理仿真，建议也准备：

```text
assets/objects/<NNN_name>/collision/baseX.glb
```

如果指定场景使用 URDF 资产，应先决定是：

- 直接在 Isaac Sim 中导入 URDF；
- 还是先转成 USD/GLB 后作为静态或动态物体引用。

### 13.3 处理坐标与单位

统一规则：

- 使用米。
- USD stage 使用 Z-up。
- TableSim GLB 资产需要从 Y-up 修正到 Z-up。
- 物体最终 `z` 坐标应大于桌面高度，避免初始穿透。

如果桌面顶面高度是 `0.74`，物体放置高度可从以下值起步：

```text
z = 0.74 + 物体高度 / 2 + 0.002
```

薄片类物体可以直接略高于桌面：

```text
z = 0.742
```

### 13.4 创建 USD 场景

运行：

```bash
python script/build_isaac_scene.py \
  --config configs/isaac_scenes/messy_desk_demo.json \
  --headless
```

预期输出：

```text
assets/isaac_scenes/messy_desk_demo/messy_desk_demo.usd
```

### 13.5 打开检查

用 Isaac Sim GUI 打开 USD，检查：

- 物体是否都出现。
- 物体方向是否正确。
- 尺寸是否合理。
- 是否存在穿透桌面。
- 相机是否能看到目标区域。
- 灯光是否过暗或过曝。
- 物理播放后物体是否稳定。

### 13.6 保存预览图和 manifest

建议每个指定场景都保存一个 manifest：

```json
{
  "scene_name": "messy_desk_demo",
  "usd": "assets/isaac_scenes/messy_desk_demo/messy_desk_demo.usd",
  "objects": [
    {
      "name": "notebook_0",
      "asset": "assets/objects/092_notebook/visual/base0.glb",
      "pose": [-0.20, -0.05, 0.77, 90.0, 0.0, 18.0]
    }
  ],
  "camera": {
    "position": [0.35, -1.05, 1.35],
    "target": [0.0, 0.0, 0.82]
  }
}
```

推荐输出结构：

```text
assets/isaac_scenes/messy_desk_demo/
├── messy_desk_demo.usd
├── manifest.json
└── preview.png
```

## 14. 质量检查清单

新增 `assets/objects` 物品时，至少确认：

- 目录名是 `NNN_name`。
- `visual/base0.glb` 可以被 trimesh 和 SAPIEN 读取。
- `collision/base0.glb` 闭合、低复杂度、尺寸合理。
- `model_data0.json` 包含 `center`、`extents`、`scale`、`stable`。
- `stable` 是 `true`。
- `points_info.json` 存在。
- `create_actor(..., convex=True)` 能成功创建。
- 仿真 250 步后物体没有飞走或穿地。
- 如果要用于 messy desk，已更新 `image_collectors/object_asset_map.json`。
- 如果是同类替代资产，已评估是否需要更新 `assets/objects/same.json`。

生成 `image_collectors/scenes` 随机桌面场景时，至少确认：

- `messy_desk.json` 包含 `objects`、`co-occurence`、`relationship`。
- `object_asset_map.json` 覆盖了准备采样的语义类别。
- 生成命令在 Docker 中执行，SAPIEN 渲染正常。
- `manifest.json` 中 `summary.rendered_scenes` 等于预期数量。
- `manifest.json` 中 `failures` 为空，或失败 scene 不会被使用。
- 每个可用 scene 的 `placed_objects` 数量足够。
- 每个可用 scene 至少有一个 `on_relationships`。
- 每个可用 scene 有 `occlusion` 记录。
- 图片中目标物体没有完全不可见。

构建并采集新 task 时，至少确认：

- `envs/<task>.py` 文件名和 class 名完全一致。
- `task_config/<task>.yml` 存在，并且 `embodiment`、`camera`、`data_type` 合理。
- `description/task_instruction/<task>.json` 存在。
- `self.info["info"]` 的占位符和 instruction JSON 模板一致。
- 固定 messy scene task 已关闭 `domain_randomization.cluttered_table`。
- 非目标 clutter 已设置为静态或固定 root，并加入 prohibit area。
- 目标物体是动态物体，可以被抓起。
- `script/check_sim.py --setup` 能通过。
- 正式采集前已备份或移走旧 `data/<task>/<config>/`。
- HDF5、video、traj、instruction 数量和 `episode_num` 一致。

搭建 Isaac Sim 指定场景时，至少确认：

- 当前运行环境能通过 `script/smoke_isaacsim.py`。
- USD stage 是 meters + Z-up。
- TableSim GLB 已做 Y-up 到 Z-up 姿态修正。
- 所有引用路径在目标运行环境中可访问。
- 桌子、物体、相机、灯光都写入 USD。
- 物体初始位置没有明显穿透。
- 物理播放后场景稳定。
- 输出了 USD、manifest 和预览图。

## 15. 常见问题

### 15.1 新资产没有被 clutter 发现

优先检查：

- 目录名是否满足 `^[0-9]{3}_`。
- 是否有 `model_data0.json`。
- `model_data0.json` 是否有 `center` 和 `extents`。
- `stable` 是否为 `true`。

### 15.2 SAPIEN 加载时报 convex 或 mesh 错误

通常是碰撞网格问题：

- 网格不闭合。
- 面法线混乱。
- 顶点存在 NaN/Inf。
- 碰撞模型过细、过薄或自相交。

处理方式：

- 用更简单的盒体、圆柱或多凸包替代复杂碰撞。
- 视觉模型保留细节，碰撞模型做近似。

### 15.3 图片里物体方向不对

编号 GLB 资产是局部 Y-up。SAPIEN clutter 会统一旋转，Isaac Sim USD 里也需要
显式做姿态修正。先检查是否漏了约 `90` 度的轴向转换。

### 15.4 `scence_generator.py` 生成 scene 失败

优先检查：

- 是否有足够资产映射到 `object_asset_map.json`。
- 是否有大物体、房间物体或不可桌面放置物体进入候选池。
- `--min-placed` 是否设置过高。
- `relationship` 中的 `on` 关系是否过少或对应资产缺失。
- 遮挡 pair 中的两个对象是否都能被采样到并成功放置。

快速调试可临时加：

```bash
python script/scence_generator.py --scenes 1 --fail-fast
```

如果只是验证渲染链路，可以临时加 `--allow-missing-on` 和
`--allow-missing-occlusion`，但正式生成 task 场景不建议放宽。

### 15.5 `collect_data` 没有重新生成数据

`collect_data.py` 会复用已有 `seed.txt`，并跳过已存在的
`data/episodeN.hdf5`。如果你改了 scene、相机、动作或成功条件，应先备份旧目录：

```bash
mv data/<task>/<config> data/<task>/<config>_backup_$(date +%Y%m%d_%H%M%S)
```

然后重新运行：

```bash
bash collect_data.sh <task> <config> 0
```

### 15.6 新 task 指令生成失败

常见原因是 `description/task_instruction/<task>.json` 中的占位符和
`self.info["info"]` 不一致。处理方式：

- JSON 模板用了 `{A}`，`self.info["info"]` 就必须提供 `{A}`。
- JSON 模板用了 `{a}`，`self.info["info"]` 就必须提供 `{a}`。
- 不要在模板里保留 `{B}`、`{C}` 等未提供字段。
- `task_config/<task>.yml` 中的 `language_num` 不应小于 1。
- 如果占位符值含 `/`，例如 `015_laptop/10125`，必须存在
  `description/objects_description/015_laptop/10125.json`。
- `generate_episode_instructions.py` 要求模板中的非 arm 占位符集合与
  `self.info["info"]` 完全一致；不要用“`{H}` 到 `{L}`”代替中间的
  `{I}`、`{J}`、`{K}`。

### 15.7 Isaac Sim smoke test 失败

这说明当前 Python 环境没有完整 Isaac Sim。TableSim Docker 主要为 SAPIEN 数据
采集准备，不保证包含 Isaac Sim。需要换到已安装 Isaac Sim 的环境后，再运行
`script/build_isaac_scene.py`。

### 15.8 物体在 Isaac Sim 中有视觉但没有碰撞

GLB 引用不等于一定有可用物理碰撞。建议显式添加 `UsdPhysics.CollisionAPI`，并
根据 Isaac Sim 版本选择 convex hull、convex decomposition 或单独 collision mesh。
