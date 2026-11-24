# 可夹取文件夹与带支脚键盘

数据集映射现在统一使用下面两套新资产，旧的 `122_folder`、
`116_keyboard` 不覆盖，历史 manifest 和固定任务不会被悄悄换模型。

| 类别 | 资产 | 尺寸与抓取空间 |
| --- | --- | --- |
| folder / file folder | `136_file-folder` | 竖放 A4 厚硬皮档案夹；厚 55 mm、高 310 mm、深 260 mm |
| keyboard | `137_keyboard-supported` | 宽 300 mm、深约 150 mm；机身倾斜 8°；前沿直接接桌，后沿底部抬高约 21 mm |

键盘只在后沿有两个支脚，前沿没有支脚、没有整体架空。侧视图中，
倾斜机身、后部支撑与桌面之间围出近似直角三角形的空隙。后沿支脚
与机身属于同一刚体，不会在搬运时遗落一个独立支架。
碰撞文件包含机身、键面简化体和两个独立的凸支脚，不能合并为包住
底部空隙的单一凸包。短侧中央空隙约 10 mm，候选抓取位置移至靠后
区域，底部空隙约 15 mm。前沿不能插入下夹指，不能继续沿用此前的
前侧抓取假设。侧边上下夹持约 20 mm 厚的机身与键帽，不是用夹爪
跨过整个 150 mm 深度。

文件夹包含厚封面、填充纸芯、脊背标签与装饰孔。碰撞模型把关闭且
装满文件的档案夹视作实体；脊背装饰孔不作为穿孔抓取位置。

## 服务器生成与验收

在仓库根目录激活 `.venv-sim` 后：

```bash
python -m pip install 'trimesh>=4.4,<5' Pillow
python script/generate_messy_desk_assets.py \
  --only 136_file-folder \
  --only 137_keyboard-supported --force
```

两套资产的 GLB 不进入 Git：同步生成脚本和类别映射后，必须在服务器
执行上面的生成命令，再启动场景生成器。只复制 JSON 不会生成网格。
`--force` 兼容 Git 已同步模型 JSON 的情况，仅覆盖生成器负责的文件，
不删除其他文件。
不要省略 `--only`，以免重新生成无关资产。

只渲染这两个模型检查外观：

```bash
python script/render_messy_desk_objects.py \
  --only 136_file-folder \
  --only 137_keyboard-supported \
  --resolution 512 --poses 3
```

预览会放在 `image_collectors/objects/`。随后在全新输出目录生成少量
场景检查相对尺寸、遮挡和摆放，旧图片与旧 manifest 不会自动更新。

本机无 SAPIEN 时，可追加 `--preview-dir image_collectors/graspable_asset_preview`
生成 CPU 网格预览，左侧显示外观与内嵌贴图，右侧显示碰撞几何；它不是
仿真渲染。模型数据与碰撞文件均由同一生成脚本输出，避免手工同步失配。

## 验证边界

生成器写入与模型坐标一致的候选 `contact_points_pose`，文件夹建议
开口 65 mm，键盘建议开口 30 mm。它们是候选接触位姿，不能保证任何
夹爪和规划器都能成功；仍需检查开口、指长、碰撞、摩擦、质量与可达性。
`grasp_design.physics_grasp_verified` 明确为 `false`。

当前本机未安装 SAPIEN，CPU 测试验证了尺度、闭合碰撞体、支脚接地、
支脚与机身相连、短侧夹指空间，以及导出和重新加载后的几何一致性。
真实夹取成功率需要服务器进行动态物理测试，不能把静态场景渲染成功
等同于抓取成功。固定使用旧资产的 `envs` 任务仍需显式选用新资产，
并以 `create_actor(..., convex=True, is_static=False)` 加载可移动刚体；
数据集图片生成器保持静态物体，不会因增加抓取空间自动变成动态任务。
