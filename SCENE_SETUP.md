# Task Scene Setup

## Messy Desk

### Step1: Statistics of Key Elements

scrape images from google -> an extractor: analize images -> write a report with markdown 

1. occurence frequency and proportion of objects
2. co-occurence combinations of high-frequency objects
3. object relationships
4. high-frequency connection relatinships

- images: 259/300
- vision model: `Qwen2.5-VL-3B-Instruct`
- the element table only shows categoried that appear at least twice



# ALOHA 顶部相机位置在这里改：

  assets/embodiments/aloha-agilex/config.yml:27

  我修改的是 static_camera_list -> head_camera：

  position:

- -0.032
- -0.45
- 2.2
  forward:
- 0.03
- 0.45
- -1.45

  变化：

- 原高度 z=1.65
- 现高度 z=2.20
- 同步把 forward.z 从 -0.9 改成 -1.45，让相机升高后仍然看向桌面中心附近，而不
  是平视/偏离桌面。

## 实际读取这个配置的位置在：

  envs/camera/camera.py:92

  这里会用 position / forward / left 创建 head_camera。

  另外区分一下：

- 相机位置/朝向：assets/embodiments/aloha-agilex/config.yml
- 相机分辨率/FOV/type：task_config/_camera_config.yml
- 是否采集 head camera、是否随机扰动：各 task 的 task_config/*.yml 里的 camera
  和 domain_randomization.random_head_camera_dis

  我已在 docker 里验证配置能正常加载，并跑了：

  docker compose exec -T tablesim python script/check_sim.py --task
  pick_an_object_messy --config pick_an_object_messy --setup

  结果通过：Full simulation setup OK。
