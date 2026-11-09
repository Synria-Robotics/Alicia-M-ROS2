# Alicia-M-ROS2

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)

## 项目简介 | Project & Feature Overview

`Alicia-M-ROS2` 是用于控制 Alicia-M 系列机械臂的 ROS2 工作区。项目基于 ROS 2 Humble，提供从底层驱动到上层规划的完整控制链路，覆盖硬件启动、MoveIt 运动规划、模型可视化与手眼标定等核心能力。

主要功能：

- `ros2_control` 硬件驱动集成（串口通信）
- MoveIt2 规划与执行支持
- 机械臂模型显示与 RViz 可视化
- 手眼标定（眼在手上，适配intel realsense d405）与标定结果验证
- 固定外部相机（eye-to-hand）标定
- Python 脚本示例（如 Pick and Place）
- (待开发)物块分拣工作流（2D抓取）
- (待开发)基于 GraspGen（NVIDIA 2025）的 6D 抓取工作流。

## 项目结构

```text
src/
├── alicia_m_bringup/          # 机械臂硬件与 MoveIt 启动入口
├── alicia_m_calibration/      # 手眼标定与验证脚本
├── alicia_m_descriptions/     # URDF、mesh、RViz 显示配置
├── alicia_m_driver/           # ros2_control 硬件驱动（串口通信）
├── alicia_m_grasp_6d/         # 6D 抓取模块（待开发）
├── alicia_m_moveit_config/    # MoveIt2 配置与控制器配置
└── examples/                  # MoveIt 示例脚本
```

## 安装方式

### 1. 环境要求

- Ubuntu 22.04
- ROS2 Humble

### 2. 依赖安装

```bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

### 3. 编译工作区

在工作区根目录（包含 `src/`）执行：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 基本使用（启动机械臂）

设置临时串口权限：

```bash
sudo chmod 666 /dev/ttyACM*
```

加载工作区环境：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```



### Option 1. 启动硬件 + MoveIt（推荐）

```bash
ros2 launch alicia_m_bringup moveit_hardware.launch.py
```

### Option 2. 单独启动硬件控制

```bash
ros2 launch alicia_m_bringup hardware.launch.py
```

### Option 3. 启动仿真 + MoveIt

```bash
ros2 launch alicia_m_bringup moveit_sim.launch.py
```

## 手眼标定

标定参数位于 `alicia_m_calibration/config/`。以下命令均假定工作区已经编译，并且每个新终端都已加载环境：

```bash
cd /home/ubuntu/Alicia/Alicia-M-ROS2
source /opt/ros/humble/setup.bash
source install/setup.bash
```

先使用实际串口和 PV 模式启动机械臂、MoveIt 与 RViz：

```bash
ros2 launch alicia_m_bringup moveit_hardware.launch.py \
  serial_port:=/dev/ttyACM3 \
  control_mode:=pv
```

在另一个终端启动 RealSense D405。固定外部相机使用 `cam1`，对应图像 topic 为 `/camera/cam1/color/image_raw`：

```bash
ros2 launch realsense2_camera rs_launch.py \
  camera_name:=cam1 \
  enable_depth:=false \
  enable_infra1:=false \
  enable_infra2:=false
```

然后在第三个终端运行标定：

```bash
# Eye-in-hand：相机安装在机械臂末端
ros2 launch alicia_m_calibration calibrate.launch.py mode:=eye_in_hand

# Eye-to-hand：相机固定在机械臂外部
ros2 launch alicia_m_calibration calibrate.launch.py mode:=eye_to_hand
```

标定板尺寸、ID、最少样本数和求解方法可在启动时覆盖：

```bash
ros2 launch alicia_m_calibration calibrate.launch.py \
  mode:=eye_in_hand \
  marker_size:=0.0765 \
  marker_id:=0 \
  min_samples:=15 \
  calibration_method:=daniilidis
```

当前标定板参数为 `marker_id=0`、`marker_size=0.0765 m`。Eye-to-hand 推荐至少采集 18 个有效样本，并使用具有充分旋转变化且能让 marker 完整留在画面内的机器人姿态。

### 在 MoveIt 中验证 eye-to-hand 结果（一条命令）

下面一条命令会同时启动 `/dev/ttyACM3` PV 硬件、MoveIt、RViz、RealSense D405 `cam1` 和验证节点，不需要分别打开多个终端：

```bash
cd /home/ubuntu/Alicia/Alicia-M-ROS2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch alicia_m_calibration verify_calibration.launch.py
```

默认参数已经是：

```text
mode:=eye_to_hand
serial_port:=/dev/ttyACM3
control_mode:=pv
marker_id:=0
marker_size:=0.0765
use_rviz:=true
```

如果硬件/MoveIt 或相机已经在运行，可以避免重复启动：

```bash
# 已经启动硬件、MoveIt 和 RViz
ros2 launch alicia_m_calibration verify_calibration.launch.py \
  start_robot:=false

# 已经启动 D405
ros2 launch alicia_m_calibration verify_calibration.launch.py \
  start_camera:=false

# 两者都已启动时，仅运行验证节点
ros2 launch alicia_m_calibration verify_calibration.launch.py \
  start_robot:=false \
  start_camera:=false
```

验证节点会对每一张新图像执行以下转换：

```text
base_link -> camera optical frame（标定结果）
camera optical frame -> ArUco（当前图像检测）
= base_link -> aruco_marker_frame
```

检测结果会同时发布为：

- TF：`base_link -> aruco_marker_frame`
- MoveIt world object topic：`/collision_object`
- CollisionObject ID：`detected_aruco_marker`
- 几何尺寸：`0.0765 x 0.0765 x 0.002 m`

在 RViz 中将 Fixed Frame 设置为 `base_link`，并在 MotionPlanning display 中启用 Scene Geometry，即可看到最新检测到的 marker。也可以用命令行确认发布内容：

```bash
ros2 topic echo /collision_object moveit_msgs/msg/CollisionObject --once
```

确认 MoveIt planning scene 已经接收该对象：

```bash
ros2 service call /get_planning_scene moveit_msgs/srv/GetPlanningScene \
  "{components: {components: 24}}" | grep detected_aruco_marker
```

查看实时检测坐标：

```bash
ros2 run tf2_ros tf2_echo base_link aruco_marker_frame
```

URDF 文件中保留了一段被注释的 `marker` link/joint，仅用于人工核对实体 marker 的预定义参考位姿。它不会进入运行时 robot model，验证节点也不会使用该参考位姿修正标定结果或在项目内计算误差。如需做绝对精度测试，可在临时测试环境中启用该参考 frame，再将其与 `aruco_marker_frame` 比较；不要把误差计算加入运行代码。

Eye-in-hand 结果也可以一条命令验证；launch 会自动将相机名切换为 `camera`：

```bash
ros2 launch alicia_m_calibration verify_calibration.launch.py \
  mode:=eye_in_hand
```
