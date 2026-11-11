# Alicia-M-ROS2

[![ROS 2](https://img.shields.io/badge/ROS%202-Jazzy-blue.svg)](https://docs.ros.org/en/jazzy/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-24.04-E95420.svg)](https://releases.ubuntu.com/24.04/)

## 项目简介

本分支 `jazzy` 面向 **Ubuntu 24.04、ROS 2 Jazzy 和 Alicia-M v1.2**，提供：

- `ros2_control` 串口硬件驱动
- MoveIt 2 规划与轨迹执行
- Alicia-M v1.2 URDF、mesh 与碰撞配置
- RealSense D405 眼在手上手眼标定
- ArUco 检测和标定结果验证
- MoveIt 示例程序

Ubuntu 22.04 / ROS 2 Humble 版本请使用项目的 `main` 分支。

## 工作区结构

```text
alicia_m_bringup/          # 硬件、MoveIt 和仿真启动入口
alicia_m_calibration/      # 手眼标定、ArUco 检测与验证
alicia_m_descriptions/     # Alicia-M v1.2 URDF、mesh 和 RViz 配置
alicia_m_driver/           # ros2_control 串口硬件驱动
alicia_m_grasp_6d/         # 6D 抓取模块（开发中）
alicia_m_moveit_config/    # MoveIt 2 与控制器配置
examples/                  # 示例程序
setup_jazzy.sh             # Jazzy 依赖安装与构建脚本
```

## 安装

### 1. 获取 Jazzy 分支

```bash
git clone -b jazzy git@github.com:Synria-Robotics/Alicia-M-ROS2.git
cd Alicia-M-ROS2
```

### 2. 安装依赖并构建

系统需要预先安装 ROS 2 Jazzy。首次使用 `rosdep` 时执行：

```bash
sudo rosdep init
```

随后在仓库根目录运行：

```bash
./setup_jazzy.sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

脚本会运行 `rosdep update`、安装 Jazzy 依赖，并使用系统 Python 完成
`colcon build --symlink-install`。

## 启动机械臂

确认当前用户属于 `dialout` 组；如果不是，请执行以下命令并重新登录：

```bash
sudo usermod -aG dialout "$USER"
```

检查串口：

```bash
ls -l /dev/ttyACM*
```

启动硬件和 MoveIt，串口名称请按实际设备修改：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch alicia_m_bringup moveit_hardware.launch.py \
  serial_port:=/dev/ttyACM0 default_speed:=0.3
```

其他启动方式：

```bash
# 只启动硬件控制
ros2 launch alicia_m_bringup hardware.launch.py serial_port:=/dev/ttyACM0

# 使用 mock hardware 启动 MoveIt 仿真
ros2 launch alicia_m_bringup moveit_sim.launch.py
```

## RealSense D405 手眼标定

标定参数位于 `alicia_m_calibration/config/`。固定外部相机的默认参数为
`marker_id=0`、`marker_size=0.0765 m`、18 个样本和 Park 方法。

以 PV 模式启动机械臂与 MoveIt：

```bash
ros2 launch alicia_m_bringup moveit_hardware.launch.py \
  serial_port:=/dev/ttyACM0 control_mode:=pv
```

启动固定的 D405；eye-to-hand 使用相机名称 `cam1`：

```bash
ros2 launch realsense2_camera rs_launch.py \
  camera_name:=cam1 enable_depth:=false enable_infra1:=false enable_infra2:=false
```

运行标定：

```bash
# 相机安装在末端
ros2 launch alicia_m_calibration calibrate.launch.py \
  mode:=eye_in_hand camera_name:=cam1 calibration_method:=daniilidis

# 相机固定在机械臂外部
ros2 launch alicia_m_calibration calibrate.launch.py \
  mode:=eye_to_hand camera_name:=cam1 calibration_method:=park
```

两个模式都只配置一个六关节起始姿态，程序会根据 `min_samples` 自动生成
单关节与组合扰动姿态，并预留检测失败后的补采姿态。可按实际安装覆盖起始姿态：

```bash
ros2 launch alicia_m_calibration calibrate.launch.py \
  mode:=eye_in_hand camera_name:=cam1 min_samples:=15 \
  calibration_method:=daniilidis \
  'reference_pose:=[0.0,-1.09,-0.87,0.0,-0.77,0.0]'
```

高级用法可通过 `pose_amplitudes:=[...]` 调整六个关节相对起始姿态的最大扰动。

Eye-to-hand 应使用具有充分旋转变化、且 marker 始终完整位于画面内的机器人姿态。
标定程序会等待关节实际到达目标后再采样，并输出各求解方法的平移与旋转残差。

### 在 MoveIt 中验证 eye-to-hand

以下命令一次启动 `/dev/ttyACM0` PV 硬件、MoveIt、RViz、D405 `cam1` 和验证节点：

```bash
ros2 launch alicia_m_calibration verify_calibration.launch.py
```

若硬件或相机已经启动，可分别设置 `start_robot:=false` 或
`start_camera:=false`。仅启动验证节点：

```bash
ros2 launch alicia_m_calibration verify_calibration.launch.py \
  start_robot:=false start_camera:=false
```

验证节点组合标定的 `base_link -> camera optical frame` 与实时检测的
`camera optical frame -> ArUco`，发布：

- TF：`base_link -> aruco_marker_frame`
- MoveIt CollisionObject：`detected_aruco_marker`
- Topic：`/collision_object`

检查实时坐标和 MoveIt 对象：

```bash
ros2 run tf2_ros tf2_echo base_link aruco_marker_frame
ros2 topic echo /collision_object moveit_msgs/msg/CollisionObject --once
```

当前桌面布局下，marker 的期望位置约为
`[x, y, z] = [0.300, -0.005, 0.002] m`，数毫米误差属于正常范围。
