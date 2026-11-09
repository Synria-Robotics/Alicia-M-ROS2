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
  serial_port:=/dev/ttyACM1 default_speed:=0.3
```

其他启动方式：

```bash
# 只启动硬件控制
ros2 launch alicia_m_bringup hardware.launch.py serial_port:=/dev/ttyACM1

# 使用 mock hardware 启动 MoveIt 仿真
ros2 launch alicia_m_bringup moveit_sim.launch.py
```

## RealSense D405 手眼标定

启动相机：

```bash
ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=false
```

保持相机和硬件 MoveIt 运行，在另一个终端执行：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch alicia_m_calibration hand_eye_calibration.launch.py \
  aruco_dict:=DICT_4X4_50 marker_id:=0 marker_size:=0.0765
```

标定结果保存到：

```text
alicia_m_calibration/config/hand_eye_calibration_result.yaml
```

验证结果：

```bash
ros2 launch alicia_m_calibration verify_calibration.launch.py \
  calibration_file:="$PWD/alicia_m_calibration/config/hand_eye_calibration_result.yaml"

ros2 run tf2_ros tf2_echo base_link aruco_marker_frame
```

当前实机配置中，机械臂底座和 ArUco 位于同一桌面，验证得到的标记高度约为
`z = 0.002 m`。
