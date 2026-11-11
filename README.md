# Alicia-M-ROS2

[![ROS 2](https://img.shields.io/badge/ROS%202-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420.svg)](https://releases.ubuntu.com/22.04/)

## 项目简介

当前 `main` 分支面向 **Ubuntu 22.04 和 ROS 2 Humble**，为 **Alicia-M v1.2** 提供：

- `ros2_control` 串口硬件驱动
- MoveIt 2 规划与轨迹执行
- Alicia-M v1.2 URDF、mesh 与碰撞配置
- Eye-in-hand 与 eye-to-hand 手眼标定
- ArUco 检测和标定结果验证
- MoveIt 示例程序

## 工作区结构

```text
alicia_m_bringup/          # 硬件、MoveIt 和仿真启动入口
alicia_m_calibration/      # 手眼标定、ArUco 检测与验证
alicia_m_descriptions/     # Alicia-M v1.2 URDF、mesh 和 RViz 配置
alicia_m_driver/           # ros2_control 串口硬件驱动
alicia_m_grasp_6d/         # 6D 抓取模块（开发中）
alicia_m_moveit_config/    # MoveIt 2 与控制器配置
examples/                  # 示例程序
```

## 安装

### 1. 获取 main 分支

```bash
git clone git@github.com:Synria-Robotics/Alicia-M-ROS2.git
cd Alicia-M-ROS2
```

### 2. 安装依赖并构建

在仓库根目录安装依赖并构建：

```bash
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths . --ignore-src -r -y --rosdistro humble
colcon build --symlink-install
source install/setup.bash
```

如果 `rosdep update` 提示尚未初始化，先执行 `sudo rosdep init`，然后重新执行
`rosdep update`。使用 Zsh 时，将上述两个 `setup.bash` 分别替换为 `setup.zsh`。

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
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch alicia_m_bringup moveit_hardware.launch.py \
  serial_port:=/dev/ttyACM0
```

其他启动方式：

```bash
# 只启动硬件控制
ros2 launch alicia_m_bringup hardware.launch.py serial_port:=/dev/ttyACM0

# 使用 mock hardware 启动 MoveIt 仿真
ros2 launch alicia_m_bringup moveit_sim.launch.py
```

## RealSense D405 手眼标定

标定参数位于 `alicia_m_calibration/config/`。当前标定板参数为
`marker_id=0`、`marker_size=0.0765 m`；eye-in-hand 默认采集 18 个有效样本。
求解方法必须显式指定，推荐 eye-in-hand 使用 `daniilidis`，eye-to-hand 使用 `park`。

标定会自动规划并执行多个机械臂姿态。运行前必须清空机械臂工作空间、确认参考姿态和
扰动范围不会碰撞，并确保急停可触达。不要在有人进入机械臂运动范围时运行标定。

### 1. 启动机械臂和相机

以 PV 模式启动机械臂与 MoveIt：

```bash
ros2 launch alicia_m_bringup moveit_hardware.launch.py \
  serial_port:=/dev/ttyACM0 control_mode:=pv
```

启动 D405。相机名称是可选参数；单相机环境可直接使用 RealSense 默认名称：

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_depth:=false enable_infra1:=false enable_infra2:=false
```

如果启动相机时显式设置了名称，例如 `camera_name:=cam1`，标定和验证命令也可传入
相同的 `camera_name:=cam1`。在只有一个正在发布彩色图像的相机时，省略
`camera_name` 会自动发现该相机；多相机环境必须指定 `camera_name`，或者直接传入
`camera_topic` 和 `camera_info_topic`。

### 2. 运行标定

Eye-in-hand，相机安装在机械臂末端；默认单相机环境无需传 `camera_name`：

```bash
ros2 launch alicia_m_calibration calibrate.launch.py \
  mode:=eye_in_hand \
  marker_id:=0 \
  marker_size:=0.0765 \
  min_samples:=18 \
  calibration_method:=daniilidis
```

如果相机使用非默认名称：

```bash
ros2 launch alicia_m_calibration calibrate.launch.py \
  mode:=eye_in_hand camera_name:=cam1 \
  marker_id:=0 marker_size:=0.0765 min_samples:=18 \
  calibration_method:=daniilidis
```

Eye-to-hand，相机固定在机械臂外部，ArUco marker 安装在机械臂末端：

```bash
ros2 launch alicia_m_calibration calibrate.launch.py \
  mode:=eye_to_hand min_samples:=18 calibration_method:=park
```

两个模式都只配置一个六关节参考姿态。程序根据 `min_samples` 自动生成单关节与组合
扰动姿态，并预留 marker 检测失败后的补采姿态。可按实际安装覆盖参考姿态：

```bash
ros2 launch alicia_m_calibration calibrate.launch.py \
  mode:=eye_in_hand min_samples:=18 calibration_method:=daniilidis \
  'reference_pose:=[0.0,-1.09,-0.87,0.0,-0.77,0.0]'
```

高级用法可通过 `pose_amplitudes:=[...]` 调整六个关节相对参考姿态的最大扰动。
标定程序会等待关节实际到达目标后再采样；只有达到 `min_samples` 才会覆盖结果文件。
结果分别保存到：

- Eye-in-hand：`alicia_m_calibration/config/hand_eye_calibration_result.yaml`
- Eye-to-hand：`alicia_m_calibration/config/eye_to_hand_calibration_result.yaml`

### 3. 结果单位

标定结果统一使用以下单位：

- `marker_size`、`distance`、`transform.translation`、矩阵最后一列和所有 `*_xyz_m`：米（m）
- `translation_rmse_m`：米（m），换算为毫米时乘以 1000
- `euler_xyz_deg` 和 `rotation_rmse_deg`：度（°）
- `quaternion`：无量纲，字段顺序为 `x, y, z, w`

例如 `translation_rmse_m: 0.001279` 表示 `1.279 mm`，
`tool_camera_xyz_m` 中的 `z=-0.0717` 表示 `-71.7 mm`。重复标定得到毫米级差异属于
测量重复性差异，不表示单位发生变化。结果文件同时保存显式的 `units` 元数据。

### 4. 验证 eye-in-hand

如果机械臂、MoveIt 和 D405 尚未启动，可用一个命令启动完整验证环境：

```bash
ros2 launch alicia_m_calibration verify_calibration.launch.py \
  mode:=eye_in_hand serial_port:=/dev/ttyACM0
```

如果机械臂、MoveIt 和 D405 已经运行，只启动验证节点即可。单相机环境无需指定
相机名称：

```bash
ros2 launch alicia_m_calibration verify.launch.py mode:=eye_in_hand
```

如果 D405 使用 `camera_name:=cam1` 启动，则执行：

```bash
ros2 launch alicia_m_calibration verify.launch.py \
  mode:=eye_in_hand camera_name:=cam1
```

验证节点读取 `config/hand_eye_calibration_result.yaml`，发布标定得到的
`tool0 -> <camera_link>` 静态 TF，并将实时 ArUco 检测转换为：

- TF：`base_link -> aruco_marker_frame`
- MoveIt CollisionObject：`detected_aruco_marker`
- Topic：`/collision_object`

检查标定 TF、marker 的实时坐标以及 MoveIt 对象：

```bash
# 查看验证节点日志中的 Published tool0 -> <camera_link>，使用实际 frame 名称
# 默认相机名通常对应 camera_link
ros2 run tf2_ros tf2_echo tool0 camera_link

# camera_name:=cam1 通常对应 cam1_link
ros2 run tf2_ros tf2_echo tool0 cam1_link

# marker 在机器人基座坐标系中的实时位姿
ros2 run tf2_ros tf2_echo base_link aruco_marker_frame

# marker 是否已发布到 MoveIt planning scene
ros2 topic echo /collision_object moveit_msgs/msg/CollisionObject --once
```

Eye-in-hand 验证的核心判据是：保持 ArUco marker 固定，安全地改变机械臂姿态后，
`base_link -> aruco_marker_frame` 应基本保持不变。应比较多个姿态下的位置离散程度，
而不是只检查单个姿态；平移波动应为毫米级，旋转波动通常应小于约 1°。

### 5. 验证 eye-to-hand

以下命令可一次启动 `/dev/ttyACM0` PV 硬件、MoveIt、RViz、D405 和验证节点。
Eye-to-hand 的集成启动默认使用相机名称 `cam1`：

```bash
ros2 launch alicia_m_calibration verify_calibration.launch.py mode:=eye_to_hand
```

若硬件或相机已经启动，可分别设置 `start_robot:=false` 或
`start_camera:=false`。仅启动验证节点：

```bash
ros2 launch alicia_m_calibration verify_calibration.launch.py \
  mode:=eye_to_hand start_robot:=false start_camera:=false
```

Eye-to-hand 验证时相机固定在基座外部，移动机械臂上的 marker 后，检测到的
`base_link -> aruco_marker_frame` 应随实际 marker 运动，并持续发布到 MoveIt 场景。
