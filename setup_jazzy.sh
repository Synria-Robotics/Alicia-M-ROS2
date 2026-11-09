#!/usr/bin/env bash
set -eo pipefail

workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS 2 Jazzy is not installed at /opt/ros/jazzy." >&2
  exit 1
fi

source /opt/ros/jazzy/setup.bash
set -u
cd "$workspace_dir"

if [[ ! -d /etc/ros/rosdep/sources.list.d ]]; then
  echo "rosdep is not initialized. Run: sudo rosdep init" >&2
  exit 1
fi

rosdep update
rosdep install --from-paths . --ignore-src -r -y --rosdistro jazzy

# This host also has a user-local Python. Force the Ubuntu Python that owns
# the ROS apt modules (catkin_pkg, ament, rclpy, cv_bridge).
colcon build \
  --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

echo
echo "Build complete. Run: source $workspace_dir/install/setup.bash"
