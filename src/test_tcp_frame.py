# -*- coding: utf-8 -*-
"""测试Franka Panda的TCP坐标系"""
import pybullet as p
import pybullet_data
import numpy as np
import time

# 连接PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# 加载机械臂
robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

# 等待稳定
for _ in range(100):
    p.stepSimulation()
    time.sleep(1./240.)

# 获取末端执行器状态
ee_link = 11
ee_state = p.getLinkState(robot_id, ee_link)
ee_pos = ee_state[0]
ee_ori = ee_state[1]

print(f"末端执行器位置: {ee_pos}")
print(f"末端执行器四元数: {ee_ori}")

# 转换为旋转矩阵
rot_matrix = p.getMatrixFromQuaternion(ee_ori)
rot_matrix = np.array(rot_matrix).reshape(3, 3)

print(f"\n旋转矩阵:")
print(f"X轴（红色）: {rot_matrix[:, 0]}")
print(f"Y轴（绿色）: {rot_matrix[:, 1]}")
print(f"Z轴（蓝色）: {rot_matrix[:, 2]}")

# 可视化默认TCP坐标系
axis_length = 0.2
p.addUserDebugLine(ee_pos, ee_pos + rot_matrix[:, 0] * axis_length, [1, 0, 0], 5, 0)  # X-红
p.addUserDebugLine(ee_pos, ee_pos + rot_matrix[:, 1] * axis_length, [0, 1, 0], 5, 0)  # Y-绿
p.addUserDebugLine(ee_pos, ee_pos + rot_matrix[:, 2] * axis_length, [0, 0, 1], 5, 0)  # Z-蓝
p.addUserDebugText("默认TCP", ee_pos + np.array([0, 0, 0.1]), [1, 1, 1], 2, 0)

print("\n观察PyBullet GUI中的坐标系：")
print("- 红色 = X轴")
print("- 绿色 = Y轴")  
print("- 蓝色 = Z轴")

# 测试不同的抓取姿态
print("\n\n测试不同的欧拉角组合（让Z轴朝下）：")

test_eulers = [
    ([0, 0, 0], "默认姿态"),
    ([np.pi, 0, 0], "[π, 0, 0] - 绕X旋转180°"),
    ([0, np.pi, 0], "[0, π, 0] - 绕Y旋转180°"),
    ([0, 0, np.pi], "[0, 0, π] - 绕Z旋转180°"),
    ([-np.pi/2, 0, 0], "[-π/2, 0, 0] - 绕X旋转-90°"),
    ([np.pi/2, 0, 0], "[π/2, 0, 0] - 绕X旋转90°"),
]

target_pos = [0.5, 0, 0.8]  # 测试目标位置

for i, (euler, desc) in enumerate(test_eulers):
    quat = p.getQuaternionFromEuler(euler)
    rot = p.getMatrixFromQuaternion(quat)
    rot = np.array(rot).reshape(3, 3)
    
    print(f"\n{i+1}. {desc}")
    print(f"   四元数: {quat}")
    print(f"   Z轴方向: {rot[:, 2]} (应该是[0, 0, -1]朝下)")
    
    # 可视化这个姿态
    offset = np.array([0, i * 0.15 - 0.3, 0])
    vis_pos = target_pos + offset
    axis_len = 0.08
    p.addUserDebugLine(vis_pos, vis_pos + rot[:, 0] * axis_len, [1, 0.5, 0.5], 2, 0)
    p.addUserDebugLine(vis_pos, vis_pos + rot[:, 1] * axis_len, [0.5, 1, 0.5], 2, 0)
    p.addUserDebugLine(vis_pos, vis_pos + rot[:, 2] * axis_len, [0.5, 0.5, 1], 2, 0)
    p.addUserDebugText(f"{i+1}", vis_pos + np.array([0, 0, 0.1]), [1, 1, 0], 1, 0)

print("\n在PyBullet GUI中观察6个测试姿态，看哪个的Z轴（浅蓝色）朝下")
print("按 Ctrl+C 退出...")

try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    p.disconnect()
